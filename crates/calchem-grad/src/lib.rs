pub mod derivatives;

use calchem_basis::BasisSet;
use calchem_cuda::{
    cuda_grad4c_contract, is_cuda_available, Grad4cPrimPairGpu, Grad4cShellGpu, Grad4cShellPairGpu,
};
use calchem_dft::{xc_nuclear_gradient_with_gpu, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::primitive::GaussianPair;
use calchem_int::screening::SchwarzScreening;
use calchem_model::{CalcError, Molecule, ScfWavefunction, UhfWavefunction};
use calchem_tensor::Matrix2D;
use derivatives::{nuclear_repulsion_derivative_with_basis, one_electron_derivative};
use rayon::prelude::*;

/// Analytical nuclear gradient result with GPU kernel and fallback telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientResult {
    pub gradient: Vec<[f64; 3]>,
    pub gpu_kernel_invocations: usize,
    pub gpu_fallback_invocations: usize,
}

#[inline]
pub fn distinct_shell_permutations(i: usize, j: usize, k: usize, l: usize) -> Vec<[usize; 4]> {
    let raw = [
        [i, j, k, l],
        [j, i, k, l],
        [i, j, l, k],
        [j, i, l, k],
        [k, l, i, j],
        [l, k, i, j],
        [k, l, j, i],
        [l, k, j, i],
    ];
    let mut out: Vec<[usize; 4]> = Vec::with_capacity(8);
    for p in raw {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Validates that coordinates are finite and the basis shells sit on the molecule's nuclei.
fn check_geometry(mol: &Molecule, basis: &BasisSet) -> Result<(), CalcError> {
    if mol.atoms.is_empty() {
        return Err(CalcError::InvalidMolecule("No atoms in molecule".into()));
    }
    for atom in &mol.atoms {
        if !atom.position_bohr.iter().all(|c| c.is_finite()) {
            return Err(CalcError::InvalidMolecule(
                "Non-finite coordinates detected in molecule".into(),
            ));
        }
    }
    for shell in &basis.shells {
        let atom_pos = mol
            .atoms
            .get(shell.atom_index)
            .ok_or_else(|| {
                CalcError::InvalidMolecule(format!(
                    "Basis shell atom_index {} out of bounds for molecule with {} atoms",
                    shell.atom_index,
                    mol.atoms.len()
                ))
            })?
            .position_bohr;
        for c in 0..3 {
            if (shell.center[c] - atom_pos[c]).abs() > 1e-10 {
                return Err(CalcError::InvalidMolecule(format!(
                    "Basis shell center mismatch on atom {}: shell center {:?} vs atom pos {:?}",
                    shell.atom_index, shell.center, atom_pos
                )));
            }
        }
    }
    for ecp in &basis.ecp_potentials {
        let atom_pos = mol
            .atoms
            .get(ecp.atom_index)
            .ok_or_else(|| {
                CalcError::InvalidMolecule(format!(
                    "ECP atom_index {} out of bounds for molecule with {} atoms",
                    ecp.atom_index,
                    mol.atoms.len()
                ))
            })?
            .position_bohr;
        for c in 0..3 {
            if (ecp.center[c] - atom_pos[c]).abs() > 1e-10 {
                return Err(CalcError::InvalidMolecule(format!(
                    "ECP center mismatch on atom {}: ecp center {:?} vs atom pos {:?}",
                    ecp.atom_index, ecp.center, atom_pos
                )));
            }
        }
    }
    Ok(())
}

fn to_matrix(flat: &[f64], n: usize) -> Matrix2D {
    let mut m = Matrix2D::zeros(n, n);
    m.data.copy_from_slice(&flat[..n * n]);
    m
}

/// `W = sum_i occ_i eps_i C_{mu i} C_{nu i}` using per-MO occupations when provided,
/// or uniform `occ` over the first `n_occ` columns when `mo_occupations` is empty.
fn energy_weighted_density_with_occupations(
    c: &[f64],
    eps: &[f64],
    mo_occupations: &[f64],
    n: usize,
    n_occ: usize,
    occ: f64,
) -> Matrix2D {
    let mut w = Matrix2D::zeros(n, n);
    if mo_occupations.len() == n && eps.len() >= n && c.len() >= n * n {
        for mu in 0..n {
            for nu in 0..n {
                let mut s = 0.0;
                for i in 0..n {
                    let o = mo_occupations[i];
                    if o.abs() > 1e-15 {
                        s += o * eps[i] * c[mu * n + i] * c[nu * n + i];
                    }
                }
                w.set(mu, nu, s);
            }
        }
    } else {
        for mu in 0..n {
            for nu in 0..n {
                let mut s = 0.0;
                for i in 0..n_occ {
                    s += eps[i] * c[mu * n + i] * c[nu * n + i];
                }
                w.set(mu, nu, occ * s);
            }
        }
    }
    w
}

/// `W = sum_i eps_i C_{mu i} C_{nu i}` over the first `n_occ` columns, scaled by `occ`.
fn energy_weighted_density(c: &[f64], eps: &[f64], n: usize, n_occ: usize, occ: f64) -> Matrix2D {
    energy_weighted_density_with_occupations(c, eps, &[], n, n_occ, occ)
}

#[allow(clippy::too_many_arguments)]
fn try_gpu_grad4c(
    basis: &BasisSet,
    n_atoms: usize,
    pt: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    d_tot_max: &[f64],
    d_spin_max: &[f64],
    sig_pairs: &[(usize, usize, f64)],
    do_coul: bool,
    std_exx: f64,
    lr_exx: f64,
    lr_omega: Option<f64>,
    grad_thresh: f64,
) -> Option<(Vec<[f64; 3]>, usize)> {
    let need_pass1 = do_coul || std_exx.abs() > 1e-15;
    let need_pass2 = lr_omega.is_some() && lr_exx.abs() > 1e-15;
    if !need_pass1 && !need_pass2 {
        return Some((vec![[0.0; 3]; n_atoms], 0));
    }
    if sig_pairs.is_empty() {
        return Some((vec![[0.0; 3]; n_atoms], 0));
    }
    if !is_cuda_available()
        || basis
            .shells
            .iter()
            .any(|s| s.shell_type.angular_momentum() > 3)
    {
        return None;
    }

    let mut gpu_shells = Vec::with_capacity(basis.shells.len());
    for shell in &basis.shells {
        let l = shell.shell_type.angular_momentum();
        let lead_norm_1 = calchem_basis::primitive_norm(1.0, [l, 0, 0]);
        let mut comp_scales = [0.0f64; 10];
        for (idx, &lmn) in shell.shell_type.components().iter().enumerate() {
            comp_scales[idx] = calchem_basis::primitive_norm(1.0, lmn) / lead_norm_1;
        }
        gpu_shells.push(Grad4cShellGpu {
            center: shell.center,
            comp_scales,
            l: l as i32,
            ao_offset: shell.ao_offset as i32,
            n_funcs: shell.n_functions() as i32,
            atom_index: shell.atom_index as i32,
        });
    }

    let mut gpu_pairs = Vec::with_capacity(sig_pairs.len());
    let mut gpu_prims = Vec::new();
    let mut ends = Vec::with_capacity(sig_pairs.len());

    for (pi, &(i, j, bound)) in sig_pairs.iter().enumerate() {
        let sa = &basis.shells[i];
        let sb = &basis.shells[j];
        let la = sa.shell_type.angular_momentum();
        let lb = sb.shell_type.angular_momentum();
        let prim_pair_start = gpu_prims.len() as i32;
        let max_raw = sa.primitives.len() * sb.primitives.len();
        for (pa_idx, pa_prim) in sa.primitives.iter().enumerate() {
            let ca_lead = pa_prim.coeff * sa.primitive_norm_for(pa_idx, [la, 0, 0]);
            for (pb_idx, pb_prim) in sb.primitives.iter().enumerate() {
                let gp = GaussianPair::new(pa_prim.exponent, sa.center, pb_prim.exponent, sb.center);
                if gp.kab < 1e-18 && max_raw > 4 {
                    continue;
                }
                let cb_lead = pb_prim.coeff * sb.primitive_norm_for(pb_idx, [lb, 0, 0]);
                gpu_prims.push(Grad4cPrimPairGpu {
                    gamma: gp.gamma,
                    kab_w: gp.kab * ca_lead * cb_lead,
                    alpha: pa_prim.exponent,
                    beta: pb_prim.exponent,
                    p: gp.p,
                    pa: gp.pa,
                });
            }
        }
        let n_prim_pairs = (gpu_prims.len() as i32) - prim_pair_start;
        gpu_pairs.push(Grad4cShellPairGpu {
            shell_i: i as i32,
            shell_j: j as i32,
            bound,
            prim_pair_start,
            n_prim_pairs,
        });
        let end_q = sig_pairs[..=pi].partition_point(|&(_, _, q_kl)| bound * q_kl >= 1e-11);
        ends.push(end_q as i32);
    }

    if gpu_prims.is_empty() {
        return Some((vec![[0.0; 3]; n_atoms], 0));
    }

    let mut total = vec![[0.0; 3]; n_atoms];
    let mut kernels = 0usize;
    let mut buf = vec![0.0f64; n_atoms * 3];

    if need_pass1 {
        if !cuda_grad4c_contract(
            &gpu_shells,
            &gpu_pairs,
            &gpu_prims,
            &pt.data,
            &pa.data,
            &pb.data,
            d_tot_max,
            d_spin_max,
            &ends,
            basis.n_ao,
            do_coul,
            std_exx,
            None,
            grad_thresh,
            &mut buf,
        ) {
            return None;
        }
        for a in 0..n_atoms {
            for q in 0..3 {
                total[a][q] += buf[a * 3 + q];
            }
        }
        kernels += 1;
    }

    if need_pass2 {
        if !cuda_grad4c_contract(
            &gpu_shells,
            &gpu_pairs,
            &gpu_prims,
            &pt.data,
            &pa.data,
            &pb.data,
            d_tot_max,
            d_spin_max,
            &ends,
            basis.n_ao,
            false,
            lr_exx,
            lr_omega,
            grad_thresh,
            &mut buf,
        ) {
            return None;
        }
        for a in 0..n_atoms {
            for q in 0..3 {
                total[a][q] += buf[a * 3 + q];
            }
        }
        kernels += 1;
    }

    Some((total, kernels))
}

/// Shared gradient engine for RHF, UHF, RKS, and UKS (`1e + 2e + E_nn + Pulay`).
///
/// `dE/dX = dE_nn/dX + sum P^t dh/dX - sum W dS/dX + sum (mu nu|lam sig)^X Gamma_{mu nu lam sig}`
/// with `Gamma = 1/2 P^t_{mn} P^t_{ls} - c_HF / 2 * (P^a_{ml} P^a_{ns} + P^b_{ml} P^b_{ns})`.
#[allow(clippy::too_many_arguments)]
fn scf_gradient_core(
    mol: &Molecule,
    basis: &BasisSet,
    pt: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    w: &Matrix2D,
    exx_fraction: f64,
    range_sep: Option<(f64, f64, f64)>,
    density_fitting: bool,
    jfit_basis: calchem_model::JFitBasis,
    use_gpu: bool,
) -> Result<(Vec<[f64; 3]>, usize, usize), CalcError> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;
    let mut gpu_kernel_invocations = 0usize;
    let mut gpu_fallback_invocations = 0usize;

    let (std_exx, lr_exx, lr_omega) = match range_sep {
        Some((alpha, beta, omega)) => (alpha, beta, Some(omega)),
        None => (exx_fraction, 0.0, None),
    };

    let g_j_ri = if density_fitting {
        let rij = calchem_int::RiJEngine::with_basis_for_gradient(mol, basis, use_gpu, jfit_basis)?;
        let (g_ri, k_ri, fb_ri) = rij.compute_coulomb_gradient_with_telemetry(n_atoms, basis, pt)?;
        gpu_kernel_invocations += k_ri;
        gpu_fallback_invocations += fb_ri;
        Some(g_ri)
    } else {
        None
    };

    let need_4c = !density_fitting || std_exx.abs() > 1e-15 || lr_exx.abs() > 1e-15;

    let g_2e_total = if need_4c {
        let n_shells = basis.shells.len();
        let screening = SchwarzScreening::new(basis, 1e-11);
        let grad_thresh = 1e-12_f64;

        // Precompute shell-pair max density magnitudes for rigorous Coulomb & Exchange screening
        let mut d_tot_max = vec![0.0f64; n_shells * n_shells];
        let mut d_spin_max = vec![0.0f64; n_shells * n_shells];
        for i in 0..n_shells {
            let si = &basis.shells[i];
            for j in i..n_shells {
                let sj = &basis.shells[j];
                let mut mt = 0.0f64;
                let mut ms = 0.0f64;
                for a in si.ao_offset..si.ao_offset + si.n_functions() {
                    for b in sj.ao_offset..sj.ao_offset + sj.n_functions() {
                        mt = mt.max(pt.get(a, b).abs());
                        ms = ms.max(pa.get(a, b).abs()).max(pb.get(a, b).abs());
                    }
                }
                d_tot_max[i * n_shells + j] = mt;
                d_tot_max[j * n_shells + i] = mt;
                d_spin_max[i * n_shells + j] = ms;
                d_spin_max[j * n_shells + i] = ms;
            }
        }

        // Significant canonical shell pairs (i <= j) sorted by bound descending (zero par_bridge contention)
        let mut sig_pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n_shells {
            for j in i..n_shells {
                let b = screening.bound(i, j);
                if b >= 1e-11 {
                    sig_pairs.push((i, j, b));
                }
            }
        }
        sig_pairs.sort_by(|a, b| b.2.total_cmp(&a.2));
        let exx_abs = std_exx.abs() + lr_exx.abs();

        let gpu_4c_res = if use_gpu {
            match try_gpu_grad4c(
                basis,
                n_atoms,
                pt,
                pa,
                pb,
                &d_tot_max,
                &d_spin_max,
                &sig_pairs,
                !density_fitting,
                std_exx,
                lr_exx,
                lr_omega,
                grad_thresh,
            ) {
                Some((g_gpu, k_4c)) => {
                    gpu_kernel_invocations += k_4c;
                    Some(g_gpu)
                }
                None => {
                    gpu_fallback_invocations += 1;
                    None
                }
            }
        } else {
            None
        };

        if let Some(g_gpu) = gpu_4c_res {
            g_gpu
        } else {
            (0..sig_pairs.len())
                .into_par_iter()
                .fold(
                    || vec![[0.0; 3]; n_atoms],
                    |mut local_grad, p_idx| {
                        let (i, j, q_ij) = sig_pairs[p_idx];
                        let sa = &basis.shells[i];
                        let sb = &basis.shells[j];
                        let d_ij = d_tot_max[i * n_shells + j];

                        for q_idx in 0..=p_idx {
                            let (k, l, q_kl) = sig_pairs[q_idx];
                            let q_ijkl = q_ij * q_kl;
                            if q_ijkl < 1e-11 {
                                continue;
                            }

                            let sc = &basis.shells[k];
                            let sd = &basis.shells[l];
                            let (aa, ab, ac, ad) =
                                (sa.atom_index, sb.atom_index, sc.atom_index, sd.atom_index);
                            if aa == ab && ab == ac && ac == ad {
                                continue;
                            }

                            // Density-weighted Schwarz pre-screening before assembling AO weights or calling ERI derivatives
                            let d_coul = if density_fitting {
                                0.0
                            } else {
                                d_ij * d_tot_max[k * n_shells + l]
                            };
                            let d_exch = d_spin_max[i * n_shells + k] * d_spin_max[j * n_shells + l]
                                + d_spin_max[i * n_shells + l] * d_spin_max[j * n_shells + k];
                            let d_upper = 4.0 * (d_coul + exx_abs * d_exch);
                            if q_ijkl * d_upper < grad_thresh {
                                continue;
                            }

                            let shells = [sa, sb, sc, sd];
                            let multiplicity = (if i == j { 1 } else { 2 })
                                * (if k == l { 1 } else { 2 })
                                * (if (i, j) == (k, l) { 1 } else { 2 });
                            let weight = 0.5 * f64::from(multiplicity);
                            let size = sa.n_functions()
                                * sb.n_functions()
                                * sc.n_functions()
                                * sd.n_functions();
                            let mut full_weights = Vec::with_capacity(size);
                            let mut long_weights =
                                Vec::with_capacity(if lr_omega.is_some() { size } else { 0 });
                            let mut max_full_w = 0.0f64;
                            let mut max_long_w = 0.0f64;

                            for a in sa.ao_offset..sa.ao_offset + sa.n_functions() {
                                for b in sb.ao_offset..sb.ao_offset + sb.n_functions() {
                                    for c in sc.ao_offset..sc.ao_offset + sc.n_functions() {
                                        for d in sd.ao_offset..sd.ao_offset + sd.n_functions() {
                                            let coul = if density_fitting {
                                                0.0
                                            } else {
                                                pt.get(a, b) * pt.get(c, d)
                                            };
                                            let exch = 0.5
                                                * (pa.get(a, c) * pa.get(b, d)
                                                    + pb.get(a, c) * pb.get(b, d)
                                                    + pa.get(a, d) * pa.get(b, c)
                                                    + pb.get(a, d) * pb.get(b, c));
                                            let fw = weight * (coul - std_exx * exch);
                                            max_full_w = max_full_w.max(fw.abs());
                                            full_weights.push(fw);
                                            if lr_omega.is_some() {
                                                let lw = -weight * lr_exx * exch;
                                                max_long_w = max_long_w.max(lw.abs());
                                                long_weights.push(lw);
                                            }
                                        }
                                    }
                                }
                            }

                            let atoms = [aa, ab, ac, ad];
                            if q_ijkl * max_full_w >= grad_thresh {
                                let dg = derivatives::contract_eri_derivative_quartet_all_centers(
                                    shells,
                                    &full_weights,
                                    None,
                                );
                                for c_idx in 0..4 {
                                    let at = atoms[c_idx];
                                    for axis in 0..3 {
                                        local_grad[at][axis] += dg[c_idx][axis];
                                    }
                                }
                            }
                            if let Some(omega) = lr_omega {
                                if q_ijkl * max_long_w >= grad_thresh {
                                    let dg_lr =
                                        derivatives::contract_eri_derivative_quartet_all_centers(
                                            shells,
                                            &long_weights,
                                            Some(omega),
                                        );
                                    for c_idx in 0..4 {
                                        let at = atoms[c_idx];
                                        for axis in 0..3 {
                                            local_grad[at][axis] += dg_lr[c_idx][axis];
                                        }
                                    }
                                }
                            }
                        }
                        local_grad
                    },
                )
                .reduce(
                    || vec![[0.0; 3]; n_atoms],
                    |mut a, b| {
                        for i in 0..n_atoms {
                            for q in 0..3 {
                                a[i][q] += b[i][q];
                            }
                        }
                        a
                    },
                )
        }
    } else {
        vec![[0.0; 3]; n_atoms]
    };

    let mut grad = vec![[0.0; 3]; n_atoms];
    for a in 0..n_atoms {
        for q in 0..3 {
            let g_nuc = nuclear_repulsion_derivative_with_basis(mol, basis, a, q);
            let (ds, dt, dv) = one_electron_derivative(mol, basis, a, q);
            let mut g_1e = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let h_val = dt.get(mu, nu) + dv.get(mu, nu);
                    g_1e += pt.get(mu, nu) * h_val - w.get(mu, nu) * ds.get(mu, nu);
                }
            }
            let g_ri = g_j_ri.as_ref().map_or(0.0, |g| g[a][q]);
            grad[a][q] = g_nuc + g_1e + g_2e_total[a][q] + g_ri;
        }
    }
    Ok((grad, gpu_kernel_invocations, gpu_fallback_invocations))
}

/// Computes the analytical RHF Cartesian energy gradient with respect to nuclear coordinates.
///
/// Output units: Hartree/Bohr (Eh/Bohr), shaped as `[ [dE/dx, dE/dy, dE/dz], ... ]` for each atom.
/// The nuclear force is `-G`.
pub fn rhf_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    rhf_gradient_with_gpu(mol, basis, wf, false).map(|r| r.gradient)
}

/// Computes the analytical RHF Cartesian energy gradient with optional CUDA GPU acceleration.
pub fn rhf_gradient_with_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &ScfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    if mol.multiplicity != 1 {
        return Err(CalcError::UnsupportedMethod(
            "Analytical RHF gradient requires singlet closed-shell state".into(),
        ));
    }
    let n = basis.n_ao;
    if wf.n_ao != n || wf.n_mo != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({}) does not match wavefunction AO count ({})",
            n, wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(GradientResult {
            gradient: vec![[0.0, 0.0, 0.0]],
            gpu_kernel_invocations: 0,
            gpu_fallback_invocations: 0,
        });
    }

    // RHF stores the spin-summed density P = 2 C_occ C_occ^T in `density_alpha`.
    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density_with_occupations(
        &wf.mo_coefficients,
        &wf.mo_energies,
        &wf.mo_occupations,
        n,
        n_occ,
        2.0,
    );

    let (gradient, gpu_kernel_invocations, gpu_fallback_invocations) = scf_gradient_core(
        mol,
        basis,
        &pt,
        &half,
        &half,
        &w,
        1.0,
        None,
        wf.density_fitting,
        wf.diagnostics.jfit_basis,
        use_gpu,
    )?;
    Ok(GradientResult {
        gradient,
        gpu_kernel_invocations,
        gpu_fallback_invocations,
    })
}

/// Computes the analytical UHF Cartesian energy gradient (Eh/Bohr) for open-shell systems.
pub fn uhf_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    uhf_gradient_with_gpu(mol, basis, wf, false).map(|r| r.gradient)
}

/// Computes the analytical UHF Cartesian energy gradient with optional CUDA GPU acceleration.
pub fn uhf_gradient_with_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &UhfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    let n = basis.n_ao;
    if wf.n_ao != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({}) does not match wavefunction AO count ({})",
            n, wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(GradientResult {
            gradient: vec![[0.0, 0.0, 0.0]],
            gpu_kernel_invocations: 0,
            gpu_fallback_invocations: 0,
        });
    }

    let pa = to_matrix(&wf.density_alpha, n);
    let pb = to_matrix(&wf.density_beta, n);
    let mut pt = pa.clone();
    for (t, b) in pt.data.iter_mut().zip(&pb.data) {
        *t += b;
    }
    let wa = energy_weighted_density(
        &wf.mo_coefficients_alpha,
        &wf.mo_energies_alpha,
        n,
        wf.n_alpha,
        1.0,
    );
    let wb = energy_weighted_density(
        &wf.mo_coefficients_beta,
        &wf.mo_energies_beta,
        n,
        wf.n_beta,
        1.0,
    );
    let mut w = wa;
    for (t, b) in w.data.iter_mut().zip(&wb.data) {
        *t += b;
    }

    let (gradient, gpu_kernel_invocations, gpu_fallback_invocations) = scf_gradient_core(
        mol,
        basis,
        &pt,
        &pa,
        &pb,
        &w,
        1.0,
        None,
        wf.density_fitting,
        wf.diagnostics.jfit_basis,
        use_gpu,
    )?;
    Ok(GradientResult {
        gradient,
        gpu_kernel_invocations,
        gpu_fallback_invocations,
    })
}

/// Computes the exact analytical Restricted Kohn-Sham (RKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn rks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    rks_gradient_with_gpu(mol, basis, xc, grid_level, wf, false).map(|r| r.gradient)
}

/// Computes the exact analytical RKS Cartesian nuclear gradient with optional CUDA GPU acceleration.
pub fn rks_gradient_with_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &ScfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    rks_gradient_with_grid_and_gpu(mol, basis, xc, &grid, wf, use_gpu)
}

/// Computes the exact analytical RKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn rks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    rks_gradient_with_grid_and_gpu(mol, basis, xc, grid, wf, false).map(|r| r.gradient)
}

/// Computes the exact analytical RKS Cartesian nuclear gradient on a prebuilt [`MolecularGrid`]
/// with optional CUDA GPU acceleration.
pub fn rks_gradient_with_grid_and_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &ScfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    if mol.multiplicity != 1 {
        return Err(CalcError::UnsupportedMethod(
            "Analytical RKS gradient requires singlet closed-shell state".into(),
        ));
    }
    let n = basis.n_ao;
    if wf.n_ao != n || wf.n_mo != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({n}) does not match wavefunction AO count ({})",
            wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(GradientResult {
            gradient: vec![[0.0, 0.0, 0.0]],
            gpu_kernel_invocations: 0,
            gpu_fallback_invocations: 0,
        });
    }

    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density_with_occupations(
        &wf.mo_coefficients,
        &wf.mo_energies,
        &wf.mo_occupations,
        n,
        n_occ,
        2.0,
    );

    let (mut grad, mut gpu_kernel_invocations, mut gpu_fallback_invocations) = scf_gradient_core(
        mol,
        basis,
        &pt,
        &half,
        &half,
        &w,
        xc.exact_exchange_fraction(),
        xc.range_separated_params(),
        wf.density_fitting,
        wf.diagnostics.jfit_basis,
        use_gpu,
    )?;
    let (g_xc, xc_k, xc_fb) =
        xc_nuclear_gradient_with_gpu(mol, basis, grid, xc, &half, &half, use_gpu);
    gpu_kernel_invocations += xc_k;
    gpu_fallback_invocations += xc_fb;
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
    }
    if xc == XcFunctional::Wb97xd {
        let (_, g_disp) = calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?;
        for (ga, gd) in grad.iter_mut().zip(g_disp.iter()) {
            ga[0] += gd[0];
            ga[1] += gd[1];
            ga[2] += gd[2];
        }
    }
    Ok(GradientResult {
        gradient: grad,
        gpu_kernel_invocations,
        gpu_fallback_invocations,
    })
}

/// Computes the exact analytical Unrestricted Kohn-Sham (UKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn uks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    uks_gradient_with_gpu(mol, basis, xc, grid_level, wf, false).map(|r| r.gradient)
}

/// Computes the exact analytical UKS Cartesian nuclear gradient with optional CUDA GPU acceleration.
pub fn uks_gradient_with_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &UhfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    uks_gradient_with_grid_and_gpu(mol, basis, xc, &grid, wf, use_gpu)
}

/// Computes the exact analytical UKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn uks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    uks_gradient_with_grid_and_gpu(mol, basis, xc, grid, wf, false).map(|r| r.gradient)
}

/// Computes the exact analytical UKS Cartesian nuclear gradient on a prebuilt [`MolecularGrid`]
/// with optional CUDA GPU acceleration.
pub fn uks_gradient_with_grid_and_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &UhfWavefunction,
    use_gpu: bool,
) -> Result<GradientResult, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    let n = basis.n_ao;
    if wf.n_ao != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({n}) does not match wavefunction AO count ({})",
            wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(GradientResult {
            gradient: vec![[0.0, 0.0, 0.0]],
            gpu_kernel_invocations: 0,
            gpu_fallback_invocations: 0,
        });
    }

    let pa = to_matrix(&wf.density_alpha, n);
    let pb = to_matrix(&wf.density_beta, n);
    let mut pt = pa.clone();
    for (t, b) in pt.data.iter_mut().zip(&pb.data) {
        *t += b;
    }
    let wa = energy_weighted_density(
        &wf.mo_coefficients_alpha,
        &wf.mo_energies_alpha,
        n,
        wf.n_alpha,
        1.0,
    );
    let wb = energy_weighted_density(
        &wf.mo_coefficients_beta,
        &wf.mo_energies_beta,
        n,
        wf.n_beta,
        1.0,
    );
    let mut w = wa;
    for (t, b) in w.data.iter_mut().zip(&wb.data) {
        *t += b;
    }

    let (mut grad, mut gpu_kernel_invocations, mut gpu_fallback_invocations) = scf_gradient_core(
        mol,
        basis,
        &pt,
        &pa,
        &pb,
        &w,
        xc.exact_exchange_fraction(),
        xc.range_separated_params(),
        wf.density_fitting,
        wf.diagnostics.jfit_basis,
        use_gpu,
    )?;
    let (g_xc, xc_k, xc_fb) =
        xc_nuclear_gradient_with_gpu(mol, basis, grid, xc, &pa, &pb, use_gpu);
    gpu_kernel_invocations += xc_k;
    gpu_fallback_invocations += xc_fb;
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
    }
    if xc == XcFunctional::Wb97xd {
        let (_, g_disp) = calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?;
        for (ga, gd) in grad.iter_mut().zip(g_disp.iter()) {
            ga[0] += gd[0];
            ga[1] += gd[1];
            ga[2] += gd[2];
        }
    }
    Ok(GradientResult {
        gradient: grad,
        gpu_kernel_invocations,
        gpu_fallback_invocations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{primitive_norm, Primitive, Shell, ShellType};
    use calchem_model::{Atom, JFitBasis, ScfOptions};
    use std::sync::Mutex;

    static GRAD_GPU_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct DisableGpuGuard(Option<String>);
    impl DisableGpuGuard {
        fn activate() -> Self {
            let prev = std::env::var("CALCHEM_DISABLE_GPU").ok();
            std::env::set_var("CALCHEM_DISABLE_GPU", "1");
            Self(prev)
        }
    }
    impl Drop for DisableGpuGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var("CALCHEM_DISABLE_GPU", v),
                None => std::env::remove_var("CALCHEM_DISABLE_GPU"),
            }
        }
    }

    fn max_abs_grad_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
            .fold(0.0_f64, f64::max)
    }

    fn distorted_water() -> Molecule {
        Molecule::new(
            vec![
                Atom::from_bohr("O", [0.05, -0.03, 0.12]).unwrap(),
                Atom::from_bohr("H", [1.45, 0.10, -1.05]).unwrap(),
                Atom::from_bohr("H", [-1.38, -0.08, -1.12]).unwrap(),
            ],
            0,
            1,
        )
    }

    fn oh_radical() -> Molecule {
        Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.15]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, -1.75]).unwrap(),
            ],
            0,
            2,
        )
    }

    fn agh_molecule() -> Molecule {
        Molecule::new(
            vec![
                Atom::from_bohr("Ag", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 3.10]).unwrap(),
            ],
            0,
            1,
        )
    }

    #[test]
    fn test_gpu_gradient_parity_across_methods_and_f_shells() {
        let _guard = GRAD_GPU_ENV_LOCK.lock().unwrap();
        if !is_cuda_available() {
            return;
        }
        let tol = 1e-6;
        let h2o = distorted_water();
        let bs_631gd = BasisSet::from_name("6-31g(d)", &h2o).unwrap();

        // 1. RHF Direct4c
        let mut opts_dir = ScfOptions::default();
        opts_dir.conv_energy = 1e-10;
        opts_dir.conv_density = 1e-10;
        let wf_rhf = calchem_scf::rhf(&h2o, &bs_631gd, &opts_dir).unwrap();
        let cpu_rhf = rhf_gradient(&h2o, &bs_631gd, &wf_rhf).unwrap();
        let gpu_rhf = rhf_gradient_with_gpu(&h2o, &bs_631gd, &wf_rhf, true).unwrap();
        assert!(gpu_rhf.gpu_kernel_invocations > 0);
        assert_eq!(gpu_rhf.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_rhf, &gpu_rhf.gradient) <= tol,
            "RHF Direct4c diff={}",
            max_abs_grad_diff(&cpu_rhf, &gpu_rhf.gradient)
        );

        // 2. RHF RI-J (auto-jfit & def2-jfit-f)
        for jfit in [JFitBasis::Auto, JFitBasis::Def2UpToF] {
            let mut opts_ri = opts_dir.clone();
            opts_ri.density_fitting = true;
            opts_ri.jfit_basis = jfit;
            let wf_ri = calchem_scf::rhf(&h2o, &bs_631gd, &opts_ri).unwrap();
            let cpu_ri = rhf_gradient(&h2o, &bs_631gd, &wf_ri).unwrap();
            let gpu_ri = rhf_gradient_with_gpu(&h2o, &bs_631gd, &wf_ri, true).unwrap();
            assert!(gpu_ri.gpu_kernel_invocations > 0);
            assert_eq!(gpu_ri.gpu_fallback_invocations, 0);
            assert!(
                max_abs_grad_diff(&cpu_ri, &gpu_ri.gradient) <= tol,
                "RHF RI-J ({:?}) diff={}",
                jfit,
                max_abs_grad_diff(&cpu_ri, &gpu_ri.gradient)
            );
        }

        // 3. RKS PBE RI-J (pure DFT)
        let mut opts_ri = opts_dir.clone();
        opts_ri.density_fitting = true;
        let wf_pbe =
            calchem_scf::rks(&h2o, &bs_631gd, XcFunctional::Pbe, GridLevel::Fine, &opts_ri)
                .unwrap();
        let cpu_pbe =
            rks_gradient(&h2o, &bs_631gd, XcFunctional::Pbe, GridLevel::Fine, &wf_pbe).unwrap();
        let gpu_pbe = rks_gradient_with_gpu(
            &h2o,
            &bs_631gd,
            XcFunctional::Pbe,
            GridLevel::Fine,
            &wf_pbe,
            true,
        )
        .unwrap();
        assert!(gpu_pbe.gpu_kernel_invocations > 0);
        assert_eq!(gpu_pbe.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_pbe, &gpu_pbe.gradient) <= tol,
            "RKS PBE RI-J diff={}",
            max_abs_grad_diff(&cpu_pbe, &gpu_pbe.gradient)
        );

        // 4. RKS B3LYP Direct4c and RI-J
        for df in [false, true] {
            let mut o = opts_dir.clone();
            o.density_fitting = df;
            let wf_b3 =
                calchem_scf::rks(&h2o, &bs_631gd, XcFunctional::B3lyp, GridLevel::Fine, &o)
                    .unwrap();
            let cpu_b3 =
                rks_gradient(&h2o, &bs_631gd, XcFunctional::B3lyp, GridLevel::Fine, &wf_b3)
                    .unwrap();
            let gpu_b3 = rks_gradient_with_gpu(
                &h2o,
                &bs_631gd,
                XcFunctional::B3lyp,
                GridLevel::Fine,
                &wf_b3,
                true,
            )
            .unwrap();
            assert!(gpu_b3.gpu_kernel_invocations > 0);
            assert_eq!(gpu_b3.gpu_fallback_invocations, 0);
            assert!(
                max_abs_grad_diff(&cpu_b3, &gpu_b3.gradient) <= tol,
                "RKS B3LYP (df={df}) diff={}",
                max_abs_grad_diff(&cpu_b3, &gpu_b3.gradient)
            );
        }

        // 5. RSH RKS (CAM-B3LYP and wB97X-D)
        for xc in [XcFunctional::CamB3lyp, XcFunctional::Wb97xd] {
            let wf_rsh = calchem_scf::rks(&h2o, &bs_631gd, xc, GridLevel::Fine, &opts_dir).unwrap();
            let cpu_rsh = rks_gradient(&h2o, &bs_631gd, xc, GridLevel::Fine, &wf_rsh).unwrap();
            let gpu_rsh =
                rks_gradient_with_gpu(&h2o, &bs_631gd, xc, GridLevel::Fine, &wf_rsh, true).unwrap();
            assert!(gpu_rsh.gpu_kernel_invocations > 0);
            assert_eq!(gpu_rsh.gpu_fallback_invocations, 0);
            assert!(
                max_abs_grad_diff(&cpu_rsh, &gpu_rsh.gradient) <= tol,
                "RSH {:?} diff={}",
                xc,
                max_abs_grad_diff(&cpu_rsh, &gpu_rsh.gradient)
            );
        }

        // 6. UHF and UKS on OH radical
        let oh = oh_radical();
        let bs_oh = BasisSet::from_name("6-31g(d)", &oh).unwrap();
        let wf_uhf = calchem_scf::uhf(&oh, &bs_oh, &opts_dir).unwrap();
        let cpu_uhf = uhf_gradient(&oh, &bs_oh, &wf_uhf).unwrap();
        let gpu_uhf = uhf_gradient_with_gpu(&oh, &bs_oh, &wf_uhf, true).unwrap();
        assert!(gpu_uhf.gpu_kernel_invocations > 0);
        assert_eq!(gpu_uhf.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_uhf, &gpu_uhf.gradient) <= tol,
            "UHF diff={}",
            max_abs_grad_diff(&cpu_uhf, &gpu_uhf.gradient)
        );

        let wf_uks =
            calchem_scf::uks(&oh, &bs_oh, XcFunctional::B3lyp, GridLevel::Fine, &opts_dir).unwrap();
        let cpu_uks =
            uks_gradient(&oh, &bs_oh, XcFunctional::B3lyp, GridLevel::Fine, &wf_uks).unwrap();
        let gpu_uks = uks_gradient_with_gpu(
            &oh,
            &bs_oh,
            XcFunctional::B3lyp,
            GridLevel::Fine,
            &wf_uks,
            true,
        )
        .unwrap();
        assert!(gpu_uks.gpu_kernel_invocations > 0);
        assert_eq!(gpu_uks.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_uks, &gpu_uks.gradient) <= tol,
            "UKS B3LYP diff={}",
            max_abs_grad_diff(&cpu_uks, &gpu_uks.gradient)
        );

        // 7. ECP molecule (AgH LANL2DZ)
        let agh = agh_molecule();
        let bs_ecp = BasisSet::from_name("lanl2dz", &agh).unwrap();
        let wf_ecp = calchem_scf::rhf(&agh, &bs_ecp, &opts_dir).unwrap();
        let cpu_ecp = rhf_gradient(&agh, &bs_ecp, &wf_ecp).unwrap();
        let gpu_ecp = rhf_gradient_with_gpu(&agh, &bs_ecp, &wf_ecp, true).unwrap();
        assert!(gpu_ecp.gpu_kernel_invocations > 0);
        assert_eq!(gpu_ecp.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_ecp, &gpu_ecp.gradient) <= tol,
            "ECP AgH diff={}",
            max_abs_grad_diff(&cpu_ecp, &gpu_ecp.gradient)
        );

        // 8. L=3 F-shell parity on H2O augmented with Cartesian F shell
        let mut bs_f = bs_631gd.clone();
        let o_pos = h2o.atoms[0].position_bohr;
        let f_ao_off = bs_f.n_ao;
        bs_f.shells.push(Shell {
            atom_index: 0,
            center: o_pos,
            shell_type: ShellType::F,
            primitives: vec![Primitive {
                exponent: 1.15,
                coeff: 1.0,
                raw_coeff: 1.0,
                norm: primitive_norm(1.15, [3, 0, 0]),
            }],
            ao_offset: f_ao_off,
        });
        bs_f.n_ao += ShellType::F.components().len();
        let wf_f = calchem_scf::rks(&h2o, &bs_f, XcFunctional::B3lyp, GridLevel::Fine, &opts_dir)
            .unwrap();
        let cpu_f =
            rks_gradient(&h2o, &bs_f, XcFunctional::B3lyp, GridLevel::Fine, &wf_f).unwrap();
        let gpu_f =
            rks_gradient_with_gpu(&h2o, &bs_f, XcFunctional::B3lyp, GridLevel::Fine, &wf_f, true)
                .unwrap();
        assert!(gpu_f.gpu_kernel_invocations > 0);
        assert_eq!(gpu_f.gpu_fallback_invocations, 0);
        assert!(
            max_abs_grad_diff(&cpu_f, &gpu_f.gradient) <= tol,
            "F-shell B3LYP diff={}",
            max_abs_grad_diff(&cpu_f, &gpu_f.gradient)
        );
    }

    #[test]
    fn test_gpu_gradient_disable_gpu_records_fallbacks_and_matches_cpu() {
        let _guard = GRAD_GPU_ENV_LOCK.lock().unwrap();
        let h2o = distorted_water();
        let bs = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let mut opts = ScfOptions::default();
        opts.conv_energy = 1e-10;
        opts.conv_density = 1e-10;

        let wf_rhf = calchem_scf::rhf(&h2o, &bs, &opts).unwrap();
        let cpu_rhf = rhf_gradient(&h2o, &bs, &wf_rhf).unwrap();

        let mut opts_ri = opts.clone();
        opts_ri.density_fitting = true;
        let wf_pbe =
            calchem_scf::rks(&h2o, &bs, XcFunctional::Pbe, GridLevel::Fine, &opts_ri).unwrap();
        let cpu_pbe = rks_gradient(&h2o, &bs, XcFunctional::Pbe, GridLevel::Fine, &wf_pbe).unwrap();

        let wf_b3_ri =
            calchem_scf::rks(&h2o, &bs, XcFunctional::B3lyp, GridLevel::Fine, &opts_ri).unwrap();
        let cpu_b3_ri =
            rks_gradient(&h2o, &bs, XcFunctional::B3lyp, GridLevel::Fine, &wf_b3_ri).unwrap();

        let _disable = DisableGpuGuard::activate();

        let res_rhf = rhf_gradient_with_gpu(&h2o, &bs, &wf_rhf, true).unwrap();
        assert_eq!(res_rhf.gpu_kernel_invocations, 0);
        assert!(res_rhf.gpu_fallback_invocations > 0);
        assert!(max_abs_grad_diff(&cpu_rhf, &res_rhf.gradient) <= 1e-12);

        let res_pbe =
            rks_gradient_with_gpu(&h2o, &bs, XcFunctional::Pbe, GridLevel::Fine, &wf_pbe, true)
                .unwrap();
        assert_eq!(res_pbe.gpu_kernel_invocations, 0);
        assert!(res_pbe.gpu_fallback_invocations > 0);
        assert!(max_abs_grad_diff(&cpu_pbe, &res_pbe.gradient) <= 1e-12);

        let res_b3_ri = rks_gradient_with_gpu(
            &h2o,
            &bs,
            XcFunctional::B3lyp,
            GridLevel::Fine,
            &wf_b3_ri,
            true,
        )
        .unwrap();
        assert_eq!(res_b3_ri.gpu_kernel_invocations, 0);
        assert!(res_b3_ri.gpu_fallback_invocations > 0);
        assert!(max_abs_grad_diff(&cpu_b3_ri, &res_b3_ri.gradient) <= 1e-12);
    }

    struct GradEnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }
    impl GradEnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            Self {
                saved: keys.iter().map(|&k| (k, std::env::var(k).ok())).collect(),
            }
        }
    }
    impl Drop for GradEnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn test_gradient_error_contracts_and_single_atom_gpu_telemetry() {
        let _guard = GRAD_GPU_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let h2o = distorted_water();
        let bs = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let mut opts = ScfOptions::default();
        opts.conv_energy = 1e-10;
        opts.conv_density = 1e-10;
        let wf = calchem_scf::rhf(&h2o, &bs, &opts).unwrap();

        let mut bad_wf = wf.clone();
        bad_wf.converged = false;
        assert!(matches!(rhf_gradient(&h2o, &bs, &bad_wf), Err(CalcError::NotConverged { .. })));

        let mut trip_mol = h2o.clone();
        trip_mol.multiplicity = 3;
        assert!(matches!(rhf_gradient(&trip_mol, &bs, &wf), Err(CalcError::UnsupportedMethod(_))));
        assert!(matches!(
            rks_gradient(&trip_mol, &bs, XcFunctional::B3lyp, GridLevel::Coarse, &wf),
            Err(CalcError::UnsupportedMethod(_))
        ));

        let mut bad_ao = wf.clone();
        bad_ao.n_ao += 1;
        assert!(matches!(rhf_gradient(&h2o, &bs, &bad_ao), Err(CalcError::InvalidMolecule(_))));

        let mut shifted_mol = h2o.clone();
        shifted_mol.atoms[0].position_bohr[0] += 0.1;
        assert!(matches!(rhf_gradient(&shifted_mol, &bs, &wf), Err(CalcError::InvalidMolecule(_))));

        let mut nan_mol = h2o.clone();
        nan_mol.atoms[0].position_bohr[0] = f64::NAN;
        assert!(matches!(rhf_gradient(&nan_mol, &bs, &wf), Err(CalcError::InvalidMolecule(_))));

        // Out-of-bounds shell atom_index must return InvalidMolecule instead of panicking (LOW-01)
        let mut bad_idx_bs = bs.clone();
        bad_idx_bs.shells[0].atom_index = 99;
        assert!(matches!(rhf_gradient(&h2o, &bad_idx_bs, &wf), Err(CalcError::InvalidMolecule(_))));

        let he = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 1, 2);
        let bs_he = BasisSet::sto_3g(&he).unwrap();
        let wf_he = calchem_scf::uks(&he, &bs_he, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
        let g_he = uks_gradient_with_gpu(&he, &bs_he, XcFunctional::B3lyp, GridLevel::Coarse, &wf_he, true).unwrap();
        assert_eq!(g_he.gradient, vec![[0.0, 0.0, 0.0]]);
        assert_eq!((g_he.gpu_kernel_invocations, g_he.gpu_fallback_invocations), (0, 0));
    }

    #[test]
    fn test_gpu_gradient_contracted_fshell_uks_rij_rsh_ecp_linear_and_vram_fallback() {
        let _guard = GRAD_GPU_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _env = GradEnvGuard::new(&["CALCHEM_MAX_GPU_MEM_MB"]);
        if !is_cuda_available() {
            return;
        }
        let tol = 1e-6;
        let mut opts = ScfOptions::default();
        opts.conv_energy = 1e-10;
        opts.conv_density = 1e-10;

        let zno = Molecule::new(
            vec![
                Atom::from_bohr("Zn", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("O", [0.0, 0.0, 3.2]).unwrap(),
            ],
            0,
            1,
        );
        let mut bs_zno = BasisSet::from_name("6-31g(d)", &zno).unwrap();
        for sh in &mut bs_zno.shells {
            if sh.shell_type == ShellType::F && sh.primitives.len() == 1 {
                let exps = [1.4, 0.55];
                let raw = [0.45, 0.65];
                let norm_c = calchem_basis::normalized_contraction(&exps, &raw, [3, 0, 0]).unwrap();
                sh.primitives = exps
                    .iter()
                    .zip(norm_c.iter())
                    .zip(raw.iter())
                    .map(|((&e, &c), &r)| calchem_basis::Primitive {
                        exponent: e,
                        coeff: c,
                        raw_coeff: r,
                        norm: calchem_basis::primitive_norm(e, [3, 0, 0]),
                    })
                    .collect();
            }
        }
        assert!(bs_zno.shells.iter().any(|s| s.shell_type == ShellType::F && s.primitives.len() > 1));
        let wf_zno = calchem_scf::rks(&zno, &bs_zno, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
        let cpu_zno = rks_gradient(&zno, &bs_zno, XcFunctional::B3lyp, GridLevel::Coarse, &wf_zno).unwrap();
        let gpu_zno = rks_gradient_with_gpu(&zno, &bs_zno, XcFunctional::B3lyp, GridLevel::Coarse, &wf_zno, true).unwrap();
        assert!(gpu_zno.gpu_kernel_invocations > 0);
        assert_eq!(gpu_zno.gpu_fallback_invocations, 0);
        assert!(max_abs_grad_diff(&cpu_zno, &gpu_zno.gradient) <= tol);

        let oh = oh_radical();
        let bs_oh = BasisSet::from_name("6-31g(d)", &oh).unwrap();
        let mut opts_ri = opts.clone();
        opts_ri.density_fitting = true;
        for xc in [XcFunctional::Pbe, XcFunctional::B3lyp] {
            let wf_u = calchem_scf::uks(&oh, &bs_oh, xc, GridLevel::Fine, &opts_ri).unwrap();
            let cpu_u = uks_gradient(&oh, &bs_oh, xc, GridLevel::Fine, &wf_u).unwrap();
            let gpu_u = uks_gradient_with_gpu(&oh, &bs_oh, xc, GridLevel::Fine, &wf_u, true).unwrap();
            assert!(gpu_u.gpu_kernel_invocations > 0);
            assert_eq!(gpu_u.gpu_fallback_invocations, 0);
            assert!(max_abs_grad_diff(&cpu_u, &gpu_u.gradient) <= tol);
        }
        for xc in [XcFunctional::CamB3lyp, XcFunctional::Wb97xd] {
            let wf_u = calchem_scf::uks(&oh, &bs_oh, xc, GridLevel::Fine, &opts).unwrap();
            let cpu_u = uks_gradient(&oh, &bs_oh, xc, GridLevel::Fine, &wf_u).unwrap();
            let gpu_u = uks_gradient_with_gpu(&oh, &bs_oh, xc, GridLevel::Fine, &wf_u, true).unwrap();
            assert!(gpu_u.gpu_kernel_invocations > 0);
            assert_eq!(gpu_u.gpu_fallback_invocations, 0);
            assert!(max_abs_grad_diff(&cpu_u, &gpu_u.gradient) <= tol);
        }

        let hbr_cat = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("Br", [0.0, 0.0, 2.7]).unwrap(),
            ],
            1,
            2,
        );
        let bs_hbr = BasisSet::from_name("lanl2dzdp", &hbr_cat).unwrap();
        let wf_hbr = calchem_scf::uhf(&hbr_cat, &bs_hbr, &opts).unwrap();
        let cpu_hbr = uhf_gradient(&hbr_cat, &bs_hbr, &wf_hbr).unwrap();
        let gpu_hbr = uhf_gradient_with_gpu(&hbr_cat, &bs_hbr, &wf_hbr, true).unwrap();
        assert!(gpu_hbr.gpu_kernel_invocations > 0);
        assert_eq!(gpu_hbr.gpu_fallback_invocations, 0);
        assert!(max_abs_grad_diff(&cpu_hbr, &gpu_hbr.gradient) <= tol);

        let h2o = distorted_water();
        let bs_w = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let wf_w = calchem_scf::rks(&h2o, &bs_w, XcFunctional::B3lyp, GridLevel::Fine, &opts).unwrap();
        let cpu_w = rks_gradient(&h2o, &bs_w, XcFunctional::B3lyp, GridLevel::Fine, &wf_w).unwrap();
        std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "1");
        let oom_w = rks_gradient_with_gpu(&h2o, &bs_w, XcFunctional::B3lyp, GridLevel::Fine, &wf_w, true).unwrap();
        assert!(oom_w.gpu_fallback_invocations > 0);
        assert!(max_abs_grad_diff(&cpu_w, &oom_w.gradient) <= 1e-11);
    }
}
