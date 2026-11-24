pub mod derivatives;

use calchem_basis::BasisSet;
use calchem_dft::{xc_nuclear_gradient, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::screening::SchwarzScreening;
use calchem_model::{CalcError, Molecule, ScfWavefunction, UhfWavefunction};
use calchem_tensor::Matrix2D;
use derivatives::{nuclear_repulsion_derivative_with_basis, one_electron_derivative};
use rayon::prelude::*;

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
        let atom_pos = mol.atoms[shell.atom_index].position_bohr;
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
        let atom_pos = mol.atoms[ecp.atom_index].position_bohr;
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

/// `W = sum_i eps_i C_{mu i} C_{nu i}` over the first `n_occ` columns, scaled by `occ`.
fn energy_weighted_density(c: &[f64], eps: &[f64], n: usize, n_occ: usize, occ: f64) -> Matrix2D {
    let mut w = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for i in 0..n_occ {
                s += eps[i] * c[mu * n + i] * c[nu * n + i];
            }
            w.set(mu, nu, occ * s);
        }
    }
    w
}

/// Shared gradient engine for RHF, UHF, RKS, and UKS (`1e + 2e + E_nn + Pulay`).
///
/// `dE/dX = dE_nn/dX + sum P^t dh/dX - sum W dS/dX + sum (mu nu|lam sig)^X Gamma_{mu nu lam sig}`
/// with `Gamma = 1/2 P^t_{mn} P^t_{ls} - c_HF / 2 * (P^a_{ml} P^a_{ns} + P^b_{ml} P^b_{ns})`.
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
) -> Result<Vec<[f64; 3]>, CalcError> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;

    let (std_exx, lr_exx, lr_omega) = match range_sep {
        Some((alpha, beta, omega)) => (alpha, beta, Some(omega)),
        None => (exx_fraction, 0.0, None),
    };

    let g_j_ri = if density_fitting {
        let rij = calchem_int::RiJEngine::with_basis(mol, basis, false, jfit_basis)?;
        Some(rij.compute_coulomb_gradient(n_atoms, basis, pt)?)
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
        let exx_abs = std_exx.abs() + lr_exx.abs();

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
                        let (aa, ab, ac, ad) = (sa.atom_index, sb.atom_index, sc.atom_index, sd.atom_index);
                        if aa == ab && ab == ac && ac == ad {
                            continue;
                        }

                        // Density-weighted Schwarz pre-screening before assembling AO weights or calling ERI derivatives
                        let d_coul = if density_fitting { 0.0 } else { d_ij * d_tot_max[k * n_shells + l] };
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
                        let size = sa.n_functions() * sb.n_functions() * sc.n_functions() * sd.n_functions();
                        let mut full_weights = Vec::with_capacity(size);
                        let mut long_weights = Vec::with_capacity(if lr_omega.is_some() { size } else { 0 });
                        let mut max_full_w = 0.0f64;
                        let mut max_long_w = 0.0f64;

                        for a in sa.ao_offset..sa.ao_offset + sa.n_functions() {
                            for b in sb.ao_offset..sb.ao_offset + sb.n_functions() {
                                for c in sc.ao_offset..sc.ao_offset + sc.n_functions() {
                                    for d in sd.ao_offset..sd.ao_offset + sd.n_functions() {
                                        let coul = if density_fitting { 0.0 } else { pt.get(a, b) * pt.get(c, d) };
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
                                let dg_lr = derivatives::contract_eri_derivative_quartet_all_centers(
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
    Ok(grad)
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
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    // RHF stores the spin-summed density P = 2 C_occ C_occ^T in `density_alpha`.
    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density(&wf.mo_coefficients, &wf.mo_energies, n, n_occ, 2.0);

    scf_gradient_core(
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
    )
}

/// Computes the analytical UHF Cartesian energy gradient (Eh/Bohr) for open-shell systems.
pub fn uhf_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
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
        return Ok(vec![[0.0, 0.0, 0.0]]);
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

    scf_gradient_core(mol, basis, &pt, &pa, &pb, &w, 1.0, None, wf.density_fitting, wf.diagnostics.jfit_basis)
}

/// Computes the exact analytical Restricted Kohn-Sham (RKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn rks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    rks_gradient_with_grid(mol, basis, xc, &grid, wf)
}

/// Computes the exact analytical RKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn rks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
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
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density(&wf.mo_coefficients, &wf.mo_energies, n, n_occ, 2.0);

    let mut grad = scf_gradient_core(
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
    )?;
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &half, &half);
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
    Ok(grad)
}

/// Computes the exact analytical Unrestricted Kohn-Sham (UKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn uks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    uks_gradient_with_grid(mol, basis, xc, &grid, wf)
}

/// Computes the exact analytical UKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn uks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
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
        return Ok(vec![[0.0, 0.0, 0.0]]);
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

    let mut grad = scf_gradient_core(
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
    )?;
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &pa, &pb);
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
    Ok(grad)
}
