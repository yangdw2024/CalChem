pub mod derivatives;

use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional, xc_nuclear_gradient};
use calchem_int::screening::SchwarzScreening;
use calchem_int::symmetry::shell_quartets;
use calchem_model::{CalcError, Molecule, ScfWavefunction, UhfWavefunction};
use calchem_tensor::Matrix2D;
use derivatives::{
    eri_derivative_quartet, nuclear_repulsion_derivative_with_basis, one_electron_derivative,
};
use rayon::prelude::*;

#[inline]
fn distinct_shell_permutations(i: usize, j: usize, k: usize, l: usize) -> Vec<[usize; 4]> {
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
) -> Vec<[f64; 3]> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;

    let screening = SchwarzScreening::new(basis, 1e-11);
    let canonical_quartets = shell_quartets(basis.shells.len());

    let g_2e_total = canonical_quartets
        .par_bridge()
        .fold(
            || vec![[0.0; 3]; n_atoms],
            |mut local_grad, [i, j, k, l]| {
                if screening.can_skip(i, j, k, l) {
                    return local_grad;
                }
                for [a_idx, b_idx, c_idx, d_idx] in distinct_shell_permutations(i, j, k, l) {
                    let sh_a = &basis.shells[a_idx];
                    let sh_b = &basis.shells[b_idx];
                    let sh_c = &basis.shells[c_idx];
                    let sh_d = &basis.shells[d_idx];

                    let mut involved = [
                        sh_a.atom_index,
                        sh_b.atom_index,
                        sh_c.atom_index,
                        sh_d.atom_index,
                    ];
                    involved.sort_unstable();
                    let mut n_unique_atoms = 0;
                    for idx in 0..4 {
                        if idx == 0 || involved[idx] != involved[idx - 1] {
                            involved[n_unique_atoms] = involved[idx];
                            n_unique_atoms += 1;
                        }
                    }

                    let na = sh_a.n_functions();
                    let nb = sh_b.n_functions();
                    let nc = sh_c.n_functions();
                    let nd = sh_d.n_functions();

                    for &atom in &involved[..n_unique_atoms] {
                        for q in 0..3 {
                            let d_block = eri_derivative_quartet([sh_a, sh_b, sh_c, sh_d], atom, q);
                            let mut d_contrib = 0.0;
                            for ia in 0..na {
                                let mu = sh_a.ao_offset + ia;
                                for ib in 0..nb {
                                    let nu = sh_b.ao_offset + ib;
                                    for ic in 0..nc {
                                        let lam = sh_c.ao_offset + ic;
                                        for id in 0..nd {
                                            let sig = sh_d.ao_offset + id;
                                            let d_val =
                                                d_block[((ia * nb + ib) * nc + ic) * nd + id];
                                            if d_val.abs() > 1e-16 {
                                                let coul = 0.5 * pt.get(mu, nu) * pt.get(lam, sig);
                                                let gamma = if exx_fraction.abs() < 1e-15 {
                                                    coul
                                                } else {
                                                    let exch = 0.5
                                                        * exx_fraction
                                                        * (pa.get(mu, lam) * pa.get(nu, sig)
                                                            + pb.get(mu, lam) * pb.get(nu, sig));
                                                    coul - exch
                                                };
                                                d_contrib += d_val * gamma;
                                            }
                                        }
                                    }
                                }
                            }
                            local_grad[atom][q] += d_contrib;
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
        );

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
            grad[a][q] = g_nuc + g_1e + g_2e_total[a][q];
        }
    }
    grad
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

    Ok(scf_gradient_core(mol, basis, &pt, &half, &half, &w, 1.0))
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
    let wa = energy_weighted_density(&wf.mo_coefficients_alpha, &wf.mo_energies_alpha, n, wf.n_alpha, 1.0);
    let wb = energy_weighted_density(&wf.mo_coefficients_beta, &wf.mo_energies_beta, n, wf.n_beta, 1.0);
    let mut w = wa;
    for (t, b) in w.data.iter_mut().zip(&wb.data) {
        *t += b;
    }

    Ok(scf_gradient_core(mol, basis, &pt, &pa, &pb, &w, 1.0))
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
    );
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &half, &half);
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
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
    );
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &pa, &pb);
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
    }
    Ok(grad)
}
