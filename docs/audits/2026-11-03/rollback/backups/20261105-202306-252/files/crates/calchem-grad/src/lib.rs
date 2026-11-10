pub mod derivatives;

use calchem_basis::BasisSet;
use calchem_int::screening::SchwarzScreening;
use calchem_int::symmetry::unique_shell_quartets;
use calchem_model::{CalcError, Molecule, ScfWavefunction};
use calchem_tensor::Matrix2D;
use derivatives::{eri_derivative_quartet, nuclear_repulsion_derivative, one_electron_derivative};
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

    let n_atoms = mol.atoms.len();
    if n_atoms == 0 {
        return Err(CalcError::InvalidMolecule("No atoms in molecule".into()));
    }

    let n = basis.n_ao;
    if wf.n_ao != n || wf.n_mo != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({}) does not match wavefunction AO count ({})",
            n, wf.n_ao
        )));
    }

    // Verify coordinates are finite and match basis set centers
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

    // Single-atom gradient must vanish exactly
    if n_atoms == 1 {
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    // Extract spin-summed density matrix P (P = 2 * C_occ * C_occ^T)
    let mut p = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p.set(i, j, wf.density_alpha[i * n + j]);
        }
    }

    // Energy-weighted density matrix W_mu,nu = sum_i occ_i * eps_i * C_mu,i * C_nu,i
    let mut w = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum_w = 0.0;
            for i in 0..n {
                let occ_i = wf.mo_occupations[i];
                if occ_i > 1e-12 {
                    let eps_i = wf.mo_energies[i];
                    let c_mu_i = wf.mo_coefficients[mu * n + i];
                    let c_nu_i = wf.mo_coefficients[nu * n + i];
                    sum_w += occ_i * eps_i * c_mu_i * c_nu_i;
                }
            }
            w.set(mu, nu, sum_w);
        }
    }

    // 1. Pre-screen unique shell quartets
    let screening = SchwarzScreening::new(basis, 1e-11);
    let canonical_quartets = unique_shell_quartets(basis.shells.len());

    // 2. Evaluate 2-electron gradient contributions across unique quartets in parallel
    let g_2e_total = canonical_quartets
        .into_par_iter()
        .fold(
            || vec![[0.0; 3]; n_atoms],
            |mut local_grad, [i, j, k, l]| {
                if screening.can_skip(i, j, k, l) {
                    return local_grad;
                }

                let perms = distinct_shell_permutations(i, j, k, l);
                for [a_idx, b_idx, c_idx, d_idx] in perms {
                    let sh_a = &basis.shells[a_idx];
                    let sh_b = &basis.shells[b_idx];
                    let sh_c = &basis.shells[c_idx];
                    let sh_d = &basis.shells[d_idx];

                    // Determine unique atoms involved in these four shells
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
                            let d_block =
                                eri_derivative_quartet([sh_a, sh_b, sh_c, sh_d], atom, q);
                            let mut d_contrib = 0.0;

                            for ia in 0..na {
                                let mu = sh_a.ao_offset + ia;
                                for ib in 0..nb {
                                    let nu = sh_b.ao_offset + ib;
                                    for ic in 0..nc {
                                        let lam = sh_c.ao_offset + ic;
                                        for id in 0..nd {
                                            let sig = sh_d.ao_offset + id;
                                            let block_idx =
                                                ((ia * nb + ib) * nc + ic) * nd + id;
                                            let d_val = d_block[block_idx];

                                            if d_val.abs() > 1e-16 {
                                                let p_mn = p.get(mu, nu);
                                                let p_ls = p.get(lam, sig);
                                                let p_ml = p.get(mu, lam);
                                                let p_ns = p.get(nu, sig);
                                                let gamma =
                                                    0.5 * p_mn * p_ls - 0.25 * p_ml * p_ns;
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

    // 3. Assemble total nuclear forces (Nuclear + 1-Electron + 2-Electron)
    let mut grad = vec![[0.0; 3]; n_atoms];
    for a in 0..n_atoms {
        for q in 0..3 {
            let g_nuc = nuclear_repulsion_derivative(mol, a, q);
            let (ds, dt, dv) = one_electron_derivative(mol, basis, a, q);
            let mut g_1e = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let p_val = p.get(mu, nu);
                    let w_val = w.get(mu, nu);
                    let h_val = dt.get(mu, nu) + dv.get(mu, nu);
                    let s_val = ds.get(mu, nu);
                    g_1e += p_val * h_val - w_val * s_val;
                }
            }
            grad[a][q] = g_nuc + g_1e + g_2e_total[a][q];
        }
    }

    Ok(grad)
}
