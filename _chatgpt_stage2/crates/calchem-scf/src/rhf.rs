use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_int::{compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D, Tensor4D};

fn build_fock_matrix(h_core: &Matrix2D, p: &Matrix2D, eri: &Tensor4D) -> Matrix2D {
    let n = h_core.rows;
    let mut f = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut g_munu = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let p_ls = p.get(lam, sig);
                    let j = eri.get(mu, nu, lam, sig);
                    let k = eri.get(mu, lam, nu, sig);
                    g_munu += p_ls * (j - 0.5 * k);
                }
            }
            f.set(mu, nu, h_core.get(mu, nu) + g_munu);
        }
    }
    f
}

fn compute_electronic_energy(p: &Matrix2D, h_core: &Matrix2D, fock: &Matrix2D) -> f64 {
    let n = h_core.rows;
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * p.get(mu, nu) * (h_core.get(mu, nu) + fock.get(mu, nu));
        }
    }
    e_elec
}

fn diagonalize_fock(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, 2.0 * sum);
        }
    }
    Ok((eps, c, p))
}

/// Runs a closed-shell Restricted Hartree-Fock (RHF) calculation in the given basis set.
pub fn rhf(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    let (n_alpha, n_beta) = mol.validate_rhf()?;
    if n_alpha != n_beta {
        return Err(CalcError::UnsupportedMethod(
            "RHF requires a closed-shell singlet state (n_alpha == n_beta)".into(),
        ));
    }
    let n_occ = n_alpha;
    let n = basis.n_ao;
    if n == 0 || n_occ > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({}) for {} occupied orbitals",
            n, n_occ
        )));
    }

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i];
    }

    let eri = compute_eri(basis);
    let e_nuc = mol.nuclear_repulsion_energy();

    // Symmetric orthogonalization X = U * s^{-1/2} * U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 0.0 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is not positive definite (eigenvalue {:.3e})",
                val
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Core Hamiltonian initial guess
    let (mut mo_energies, mut c_mat, mut p) = diagonalize_fock(&h_core, &x, &xt, n_occ)?;
    let mut f = h_core.clone();

    let mut diis = DiisState::new(opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        f = build_fock_matrix(&h_core, &p, &eri);
        let e_elec = compute_electronic_energy(&p, &h_core, &f);
        let e_tot = e_elec + e_nuc;
        energy_history.push(e_tot);

        // Pulay commutator error e = F P S - S P F
        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = if opts.use_diis && iter >= 2 {
            diis.push_and_extrapolate(&f, &err)
        } else {
            if opts.use_diis {
                let _ = diis.push_and_extrapolate(&f, &err);
            }
            f.clone()
        };

        let (eps_new, c_new, mut p_new) = diagonalize_fock(&f_extrap, &x, &xt, n_occ)?;

        if opts.damping > 0.0 && opts.damping < 1.0 {
            let d = opts.damping;
            for idx in 0..p_new.data.len() {
                p_new.data[idx] = (1.0 - d) * p_new.data[idx] + d * p.data[idx];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        last_delta_e = delta_e;
        last_rms_d = rms_d;
        p = p_new;
        c_mat = c_new;
        mo_energies = eps_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        if iter > 1 && delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Recompute consistent final Fock matrix, variational energy, and canonical orbitals
    // from the converged density matrix P.
    f = build_fock_matrix(&h_core, &p, &eri);
    let e_elec_final = compute_electronic_energy(&p, &h_core, &f);
    let e_tot_final = e_elec_final + e_nuc;
    let (final_eps, final_c, final_p) = diagonalize_fock(&f, &x, &xt, n_occ)?;
    mo_energies = final_eps;
    c_mat = final_c;
    p = final_p;

    let mut occ = vec![0.0; n];
    for slot in occ.iter_mut().take(n_occ) {
        *slot = 2.0;
    }

    Ok(ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: e_tot_final,
        electronic_energy_hartree: e_elec_final,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies,
        mo_occupations: occ,
        mo_coefficients: c_mat.data,
        density_alpha: p.data,
        density_beta: None,
        fock_alpha: f.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: ScfDiagnostics {
            iterations: used_iter,
            energy_history,
            rms_density_history: rms_history,
            diis_error_history: diis_err_history,
        },
    })
}
