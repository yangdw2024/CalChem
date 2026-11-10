use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_int::{
    SparseSymmetricEri, compute_ecp, compute_kinetic, compute_nuclear, compute_overlap,
    compute_sparse_eri,
};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{Matrix2D, eigh_sym, frobenius_norm, matmul, rms_diff, transpose};

fn build_fock_matrix(h_core: &Matrix2D, p: &Matrix2D, eri: &SparseSymmetricEri) -> Matrix2D {
    let n = h_core.rows;
    let mut f = eri.contract_rks(&p.data, 1.0);
    for i in 0..n * n {
        f[i] += h_core.data[i];
    }
    Matrix2D {
        rows: n,
        cols: n,
        data: f,
    }
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
    crate::validate_resources_and_options(basis, opts)?;
    let (n_alpha, n_beta) = basis.validate_rhf_valence(mol)?;
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
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }

    let eri = compute_sparse_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // Symmetric orthogonalization X = U * s^{-1/2} * U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {:.3e})",
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
    let (_, _, mut p) = diagonalize_fock(&h_core, &x, &xt, n_occ)?;
    let mut f;

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

        let (_eps_new, _c_new, mut p_new) = diagonalize_fock(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if opts.damping > 0.0 && opts.damping < 1.0 {
            opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
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
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        let tight_plateau = iter > 8
            && delta_e < (opts.conv_energy * 0.1).min(1e-12)
            && rms_d < opts.conv_density.max(1e-8)
            && diis_err < grad_tol;
        if iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau)
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

    // Tighten the final fixed-point step to ensure gradient-grade stationarity:
    // canonical (C, epsilon, P, F) with F = F[P], P = 2 C_occ C_occ^T,
    // and ||FPS - SPF|| below gradient tolerance.
    let mut final_f = build_fock_matrix(&h_core, &p, &eri);
    let (eps_init, c_init, p_init) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
    let f_init = build_fock_matrix(&h_core, &p_init, &eri);
    let fps_init = matmul(&matmul(&f_init, &p_init), &s);
    let spf_init = matmul(&matmul(&s, &p_init), &f_init);
    let mut comm_init = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        comm_init.data[idx] = fps_init.data[idx] - spf_init.data[idx];
    }
    let comm_norm_init = frobenius_norm(&comm_init);

    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-8);

    let mut final_eps = eps_init;
    let mut final_c = c_init;
    let mut final_p = p_init;
    final_f = f_init;

    if comm_norm_init > fp_comm_tol {
        let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));
        for fp_step in 0..20 {
            let fps = matmul(&matmul(&final_f, &final_p), &s);
            let spf = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm.data[idx] = fps.data[idx] - spf.data[idx];
            }
            let f_step = if fp_step >= 2 {
                fp_diis.push_and_extrapolate(&final_f, &comm)
            } else {
                let _ = fp_diis.push_and_extrapolate(&final_f, &comm);
                final_f.clone()
            };
            let (eps_new, c_new, p_new) = diagonalize_fock(&f_step, &x, &xt, n_occ)?;
            let diff = rms_diff(&p_new, &final_p);
            final_p = p_new;
            final_eps = eps_new;
            final_c = c_new;
            final_f = build_fock_matrix(&h_core, &final_p, &eri);
            let fps2 = matmul(&matmul(&final_f, &final_p), &s);
            let spf2 = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm2 = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm2.data[idx] = fps2.data[idx] - spf2.data[idx];
            }
            let comm_norm2 = frobenius_norm(&comm2);
            if diff < fp_density_tol && comm_norm2 < fp_comm_tol {
                let (eps_exact, c_exact, p_exact) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
                final_eps = eps_exact;
                final_c = c_exact;
                final_p = p_exact;
                final_f = build_fock_matrix(&h_core, &final_p, &eri);
                break;
            }
        }
    }

    let fps_chk = matmul(&matmul(&final_f, &final_p), &s);
    let spf_chk = matmul(&matmul(&s, &final_p), &final_f);
    let mut comm_chk = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        comm_chk.data[idx] = fps_chk.data[idx] - spf_chk.data[idx];
    }
    let final_comm_norm = frobenius_norm(&comm_chk);
    if !final_comm_norm.is_finite() || final_comm_norm > 1e-6 {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }
    let e_elec_final = compute_electronic_energy(&final_p, &h_core, &final_f);
    let e_tot_final = e_elec_final + e_nuc;

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
        mo_energies: final_eps,
        mo_occupations: occ,
        mo_coefficients: final_c.data,
        density_alpha: final_p.data,
        density_beta: None,
        fock_alpha: final_f.data,
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
