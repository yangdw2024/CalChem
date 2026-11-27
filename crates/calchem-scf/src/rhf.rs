use crate::diis::DiisState;
use crate::fock_builder::{RksFockAccumulator, RksFockBuilder};
use calchem_basis::BasisSet;
use calchem_int::{compute_ecp, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};

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
    rhf_with_guess(mol, basis, opts, None)
}

pub fn rhf_with_guess(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
    guess: Option<&crate::guess::InitialGuess>,
) -> Result<ScfWavefunction, CalcError> {
    let use_direct = crate::should_use_direct(basis, opts);
    if use_direct {
        crate::validate_direct_resources_and_options(basis, opts)?;
    } else {
        crate::validate_resources_and_options(basis, opts)?;
    }
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

    let fock_builder = RksFockBuilder::with_jfit_basis(
        mol,
        basis,
        use_direct,
        opts.density_fitting,
        opts.use_gpu,
        None,
        opts.jfit_basis,
    )?;
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

    // Projected adjacent-geometry orbital guess (falls back to SAD, then H_core)
    let projected = match guess {
        Some(crate::guess::InitialGuess::Restricted {
            n_ao,
            mo_coefficients,
        }) if *n_ao == n => crate::guess::project_occupied_density(mo_coefficients, n, n_occ, &s, 2.0),
        _ => None,
    };
    let (mut p, used_sad_guess) = if let Some(p_proj) = projected {
        (p_proj, false)
    } else {
        match crate::guess::sad_guess(mol, basis, &s, n_occ) {
            Ok(p_sad) => (p_sad, true),
            Err(_) => (diagonalize_fock(&h_core, &x, &xt, n_occ)?.2, false),
        }
    };
    let mut f;

    let mut diis = DiisState::from_scf_options(opts);
    let mut diis_pushed = 0usize;
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;
    let mut prev_diis_err = f64::INFINITY;

    let mut acc = RksFockAccumulator::new(n, &p, opts.conv_density);

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        let g_2e = acc.update(
            &fock_builder,
            basis,
            &p,
            1.0,
            None,
            iter,
            use_direct,
            last_rms_d,
            prev_diis_err,
        );

        f = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            f.data[idx] = h_core.data[idx] + g_2e.data[idx];
        }

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

        let skip_diis_this_step = used_sad_guess && iter == 1;
        let f_extrap = if opts.use_diis && !skip_diis_this_step {
            diis_pushed += 1;
            let extrap = diis
                .try_push_and_extrapolate(&f, &err)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            if diis_pushed >= 2 {
                extrap
            } else {
                f.clone()
            }
        } else {
            f.clone()
        };

        let (_eps_new, _c_new, mut p_new) = diagonalize_fock(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if opts.damping > 0.0 && opts.damping < 1.0 {
            opts.damping
        } else if !skip_diis_this_step && diis_err > 0.1 && diis_err > prev_diis_err * 1.15 {
            0.3 // damp only if DIIS oscillates or diverges
        } else {
            0.0 // full DIIS speed
        };
        if !skip_diis_this_step {
            prev_diis_err = diis_err;
        }
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
        let e_plateau = (1e-11 * (e_tot.abs() / 100.0).max(1.0)).max(opts.conv_energy * 0.1);
        let d_plateau = (opts.conv_density * 0.5)
            .max(1e-10)
            .min(opts.conv_density.max(1e-8));
        let tight_plateau =
            iter > 8 && delta_e < e_plateau && rms_d < d_plateau && diis_err < grad_tol;
        let criteria_met = iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau);
        if criteria_met {
            if acc.in_tight_pass() || !use_direct {
                converged = true;
                break;
            } else {
                acc.request_tight_pass();
            }
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Tighten the final fixed-point step to ensure gradient-grade stationarity with 1 clean build.
    let tight_base = acc.tight_base();
    let mut final_f = fock_builder.build_fock_with_thresh(&h_core, basis, &p, 1.0, tight_base);
    let mut final_p = p.clone();
    let fp_density_tol = opts.conv_density;
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-6);

    let comm_0 = {
        let fps = matmul(&matmul(&final_f, &final_p), &s);
        let spf = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            comm.data[idx] = fps.data[idx] - spf.data[idx];
        }
        comm
    };
    let comm_norm_0 = frobenius_norm(&comm_0);
    let (eps_cand, c_cand, p_cand) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
    let diff_cand = rms_diff(&p_cand, &final_p);

    let mut final_eps = eps_cand;
    let mut final_c = c_cand;

    if !(comm_norm_0 <= fp_comm_tol && diff_cand < fp_density_tol) {
        let mut fp_diis = opts.use_diis.then(|| DiisState::new(opts.diis_max_vecs.max(6)));
        if let Some(ref mut d) = fp_diis {
            let _ = d
                .try_push_and_extrapolate(&final_f, &comm_0)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
        }
        final_p = p_cand;
        final_f = fock_builder.build_fock_with_thresh(&h_core, basis, &final_p, 1.0, tight_base);

        for fp_step in 1..20 {
            let fps = matmul(&matmul(&final_f, &final_p), &s);
            let spf = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm.data[idx] = fps.data[idx] - spf.data[idx];
            }
            let comm_norm = frobenius_norm(&comm);
            if comm_norm <= fp_comm_tol && fp_step == 1 {
                let (eps_c, c_c, p_c) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
                if rms_diff(&p_c, &final_p) < fp_density_tol {
                    final_eps = eps_c;
                    final_c = c_c;
                    break;
                }
            }
            let f_step = if let Some(ref mut d) = fp_diis {
                let ex = d
                    .try_push_and_extrapolate(&final_f, &comm)
                    .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
                if fp_step >= 2 {
                    ex
                } else {
                    final_f.clone()
                }
            } else {
                final_f.clone()
            };
            let (eps_new, c_new, mut p_new) = diagonalize_fock(&f_step, &x, &xt, n_occ)?;
            if !opts.use_diis {
                let damp = if opts.damping > 0.0 && opts.damping < 1.0 {
                    opts.damping
                } else {
                    0.2
                };
                for idx in 0..p_new.data.len() {
                    p_new.data[idx] = (1.0 - damp) * p_new.data[idx] + damp * final_p.data[idx];
                }
            }
            let diff = rms_diff(&p_new, &final_p);
            final_p = p_new;
            final_eps = eps_new;
            final_c = c_new;
            final_f =
                fock_builder.build_fock_with_thresh(&h_core, basis, &final_p, 1.0, tight_base);
            let fps2 = matmul(&matmul(&final_f, &final_p), &s);
            let spf2 = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm2 = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm2.data[idx] = fps2.data[idx] - spf2.data[idx];
            }
            let comm_norm2 = frobenius_norm(&comm2);
            if diff < fp_density_tol && comm_norm2 < fp_comm_tol {
                if fp_step < 2 {
                    break;
                }
                let (eps_exact, c_exact, p_exact) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
                final_eps = eps_exact;
                final_c = c_exact;
                final_p = p_exact;
                final_f = fock_builder
                    .build_fock_with_thresh(&h_core, basis, &final_p, 1.0, tight_base);
                let fps_exact = matmul(&matmul(&final_f, &final_p), &s);
                let spf_exact = matmul(&matmul(&s, &final_p), &final_f);
                let norm_exact = fps_exact
                    .data
                    .iter()
                    .zip(&spf_exact.data)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm_exact.is_finite() && norm_exact <= fp_comm_tol {
                    break;
                }
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
    if !final_comm_norm.is_finite() || final_comm_norm > fp_comm_tol {
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

    let (gpu_kernel_invocations, gpu_fallback_invocations, rij_metric_truncated_modes) =
        fock_builder.telemetry();
    let diis_disk_backed = diis.is_disk_backed();

    Ok(ScfWavefunction {
        density_fitting: opts.density_fitting,
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
            gpu_kernel_invocations,
            gpu_fallback_invocations,
            diis_disk_backed,
            rij_metric_truncated_modes,
            jfit_basis: opts.jfit_basis,
        },
    })
}
