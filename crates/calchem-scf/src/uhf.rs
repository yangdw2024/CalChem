//! Unrestricted Hartree-Fock (UHF) for open-shell systems (radical ions, doublets, triplets).
//!
//! Fock operators (per spin, `P^t = P^a + P^b`):
//!   `F^a = h + J[P^t] - K[P^a]`,  `F^b = h + J[P^t] - K[P^b]`
//! Energy:
//!   `E = 1/2 * sum_{mu nu} [ P^t h + P^a F^a + P^b F^b ]_{mu nu} + E_nn`
//! DIIS runs on the stacked `[F^a; F^b]` / `[e^a; e^b]` matrices so both spins share
//! one set of extrapolation coefficients.

use crate::diis::DiisState;
use crate::fock_builder::{UksFockAccumulator, UksFockBuilder};
use calchem_basis::BasisSet;
use calchem_int::{compute_ecp, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, UhfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};

/// Builds `(F^a, F^b)` from per-spin densities with explicit ERI screening threshold.
fn build_uhf_fock_with_thresh(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    builder: &UksFockBuilder,
    basis: &BasisSet,
    base_thresh: f64,
) -> (Matrix2D, Matrix2D) {
    let (ga, gb) = builder.build_2e_with_thresh(basis, pa, pb, 1.0, None, base_thresh);
    let n = h_core.rows;
    let mut fa = Matrix2D::zeros(n, n);
    let mut fb = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        fa.data[idx] = h_core.data[idx] + ga.data[idx];
        fb.data[idx] = h_core.data[idx] + gb.data[idx];
    }
    (fa, fb)
}

fn uhf_electronic_energy(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    fa: &Matrix2D,
    fb: &Matrix2D,
) -> f64 {
    let mut e = 0.0;
    for i in 0..h_core.data.len() {
        let pt = pa.data[i] + pb.data[i];
        e += pt * h_core.data[i] + pa.data[i] * fa.data[i] + pb.data[i] * fb.data[i];
    }
    0.5 * e
}

/// Diagonalizes `F` in the orthogonal basis; returns `(eps, C, P_spin)` with `P = C_occ C_occ^T`.
fn diagonalize_spin(
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
            let mut s = 0.0;
            for m in 0..n_occ {
                s += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, s);
        }
    }
    Ok((eps, c, p))
}

fn commutator(f: &Matrix2D, p: &Matrix2D, s: &Matrix2D) -> Matrix2D {
    let fps = matmul(&matmul(f, p), s);
    let spf = matmul(&matmul(s, p), f);
    let mut e = Matrix2D::zeros(f.rows, f.cols);
    for i in 0..e.data.len() {
        e.data[i] = fps.data[i] - spf.data[i];
    }
    e
}

fn stack(a: &Matrix2D, b: &Matrix2D) -> Matrix2D {
    let mut out = Matrix2D::zeros(a.rows + b.rows, a.cols);
    out.data[..a.data.len()].copy_from_slice(&a.data);
    out.data[a.data.len()..].copy_from_slice(&b.data);
    out
}

fn unstack(m: &Matrix2D, n: usize) -> (Matrix2D, Matrix2D) {
    let mut a = Matrix2D::zeros(n, n);
    let mut b = Matrix2D::zeros(n, n);
    a.data.copy_from_slice(&m.data[..n * n]);
    b.data.copy_from_slice(&m.data[n * n..]);
    (a, b)
}

/// `<S^2> = S_z(S_z+1) + n_beta - tr(P^a S P^b S)`.
pub fn uhf_s_squared(
    pa: &Matrix2D,
    pb: &Matrix2D,
    s: &Matrix2D,
    n_alpha: usize,
    n_beta: usize,
) -> f64 {
    let sz = 0.5 * (n_alpha as f64 - n_beta as f64);
    let m = matmul(&matmul(&matmul(pa, s), pb), s);
    let mut tr = 0.0;
    for i in 0..m.rows {
        tr += m.get(i, i);
    }
    sz * (sz + 1.0) + n_beta as f64 - tr
}

/// Runs an Unrestricted Hartree-Fock calculation.
pub fn uhf(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    uhf_with_guess(mol, basis, opts, None)
}

/// Runs an Unrestricted Hartree-Fock calculation with an optional adjacent-geometry orbital guess.
pub fn uhf_with_guess(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
    guess: Option<&crate::guess::InitialGuess>,
) -> Result<UhfWavefunction, CalcError> {
    let use_direct = crate::should_use_direct(basis, opts);
    if use_direct {
        crate::validate_direct_resources_for_method(basis, opts, true)?;
    } else {
        crate::validate_open_shell_stored_resources(basis, opts)?;
    }
    let (n_alpha, n_beta) = basis.validate_uhf_valence(mol)?;
    let n = basis.n_ao;
    if n == 0 || n_alpha > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_alpha} alpha electrons"
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
    let fock_builder = UksFockBuilder::with_jfit_basis(
        mol,
        basis,
        use_direct,
        opts.density_fitting,
        opts.use_gpu,
        None,
        opts.jfit_basis,
    )?;
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // Symmetric orthogonalization X = U s^{-1/2} U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {val:.3e})"
            )));
        }
        s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Projected adjacent-geometry orbital guess (falls back to Core-Hamiltonian guess)
    let projected = match guess {
        Some(crate::guess::InitialGuess::Unrestricted {
            n_ao,
            mo_alpha,
            mo_beta,
        }) if *n_ao == n => {
            let pa_opt = crate::guess::project_occupied_density(mo_alpha, n, n_alpha, &s, 1.0);
            let pb_opt = if n_beta == 0 {
                Some(Matrix2D::zeros(n, n))
            } else {
                crate::guess::project_occupied_density(mo_beta, n, n_beta, &s, 1.0)
            };
            pa_opt.zip(pb_opt)
        }
        _ => None,
    };
    let (mut pa, mut pb, used_sad_guess) = if let Some((pa_proj, pb_proj)) = projected {
        (pa_proj, pb_proj, false)
    } else {
        match crate::guess::sad_guess_uhf(mol, basis, &s, n_alpha, n_beta) {
            Ok((pa_sad, pb_sad)) => (pa_sad, pb_sad, true),
            Err(_) => {
                let (_, _, pa_h) = diagonalize_spin(&h_core, &x, &xt, n_alpha)?;
                let (_, _, pb_h) = diagonalize_spin(&h_core, &x, &xt, n_beta)?;
                (pa_h, pb_h, false)
            }
        }
    };

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

    let mut acc = UksFockAccumulator::new(n, &pa, &pb, opts.conv_density);

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        let (ga_cur, gb_cur) = acc.update(
            &fock_builder,
            basis,
            &pa,
            &pb,
            1.0,
            None,
            iter,
            use_direct,
            last_rms_d,
            prev_diis_err,
        );

        let mut fa = Matrix2D::zeros(n, n);
        let mut fb = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            fa.data[idx] = h_core.data[idx] + ga_cur.data[idx];
            fb.data[idx] = h_core.data[idx] + gb_cur.data[idx];
        }

        let e_tot = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb) + e_nuc;
        energy_history.push(e_tot);

        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let skip_diis_this_step = used_sad_guess && iter == 1;
        let f_stacked = stack(&fa, &fb);
        let (fa_ex, fb_ex) = if opts.use_diis && !skip_diis_this_step {
            diis_pushed += 1;
            let ex = diis
                .try_push_and_extrapolate(&f_stacked, &err)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            if diis_pushed >= 2 {
                unstack(&ex, n)
            } else {
                (fa.clone(), fb.clone())
            }
        } else {
            (fa.clone(), fb.clone())
        };

        let (_, _, mut pa_new) = diagonalize_spin(&fa_ex, &x, &xt, n_alpha)?;
        let (_, _, mut pb_new) = diagonalize_spin(&fb_ex, &x, &xt, n_beta)?;

        let effective_damping = if opts.damping > 0.0 && opts.damping < 1.0 {
            opts.damping
        } else if !skip_diis_this_step && diis_err > 0.15 && diis_err > prev_diis_err * 1.2 {
            0.3
        } else {
            0.0
        };
        if !skip_diis_this_step {
            prev_diis_err = diis_err;
        }
        if effective_damping > 0.0 {
            for i in 0..pa_new.data.len() {
                pa_new.data[i] =
                    (1.0 - effective_damping) * pa_new.data[i] + effective_damping * pa.data[i];
                pb_new.data[i] =
                    (1.0 - effective_damping) * pb_new.data[i] + effective_damping * pb.data[i];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        rms_history.push(rms_d);
        last_delta_e = delta_e;
        last_rms_d = rms_d;
        pa = pa_new;
        pb = pb_new;
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

    // Fixed-point verification: 1 clean full tight rebuild at (pa, pb), skipping Build #2 when already converged.
    let tight_base = acc.tight_base();
    let (mut fa, mut fb) =
        build_uhf_fock_with_thresh(&h_core, &pa, &pb, &fock_builder, basis, tight_base);
    let fp_density_tol = opts.conv_density;
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-6);

    let err_0 = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
    let comm_0 = frobenius_norm(&err_0);
    let (ea_cand, ca_cand, pa_cand) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
    let (eb_cand, cb_cand, pb_cand) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
    let diff_cand = rms_diff(&pa_cand, &pa).max(rms_diff(&pb_cand, &pb));

    let mut eps_a = ea_cand;
    let mut eps_b = eb_cand;
    let mut ca = ca_cand;
    let mut cb = cb_cand;

    if !(comm_0 <= fp_comm_tol && diff_cand < fp_density_tol) {
        let mut polish_diis = opts.use_diis.then(|| DiisState::new(opts.diis_max_vecs.max(6)));
        if let Some(ref mut d) = polish_diis {
            let _ = d
                .try_push_and_extrapolate(&stack(&fa, &fb), &err_0)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
        }
        pa = pa_cand;
        pb = pb_cand;
        let (fa_1, fb_1) =
            build_uhf_fock_with_thresh(&h_core, &pa, &pb, &fock_builder, basis, tight_base);
        fa = fa_1;
        fb = fb_1;

        for fp_step in 1..60 {
            let err_a = commutator(&fa, &pa, &s);
            let err_b = commutator(&fb, &pb, &s);
            let err_st = stack(&err_a, &err_b);
            if frobenius_norm(&err_st) <= fp_comm_tol && fp_step == 1 {
                let (ea_c, ca_c, pa_c) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
                let (eb_c, cb_c, pb_c) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
                if rms_diff(&pa_c, &pa).max(rms_diff(&pb_c, &pb)) < fp_density_tol {
                    eps_a = ea_c;
                    eps_b = eb_c;
                    ca = ca_c;
                    cb = cb_c;
                    break;
                }
            }
            let (fa_step, fb_step) = if let Some(ref mut d) = polish_diis {
                let extrap = d
                    .try_push_and_extrapolate(&stack(&fa, &fb), &err_st)
                    .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
                if fp_step >= 2 {
                    unstack(&extrap, n)
                } else {
                    (fa.clone(), fb.clone())
                }
            } else {
                (fa.clone(), fb.clone())
            };
            let (ea, ca_new, mut pa_new) = diagonalize_spin(&fa_step, &x, &xt, n_alpha)?;
            let (eb, cb_new, mut pb_new) = diagonalize_spin(&fb_step, &x, &xt, n_beta)?;
            if !opts.use_diis {
                let damp = if opts.damping > 0.0 && opts.damping < 1.0 {
                    opts.damping
                } else {
                    0.2
                };
                for idx in 0..pa_new.data.len() {
                    pa_new.data[idx] = (1.0 - damp) * pa_new.data[idx] + damp * pa.data[idx];
                    pb_new.data[idx] = (1.0 - damp) * pb_new.data[idx] + damp * pb.data[idx];
                }
            }
            let diff = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
            pa = pa_new;
            pb = pb_new;
            eps_a = ea;
            eps_b = eb;
            ca = ca_new;
            cb = cb_new;
            let (fa_n, fb_n) =
                build_uhf_fock_with_thresh(&h_core, &pa, &pb, &fock_builder, basis, tight_base);
            fa = fa_n;
            fb = fb_n;
            let comm_norm =
                frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
            if diff < fp_density_tol && comm_norm < fp_comm_tol {
                if fp_step < 2 {
                    break;
                }
                let (ea_ex, ca_ex, pa_ex) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
                let (eb_ex, cb_ex, pb_ex) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
                eps_a = ea_ex;
                eps_b = eb_ex;
                ca = ca_ex;
                cb = cb_ex;
                pa = pa_ex;
                pb = pb_ex;
                let (fa_ex, fb_ex) = build_uhf_fock_with_thresh(
                    &h_core,
                    &pa,
                    &pb,
                    &fock_builder,
                    basis,
                    tight_base,
                );
                fa = fa_ex;
                fb = fb_ex;
                let exact_comm =
                    frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
                if exact_comm.is_finite() && exact_comm <= fp_comm_tol {
                    break;
                }
            }
        }
    }

    let final_comm = frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
    if !final_comm.is_finite() || final_comm > fp_comm_tol {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }
    let e_elec = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb);
    let s2 = uhf_s_squared(&pa, &pb, &s, n_alpha, n_beta);
    let (gpu_kernel_invocations, gpu_fallback_invocations, rij_metric_truncated_modes) =
        fock_builder.telemetry();
    let diis_disk_backed = diis.is_disk_backed();

    Ok(UhfWavefunction {
        density_fitting: opts.density_fitting,
        converged: true,
        iterations: used_iter,
        total_energy_hartree: e_elec + e_nuc,
        electronic_energy_hartree: e_elec,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_alpha,
        n_beta,
        mo_energies_alpha: eps_a,
        mo_energies_beta: eps_b,
        mo_coefficients_alpha: ca.data,
        mo_coefficients_beta: cb.data,
        density_alpha: pa.data,
        density_beta: pb.data,
        s_squared: s2,
        overlap_matrix: s.data,
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
