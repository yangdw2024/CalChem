//! Closed-shell Restricted Kohn-Sham (RKS) Density Functional Theory solver in pure Rust.
//!
//! Supports pure LDA/GGA (`c_HF = 0`, skipping exact-exchange ERI contraction) and
//! global hybrid functionals (`B3LYP`, `B3LYP5`, `PBE0`) as well as scalar relativistic ECPs.

use crate::diis::DiisState;
use crate::fock_builder::RksFockBuilder;
use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, RksVxcEngine, XcFunctional};
use calchem_int::{compute_ecp, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};
use std::time::{Duration, Instant};

fn assemble_rks_fock_and_energy(
    h_core: &Matrix2D,
    p: &Matrix2D,
    builder: &RksFockBuilder,
    vxc_engine: &RksVxcEngine,
    basis: &BasisSet,
    grid: &MolecularGrid,
    xc: XcFunctional,
) -> (Matrix2D, f64, f64) {
    let n = h_core.rows;
    let g_2e = builder.build_2e(
        basis,
        p,
        xc.exact_exchange_fraction(),
        xc.range_separated_params(),
    );
    let (e_xc, v_xc) = vxc_engine.evaluate(basis, grid, xc, p);

    let mut f = Matrix2D::zeros(n, n);
    let mut e_1e_2e = 0.0;
    for i in 0..n * n {
        f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i];
        e_1e_2e += p.data[i] * (h_core.data[i] + 0.5 * g_2e.data[i]);
    }
    (f, e_1e_2e + e_xc, e_xc)
}

fn diagonalize_fock_with_shift(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
    cp_prev: Option<&Matrix2D>,
    level_shift: f64,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let mut fp = matmul(&matmul(xt, fock), x);
    if level_shift > 0.0 {
        if let Some(cp_old) = cp_prev {
            // Shift virtual subspace by +level_shift: F'_shift = F' + shift * (I - P'_occ)
            // where P'_occ = C'_occ * C'_occ^T
            for i in 0..n {
                fp.data[i * n + i] += level_shift;
                for j in 0..=i {
                    let mut p_occ_ij = 0.0;
                    for m in 0..n_occ {
                        p_occ_ij += cp_old.data[i * n + m] * cp_old.data[j * n + m];
                    }
                    let delta = level_shift * p_occ_ij;
                    fp.data[i * n + j] -= delta;
                    if i != j {
                        fp.data[j * n + i] -= delta;
                    }
                }
            }
        }
    }
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    // Compute P = 2 * C_occ * C_occ^T using faer-backed matmul for large n or direct loop
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        let c_mu = &c.data[mu * n..mu * n + n_occ];
        for nu in 0..=mu {
            let c_nu = &c.data[nu * n..nu * n + n_occ];
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c_mu[m] * c_nu[m];
            }
            let val = 2.0 * sum;
            p.data[mu * n + nu] = val;
            p.data[nu * n + mu] = val;
        }
    }
    Ok((eps, c, p, cp))
}

fn diagonalize_fock(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let (eps, c, p, _) = diagonalize_fock_with_shift(fock, x, xt, n_occ, None, 0.0)?;
    Ok((eps, c, p))
}

/// Runs a closed-shell Restricted Kohn-Sham (RKS) DFT calculation.
pub fn rks(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    rks_with_grid(mol, basis, xc, &grid, opts)
}

/// Runs a closed-shell Restricted Kohn-Sham (RKS) DFT calculation on a prebuilt [`MolecularGrid`].
pub fn rks_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    rks_with_grid_and_guess(mol, basis, xc, grid, opts, None)
}

/// Runs a closed-shell Restricted Kohn-Sham (RKS) DFT calculation on a prebuilt [`MolecularGrid`]
/// with an optional adjacent-geometry orbital projection guess (falls back to SAD if invalid).
pub fn rks_with_grid_and_guess(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    opts: &ScfOptions,
    guess: Option<&crate::guess::InitialGuess>,
) -> Result<ScfWavefunction, CalcError> {
    let use_direct = crate::should_use_direct(basis, opts);
    if use_direct {
        crate::validate_direct_resources_and_options(basis, opts)?;
    } else {
        crate::validate_resources_and_options(basis, opts)?;
    }
    xc.validate_native()?;
    let (n_alpha, n_beta) = basis.validate_rhf_valence(mol)?;
    if n_alpha != n_beta {
        return Err(CalcError::UnsupportedMethod(
            "RKS requires a closed-shell singlet state (n_alpha == n_beta)".into(),
        ));
    }
    let n_occ = n_alpha;
    let n = basis.n_ao;
    if n == 0 || n_occ > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_occ} occupied orbitals"
        )));
    }

    let profile = std::env::var_os("CALCHEM_PROFILE").is_some();
    let t_start = Instant::now();
    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let d_1e = t_start.elapsed();

    let t_b0 = Instant::now();
    let fock_builder = RksFockBuilder::with_jfit_basis(
        mol,
        basis,
        use_direct,
        opts.density_fitting,
        opts.use_gpu,
        xc.range_separated_params(),
        opts.jfit_basis,
    )?;
    let d_builder = t_b0.elapsed();
    let t_vxc_init = Instant::now();
    let vxc_engine = RksVxcEngine::new_with_gpu(basis, grid, opts.use_gpu);
    if profile {
        eprintln!(
            "[rks] vxc_engine ready in {:.3}s",
            t_vxc_init.elapsed().as_secs_f64()
        );
    }
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let e_disp = if xc == XcFunctional::Wb97xd {
        calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?.0
    } else {
        0.0
    };

    let t_o0 = Instant::now();
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {val:.3e})"
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let projected = match guess {
        Some(crate::guess::InitialGuess::Restricted {
            n_ao,
            mo_coefficients,
        }) if *n_ao == n => crate::guess::project_occupied_density(mo_coefficients, n, n_occ, &s, 2.0),
        _ => None,
    };
    let (mut p, used_sad_guess, mut cp_prev) = if let Some(p_proj) = projected {
        (p_proj, false, None)
    } else {
        match crate::guess::sad_guess(mol, basis, &s, n_occ) {
            Ok(p_sad) => (p_sad, true, None),
            Err(_) => {
                let (_, _, p_h, cp_h) = diagonalize_fock_with_shift(&h_core, &x, &xt, n_occ, None, 0.0)?;
                (p_h, false, Some(cp_h))
            }
        }
    };
    let d_orth = t_o0.elapsed();
    let mut f;

    let mut d_2e = Duration::ZERO;
    let mut d_vxc = Duration::ZERO;
    let mut d_diag = Duration::ZERO;
    let mut d_comm = Duration::ZERO;

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

    let mut g_2e = Matrix2D::zeros(n, n);
    let mut p_prev = p.clone();

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        let t_2e0 = Instant::now();
        if iter == 1 || iter % 8 == 0 || !use_direct {
            g_2e = fock_builder.build_2e(
                basis,
                &p,
                xc.exact_exchange_fraction(),
                xc.range_separated_params(),
            );
        } else {
            let mut delta_p = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                delta_p.data[idx] = p.data[idx] - p_prev.data[idx];
            }
            let delta_g = fock_builder.build_2e(
                basis,
                &delta_p,
                xc.exact_exchange_fraction(),
                xc.range_separated_params(),
            );
            for idx in 0..n * n {
                g_2e.data[idx] += delta_g.data[idx];
            }
        }
        d_2e += t_2e0.elapsed();
        p_prev = p.clone();

        let t_vx0 = Instant::now();
        let (e_xc, v_xc) = vxc_engine.evaluate(basis, grid, xc, &p);
        d_vxc += t_vx0.elapsed();
        f = Matrix2D::zeros(n, n);
        let mut e_1e_2e = 0.0;
        for i in 0..n * n {
            f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i];
            e_1e_2e += p.data[i] * (h_core.data[i] + 0.5 * g_2e.data[i]);
        }
        let e_tot = e_1e_2e + e_xc + e_nuc + e_disp;
        energy_history.push(e_tot);

        let t_c0 = Instant::now();
        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);
        d_comm += t_c0.elapsed();

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

        let level_shift = if n >= 40 && !skip_diis_this_step && diis_err > 0.08 {
            0.2
        } else {
            0.0
        };
        let t_d0 = Instant::now();
        let (_eps_new, _c_new, mut p_new, cp_new) = diagonalize_fock_with_shift(
            &f_extrap,
            &x,
            &xt,
            n_occ,
            cp_prev.as_ref(),
            level_shift,
        )?;
        cp_prev = Some(cp_new);
        d_diag += t_d0.elapsed();

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

    // Gradient-grade fixed-point tightening (using DIIS-assisted polishing if needed).
    let t_pol0 = Instant::now();
    let (mut final_f, mut e_elec_final, _) =
        assemble_rks_fock_and_energy(&h_core, &p, &fock_builder, &vxc_engine, basis, grid, xc);
    let mut final_eps = Vec::new();
    let mut final_c = Matrix2D::zeros(n, n);
    let mut final_p = p.clone();
    let fp_density_tol = opts.conv_density;
    // A fixed quadrature defines a deterministic discretized functional. Integration
    // error relative to the continuum is not permission to accept an arbitrary SCF residual.
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-6);
    let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));
    let max_fp_steps = 50;

    for fp_step in 0..max_fp_steps {
        let fps = matmul(&matmul(&final_f, &final_p), &s);
        let spf = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm.data[i] = fps.data[i] - spf.data[i];
        }
        let _comm_norm = frobenius_norm(&comm);
        let f_step = if fp_step >= 2 {
            fp_diis
                .try_push_and_extrapolate(&final_f, &comm)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?
        } else {
            fp_diis
                .try_push_and_extrapolate(&final_f, &comm)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            final_f.clone()
        };

        let (eps_new, c_new, p_new) = diagonalize_fock(&f_step, &x, &xt, n_occ)?;
        let diff = rms_diff(&p_new, &final_p);
        final_p = p_new;
        final_eps = eps_new;
        final_c = c_new;
        let (f_rebuilt, e_rebuilt, _) =
            assemble_rks_fock_and_energy(&h_core, &final_p, &fock_builder, &vxc_engine, basis, grid, xc);
        final_f = f_rebuilt;
        e_elec_final = e_rebuilt;

        let fps2 = matmul(&matmul(&final_f, &final_p), &s);
        let spf2 = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm2 = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm2.data[i] = fps2.data[i] - spf2.data[i];
        }
        let comm_norm2 = frobenius_norm(&comm2);

        if diff < fp_density_tol && comm_norm2 < fp_comm_tol {
            // A physical (non-extrapolated) diagonalization already produced these orbitals.
            if fp_step < 2 {
                break;
            }
            // Do one final pure diagonalization of `final_f = F[final_p]` so (C, eps) match F[P]
            let (eps_exact, c_exact, p_exact) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
            final_eps = eps_exact;
            final_c = c_exact;
            final_p = p_exact;
            let (f_exact, e_exact, _) =
                assemble_rks_fock_and_energy(&h_core, &final_p, &fock_builder, &vxc_engine, basis, grid, xc);
            final_f = f_exact;
            e_elec_final = e_exact;
            // A pure canonicalization can amplify a small DIIS residual. Verify
            // the rebuilt physical Fock again instead of accepting that extra step.
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

    {
        let fps = matmul(&matmul(&final_f, &final_p), &s);
        let spf = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm.data[i] = fps.data[i] - spf.data[i];
        }
        let comm_norm = frobenius_norm(&comm);
        if !comm_norm.is_finite() || comm_norm > fp_comm_tol {
            return Err(CalcError::NotConverged {
                iterations: used_iter,
                delta_e: last_delta_e,
                rms_d: last_rms_d,
            });
        }
    }

    let d_polish = t_pol0.elapsed();
    if profile {
        eprintln!(
            "[rks_profile] iters={} 1e={:.3}s builder_init={:.3}s orth_guess={:.3}s scf_2e={:.3}s scf_vxc={:.3}s scf_comm={:.3}s scf_diag={:.3}s polish={:.3}s",
            used_iter,
            d_1e.as_secs_f64(),
            d_builder.as_secs_f64(),
            d_orth.as_secs_f64(),
            d_2e.as_secs_f64(),
            d_vxc.as_secs_f64(),
            d_comm.as_secs_f64(),
            d_diag.as_secs_f64(),
            d_polish.as_secs_f64()
        );
    }

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
