//! Unrestricted Kohn-Sham (UKS) Density Functional Theory solver for open-shell systems
//! (radical cations/anions, doublets, triplets, single-electron atoms) in pure Rust.

use crate::diis::DiisState;
use crate::fock_builder::UksFockBuilder;
use crate::uhf::uhf_s_squared;
use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_uks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{compute_ecp, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, UhfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};

fn assemble_uks_fock_and_energy(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    builder: &UksFockBuilder,
    basis: &BasisSet,
    grid: &MolecularGrid,
    xc: XcFunctional,
) -> (Matrix2D, Matrix2D, f64, f64) {
    let n = h_core.rows;
    let (ga, gb) = builder.build_2e(
        basis,
        pa,
        pb,
        xc.exact_exchange_fraction(),
        xc.range_separated_params(),
    );
    let (e_xc, va, vb) = evaluate_vxc_uks(basis, grid, xc, pa, pb);

    let mut fa = Matrix2D::zeros(n, n);
    let mut fb = Matrix2D::zeros(n, n);
    let mut e_1e_2e = 0.0;
    for i in 0..n * n {
        fa.data[i] = h_core.data[i] + ga.data[i] + va.data[i];
        fb.data[i] = h_core.data[i] + gb.data[i] + vb.data[i];
        let pt_i = pa.data[i] + pb.data[i];
        e_1e_2e +=
            pt_i * h_core.data[i] + 0.5 * (pa.data[i] * ga.data[i] + pb.data[i] * gb.data[i]);
    }
    (fa, fb, e_1e_2e + e_xc, e_xc)
}

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

/// Runs an Unrestricted Kohn-Sham (UKS) DFT calculation.
pub fn uks(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    uks_with_grid(mol, basis, xc, &grid, opts)
}

/// Runs an Unrestricted Kohn-Sham (UKS) DFT calculation on a prebuilt [`MolecularGrid`].
pub fn uks_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    let use_direct = crate::should_use_direct(basis, opts);
    if use_direct {
        crate::validate_direct_resources_for_method(basis, opts, true)?;
    } else {
        crate::validate_open_shell_stored_resources(basis, opts)?;
    }
    xc.validate_native()?;
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
    let fock_builder = UksFockBuilder::new_with_options(
        mol,
        basis,
        use_direct,
        opts.density_fitting,
        opts.use_gpu,
        xc.range_separated_params(),
    )?;
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let e_disp = if xc == XcFunctional::Wb97xd {
        calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?.0
    } else {
        0.0
    };

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

    let (_, _, mut pa) = diagonalize_spin(&h_core, &x, &xt, n_alpha)?;
    let (_, _, mut pb) = diagonalize_spin(&h_core, &x, &xt, n_beta)?;

    let mut diis = DiisState::from_scf_options(opts);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();
    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    let mut ga_cur = Matrix2D::zeros(n, n);
    let mut gb_cur = Matrix2D::zeros(n, n);
    let mut pa_prev = pa.clone();
    let mut pb_prev = pb.clone();

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        if iter == 1 || iter % 8 == 0 || !use_direct || opts.density_fitting {
            let (ga, gb) = fock_builder.build_2e(
                basis,
                &pa,
                &pb,
                xc.exact_exchange_fraction(),
                xc.range_separated_params(),
            );
            ga_cur = ga;
            gb_cur = gb;
        } else {
            let mut dpa = Matrix2D::zeros(n, n);
            let mut dpb = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                dpa.data[idx] = pa.data[idx] - pa_prev.data[idx];
                dpb.data[idx] = pb.data[idx] - pb_prev.data[idx];
            }
            let (dga, dgb) = fock_builder.build_2e(
                basis,
                &dpa,
                &dpb,
                xc.exact_exchange_fraction(),
                xc.range_separated_params(),
            );
            for idx in 0..n * n {
                ga_cur.data[idx] += dga.data[idx];
                gb_cur.data[idx] += dgb.data[idx];
            }
        }
        pa_prev = pa.clone();
        pb_prev = pb.clone();

        let (e_xc, va, vb) = evaluate_vxc_uks(basis, grid, xc, &pa, &pb);
        let mut fa = Matrix2D::zeros(n, n);
        let mut fb = Matrix2D::zeros(n, n);
        let mut e_1e_2e = 0.0;
        for i in 0..n * n {
            fa.data[i] = h_core.data[i] + ga_cur.data[i] + va.data[i];
            fb.data[i] = h_core.data[i] + gb_cur.data[i] + vb.data[i];
            let pt_i = pa.data[i] + pb.data[i];
            e_1e_2e += pt_i * h_core.data[i]
                + 0.5 * (pa.data[i] * ga_cur.data[i] + pb.data[i] * gb_cur.data[i]);
        }
        let e_tot = e_1e_2e + e_xc + e_nuc + e_disp;
        energy_history.push(e_tot);

        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_stacked = stack(&fa, &fb);
        let (fa_ex, fb_ex) = if opts.use_diis {
            let ex = diis
                .try_push_and_extrapolate(&f_stacked, &err)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            if iter >= 2 {
                unstack(&ex, n)
            } else {
                (fa.clone(), fb.clone())
            }
        } else {
            (fa.clone(), fb.clone())
        };

        let (_, _, mut pa_new) = diagonalize_spin(&fa_ex, &x, &xt, n_alpha)?;
        let (_, _, mut pb_new) = diagonalize_spin(&fb_ex, &x, &xt, n_beta)?;

        if opts.damping > 0.0 && opts.damping < 1.0 {
            let d = opts.damping;
            for i in 0..pa_new.data.len() {
                pa_new.data[i] = (1.0 - d) * pa_new.data[i] + d * pa.data[i];
                pb_new.data[i] = (1.0 - d) * pb_new.data[i] + d * pb.data[i];
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

    // Gradient-grade fixed-point tightening
    let (mut fa, mut fb, mut e_elec_final, _) =
        assemble_uks_fock_and_energy(&h_core, &pa, &pb, &fock_builder, basis, grid, xc);
    let mut eps_a = Vec::new();
    let mut eps_b = Vec::new();
    let mut ca = Matrix2D::zeros(n, n);
    let mut cb = Matrix2D::zeros(n, n);
    let fp_density_tol = opts.conv_density;
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-6);
    let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));

    for fp_step in 0..60 {
        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let f_stacked = stack(&fa, &fb);
        let (fa_step, fb_step) = if fp_step >= 2 {
            let ex = fp_diis
                .try_push_and_extrapolate(&f_stacked, &err)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            unstack(&ex, n)
        } else {
            fp_diis
                .try_push_and_extrapolate(&f_stacked, &err)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            (fa.clone(), fb.clone())
        };

        let (ea, ca_new, pa_new) = diagonalize_spin(&fa_step, &x, &xt, n_alpha)?;
        let (eb, cb_new, pb_new) = diagonalize_spin(&fb_step, &x, &xt, n_beta)?;
        let diff = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        pa = pa_new;
        pb = pb_new;
        eps_a = ea;
        eps_b = eb;
        ca = ca_new;
        cb = cb_new;
        let (fa_reb, fb_reb, e_reb, _) =
            assemble_uks_fock_and_energy(&h_core, &pa, &pb, &fock_builder, basis, grid, xc);
        fa = fa_reb;
        fb = fb_reb;
        e_elec_final = e_reb;

        let err2 = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let comm_norm = frobenius_norm(&err2);

        if diff < fp_density_tol && comm_norm < fp_comm_tol {
            let (ea_ex, ca_ex, pa_ex) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
            let (eb_ex, cb_ex, pb_ex) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
            eps_a = ea_ex;
            eps_b = eb_ex;
            ca = ca_ex;
            cb = cb_ex;
            pa = pa_ex;
            pb = pb_ex;
            let (fa_ex, fb_ex, e_ex, _) =
                assemble_uks_fock_and_energy(&h_core, &pa, &pb, &fock_builder, basis, grid, xc);
            fa = fa_ex;
            fb = fb_ex;
            e_elec_final = e_ex;
            let exact_comm =
                frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
            if exact_comm.is_finite() && exact_comm <= fp_comm_tol {
                break;
            }
        }
    }

    // Recheck after the final exact diagonalization and density rebuild too.
    let final_error = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
    let final_comm_norm = frobenius_norm(&final_error);
    if !final_comm_norm.is_finite() || final_comm_norm > fp_comm_tol {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    let s2 = uhf_s_squared(&pa, &pb, &s, n_alpha, n_beta);
    let (gpu_kernel_invocations, gpu_fallback_invocations, rij_metric_truncated_modes) =
        fock_builder.telemetry();
    let diis_disk_backed = diis.is_disk_backed();

    Ok(UhfWavefunction {
        density_fitting: opts.density_fitting,
        converged,
        iterations: used_iter,
        total_energy_hartree: e_elec_final + e_nuc,
        electronic_energy_hartree: e_elec_final,
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
        },
    })
}
