//! Unrestricted Hartree-Fock (UHF) for open-shell systems (radical ions, doublets, triplets).
//!
//! Fock operators (per spin, `P^t = P^a + P^b`):
//!   `F^a = h + J[P^t] - K[P^a]`,  `F^b = h + J[P^t] - K[P^b]`
//! Energy:
//!   `E = 1/2 * sum_{mu nu} [ P^t h + P^a F^a + P^b F^b ]_{mu nu} + E_nn`
//! DIIS runs on the stacked `[F^a; F^b]` / `[e^a; e^b]` matrices so both spins share
//! one set of extrapolation coefficients.

use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_int::{compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, UhfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D, Tensor4D};

/// Builds `(F^a, F^b)` from per-spin densities.
fn build_uhf_fock(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    eri: &Tensor4D,
) -> (Matrix2D, Matrix2D) {
    let n = h_core.rows;
    let mut fa = Matrix2D::zeros(n, n);
    let mut fb = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut j = 0.0;
            let mut ka = 0.0;
            let mut kb = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let pa_ls = pa.get(lam, sig);
                    let pb_ls = pb.get(lam, sig);
                    let coul = eri.get(mu, nu, lam, sig);
                    let exch = eri.get(mu, lam, nu, sig);
                    j += (pa_ls + pb_ls) * coul;
                    ka += pa_ls * exch;
                    kb += pb_ls * exch;
                }
            }
            let h = h_core.get(mu, nu);
            fa.set(mu, nu, h + j - ka);
            fb.set(mu, nu, h + j - kb);
        }
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
pub fn uhf_s_squared(pa: &Matrix2D, pb: &Matrix2D, s: &Matrix2D, n_alpha: usize, n_beta: usize) -> f64 {
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
    let (n_alpha, n_beta) = mol.validate_uhf()?;
    let n = basis.n_ao;
    if n == 0 || n_alpha > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_alpha} alpha electrons"
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

    // Symmetric orthogonalization X = U s^{-1/2} U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is not positive definite (eigenvalue {val:.3e})"
            )));
        }
        s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Core-Hamiltonian guess for both spins
    let (_, _, mut pa) = diagonalize_spin(&h_core, &x, &xt, n_alpha)?;
    let (_, _, mut pb) = diagonalize_spin(&h_core, &x, &xt, n_beta)?;

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
        let (fa, fb) = build_uhf_fock(&h_core, &pa, &pb, &eri);
        let e_tot = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb) + e_nuc;
        energy_history.push(e_tot);

        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_stacked = stack(&fa, &fb);
        let (fa_ex, fb_ex) = if opts.use_diis {
            let ex = diis.push_and_extrapolate(&f_stacked, &err);
            if iter >= 2 { unstack(&ex, n) } else { (fa.clone(), fb.clone()) }
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

        let delta_e = if iter == 1 { f64::INFINITY } else { (e_tot - e_prev).abs() };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        rms_history.push(rms_d);
        last_delta_e = delta_e;
        last_rms_d = rms_d;
        pa = pa_new;
        pb = pb_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        if iter > 1 && delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol {
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

    // Fixed-point tightening so (C, eps, P, F) are mutually consistent (gradient-grade).
    let (mut fa, mut fb) = build_uhf_fock(&h_core, &pa, &pb, &eri);
    let mut eps_a = Vec::new();
    let mut eps_b = Vec::new();
    let mut ca = Matrix2D::zeros(n, n);
    let mut cb = Matrix2D::zeros(n, n);
    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = (opts.conv_density * 100.0).max(1e-8);
    let mut comm_norm = f64::INFINITY;
    let mut fp_converged = false;
    for _ in 0..30 {
        let (ea, ca_new, pa_new) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
        let (eb, cb_new, pb_new) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
        let diff = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        pa = pa_new;
        pb = pb_new;
        eps_a = ea;
        eps_b = eb;
        ca = ca_new;
        cb = cb_new;
        let (fa_n, fb_n) = build_uhf_fock(&h_core, &pa, &pb, &eri);
        fa = fa_n;
        fb = fb_n;
        comm_norm = frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
        if diff < fp_density_tol && comm_norm < fp_comm_tol {
            fp_converged = true;
            break;
        }
    }
    if !fp_converged && comm_norm > 1e-6 {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    let e_elec = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb);
    let s2 = uhf_s_squared(&pa, &pb, &s, n_alpha, n_beta);

    Ok(UhfWavefunction {
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
        },
    })
}
