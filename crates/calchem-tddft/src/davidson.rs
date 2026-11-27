//! Block Davidson Subspace Eigensolvers for Hermitian TDA / CIS ($A X = \omega X$)
//! and paired Casida RPA / TD-DFT with Liouvillian blocks (A, B; -B, -A).

use calchem_model::CalcError;
use calchem_tensor::{eigh_sym, matmul, transpose, Matrix2D};

/// Threshold below which exact dense diagonalization is faster and exact in one shot.
const DENSE_FASTPATH_MAX_DIM: usize = 64;
const MGS_NORM_TOL: f64 = 1e-11;
const PRECOND_SHIFT_MIN: f64 = 1e-4;

/// Two-stage ERI screening level passed from Davidson solvers into the response evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreeningStage {
    Coarse,
    Tight,
}

#[derive(Debug, Clone)]
pub struct EigenMode {
    pub omega: f64,
    pub residual_norm: f64,
    /// Excitation vector $X$ (`n_ov`).
    pub x: Vec<f64>,
    /// De-excitation vector $Y$ (`n_ov`; all zeros in TDA/CIS).
    pub y: Vec<f64>,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn sym_matrix_residual(mat: &Matrix2D, input: &[f64], target: &[f64], omega: f64) -> f64 {
    let dim = target.len();
    let mut sq = 0.0;
    for i in 0..dim {
        let mut r = -omega * target[i];
        let row = &mat.data[i * dim..(i + 1) * dim];
        for j in 0..dim {
            r += row[j] * input[j];
        }
        sq += r * r;
    }
    sq.sqrt()
}

fn checked_dense_modes(modes: Vec<EigenMode>, tolerance: f64) -> Result<(Vec<EigenMode>,usize),CalcError> {
    if let Some(m) = modes.iter().find(|m| !m.residual_norm.is_finite() || m.residual_norm > tolerance) {
        return Err(CalcError::LinearAlgebra(format!("Dense response residual {} exceeds tolerance {tolerance}",m.residual_norm)));
    }
    Ok((modes, 1))
}

/// Appends `cand` to orthonormal basis `basis` using two-pass Modified Gram-Schmidt.
/// Returns `true` if `cand` had linearly independent component above `MGS_NORM_TOL`.
fn append_orthonormal(basis: &mut Vec<Vec<f64>>, mut cand: Vec<f64>) -> bool {
    let n0 = norm2(&cand);
    if !n0.is_finite() || n0 < MGS_NORM_TOL {
        return false;
    }
    let inv0 = 1.0 / n0;
    for v in &mut cand {
        *v *= inv0;
    }

    for _pass in 0..2 {
        for q in basis.iter() {
            let proj = dot(q, &cand);
            for (c, &q_i) in cand.iter_mut().zip(q.iter()) {
                *c -= proj * q_i;
            }
        }
    }
    let n1 = norm2(&cand);
    if !n1.is_finite() || n1 < 1e-7 {
        return false;
    }
    let inv1 = 1.0 / n1;
    for v in &mut cand {
        *v *= inv1;
    }
    basis.push(cand);
    true
}

/// Orthonormalizes `v` against `v_basis` via two-pass MGS while applying the exact same
/// linear operations to `(wp, wm)` against `(wp_basis, wm_basis)` so `(A+B)V` and `(A-B)V`
/// remain exact without re-evaluating ERIs during RPA subspace collapse.
fn append_orthonormal_triplet(
    v_basis: &mut Vec<Vec<f64>>,
    wp_basis: &mut Vec<Vec<f64>>,
    wm_basis: &mut Vec<Vec<f64>>,
    mut v: Vec<f64>,
    mut wp: Vec<f64>,
    mut wm: Vec<f64>,
) -> bool {
    let n0 = norm2(&v);
    if !n0.is_finite() || n0 < MGS_NORM_TOL {
        return false;
    }
    let inv0 = 1.0 / n0;
    for x in &mut v {
        *x *= inv0;
    }
    for x in &mut wp {
        *x *= inv0;
    }
    for x in &mut wm {
        *x *= inv0;
    }

    for _pass in 0..2 {
        for ((qv, qwp), qwm) in v_basis.iter().zip(wp_basis.iter()).zip(wm_basis.iter()) {
            let proj = dot(qv, &v);
            for (c, &qi) in v.iter_mut().zip(qv.iter()) {
                *c -= proj * qi;
            }
            for (c, &qi) in wp.iter_mut().zip(qwp.iter()) {
                *c -= proj * qi;
            }
            for (c, &qi) in wm.iter_mut().zip(qwm.iter()) {
                *c -= proj * qi;
            }
        }
    }

    let n1 = norm2(&v);
    if !n1.is_finite() || n1 < 1e-6 {
        return false;
    }
    let inv1 = 1.0 / n1;
    for x in &mut v {
        *x *= inv1;
    }
    for x in &mut wp {
        *x *= inv1;
    }
    for x in &mut wm {
        *x *= inv1;
    }
    v_basis.push(v);
    wp_basis.push(wp);
    wm_basis.push(wm);
    true
}

/// Builds initial guess unit vectors sorted by orbital energy differences $\Delta\varepsilon_{ia}$,
/// automatically completing degenerate shells so spatial irreps never split artificially.
fn build_initial_guess(diag: &[f64], n_roots: usize) -> Vec<Vec<f64>> {
    let dim = diag.len();
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| diag[a].total_cmp(&diag[b]));

    let mut m_init = (n_roots * 2 + 4).min(dim).max(n_roots.min(dim));
    while m_init < dim && (diag[order[m_init]] - diag[order[m_init - 1]]).abs() < 1e-4 {
        m_init += 1;
    }

    let mut basis = Vec::with_capacity(m_init);
    for &idx in &order[..m_init] {
        let mut v = vec![0.0; dim];
        v[idx] = 1.0;
        basis.push(v);
    }
    basis
}

/// Solves the real symmetric TDA / CIS eigenproblem $A X = \omega X$ for the lowest `n_roots` roots.
pub fn solve_tda<F>(
    diag: &[f64],
    n_roots: usize,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    apply_a: F,
) -> Result<(Vec<EigenMode>, usize), CalcError>
where
    F: Fn(&[Vec<f64>]) -> Vec<Vec<f64>>,
{
    solve_tda_with_screening(
        diag,
        n_roots,
        max_iter,
        conv_tol,
        force_davidson,
        false,
        |vecs, _| apply_a(vecs),
    )
}

/// Solves the real symmetric TDA / CIS eigenproblem $A X = \omega X$ with soft root locking
/// and optional two-stage (`Coarse -> Tight`) ERI screening with Collapse-and-Refresh.
pub fn solve_tda_with_screening<F>(
    diag: &[f64],
    n_roots: usize,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    enable_coarse_stage: bool,
    apply_a: F,
) -> Result<(Vec<EigenMode>, usize), CalcError>
where
    F: Fn(&[Vec<f64>], ScreeningStage) -> Vec<Vec<f64>>,
{
    let dim = diag.len();
    if dim == 0 || n_roots == 0 {
        return Ok((Vec::new(), 0));
    }
    let k_roots = n_roots.min(dim);

    // Exact dense fast path for compact OV spaces (always uses Tight screening)
    if !force_davidson && dim <= DENSE_FASTPATH_MAX_DIM {
        let mut unit = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut e = vec![0.0; dim];
            e[i] = 1.0;
            unit.push(e);
        }
        let cols = apply_a(&unit, ScreeningStage::Tight);
        let mut a_mat = Matrix2D::zeros(dim, dim);
        for j in 0..dim {
            for i in 0..dim {
                a_mat.set(i, j, 0.5 * (cols[j][i] + cols[i][j]));
            }
        }
        let (evals, evecs) = eigh_sym(&a_mat, 120, 1e-13)?;
        let mut modes = Vec::with_capacity(k_roots);
        for r in 0..k_roots {
            let mut x = vec![0.0; dim];
            for i in 0..dim {
                x[i] = evecs.get(i, r);
            }
            modes.push(EigenMode {
                omega: evals[r],
                residual_norm: sym_matrix_residual(&a_mat, &x, &x, evals[r]),
                x,
                y: vec![0.0; dim],
            });
        }
        return checked_dense_modes(modes, conv_tol);
    }

    let n_track = (k_roots + 4).min(dim);
    let max_subspace = (n_track * 10).max(64).min(dim);
    let mut stage = if enable_coarse_stage {
        ScreeningStage::Coarse
    } else {
        ScreeningStage::Tight
    };
    let switch_tol = (conv_tol * 500.0).clamp(5e-4, 1e-2).max(conv_tol);
    let mut v_basis = build_initial_guess(diag, k_roots);
    let mut av_basis = apply_a(&v_basis, stage);
    let mut locked = vec![false; k_roots];
    let mut last_res = 0.0f64;

    let evaluate_tda_subspace = |v_b: &[Vec<f64>], av_b: &[Vec<f64>]| -> Result<(Vec<f64>, Matrix2D, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, f64), CalcError> {
        let m = v_b.len();
        let active_roots = n_track.min(m);
        let mut sub_a = Matrix2D::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let hij = 0.5 * (dot(&v_b[i], &av_b[j]) + dot(&v_b[j], &av_b[i]));
                sub_a.set(i, j, hij);
                sub_a.set(j, i, hij);
            }
        }
        let (evals, evecs) = eigh_sym(&sub_a, 120, 1e-13)?;
        let mut max_res = 0.0f64;
        let mut residuals = Vec::with_capacity(active_roots);
        let mut ritz_vecs: Vec<Vec<f64>> = Vec::with_capacity(active_roots);
        let mut res_vecs: Vec<Vec<f64>> = Vec::with_capacity(active_roots);

        for r in 0..active_roots {
            let omega = evals[r];
            let mut x_full = vec![0.0; dim];
            let mut ax_full = vec![0.0; dim];
            for p in 0..m {
                let c = evecs.get(p, r);
                for i in 0..dim {
                    x_full[i] += c * v_b[p][i];
                    ax_full[i] += c * av_b[p][i];
                }
            }
            let mut res = vec![0.0; dim];
            for i in 0..dim {
                res[i] = ax_full[i] - omega * x_full[i];
            }
            let r_norm = norm2(&res);
            if !r_norm.is_finite() {
                return Err(CalcError::LinearAlgebra("Nonfinite TDA residual".into()));
            }
            if r < k_roots {
                max_res = max_res.max(r_norm);
            }
            residuals.push(r_norm);
            ritz_vecs.push(x_full);
            res_vecs.push(res);
        }

        Ok((evals, evecs, residuals, ritz_vecs, res_vecs, max_res))
    };

    for iter in 1..=max_iter {
        let m = v_basis.len();
        let active_roots = n_track.min(m);
        let (evals, evecs, residuals, ritz_vecs, res_vecs, max_res) =
            evaluate_tda_subspace(&v_basis, &av_basis)?;
        last_res = max_res;

        let root_tol = if stage == ScreeningStage::Coarse {
            conv_tol.max(1e-5)
        } else {
            conv_tol
        };
        let boundary_eval = evals[k_roots.min(active_roots) - 1];
        let buffer_tol = root_tol.max(1e-3);
        let mut new_cands: Vec<Vec<f64>> = Vec::new();

        for r in 0..active_roots {
            let r_norm = residuals[r];
            let expand = if r < k_roots {
                locked[r] = r_norm <= root_tol;
                !locked[r]
            } else {
                let gap = (evals[r] - boundary_eval).max(0.0);
                let eff_buffer_tol = (0.25 * gap).min(buffer_tol).max(root_tol);
                r_norm > eff_buffer_tol && gap < 0.02 + 2.0 * r_norm
            };
            if expand {
                let omega = evals[r];
                let res = &res_vecs[r];
                let mut corr = vec![0.0; dim];
                for i in 0..dim {
                    let denom = omega - diag[i];
                    let safe_denom = if denom.abs() < PRECOND_SHIFT_MIN {
                        if denom >= 0.0 {
                            PRECOND_SHIFT_MIN
                        } else {
                            -PRECOND_SHIFT_MIN
                        }
                    } else {
                        denom
                    };
                    corr[i] = res[i] / safe_denom;
                }
                new_cands.push(corr);
            }
        }

        if stage == ScreeningStage::Tight
            && max_res <= conv_tol
            && (new_cands.is_empty() || iter == max_iter)
        {
            let modes = (0..k_roots)
                .map(|r| EigenMode {
                    omega: evals[r],
                    residual_norm: residuals[r],
                    x: ritz_vecs[r].clone(),
                    y: vec![0.0; dim],
                })
                .collect();
            return Ok((modes, iter));
        }

        if stage == ScreeningStage::Coarse && (max_res <= switch_tol || new_cands.is_empty()) {
            let mut collapsed_v = Vec::with_capacity(active_roots);
            for rv in ritz_vecs {
                append_orthonormal(&mut collapsed_v, rv);
            }
            v_basis = collapsed_v;
            stage = ScreeningStage::Tight;
            av_basis = apply_a(&v_basis, ScreeningStage::Tight);
            locked.fill(false);
            if iter < max_iter {
                continue;
            }
            let (t_evals, _, t_res, t_ritz, _, t_max_res) =
                evaluate_tda_subspace(&v_basis, &av_basis)?;
            last_res = t_max_res;
            if t_max_res <= conv_tol {
                let modes = (0..k_roots)
                    .map(|r| EigenMode {
                        omega: t_evals[r],
                        residual_norm: t_res[r],
                        x: t_ritz[r].clone(),
                        y: vec![0.0; dim],
                    })
                    .collect();
                return Ok((modes, iter));
            }
            break;
        }

        // Collapse subspace to lowest Ritz vectors if approaching max_subspace
        if m + new_cands.len() > max_subspace && m > k_roots * 2 {
            let keep = (k_roots * 2).max(active_roots).min(m);
            let mut collapsed_v = Vec::with_capacity(keep);
            let mut collapsed_av = Vec::with_capacity(keep);
            for r in 0..keep {
                let mut x_f = vec![0.0; dim];
                let mut ax_f = vec![0.0; dim];
                for p in 0..m {
                    let c = evecs.get(p, r);
                    for i in 0..dim {
                        x_f[i] += c * v_basis[p][i];
                        ax_f[i] += c * av_basis[p][i];
                    }
                }
                collapsed_v.push(x_f);
                collapsed_av.push(ax_f);
            }
            v_basis = collapsed_v;
            av_basis = collapsed_av;
        }

        let old_m = v_basis.len();
        for cand in new_cands {
            append_orthonormal(&mut v_basis, cand);
        }
        if v_basis.len() == old_m {
            if stage == ScreeningStage::Coarse {
                let mut collapsed_v = Vec::with_capacity(active_roots);
                for rv in ritz_vecs {
                    append_orthonormal(&mut collapsed_v, rv);
                }
                v_basis = collapsed_v;
                stage = ScreeningStage::Tight;
                av_basis = apply_a(&v_basis, ScreeningStage::Tight);
                locked.fill(false);
                if iter < max_iter {
                    continue;
                }
                let (t_evals, _, t_res, t_ritz, _, t_max_res) =
                    evaluate_tda_subspace(&v_basis, &av_basis)?;
                last_res = t_max_res;
                if t_max_res <= conv_tol {
                    let modes = (0..k_roots)
                        .map(|r| EigenMode {
                            omega: t_evals[r],
                            residual_norm: t_res[r],
                            x: t_ritz[r].clone(),
                            y: vec![0.0; dim],
                        })
                        .collect();
                    return Ok((modes, iter));
                }
                break;
            }
            if max_res <= conv_tol {
                let modes = (0..k_roots)
                    .map(|r| EigenMode {
                        omega: evals[r],
                        residual_norm: residuals[r],
                        x: ritz_vecs[r].clone(),
                        y: vec![0.0; dim],
                    })
                    .collect();
                return Ok((modes, iter));
            }
            return Err(CalcError::NotConverged {
                iterations: iter,
                delta_e: max_res,
                rms_d: max_res,
            });
        }

        let added_av = apply_a(&v_basis[old_m..], stage);
        av_basis.extend(added_av);
    }

    Err(CalcError::NotConverged {
        iterations: max_iter,
        delta_e: last_res,
        rms_d: last_res,
    })
}

/// Solves the projected $m \times m$ Casida RPA problem given symmetric $\tilde{M}_+ = V^T(A+B)V$
/// and $\tilde{M}_- = V^T(A-B)V$, returning `(omega, sub_x_plus_y, sub_x_minus_y)` per root.
fn solve_projected_rpa(
    m_plus: &Matrix2D,
    m_minus: &Matrix2D,
    k_roots: usize,
) -> Result<Vec<(f64, Vec<f64>, Vec<f64>)>, CalcError> {
    let m = m_minus.rows;
    let (lam_m, q_m) = eigh_sym(m_minus, 120, 1e-13)?;
    let mut sqrt_m = Matrix2D::zeros(m, m);
    let mut inv_sqrt_m = Matrix2D::zeros(m, m);

    for (k, &ev) in lam_m.iter().enumerate() {
        if ev <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "TD-DFT/RPA instability detected: (A - B) has non-positive eigenvalue {ev:.3e}; use TDA=true"
            )));
        }
        let s = ev.sqrt();
        let inv_s = 1.0 / s;
        for i in 0..m {
            let qik = q_m.get(i, k);
            for j in 0..m {
                let prod = qik * q_m.get(j, k);
                sqrt_m.add_assign_at(i, j, s * prod);
                inv_sqrt_m.add_assign_at(i, j, inv_s * prod);
            }
        }
    }

    // Symmetric squared-frequency operator Omega = M_-^{1/2} M_+ M_-^{1/2}
    let tmp = matmul(&sqrt_m, m_plus);
    let mut omega_sq_mat = matmul(&tmp, &transpose(&sqrt_m));
    for i in 0..m {
        for j in (i + 1)..m {
            let avg = 0.5 * (omega_sq_mat.get(i, j) + omega_sq_mat.get(j, i));
            omega_sq_mat.set(i, j, avg);
            omega_sq_mat.set(j, i, avg);
        }
    }

    let (w2_vals, z_vecs) = eigh_sym(&omega_sq_mat, 120, 1e-13)?;
    let mut out = Vec::with_capacity(k_roots);

    for r in 0..k_roots {
        let w2 = w2_vals[r];
        if w2 <= 1e-12 {
            return Err(CalcError::LinearAlgebra(format!(
                "TD-DFT/RPA instability detected: squared excitation energy omega^2 = {w2:.3e} <= 0; use TDA=true"
            )));
        }
        let omega = w2.sqrt();
        let inv_sqrt_w = 1.0 / omega.sqrt();
        let sqrt_w = omega.sqrt();

        // x + y = M_-^{1/2} z / sqrt(omega),  x - y = sqrt(omega) M_-^{-1/2} z
        // Satisfies (x + y)^T (x - y) = z^T z = 1 <=> x^T x - y^T y = 1!
        let mut xpy = vec![0.0; m];
        let mut xmy = vec![0.0; m];
        for i in 0..m {
            let mut s_p = 0.0;
            let mut s_m = 0.0;
            for k in 0..m {
                let zk = z_vecs.get(k, r);
                s_p += sqrt_m.get(i, k) * zk;
                s_m += inv_sqrt_m.get(i, k) * zk;
            }
            xpy[i] = s_p * inv_sqrt_w;
            xmy[i] = s_m * sqrt_w;
        }
        out.push((omega, xpy, xmy));
    }

    Ok(out)
}

/// Solves the full Casida linear-response RPA / TD-DFT problem
/// `(A + B)(X + Y) = \omega (X - Y)`, `(A - B)(X - Y) = \omega (X + Y)`
/// with exact Casida normalization $X^T X - Y^T Y = 1$.
pub fn solve_rpa<F>(
    diag: &[f64],
    n_roots: usize,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    apply_apb_amb: F,
) -> Result<(Vec<EigenMode>, usize), CalcError>
where
    F: Fn(&[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>),
{
    solve_rpa_with_screening(
        diag,
        n_roots,
        max_iter,
        conv_tol,
        force_davidson,
        false,
        |vecs, _| apply_apb_amb(vecs),
    )
}

/// Solves the full Casida linear-response RPA / TD-DFT problem with soft root locking,
/// zero-ERI linear triplet subspace collapse, and optional two-stage (`Coarse -> Tight`)
/// ERI screening with Collapse-and-Refresh.
pub fn solve_rpa_with_screening<F>(
    diag: &[f64],
    n_roots: usize,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    enable_coarse_stage: bool,
    apply_apb_amb: F,
) -> Result<(Vec<EigenMode>, usize), CalcError>
where
    F: Fn(&[Vec<f64>], ScreeningStage) -> (Vec<Vec<f64>>, Vec<Vec<f64>>),
{
    let dim = diag.len();
    if dim == 0 || n_roots == 0 {
        return Ok((Vec::new(), 0));
    }
    let k_roots = n_roots.min(dim);

    if !force_davidson && dim <= DENSE_FASTPATH_MAX_DIM {
        let mut unit = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut e = vec![0.0; dim];
            e[i] = 1.0;
            unit.push(e);
        }
        let (apb_cols, amb_cols) = apply_apb_amb(&unit, ScreeningStage::Tight);
        let mut m_plus = Matrix2D::zeros(dim, dim);
        let mut m_minus = Matrix2D::zeros(dim, dim);
        for j in 0..dim {
            for i in 0..dim {
                m_plus.set(i, j, 0.5 * (apb_cols[j][i] + apb_cols[i][j]));
                m_minus.set(i, j, 0.5 * (amb_cols[j][i] + amb_cols[i][j]));
            }
        }
        let sub_sols = solve_projected_rpa(&m_plus, &m_minus, k_roots)?;
        let modes = sub_sols
            .into_iter()
            .map(|(omega, xpy, xmy)| {
                let mut x = vec![0.0; dim];
                let mut y = vec![0.0; dim];
                for i in 0..dim {
                    x[i] = 0.5 * (xpy[i] + xmy[i]);
                    y[i] = 0.5 * (xpy[i] - xmy[i]);
                }
                let plus = sym_matrix_residual(&m_plus, &xpy, &xmy, omega);
                let minus = sym_matrix_residual(&m_minus, &xmy, &xpy, omega);
                EigenMode {
                    omega,
                    x,
                    y,
                    residual_norm: plus.max(minus),
                }
            })
            .collect::<Vec<_>>();
        return checked_dense_modes(modes, conv_tol);
    }

    let n_track = (k_roots + 4).min(dim);
    let max_subspace = (n_track * 12).max(64).min(dim);
    let mut stage = if enable_coarse_stage {
        ScreeningStage::Coarse
    } else {
        ScreeningStage::Tight
    };
    let switch_tol = (conv_tol * 500.0).clamp(5e-4, 1e-2).max(conv_tol);
    let mut v_basis = build_initial_guess(diag, k_roots);
    let (mut wp_basis, mut wm_basis) = apply_apb_amb(&v_basis, stage);
    let mut locked = vec![false; k_roots];
    let mut last_res = 0.0f64;

    type RpaSubspaceEval = (
        Vec<(f64, Vec<f64>, Vec<f64>)>,
        Vec<EigenMode>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        f64,
    );

    let evaluate_rpa_subspace = |v_b: &[Vec<f64>], wp_b: &[Vec<f64>], wm_b: &[Vec<f64>]| -> Result<RpaSubspaceEval, CalcError> {
        let m = v_b.len();
        let active_roots = n_track.min(m);
        let mut m_plus = Matrix2D::zeros(m, m);
        let mut m_minus = Matrix2D::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let hp = 0.5 * (dot(&v_b[i], &wp_b[j]) + dot(&v_b[j], &wp_b[i]));
                let hm = 0.5 * (dot(&v_b[i], &wm_b[j]) + dot(&v_b[j], &wm_b[i]));
                m_plus.set(i, j, hp);
                m_plus.set(j, i, hp);
                m_minus.set(i, j, hm);
                m_minus.set(j, i, hm);
            }
        }

        let sub_sols = solve_projected_rpa(&m_plus, &m_minus, active_roots)?;
        let mut max_res = 0.0f64;
        let mut full_modes: Vec<EigenMode> = Vec::with_capacity(active_roots);
        let mut rp_vecs: Vec<Vec<f64>> = Vec::with_capacity(active_roots);
        let mut rm_vecs: Vec<Vec<f64>> = Vec::with_capacity(active_roots);

        for (r, (omega, xpy_sub, xmy_sub)) in sub_sols.iter().enumerate() {
            let mut xpy = vec![0.0; dim];
            let mut xmy = vec![0.0; dim];
            let mut apb_xpy = vec![0.0; dim];
            let mut amb_xmy = vec![0.0; dim];

            for p in 0..m {
                let cp = xpy_sub[p];
                let cm = xmy_sub[p];
                for i in 0..dim {
                    xpy[i] += cp * v_b[p][i];
                    xmy[i] += cm * v_b[p][i];
                    apb_xpy[i] += cp * wp_b[p][i];
                    amb_xmy[i] += cm * wm_b[p][i];
                }
            }

            let mut rp = vec![0.0; dim];
            let mut rm = vec![0.0; dim];
            for i in 0..dim {
                rp[i] = apb_xpy[i] - omega * xmy[i];
                rm[i] = amb_xmy[i] - omega * xpy[i];
            }
            let r_norm = norm2(&rp).max(norm2(&rm));
            if rp.iter().chain(&rm).any(|v| !v.is_finite()) {
                return Err(CalcError::LinearAlgebra("Nonfinite RPA residual".into()));
            }
            if r < k_roots {
                max_res = max_res.max(r_norm);
            }

            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            for i in 0..dim {
                x[i] = 0.5 * (xpy[i] + xmy[i]);
                y[i] = 0.5 * (xpy[i] - xmy[i]);
            }
            full_modes.push(EigenMode {
                omega: *omega,
                residual_norm: r_norm,
                x,
                y,
            });
            rp_vecs.push(rp);
            rm_vecs.push(rm);
        }

        Ok((sub_sols, full_modes, rp_vecs, rm_vecs, max_res))
    };

    for iter in 1..=max_iter {
        let m = v_basis.len();
        let active_roots = n_track.min(m);
        let (sub_sols, mut full_modes, rp_vecs, rm_vecs, max_res) =
            evaluate_rpa_subspace(&v_basis, &wp_basis, &wm_basis)?;
        last_res = max_res;

        let root_tol = if stage == ScreeningStage::Coarse {
            conv_tol.max(1e-5)
        } else {
            conv_tol
        };
        let boundary_omega = sub_sols[k_roots.min(active_roots) - 1].0;
        let buffer_tol = root_tol.max(1e-3);
        let mut new_cands: Vec<Vec<f64>> = Vec::new();

        for (r, mode) in full_modes.iter().enumerate() {
            let r_norm = mode.residual_norm;
            let expand = if r < k_roots {
                locked[r] = r_norm <= root_tol;
                !locked[r]
            } else {
                let gap = (mode.omega - boundary_omega).max(0.0);
                let eff_buffer_tol = (0.25 * gap).min(buffer_tol).max(root_tol);
                r_norm > eff_buffer_tol && gap < 0.02 + 2.0 * r_norm
            };
            if expand {
                let omega = mode.omega;
                let rp = &rp_vecs[r];
                let rm = &rm_vecs[r];
                let mut d_plus = vec![0.0; dim];
                let mut d_minus = vec![0.0; dim];
                for i in 0..dim {
                    let denom_x = omega - diag[i];
                    let safe_dx = if denom_x.abs() < PRECOND_SHIFT_MIN {
                        if denom_x >= 0.0 {
                            PRECOND_SHIFT_MIN
                        } else {
                            -PRECOND_SHIFT_MIN
                        }
                    } else {
                        denom_x
                    };
                    let safe_dy = (omega + diag[i]).max(PRECOND_SHIFT_MIN);
                    let dx = 0.5 * (rp[i] + rm[i]) / safe_dx;
                    let dy = -0.5 * (rp[i] - rm[i]) / safe_dy;
                    d_plus[i] = dx + dy;
                    d_minus[i] = dx - dy;
                }
                new_cands.push(d_plus);
                new_cands.push(d_minus);
            }
        }

        if stage == ScreeningStage::Tight
            && max_res <= conv_tol
            && (new_cands.is_empty() || iter == max_iter)
        {
            full_modes.truncate(k_roots);
            return Ok((full_modes, iter));
        }

        if stage == ScreeningStage::Coarse && (max_res <= switch_tol || new_cands.is_empty()) {
            let mut collapsed_v: Vec<Vec<f64>> = Vec::with_capacity(active_roots * 2);
            for mode in &full_modes {
                let xpy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a + b).collect();
                let xmy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a - b).collect();
                append_orthonormal(&mut collapsed_v, xpy);
                append_orthonormal(&mut collapsed_v, xmy);
            }
            v_basis = collapsed_v;
            stage = ScreeningStage::Tight;
            let (wp, wm) = apply_apb_amb(&v_basis, ScreeningStage::Tight);
            wp_basis = wp;
            wm_basis = wm;
            locked.fill(false);
            if iter < max_iter {
                continue;
            }
            let (_, mut t_modes, _, _, t_max_res) =
                evaluate_rpa_subspace(&v_basis, &wp_basis, &wm_basis)?;
            last_res = t_max_res;
            if t_max_res <= conv_tol {
                t_modes.truncate(k_roots);
                return Ok((t_modes, iter));
            }
            break;
        }

        if m + new_cands.len() > max_subspace && m > k_roots * 2 {
            // Zero-ERI linear triplet MGS collapse preserving (A+B)V and (A-B)V
            let mut collapsed_v: Vec<Vec<f64>> = Vec::with_capacity(active_roots * 2);
            let mut collapsed_wp: Vec<Vec<f64>> = Vec::with_capacity(active_roots * 2);
            let mut collapsed_wm: Vec<Vec<f64>> = Vec::with_capacity(active_roots * 2);
            for (_omega, xpy_sub, xmy_sub) in &sub_sols {
                let mut xpy = vec![0.0; dim];
                let mut xmy = vec![0.0; dim];
                let mut apb_xpy = vec![0.0; dim];
                let mut amb_xpy = vec![0.0; dim];
                let mut apb_xmy = vec![0.0; dim];
                let mut amb_xmy = vec![0.0; dim];
                for p in 0..m {
                    let cp = xpy_sub[p];
                    let cm = xmy_sub[p];
                    for i in 0..dim {
                        let vp = v_basis[p][i];
                        let wpp = wp_basis[p][i];
                        let wmp = wm_basis[p][i];
                        xpy[i] += cp * vp;
                        xmy[i] += cm * vp;
                        apb_xpy[i] += cp * wpp;
                        amb_xpy[i] += cp * wmp;
                        apb_xmy[i] += cm * wpp;
                        amb_xmy[i] += cm * wmp;
                    }
                }
                append_orthonormal_triplet(
                    &mut collapsed_v,
                    &mut collapsed_wp,
                    &mut collapsed_wm,
                    xpy,
                    apb_xpy,
                    amb_xpy,
                );
                append_orthonormal_triplet(
                    &mut collapsed_v,
                    &mut collapsed_wp,
                    &mut collapsed_wm,
                    xmy,
                    apb_xmy,
                    amb_xmy,
                );
            }
            if !collapsed_v.is_empty() {
                v_basis = collapsed_v;
                wp_basis = collapsed_wp;
                wm_basis = collapsed_wm;
            }
        }

        let old_m = v_basis.len();
        for cand in new_cands {
            append_orthonormal(&mut v_basis, cand);
        }
        if v_basis.len() == old_m {
            if stage == ScreeningStage::Coarse {
                let mut collapsed_v: Vec<Vec<f64>> = Vec::with_capacity(active_roots * 2);
                for mode in &full_modes {
                    let xpy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a + b).collect();
                    let xmy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a - b).collect();
                    append_orthonormal(&mut collapsed_v, xpy);
                    append_orthonormal(&mut collapsed_v, xmy);
                }
                v_basis = collapsed_v;
                stage = ScreeningStage::Tight;
                let (wp, wm) = apply_apb_amb(&v_basis, ScreeningStage::Tight);
                wp_basis = wp;
                wm_basis = wm;
                locked.fill(false);
                if iter < max_iter {
                    continue;
                }
                let (_, mut t_modes, _, _, t_max_res) =
                    evaluate_rpa_subspace(&v_basis, &wp_basis, &wm_basis)?;
                last_res = t_max_res;
                if t_max_res <= conv_tol {
                    t_modes.truncate(k_roots);
                    return Ok((t_modes, iter));
                }
                break;
            }
            if max_res <= conv_tol {
                full_modes.truncate(k_roots);
                return Ok((full_modes, iter));
            }
            return Err(CalcError::NotConverged {
                iterations: iter,
                delta_e: max_res,
                rms_d: max_res,
            });
        }

        let (added_wp, added_wm) = apply_apb_amb(&v_basis[old_m..], stage);
        wp_basis.extend(added_wp);
        wm_basis.extend(added_wm);
    }

    Err(CalcError::NotConverged {
        iterations: max_iter,
        delta_e: last_res,
        rms_d: last_res,
    })
}
