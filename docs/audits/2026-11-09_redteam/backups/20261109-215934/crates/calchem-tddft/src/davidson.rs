//! Block Davidson Subspace Eigensolvers for Hermitian TDA / CIS ($A X = \omega X$)
//! and Non-Hermitian Paired Casida RPA / TD-DFT ($\begin{pmatrix} A & B \\ B & A \end{pmatrix}$).

use calchem_model::CalcError;
use calchem_tensor::{Matrix2D, eigh_sym, matmul, transpose};

/// Threshold below which exact dense diagonalization is faster and exact in one shot.
const DENSE_FASTPATH_MAX_DIM: usize = 64;
const MGS_NORM_TOL: f64 = 1e-11;
const PRECOND_SHIFT_MIN: f64 = 1e-4;

#[derive(Debug, Clone)]
pub struct EigenMode {
    pub omega: f64,
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
            for (c, & q_i) in cand.iter_mut().zip(q.iter()) {
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
    let dim = diag.len();
    if dim == 0 || n_roots == 0 {
        return Ok((Vec::new(), 0));
    }
    let k_roots = n_roots.min(dim);

    // Exact dense fast path for compact OV spaces
    if !force_davidson && dim <= DENSE_FASTPATH_MAX_DIM {
        let mut unit = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut e = vec![0.0; dim];
            e[i] = 1.0;
            unit.push(e);
        }
        let cols = apply_a(&unit);
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
                x,
                y: vec![0.0; dim],
            });
        }
        return Ok((modes, 1));
    }

    let max_subspace = (k_roots * 10).max(60).min(dim);
    let mut v_basis = build_initial_guess(diag, k_roots);
    let mut av_basis = apply_a(&v_basis);
    let mut last_res = 0.0f64;

    for iter in 1..=max_iter {
        let m = v_basis.len();
        let mut sub_a = Matrix2D::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let hij = 0.5 * (dot(&v_basis[i], &av_basis[j]) + dot(&v_basis[j], &av_basis[i]));
                sub_a.set(i, j, hij);
                sub_a.set(j, i, hij);
            }
        }

        let (evals, evecs) = eigh_sym(&sub_a, 120, 1e-13)?;
        let mut max_res = 0.0f64;
        let mut ritz_vecs: Vec<Vec<f64>> = Vec::with_capacity(k_roots);
        let mut new_cands: Vec<Vec<f64>> = Vec::new();

        for r in 0..k_roots {
            let omega = evals[r];
            let mut x_full = vec![0.0; dim];
            let mut ax_full = vec![0.0; dim];
            for p in 0..m {
                let c = evecs.get(p, r);
                for i in 0..dim {
                    x_full[i] += c * v_basis[p][i];
                    ax_full[i] += c * av_basis[p][i];
                }
            }
            let mut res = vec![0.0; dim];
            for i in 0..dim {
                res[i] = ax_full[i] - omega * x_full[i];
            }
            let r_norm = norm2(&res);
            max_res = max_res.max(r_norm);
            ritz_vecs.push(x_full);

            if r_norm > conv_tol {
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
        last_res = max_res;

        if max_res <= conv_tol {
            let modes = (0..k_roots)
                .map(|r| EigenMode {
                    omega: evals[r],
                    x: ritz_vecs[r].clone(),
                    y: vec![0.0; dim],
                })
                .collect();
            return Ok((modes, iter));
        }

        // Collapse subspace to lowest Ritz vectors if approaching max_subspace
        if m + new_cands.len() > max_subspace && m > k_roots * 2 {
            let keep = (k_roots * 2).min(m);
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
            // Subspace exhausted; return current Ritz vectors
            let modes = (0..k_roots)
                .map(|r| EigenMode {
                    omega: evals[r],
                    x: ritz_vecs[r].clone(),
                    y: vec![0.0; dim],
                })
                .collect();
            return Ok((modes, iter));
        }

        let added_av = apply_a(&v_basis[old_m..]);
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
        let (apb_cols, amb_cols) = apply_apb_amb(&unit);
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
                EigenMode { omega, x, y }
            })
            .collect();
        return Ok((modes, 1));
    }

    let max_subspace = (k_roots * 12).max(64).min(dim);
    let mut v_basis = build_initial_guess(diag, k_roots);
    let (mut wp_basis, mut wm_basis) = apply_apb_amb(&v_basis);
    let mut last_res = 0.0f64;

    for iter in 1..=max_iter {
        let m = v_basis.len();
        let mut m_plus = Matrix2D::zeros(m, m);
        let mut m_minus = Matrix2D::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let hp = 0.5 * (dot(&v_basis[i], &wp_basis[j]) + dot(&v_basis[j], &wp_basis[i]));
                let hm = 0.5 * (dot(&v_basis[i], &wm_basis[j]) + dot(&v_basis[j], &wm_basis[i]));
                m_plus.set(i, j, hp);
                m_plus.set(j, i, hp);
                m_minus.set(i, j, hm);
                m_minus.set(j, i, hm);
            }
        }

        let sub_sols = solve_projected_rpa(&m_plus, &m_minus, k_roots)?;
        let mut max_res = 0.0f64;
        let mut full_modes: Vec<EigenMode> = Vec::with_capacity(k_roots);
        let mut new_cands: Vec<Vec<f64>> = Vec::new();

        for (omega, xpy_sub, xmy_sub) in &sub_sols {
            let mut xpy = vec![0.0; dim];
            let mut xmy = vec![0.0; dim];
            let mut apb_xpy = vec![0.0; dim];
            let mut amb_xmy = vec![0.0; dim];

            for p in 0..m {
                let cp = xpy_sub[p];
                let cm = xmy_sub[p];
                for i in 0..dim {
                    xpy[i] += cp * v_basis[p][i];
                    xmy[i] += cm * v_basis[p][i];
                    apb_xpy[i] += cp * wp_basis[p][i];
                    amb_xmy[i] += cm * wm_basis[p][i];
                }
            }

            // Paired Casida residuals:
            // r_+ = (A + B)(X + Y) - omega (X - Y)
            // r_- = (A - B)(X - Y) - omega (X + Y)
            let mut rp = vec![0.0; dim];
            let mut rm = vec![0.0; dim];
            for i in 0..dim {
                rp[i] = apb_xpy[i] - omega * xmy[i];
                rm[i] = amb_xmy[i] - omega * xpy[i];
            }
            let r_norm = norm2(&rp).max(norm2(&rm));
            max_res = max_res.max(r_norm);

            let mut x = vec![0.0; dim];
            let mut y = vec![0.0; dim];
            for i in 0..dim {
                x[i] = 0.5 * (xpy[i] + xmy[i]);
                y[i] = 0.5 * (xpy[i] - xmy[i]);
            }
            full_modes.push(EigenMode {
                omega: *omega,
                x,
                y,
            });

            if r_norm > conv_tol {
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

        if max_res <= conv_tol {
            return Ok((full_modes, iter));
        }
        last_res = max_res;

        if m + new_cands.len() > max_subspace && m > k_roots * 2 {
            let mut collapsed_v: Vec<Vec<f64>> = Vec::with_capacity(k_roots * 2);
            for mode in &full_modes {
                let xpy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a + b).collect();
                let xmy: Vec<f64> = mode.x.iter().zip(&mode.y).map(|(a, b)| a - b).collect();
                append_orthonormal(&mut collapsed_v, xpy);
                append_orthonormal(&mut collapsed_v, xmy);
            }
            let (wp, wm) = apply_apb_amb(&collapsed_v);
            v_basis = collapsed_v;
            wp_basis = wp;
            wm_basis = wm;
        }

        let old_m = v_basis.len();
        for cand in new_cands {
            append_orthonormal(&mut v_basis, cand);
        }
        if v_basis.len() == old_m {
            return Ok((full_modes, iter));
        }

        let (added_wp, added_wm) = apply_apb_amb(&v_basis[old_m..]);
        wp_basis.extend(added_wp);
        wm_basis.extend(added_wm);
    }

    Err(CalcError::NotConverged {
        iterations: max_iter,
        delta_e: last_res,
        rms_d: last_res,
    })
}
