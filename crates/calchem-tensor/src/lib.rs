use calchem_model::CalcError;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix2D {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    #[inline]
    pub fn add_assign_at(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] += val;
    }

    pub fn sym_frobenius_error(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        let mut s = 0.0;
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                let d = self.get(i, j) - self.get(j, i);
                s += 2.0 * d * d;
            }
        }
        s.sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor4D {
    pub n: usize,
    pub data: Vec<f64>,
}

impl Tensor4D {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n * n * n],
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        let n = self.n;
        (((i * n + j) * n + k) * n) + l
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let idx = self.idx(i, j, k, l);
        self.data[idx]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        let idx = self.idx(i, j, k, l);
        self.data[idx] = val;
    }
}

pub fn matmul(a: &Matrix2D, b: &Matrix2D) -> Matrix2D {
    assert_eq!(a.cols, b.rows);
    let mut out = Matrix2D::zeros(a.rows, b.cols);
    if a.rows.min(a.cols).min(b.cols) >= 96 {
        let left = faer::mat::from_row_major_slice::<f64, _, _>(&a.data, a.rows, a.cols);
        let right = faer::mat::from_row_major_slice::<f64, _, _>(&b.data, b.rows, b.cols);
        let c = left * right;
        for i in 0..a.rows {
            for j in 0..b.cols { out.set(i,j,c.read(i,j)); }
        }
        return out;
    }
    for i in 0..a.rows {
        for k in 0..a.cols {
            let aik = a.get(i, k);
            for j in 0..b.cols {
                out.add_assign_at(i, j, aik * b.get(k, j));
            }
        }
    }
    out
}

/// Single-threaded zero-allocation row-major GEMM: `out(m x n) = a(m x k) * b(k x n)`.
/// Uses `faer` AVX2 microkernels with `Parallelism::None` so callers can parallelize externally.
#[inline]
pub fn gemm_nn_into(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], out: &mut [f64]) {
    let a_mat = faer::mat::from_row_major_slice::<f64, _, _>(a, m, k);
    let b_mat = faer::mat::from_row_major_slice::<f64, _, _>(b, k, n);
    let out_mat = faer::mat::from_row_major_slice_mut::<f64, _, _>(out, m, n);
    faer::linalg::matmul::matmul(out_mat, a_mat, b_mat, None, 1.0, faer::Parallelism::None);
}

/// Single-threaded zero-allocation row-major GEMM: `out(m x n) = a(m x k) * b(n x k)^T`.
/// Uses `faer` AVX2 microkernels with `Parallelism::None` so callers can parallelize externally.
#[inline]
pub fn gemm_nt_into(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], out: &mut [f64]) {
    let a_mat = faer::mat::from_row_major_slice::<f64, _, _>(a, m, k);
    let b_mat = faer::mat::from_row_major_slice::<f64, _, _>(b, n, k);
    let out_mat = faer::mat::from_row_major_slice_mut::<f64, _, _>(out, m, n);
    faer::linalg::matmul::matmul(
        out_mat,
        a_mat,
        b_mat.transpose(),
        None,
        1.0,
        faer::Parallelism::None,
    );
}

pub fn transpose(a: &Matrix2D) -> Matrix2D {
    let mut out = Matrix2D::zeros(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            out.set(j, i, a.get(i, j));
        }
    }
    out
}

pub fn frobenius_norm(a: &Matrix2D) -> f64 {
    a.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn rms_diff(a: &Matrix2D, b: &Matrix2D) -> f64 {
    assert_eq!(a.data.len(), b.data.len());
    if a.data.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    (sum_sq / (a.data.len() as f64)).sqrt()
}

#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Matrix2D,
}

pub fn eigh_symmetric(a: &Matrix2D) -> Result<SymmetricEigen, CalcError> {
    let (eigenvalues, eigenvectors) = eigh_sym(a, 100, 1e-14)?;
    Ok(SymmetricEigen {
        eigenvalues,
        eigenvectors,
    })
}

/// Cyclic Jacobi eigensolver for real symmetric matrices with Rutishauser's stable rotation
/// update and post-solve residual verification `||A V - V \Lambda||_F`.
///
/// Here `max_iter` specifies the maximum number of full cyclic sweeps over all `n(n-1)/2`
/// off-diagonal pairs (minimum 50 sweeps enforced internally).
pub fn eigh_sym(
    a: &Matrix2D,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Matrix2D), CalcError> {
    if a.rows != a.cols {
        return Err(CalcError::LinearAlgebra(
            "Matrix must be square for eigh_sym".into(),
        ));
    }
    let n = a.rows;
    if n == 0 {
        return Ok((vec![], Matrix2D::zeros(0, 0)));
    }
    if a.data.iter().any(|v| !v.is_finite()) {
        return Err(CalcError::LinearAlgebra(
            "Matrix contains NaN or infinite entries".into(),
        ));
    }

    if n >= 96 {
        let sym = faer::Mat::<f64>::from_fn(n,n, |i,j| 0.5*(a.get(i,j)+a.get(j,i)));
        let eig = sym.selfadjoint_eigendecomposition(faer::Side::Lower);
        let vals: Vec<_> = (0..n).map(|i| eig.s().column_vector().read(i)).collect();
        let mut vecs = Matrix2D::zeros(n,n);
        for i in 0..n { for j in 0..n {vecs.set(i,j,eig.u().read(i,j));} }
        verify_eigen_residual(a,&vals,&vecs)?;
        return Ok((vals,vecs));
    }
    // Symmetrize input matrix A = 0.5 * (A + A^T)
    let mut d = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d.set(i, j, 0.5 * (a.get(i, j) + a.get(j, i)));
        }
    }
    let mut v = Matrix2D::identity(n);
    let max_sweeps = max_iter.max(100);
    let a_norm = frobenius_norm(&d).max(1.0);
    // Respect response/SCF accuracy requests down to a rounding-error floor.
    // A 1e-11*||A|| floor silently prevented Davidson residuals reaching 1e-10.
    let effective_tol = tol.max(4.0 * f64::EPSILON * a_norm).max(1e-15);

    let mut converged = n <= 1;
    let mut last_off_norm = 0.0;
    for _sweep in 0..max_sweeps {
        let mut off_norm_sq = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let dpq = d.get(p, q);
                off_norm_sq += 2.0 * dpq * dpq;
            }
        }
        last_off_norm = off_norm_sq.sqrt();
        if last_off_norm <= effective_tol {
            converged = true;
            break;
        }

        let skip_thresh = effective_tol / (2.0 * n as f64);
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d.get(p, q);
                if apq.abs() <= skip_thresh {
                    continue;
                }
                let app = d.get(p, p);
                let aqq = d.get(q, q);
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                d.set(p, p, app - t * apq);
                d.set(q, q, aqq + t * apq);
                d.set(p, q, 0.0);
                d.set(q, p, 0.0);

                for r in 0..n {
                    if r != p && r != q {
                        let arp = d.get(r, p);
                        let arq = d.get(r, q);
                        let new_rp = c * arp - s * arq;
                        let new_rq = s * arp + c * arq;
                        d.set(r, p, new_rp);
                        d.set(p, r, new_rp);
                        d.set(r, q, new_rq);
                        d.set(q, r, new_rq);
                    }
                }

                for r in 0..n {
                    let vrp = v.get(r, p);
                    let vrq = v.get(r, q);
                    v.set(r, p, c * vrp - s * vrq);
                    v.set(r, q, s * vrp + c * vrq);
                }
            }
        }
    }

    if !converged {
        return Err(CalcError::LinearAlgebra(format!(
            "Jacobi eigensolver did not converge within {} sweeps for N={}: last_off_norm={:.3e}, effective_tol={:.3e}, a_norm={:.3e}",
            max_sweeps, n, last_off_norm, effective_tol, a_norm
        )));
    }

    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (d.get(i, i), i)).collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut evals = vec![0.0; n];
    let mut evecs = Matrix2D::zeros(n, n);
    for (new_col, &(val, old_col)) in pairs.iter().enumerate() {
        evals[new_col] = val;
        for r in 0..n {
            evecs.set(r, new_col, v.get(r, old_col));
        }
    }

    verify_eigen_residual(a,&evals,&evecs)?;
    Ok((evals,evecs))
}

fn verify_eigen_residual(a:&Matrix2D, evals:&[f64], evecs:&Matrix2D) -> Result<(),CalcError> {
    // Verify eigen-residual ||A * V - V * diag(evals)||_F
    let n = a.rows;
    let av = matmul(a, evecs);
    let mut res_sq = 0.0;
    for r in 0..n {
        for c in 0..n {
            let diff = av.get(r, c) - evecs.get(r, c) * evals[c];
            res_sq += diff * diff;
        }
    }
    let scale = frobenius_norm(a).max(1.0);
    if !res_sq.is_finite() || res_sq.sqrt() > 1e-9 * scale {
        return Err(CalcError::LinearAlgebra(format!(
            "Symmetric eigensolver residual too large: {:.3e}",
            res_sq.sqrt()
        )));
    }

    Ok(())
}

/// Computes the symmetric (pseudo-)inverse of a diagonally equilibrated SPD matrix `A`
/// along with the count of eigenvalues truncated below `rel_cutoff * lambda_max`.
///
/// Uses `faer` Householder eigendecomposition for reliable rank detection and Cholesky
/// (`LL^T`) inversion when full rank, giving near-machine-precision `||A A^-1 - I||`
/// even for condition numbers `~1e7..1e10`.
pub fn invert_spd_with_rank_info(
    a: &Matrix2D,
    rel_cutoff: f64,
) -> Result<(Matrix2D, usize), CalcError> {
    use faer::linalg::solvers::SolverCore;
    use faer::{Mat, Side};

    let n = a.rows;
    if n == 0 || a.cols != n || a.data.len() != n * n {
        return Err(CalcError::LinearAlgebra(
            "Matrix must be nonempty and square for invert_spd_with_rank_info".into(),
        ));
    }
    if a.data.iter().any(|v| !v.is_finite()) {
        return Err(CalcError::LinearAlgebra(
            "Matrix contains NaN or infinite entries".into(),
        ));
    }

    let mut scale = Vec::with_capacity(n);
    for i in 0..n {
        let d = a.get(i, i);
        if !d.is_finite() || d <= 0.0 {
            return Err(CalcError::LinearAlgebra(
                "Nonpositive/nonfinite SPD metric diagonal".into(),
            ));
        }
        scale.push(1.0 / d.sqrt());
    }

    let norm = Mat::<f64>::from_fn(n, n, |i, j| {
        0.5 * (a.get(i, j) + a.get(j, i)) * scale[i] * scale[j]
    });

    let evd = norm.selfadjoint_eigendecomposition(Side::Lower);
    let s_col = evd.s().column_vector();
    let lambda_max = (0..n).map(|i| s_col.read(i)).fold(0.0_f64, f64::max);
    let cutoff = lambda_max * rel_cutoff;
    if (0..n).any(|i| s_col.read(i) < -cutoff) {
        return Err(CalcError::LinearAlgebra("SPD metric is indefinite".into()));
    }

    let n_truncated = (0..n).filter(|&i| s_col.read(i) <= cutoff).count();

    let norm_inv = if n_truncated == 0 {
        if let Ok(chol) = norm.cholesky(Side::Lower) {
            chol.inverse()
        } else {
            let u = evd.u();
            let us_inv = Mat::<f64>::from_fn(n, n, |i, j| {
                let lam = s_col.read(j);
                if lam > cutoff {
                    u.read(i, j) / lam
                } else {
                    0.0
                }
            });
            us_inv * u.transpose()
        }
    } else {
        let u = evd.u();
        let us_inv = Mat::<f64>::from_fn(n, n, |i, j| {
            let lam = s_col.read(j);
            if lam > cutoff {
                u.read(i, j) / lam
            } else {
                0.0
            }
        });
        us_inv * u.transpose()
    };

    let mut out = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let val = 0.5 * (norm_inv.read(i, j) + norm_inv.read(j, i)) * scale[i] * scale[j];
            out.set(i, j, val);
            out.set(j, i, val);
        }
    }
    Ok((out, n_truncated))
}

/// Solves `A x = b` via Gaussian elimination with partial pivoting.
/// Returns `Err(CalcError::LinearAlgebra)` if the matrix is singular or near-singular.
pub fn solve_linear_system(a: &Matrix2D, b: &[f64]) -> Result<Vec<f64>, CalcError> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(CalcError::LinearAlgebra(
            "Dimension mismatch in solve_linear_system".into(),
        ));
    }
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a.get(i, j);
        }
        aug[i][n] = b[i];
    }

    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for (i, row) in aug.iter().enumerate().take(n).skip(k + 1) {
            if row[k].abs() > max_val {
                max_val = row[k].abs();
                max_row = i;
            }
        }
        if !max_val.is_finite() || max_val < 1e-14 {
            return Err(CalcError::LinearAlgebra(
                "Singular or near-singular linear system".into(),
            ));
        }
        aug.swap(k, max_row);
        let pivot = aug[k][k];
        for j in k..=n {
            aug[k][j] /= pivot;
        }
        for i in 0..n {
            if i != k {
                let factor = aug[i][k];
                for j in k..=n {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }
    }

    let sol: Vec<f64> = (0..n).map(|i| aug[i][n]).collect();
    if sol.iter().any(|v| !v.is_finite()) {
        return Err(CalcError::LinearAlgebra(
            "Non-finite solution in solve_linear_system".into(),
        ));
    }
    Ok(sol)
}

/// Pre-factored symmetric positive-definite matrix (with optional rank-truncated pseudoinverse
/// fallback) for solving `A x = b` via diagonally-scaled Cholesky `L L^T` without forming `A^{-1}`.
#[derive(Debug, Clone)]
pub struct SpdFactor {
    pub n: usize,
    pub n_truncated: usize,
    pub kappa_unscaled: f64,
    pub kappa_scaled: f64,
    scale: Vec<f64>,
    /// Row-major lower-triangular Cholesky factor `L` of `D^{-1/2} A D^{-1/2}` when full-rank.
    chol_lower: Option<Vec<f64>>,
    /// Spectral pseudoinverse fallback when `n_truncated > 0` or Cholesky fails.
    pinv: Option<Matrix2D>,
}

impl SpdFactor {
    /// Factors symmetric matrix `a` using spectral rank check (`rel_cutoff * lambda_max`)
    /// on `D^{-1/2} A D^{-1/2}` and Cholesky `L L^T` when full rank.
    pub fn from_matrix(a: &Matrix2D, rel_cutoff: f64) -> Result<Self, CalcError> {
        Self::factor(a, rel_cutoff, true)
    }

    pub fn for_solve(a: &Matrix2D, rel_cutoff: f64) -> Result<Self, CalcError> {
        Self::factor(a, rel_cutoff, false)
    }

    fn factor(a: &Matrix2D, rel_cutoff: f64, diagnostics: bool) -> Result<Self, CalcError> {
        use faer::{Mat, Side};

        let n = a.rows;
        if n == 0 || a.cols != n || a.data.len() != n * n {
            return Err(CalcError::LinearAlgebra(
                "Matrix must be nonempty and square for SpdFactor".into(),
            ));
        }
        if a.data.iter().any(|v| !v.is_finite()) {
            return Err(CalcError::LinearAlgebra(
                "Matrix contains NaN or infinite entries".into(),
            ));
        }

        let mut scale = Vec::with_capacity(n);
        for i in 0..n {
            let d = a.get(i, i);
            if !d.is_finite() || d <= 0.0 {
                return Err(CalcError::LinearAlgebra(
                    "Nonpositive/nonfinite SPD metric diagonal".into(),
                ));
            }
            scale.push(1.0 / d.sqrt());
        }

        let norm = Mat::<f64>::from_fn(n, n, |i, j| {
            0.5 * (a.get(i, j) + a.get(j, i)) * scale[i] * scale[j]
        });
        if !diagnostics && n >= 128 {
            // A successful factorization after shifting past the largest possible cutoff
            // certifies full numerical rank. Use a roundoff margin near the boundary.
            let upper = (0..n).map(|i| (0..n).map(|j| norm.read(i,j).abs()).sum::<f64>())
                .fold(0.0_f64,f64::max);
            let shift = upper * (rel_cutoff + 16.0*n as f64*f64::EPSILON);
            let mut shifted = norm.clone();
            for i in 0..n { shifted.write(i,i,shifted.read(i,i)-shift); }
            if shifted.cholesky(Side::Lower).is_ok() {
                if let Ok(chol) = norm.cholesky(Side::Lower) {
                    let l = chol.compute_l();
                    let mut lower = vec![0.0; n*n];
                    for i in 0..n { for j in 0..=i { lower[i*n+j] = l.read(i,j); } }
                    return Ok(Self {n, n_truncated:0, kappa_unscaled:f64::NAN,
                        kappa_scaled:f64::NAN, scale, chol_lower:Some(lower),pinv:None});
                }
            }
        }
        let evd = norm.selfadjoint_eigendecomposition(Side::Lower);
        let s_col = evd.s().column_vector();
        let lambda_max = (0..n).map(|i| s_col.read(i)).fold(0.0_f64, f64::max);
        let lambda_min = (0..n).map(|i| s_col.read(i)).fold(f64::INFINITY, f64::min);
        let kappa_scaled = if lambda_min > 0.0 {
            lambda_max / lambda_min
        } else {
            f64::INFINITY
        };
        let kappa_unscaled = f64::NAN;

        let cutoff = lambda_max * rel_cutoff;
        if (0..n).any(|i| s_col.read(i) < -cutoff) {
            return Err(CalcError::LinearAlgebra("SPD metric is indefinite".into()));
        }
        let n_truncated = (0..n).filter(|&i| s_col.read(i) <= cutoff).count();

        if n_truncated == 0 {
            if let Ok(chol) = norm.cholesky(Side::Lower) {
                let l_mat = chol.compute_l();
                let mut chol_lower = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..=i {
                        chol_lower[i * n + j] = l_mat.read(i, j);
                    }
                }
                return Ok(Self {
                    n,
                    n_truncated,
                    kappa_unscaled,
                    kappa_scaled,
                    scale,
                    chol_lower: Some(chol_lower),
                    pinv: None,
                });
            }
        }

        let u = evd.u();
        let us_inv = Mat::<f64>::from_fn(n, n, |i, j| {
            let lam = s_col.read(j);
            if lam > cutoff {
                u.read(i, j) / lam
            } else {
                0.0
            }
        });
        let norm_inv = us_inv * u.transpose();
        let mut pinv = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = 0.5 * (norm_inv.read(i, j) + norm_inv.read(j, i)) * scale[i] * scale[j];
                pinv.set(i, j, val);
                pinv.set(j, i, val);
            }
        }
        Ok(Self {
            n,
            n_truncated,
            kappa_unscaled,
            kappa_scaled,
            scale,
            chol_lower: None,
            pinv: Some(pinv),
        })
    }

    /// Solves `A x = b` using the stored Cholesky factor (or pseudoinverse fallback).
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        assert_eq!(b.len(), n, "SpdFactor::solve dimension mismatch");

        if let Some(ref l) = self.chol_lower {
            let mut y = vec![0.0; n];
            // Forward substitution: L y = D^{-1/2} b
            for i in 0..n {
                let mut sum = self.scale[i] * b[i];
                let row = i * n;
                for j in 0..i {
                    sum -= l[row + j] * y[j];
                }
                y[i] = sum / l[row + i];
            }
            // Backward substitution: L^T z = y
            let mut z = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                for j in (i + 1)..n {
                    sum -= l[j * n + i] * z[j];
                }
                z[i] = sum / l[i * n + i];
            }
            // Unscale: x = D^{-1/2} z
            for i in 0..n {
                z[i] *= self.scale[i];
            }
            z
        } else {
            let pinv = self.pinv.as_ref().expect("SpdFactor has neither chol nor pinv");
            let mut x = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += pinv.get(i, j) * b[j];
                }
                x[i] = sum;
            }
            x
        }
    }

    /// Computes the unscaled spectral condition number `lambda_max / lambda_min` of symmetric `a`.
    pub fn unscaled_kappa(a: &Matrix2D) -> f64 {
        use faer::{Mat, Side};
        let n = a.rows;
        let raw = Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a.get(i, j) + a.get(j, i)));
        let raw_evd = raw.selfadjoint_eigendecomposition(Side::Lower);
        let raw_s = raw_evd.s().column_vector();
        let raw_max = (0..n).map(|i| raw_s.read(i)).fold(0.0_f64, f64::max);
        let raw_min = (0..n).map(|i| raw_s.read(i)).fold(f64::INFINITY, f64::min);
        if raw_min > 0.0 {
            raw_max / raw_min
        } else {
            f64::INFINITY
        }
    }

    /// Materializes the explicit inverse (or pseudoinverse) `A^{-1}` for diagnostics.
    pub fn to_explicit_inverse(&self) -> Matrix2D {
        let n = self.n;
        if let Some(ref pinv) = self.pinv {
            return pinv.clone();
        }
        let mut out = Matrix2D::zeros(n, n);
        let mut e = vec![0.0; n];
        for j in 0..n {
            e[j] = 1.0;
            let col = self.solve(&e);
            e[j] = 0.0;
            for i in 0..n {
                out.set(i, j, col[i]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigh_sym_7x7_reconstruction() {
        let n = 7;
        let mut a = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = if i == j {
                    (i as f64) * 1.7 - 2.0
                } else {
                    0.15 / ((i + j + 1) as f64)
                };
                a.set(i, j, val);
                a.set(j, i, val);
            }
        }
        let (evals, evecs) = eigh_sym(&a, 100, 1e-14).unwrap();
        for i in 1..n {
            assert!(evals[i] >= evals[i - 1]);
        }
        let vt_v = matmul(&transpose(&evecs), &evecs);
        let id = Matrix2D::identity(n);
        assert!(rms_diff(&vt_v, &id) < 1e-13);
    }

    #[test]
    fn requested_eigen_residual_is_not_limited_by_an_arbitrary_relative_floor() {
        let mut a = Matrix2D::zeros(2, 2);
        a.data = vec![100.0, 2e-10, 2e-10, 200.0];
        let (e, v) = eigh_sym(&a, 100, 1e-13).unwrap();
        let av = matmul(&a, &v);
        let residual = (0..2).flat_map(|i| (0..2).map(move |j| (i,j)))
            .map(|(i,j)| (av.get(i,j) - v.get(i,j)*e[j]).powi(2)).sum::<f64>().sqrt();
        assert!(residual < 1e-12, "eigen residual={residual:.3e}");
    }
}
