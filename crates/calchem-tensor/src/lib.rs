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

    // Symmetrize input matrix A = 0.5 * (A + A^T)
    let mut d = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d.set(i, j, 0.5 * (a.get(i, j) + a.get(j, i)));
        }
    }
    let mut v = Matrix2D::identity(n);
    let max_sweeps = max_iter.max(60);
    let threshold = tol.max(1e-15);

    let mut converged = n <= 1;
    for _sweep in 0..max_sweeps {
        let mut off_norm_sq = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let dpq = d.get(p, q);
                off_norm_sq += 2.0 * dpq * dpq;
            }
        }
        if off_norm_sq.sqrt() <= threshold {
            converged = true;
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d.get(p, q);
                if apq.abs() <= threshold * 1e-2 {
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
        return Err(CalcError::LinearAlgebra(
            "Jacobi eigensolver did not converge within maximum sweeps".into(),
        ));
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

    // Verify eigen-residual ||A * V - V * diag(evals)||_F
    let av = matmul(a, &evecs);
    let mut res_sq = 0.0;
    for r in 0..n {
        for c in 0..n {
            let diff = av.get(r, c) - evecs.get(r, c) * evals[c];
            res_sq += diff * diff;
        }
    }
    let scale = frobenius_norm(a).max(1.0);
    if res_sq.sqrt() > 1e-9 * scale {
        return Err(CalcError::LinearAlgebra(format!(
            "Symmetric eigensolver residual too large: {:.3e}",
            res_sq.sqrt()
        )));
    }

    Ok((evals, evecs))
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
}
