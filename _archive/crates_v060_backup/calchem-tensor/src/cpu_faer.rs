//! 基于 faer 的 CPU 线性代数后端。
//!
//! faer 内部通过 `pulp` 做运行时 SIMD 分派（在 Zen 5 上自动使用 AVX-512），
//! 因此这里无需在编译期硬编码 target-cpu 也能获得高性能。

use faer::{Mat, MatRef, Side};

use crate::PackedSymMatrix;

/// 由紧凑对称矩阵构造 faer 稠密矩阵。
pub fn dense_from_packed(sym: &PackedSymMatrix) -> Mat<f64> {
    let n = sym.dim;
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = sym.get(i, j);
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// 由行主序 Vec 构造 faer 矩阵。
pub fn mat_from_rowmajor(n: usize, data: &[f64]) -> Mat<f64> {
    assert_eq!(data.len(), n * n);
    Mat::from_fn(n, n, |i, j| data[i * n + j])
}

/// 对称矩阵特征分解，返回（升序特征值, 列向量为特征向量的矩阵 U）。
/// 满足 A = U * diag(λ) * U^T。
pub fn sym_eig(a: MatRef<'_, f64>) -> (Vec<f64>, Mat<f64>) {
    let n = a.nrows();
    let eig = faer::linalg::solvers::SelfAdjointEigendecomposition::new(a, Side::Lower);
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let lam = *eig.s().column_vector().get(j);
            let col: Vec<f64> = (0..n).map(|i| *eig.u().get(i, j)).collect();
            (lam, col)
        })
        .collect();
    // 按特征值升序排序（同时重排特征向量列）。
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let evals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut evecs = Mat::<f64>::zeros(n, n);
    for (j, (_, col)) in pairs.iter().enumerate() {
        for i in 0..n {
            evecs[(i, j)] = col[i];
        }
    }
    (evals, evecs)
}

/// 直接对紧凑对称矩阵做特征分解。
pub fn sym_eig_packed(sym: &PackedSymMatrix) -> (Vec<f64>, Mat<f64>) {
    let m = dense_from_packed(sym);
    sym_eig(m.as_ref())
}

/// 矩阵乘法：C = A * B（稠密）。
pub fn mat_mul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = b.ncols();
    assert_eq!(a.ncols(), b.nrows());
    let mut c = Mat::<f64>::zeros(m, n);
    // faer 语义：acc = beta·lhs·rhs（alpha=None 时不读旧 acc）。
    faer::linalg::matmul::matmul(
        c.as_mut(),
        a,
        b,
        None,
        1.0,
        faer::Parallelism::Rayon(0),
    );
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_diagonal() {
        let n = 3;
        let mut m = Mat::<f64>::zeros(n, n);
        m[(0, 0)] = 3.0;
        m[(1, 1)] = 1.0;
        m[(2, 2)] = 2.0;
        let (evals, _) = sym_eig(m.as_ref());
        assert!((evals[0] - 1.0).abs() < 1e-12);
        assert!((evals[1] - 2.0).abs() < 1e-12);
        assert!((evals[2] - 3.0).abs() < 1e-12);
    }
}
