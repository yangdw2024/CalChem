use faer::{Mat, MatRef, MatMut};
use crate::PackedSymMatrix;

pub struct FaerCpuBackend;

impl FaerCpuBackend {
    pub fn syevd_packed(sym: &PackedSymMatrix) -> (Vec<f64>, Mat<f64>) {
        let n = sym.dim;
        let mut mat = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                mat[(i, j)] = sym.get(i, j);
            }
        }
        let eig = mat.selfadjoint_eigenvalues(faer::Side::Lower);
        let mut evals: Vec<f64> = (0..n).map(|i| eig[i]).collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (evals, mat)
    }
}
