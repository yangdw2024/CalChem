//! 张量与线性代数层。

pub fn mat_vec_mul(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += a[i * n + j] * x[j];
        }
    }
    y
}
