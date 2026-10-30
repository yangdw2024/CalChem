//! 张量线性代数：对称矩阵特征值分解、矩阵运算。

/// 对称矩阵特征值分解（使用雅可比迭代，MVP 纯 Rust 实现）
pub fn eigh_sym(mat: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_iter = 100;
    for _ in 0..max_iter {
        // 找最大非对角元素
        let (mut p, mut q) = (0, 1);
        let mut max_val = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // 雅可比旋转
        let theta = (a[p][p] - a[q][q]) / (2.0 * a[p][q]);
        let t = if theta > 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let mut eigenvectors = v;

    // 按特征值排序
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_vecs = vec![vec![0.0; n]; n];
    for (new_i, &old_i) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[row][new_i] = eigenvectors[row][old_i];
        }
    }

    (sorted_vals, sorted_vecs)
}

/// 矩阵乘法：a * b
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..b.len() {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// 矩阵转置
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut t = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// 矩阵加法
pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// 矩阵数乘
pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter().map(|row| row.iter().map(|&x| x * s).collect()).collect()
}

/// 矩阵迹
pub fn trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| row[i]).sum()
}
