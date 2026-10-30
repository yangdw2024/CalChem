//! Pulay DIIS（直接逆迭代子空间）加速器。
//!
//! 在正交基（S^-1/2）下，误差向量为
//!   e_i = F'_i P'_i - P'_i F'_i
//! 极小化线性组合的误差范数，得到系数 c：
//!   [ B   -1 ] [c] = [0]
//!   [ -1ᵀ  0 ] [λ]   [1]
//! 其中 B_ij = <e_i, e_j>。

/// 对增广矩阵 [A|b] 做高斯消元（部分主元），返回解 x。
/// 系统为 n×n；失败返回 None。
fn gaussian_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // 选主元。
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        // 消元。
        for r in (col + 1)..n {
            let f = a[r * n + col] / a[col * n + col];
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    // 回代。
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}

/// DIIS 混合器。
pub struct Diis {
    /// 最大保留向量数。
    pub max_vec: usize,
    /// 历史 Fock（正交基，行主序 n²）。
    focks: Vec<Vec<f64>>,
    /// 历史误差（正交基，行主序 n²）。
    errors: Vec<Vec<f64>>,
    /// 维度。
    n: usize,
}

impl Diis {
    /// 创建 DIIS（max_vec 通常 8）。
    pub fn new(n: usize, max_vec: usize) -> Self {
        Self {
            max_vec,
            focks: Vec::new(),
            errors: Vec::new(),
            n,
        }
    }

    /// 清空历史。
    pub fn reset(&mut self) {
        self.focks.clear();
        self.errors.clear();
    }

    /// 加入一个 Fock（正交基）与对应密度（正交基），返回混合后的 Fock（正交基）。
    ///
    /// 若历史不足 2 个，直接返回当前 Fock。
    pub fn update(&mut self, f_prime: &[f64], p_prime: &[f64]) -> Vec<f64> {
        let n = self.n;
        // 误差 e = F P - P F。
        let mut e = vec![0.0; n * n];
        let at = |m: &[f64], i: usize, j: usize| -> f64 { m[i * n + j] };
        for i in 0..n {
            for j in 0..n {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                for k in 0..n {
                    s1 += at(f_prime, i, k) * at(p_prime, k, j);
                    s2 += at(p_prime, i, k) * at(f_prime, k, j);
                }
                e[i * n + j] = s1 - s2;
            }
        }

        self.focks.push(f_prime.to_vec());
        self.errors.push(e);
        if self.focks.len() > self.max_vec {
            self.focks.remove(0);
            self.errors.remove(0);
        }

        let m = self.focks.len();
        if m < 2 {
            return f_prime.to_vec();
        }

        // 组装 DIIS 增广系统（m+1 阶）。
        let nsys = m + 1;
        let mut a = vec![0.0; nsys * nsys];
        let mut bvec = vec![0.0; nsys];
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for k in 0..(n * n) {
                    dot += self.errors[i][k] * self.errors[j][k];
                }
                a[i * nsys + j] = dot;
            }
            a[i * nsys + m] = -1.0;
            a[m * nsys + i] = 1.0;
        }
        a[m * nsys + m] = 0.0;
        bvec[m] = 1.0;
        // 小阻尼。
        for i in 0..m {
            a[i * nsys + i] += 1e-10;
        }

        let coeffs = match gaussian_solve(&mut a, &mut bvec, nsys) {
            Some(c) => c,
            None => {
                // 求解失败：丢弃最旧向量，返回当前 Fock。
                self.focks.remove(0);
                self.errors.remove(0);
                return f_prime.to_vec();
            }
        };

        // 混合 Fock。
        let mut fnew = vec![0.0; n * n];
        for i in 0..m {
            let ci = coeffs[i];
            for k in 0..(n * n) {
                fnew[k] += ci * self.focks[i][k];
            }
        }
        fnew
    }
}
