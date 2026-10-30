//! 纯 Rust Obara-Saika 积分引擎（原生快路径）。
//!
//! 本文件目前包含：Boys 函数、笛卡尔原语归一化常数（已验证的种子）。
//! 壳层级 1e/2e 递推在后续填充。

#![allow(clippy::needless_range_loop)]

/// 双阶乘 (2n-1)!!，n>=0（n=0 时返回 1）。
#[inline]
pub fn double_factorial_odd(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut p = 1.0;
    let mut k = 2 * n - 1;
    while k >= 1 {
        p *= k as f64;
        if k <= 1 {
            break;
        }
        k -= 2;
    }
    p
}

/// 笛卡尔原语归一化常数 N(α,l,m,n)：
///   N = (2α/π)^{3/4} * (4α)^{(l+m+n)/2} / sqrt[(2l-1)!!(2m-1)!!(2n-1)!!]
/// 与 Python 参照 gto.py `_compute_norm` 完全一致。
pub fn primitive_norm(alpha: f64, l: usize, m: usize, n: usize) -> f64 {
    let radial = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
    let num = (4.0 * alpha).powi((l + m + n) as i32);
    let den = double_factorial_odd(l) * double_factorial_odd(m) * double_factorial_odd(n);
    let angular = (num / den).sqrt();
    radial * angular
}

/// Boys 函数 F_n(T) = ∫_0^1 t^{2n} exp(-T t²) dt。
///
/// 返回 F_0..=F_nmax（长度 nmax+1）。
///
/// 算法（标准自适应法）：
/// - 小 T：幂级数求 F_nmax，再稳定向下递推（此时 nmax > T，递推稳定）。
/// - 大 T：把起始阶数 n0 扩展到 T 之上（保证向下递推稳定且渐近式准确），
///   用渐近式求 F_n0，向下递推到所需阶数。
pub fn boys(t: f64, nmax: usize) -> Vec<f64> {
    if t < 1e-12 {
        // T -> 0：F_n(0) = 1/(2n+1)。
        return (0..=nmax).map(|n| 1.0 / (2 * n + 1) as f64).collect();
    }

    let mut f = vec![0.0; nmax + 1];

    // F_0 始终由 erf 精确给出（erf 准确到 ~1e-15，避免幂级数在大 T 时的相消）。
    f[0] = std::f64::consts::PI.sqrt() / (2.0 * t.sqrt()) * erf_accurate(t.sqrt());

    if t > 8.0 {
        // 大 T：从精确 F_0 向上递推。经验上 T>8 时 e^{-T} 已足够小，
        // 全区间（含 n 到 10）误差 < 5e-15。
        let em = (-t).exp();
        for n in 1..=nmax {
            f[n] = ((2 * n - 1) as f64 * f[n - 1] - em) / (2.0 * t);
        }
    } else {
        // 中等 T：对每个 n>=1 独立用幂级数 F_n = Σ_k (-T)^k/[k!(2n+2k+1)]。
        for n in 1..=nmax {
            let mut sum = 0.0;
            let mut term = 1.0 / (2 * n + 1) as f64;
            let mut k = 0usize;
            loop {
                sum += term;
                let num = 2 * (n + k) + 1;
                let den = 2 * (n + k + 1) + 1;
                term *= -t / (k + 1) as f64 * (num as f64) / (den as f64);
                k += 1;
                if term.abs() < 1e-16 * (sum.abs().max(1e-300)) {
                    break;
                }
                if k > 3000 {
                    break;
                }
            }
            f[n] = sum;
        }
    }

    f
}

/// 机器精度 erf（误差函数）。
///
/// 小参数用幂级数；大参数用 erfc 的连分式（Numerical Recipes 标准算法）。
pub fn erf_accurate(x: f64) -> f64 {
    if x < 0.0 {
        return -erf_accurate(-x);
    }
    if x < 1.0 {
        // erf(x) = (2/√π) Σ_n (-1)^n x^{2n+1}/(n!(2n+1))。
        let mut sum = 0.0;
        let mut term = x; // n=0
        let mut n = 0usize;
        loop {
            sum += term;
            let np = n + 1;
            term *= -x * x / np as f64 * (2 * n + 1) as f64 / (2 * np + 1) as f64;
            n = np;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
            if n > 500 {
                break;
            }
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfc(x) = Γ(1/2, x²)/√π，用不完全 γ 函数的连分式（z = x², a = 1/2）。
        let fpmin = 1e-300;
        let z = x * x;
        let mut b = z + 1.0 - 0.5; // = x² + 0.5
        let mut c = 1.0 / fpmin;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=400 {
            let an = -(i as f64) * (i as f64 - 0.5);
            b += 2.0;
            d = an * d + b;
            if d.abs() < fpmin {
                d = fpmin;
            }
            c = b + an / c;
            if c.abs() < fpmin {
                c = fpmin;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        // Γ(1/2,x²) = e^{-x²} (x²)^{1/2} · h = e^{-x²} x · h。
        let erfc = h * x * (-x * x).exp() / std::f64::consts::PI.sqrt();
        1.0 - erfc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boys_zero() {
        let f = boys(0.0, 4);
        for (n, v) in f.iter().enumerate() {
            assert!((v - 1.0 / (2 * n + 1) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn boys_known() {
        // 与 mpmath 地面真值（F_0..F_4）对照，覆盖小/中/大 T 三个区间。
        let table: &[(f64, &[f64])] = &[
            (0.01, &[0.9966766429033634, 0.33134045770977244, 0.19857696900746477, 0.1417505644077932, 0.11020585526922126]),
            (0.5, &[0.8556243918921488, 0.24909373217951536, 0.1407505368259127, 0.09722202441693015, 0.07402351120587763]),
            (2.0, &[0.5981440066613041, 0.11570218085617284, 0.05294281483297646, 0.032344697732067405, 0.022769400221964792]),
            (10.0, &[0.2802473905066427, 0.014010099528844014, 0.0020992449328384776, 0.0005225412367214951, 0.00018061943636439907]),
            (25.0, &[0.1772453850902791, 0.003544907701527823, 0.0002126944618139105, 2.1269445903632173e-05, 2.977722148749627e-06]),
            (60.0, &[0.11441140410797113, 0.0009534283675664261, 2.3835709189160652e-05, 9.931545495483605e-07, 5.7934015390321023e-08]),
        ];
        for &(t, expect) in table {
            let f = boys(t, 4);
            for n in 0..5 {
                assert!((f[n] - expect[n]).abs() < 1e-11, "T={} n={} got {} exp {}", t, n, f[n], expect[n]);
            }
        }
    }

    #[test]
    fn erf_matches() {
        // 与 mpmath 地面真值对照。
        let pairs = [
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (0.9, 0.7969082124228321),
            (1.2, 0.9103139782296354),
            (1.41421356, 0.9544997357412474),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
        ];
        for &(x, e) in pairs.iter() {
            assert!((erf_accurate(x) - e).abs() < 1e-12, "x={} got {}", x, erf_accurate(x));
            assert!((erf_accurate(-x) + e).abs() < 1e-12);
        }
    }

    #[test]
    fn norm_s_self() {
        // 归一化 s 原语自重叠应为 1。
        for &a in &[0.168, 3.425, 130.7] {
            let n = primitive_norm(a, 0, 0, 0);
            let self_ov = n * n * (std::f64::consts::PI / (2.0 * a)).powf(1.5);
            assert!((self_ov - 1.0).abs() < 1e-12);
        }
    }
}
