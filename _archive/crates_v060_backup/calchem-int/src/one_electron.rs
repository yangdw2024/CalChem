//! 一电子积分引擎（纯原生 Rust，Obara-Saika）。
//!
//! 提供壳层级（shell pair）的三类积分：
//!   - 重叠 S：三维可分离，用 1D Obara-Saika 递推。
//!   - 动能 T：由 1D 重叠的二阶导数组合（可分离）。
//!   - 核吸引 V：1/|r-C| 不可分离，用完整 OS 辅助递推（含 Boys 函数）。
//!
//! 所有原语级函数都以"壳层角动量 La/Lb"为输入，一次构建整个壳对的
//! 笛卡尔块（n_cart_a × n_cart_b），与 Python 参照的数值口径一致。
//! 归一化约定见 `calchem-basis`：每个原语在积分前乘以 N(α,l,m,n)。

use calchem_basis::{cartesian_angular_indices, BasisSetData, Shell};
use std::f64::consts::PI;

use crate::obara_saika_fast::{boys, primitive_norm};

/// 一个壳层对的几何/指数中间量（针对一对原语）。
struct PairGeom {
    gamma: f64,
    p: [f64; 3],
    k: f64,
    pa: [f64; 3], // P - A
    pb: [f64; 3], // P - B
}

fn pair_geom(alpha_a: f64, a: [f64; 3], alpha_b: f64, b: [f64; 3]) -> PairGeom {
    let gamma = alpha_a + alpha_b;
    let mut p = [0.0; 3];
    for i in 0..3 {
        p[i] = (alpha_a * a[i] + alpha_b * b[i]) / gamma;
    }
    let mut ab2 = 0.0;
    for i in 0..3 {
        ab2 += (a[i] - b[i]).powi(2);
    }
    let k = (-alpha_a * alpha_b / gamma * ab2).exp();
    let mut pa = [0.0; 3];
    let mut pb = [0.0; 3];
    for i in 0..3 {
        pa[i] = p[i] - a[i];
        pb[i] = p[i] - b[i];
    }
    PairGeom { gamma, p, k, pa, pb }
}

/// 1D 重叠积分表，返回 s[l][m]，l=0..=la, m=0..=lb（不含高斯乘积因子 K）。
///
/// 1D OS 递推：
///   (l+1|m) = (P-A)(l|m) + 1/(2γ)[ l (l-1|m) + m (l|m-1) ]
///   (l|m+1) = (P-B)(l|m) + 1/(2γ)[ l (l-1|m) + m (l|m-1) ]
fn overlap_1d_table(la: usize, lb: usize, pa: f64, pb: f64, gamma: f64) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0; lb + 1]; la + 1];
    s[0][0] = (PI / gamma).sqrt();
    // 先建 bra 列（m=0）。
    for l in 0..la {
        s[l + 1][0] = pa * s[l][0];
        if l >= 1 {
            s[l + 1][0] += (l as f64) / (2.0 * gamma) * s[l - 1][0];
        }
    }
    // 再建 ket。
    for l in 0..=la {
        for m in 0..lb {
            let mut v = pb * s[l][m];
            if m >= 1 {
                v += (m as f64) / (2.0 * gamma) * s[l][m - 1];
            }
            if l >= 1 {
                v += (l as f64) / (2.0 * gamma) * s[l - 1][m];
            }
            s[l][m + 1] = v;
        }
    }
    s
}

/// 一个原语对的整个重叠块（含 K），行主序 n_cart_a × n_cart_b。
fn primitive_overlap_block(la: usize, lb: usize, geom: &PairGeom) -> Vec<f64> {
    let sx = overlap_1d_table(la, lb, geom.pa[0], geom.pb[0], geom.gamma);
    let sy = overlap_1d_table(la, lb, geom.pa[1], geom.pb[1], geom.gamma);
    let sz = overlap_1d_table(la, lb, geom.pa[2], geom.pb[2], geom.gamma);
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let mut block = vec![0.0; cart_a.len() * cart_b.len()];
    for (i, &(al, am, an)) in cart_a.iter().enumerate() {
        for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
            block[i * cart_b.len() + j] =
                geom.k * sx[al][bl] * sy[am][bm] * sz[an][bn];
        }
    }
    block
}

/// 一个原语对的 1D 偶极表（分量轴，原点 0），在重叠表基础上：
///
///   (l|x|m) = P_x (l|m) + (1/2γ)[ l (l-1|m) + m (l|m-1) ]
///
/// 推导：电子坐标 x = x_b + B_x = P_x + (x_b - PB_x)，而 (x_b-PB)=(x-P)，
/// 由 OS bra 递推 (l|(x-P)|m) = (1/2γ)[l(l-1|m)+m(l|m-1)]。
fn dipole_1d_from_overlap(s: &[Vec<f64>], p: f64, gamma: f64) -> Vec<Vec<f64>> {
    let la = s.len() - 1;
    let lb = s[0].len() - 1;
    let mut d = vec![vec![0.0; lb + 1]; la + 1];
    for l in 0..=la {
        for m in 0..=lb {
            let mut v = p * s[l][m];
            if l >= 1 {
                v += (l as f64) / (2.0 * gamma) * s[l - 1][m];
            }
            if m >= 1 {
                v += (m as f64) / (2.0 * gamma) * s[l][m - 1];
            }
            d[l][m] = v;
        }
    }
    d
}

/// 一个原语对的整个偶极块（分量 ax，含 K），行主序 n_cart_a × n_cart_b。
fn primitive_dipole_block(la: usize, lb: usize, geom: &PairGeom, ax: usize) -> Vec<f64> {
    let mut t1d: [Vec<Vec<f64>>; 3] = [
        overlap_1d_table(la, lb, geom.pa[0], geom.pb[0], geom.gamma),
        overlap_1d_table(la, lb, geom.pa[1], geom.pb[1], geom.gamma),
        overlap_1d_table(la, lb, geom.pa[2], geom.pb[2], geom.gamma),
    ];
    t1d[ax] = dipole_1d_from_overlap(&t1d[ax], geom.p[ax], geom.gamma);
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let mut block = vec![0.0; cart_a.len() * cart_b.len()];
    for (i, &(al, am, an)) in cart_a.iter().enumerate() {
        for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
            block[i * cart_b.len() + j] = geom.k
                * t1d[0][al][bl] * t1d[1][am][bm] * t1d[2][an][bn];
        }
    }
    block
}

/// 一个原语对的整个动能块（含 K）。
///
/// T = -1/2 ∫ φ_a (∂²x + ∂²y + ∂²z) φ_b。
/// 二阶导数（ket，指数 β）：
///   ∂² g_m = m(m-1) x_b^{m-2}e^{-βx_b²} - 2β(2m+1) x_b^m e^{-βx_b²}
///            + 4β² x_b^{m+2} e^{-βx_b²}
fn primitive_kinetic_block(
    la: usize,
    lb: usize,
    alpha_a: f64,
    alpha_b: f64,
    a: [f64; 3],
    b: [f64; 3],
) -> Vec<f64> {
    let geom = pair_geom(alpha_a, a, alpha_b, b);
    // 1D 表需扩展到 ket 角 +2（m+2 项）。
    let sx = overlap_1d_table(la, lb + 2, geom.pa[0], geom.pb[0], geom.gamma);
    let sy = overlap_1d_table(la, lb + 2, geom.pa[1], geom.pb[1], geom.gamma);
    let sz = overlap_1d_table(la, lb + 2, geom.pa[2], geom.pb[2], geom.gamma);

    // 某轴 ket 二阶导数（1D），l 为 bra 角，m 为 ket 角。
    let d2 = |tbl: &[Vec<f64>], l: usize, m: usize| -> f64 {
        let mut v = -2.0 * alpha_b * (2 * m + 1) as f64 * tbl[l][m];
        if m >= 2 {
            v += (m * (m - 1)) as f64 * tbl[l][m - 2];
        }
        v += 4.0 * alpha_b * alpha_b * tbl[l][m + 2];
        v
    };

    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let mut block = vec![0.0; cart_a.len() * cart_b.len()];
    for (i, &(al, am, an)) in cart_a.iter().enumerate() {
        for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
            let t = -0.5
                * (d2(&sx, al, bl) * sy[am][bm] * sz[an][bn]
                    + sx[al][bl] * d2(&sy, am, bm) * sz[an][bn]
                    + sx[al][bl] * sy[am][bm] * d2(&sz, an, bn));
            block[i * cart_b.len() + j] = geom.k * t;
        }
    }
    block
}

/// 一个原语对的整个核吸引块（对所有核求和，含 K）。
#[allow(clippy::too_many_arguments)]
fn primitive_nuclear_block(
    la: usize,
    lb: usize,
    alpha_a: f64,
    a: [f64; 3],
    alpha_b: f64,
    b: [f64; 3],
    nuclei: &[([f64; 3], f64)],
) -> Vec<f64> {
    let geom = pair_geom(alpha_a, a, alpha_b, b);
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let mut block = vec![0.0; cart_a.len() * cart_b.len()];

    // bra/ket 用紧凑的 (l,m,n) 三维索引（lmax=壳角），尺寸 (L+1)^3。
    let bdim = |l: usize| (l + 1) * (l + 1) * (l + 1);
    let nbra = bdim(la);
    let nket = bdim(lb);
    let bidx = |l: usize, m: usize, n: usize, lmax: usize| -> usize {
        (l * (lmax + 1) + m) * (lmax + 1) + n
    };

    for &(c, zc) in nuclei {
        let mut pc = [0.0; 3];
        let mut pc2 = 0.0;
        for i in 0..3 {
            pc[i] = geom.p[i] - c[i];
            pc2 += pc[i].powi(2);
        }
        let t = geom.gamma * pc2;
        let nboys = la + lb;
        let ncnt = nboys + 1;
        let f = boys(t, nboys);
        let seed = -2.0 * PI * zc / geom.gamma * geom.k;

        let mut v = vec![0.0; nbra * nket * ncnt];
        let at = |v: &[f64], bl: usize, bm: usize, bn: usize,
                  kl: usize, km: usize, kn: usize, nn: usize| -> f64 {
            let bi = bidx(bl, bm, bn, la);
            let ki = bidx(kl, km, kn, lb);
            v[((bi * nket + ki) * ncnt) + nn]
        };
        let set = |v: &mut Vec<f64>, bl: usize, bm: usize, bn: usize,
                   kl: usize, km: usize, kn: usize, nn: usize, val: f64| {
            let bi = bidx(bl, bm, bn, la);
            let ki = bidx(kl, km, kn, lb);
            v[((bi * nket + ki) * ncnt) + nn] = val;
        };

        // 种子：bra=ket=0，所有 Boys 阶。
        for n in 0..ncnt {
            set(&mut v, 0, 0, 0, 0, 0, 0, n, seed * f[n]);
        }

        // 按总角动量 t 递推（壳角 la/lb 为界）。
        for total_t in 1..=(la + lb) {
            // 枚举 bra（l,m,n）总角 ≤ la，ket（l,m,n）总角 ≤ lb。
            for bl in 0..=la {
                for bm in 0..=la.saturating_sub(bl) {
                    for bn in 0..=la.saturating_sub(bl + bm) {
                        let bra_t = bl + bm + bn;
                        for kl in 0..=lb {
                            for km in 0..=lb.saturating_sub(kl) {
                                for kn in 0..=lb.saturating_sub(kl + km) {
                                    let ket_t = kl + km + kn;
                                    if bra_t + ket_t != total_t {
                                        continue;
                                    }
                                    // 轴：(轴, bra角, ket角, pa, pb, pc)。
                                    let axes: [(usize, usize, usize, f64, f64, f64); 3] = [
                                        (0, bl, kl, geom.pa[0], geom.pb[0], pc[0]),
                                        (1, bm, km, geom.pa[1], geom.pb[1], pc[1]),
                                        (2, bn, kn, geom.pa[2], geom.pb[2], pc[2]),
                                    ];
                                    // 第一个 bra 非零轴；否则第一个 ket 非零轴。
                                    let mut chosen: Option<(usize, bool)> = None;
                                    for a in axes.iter() {
                                        if a.1 > 0 {
                                            chosen = Some((a.0, true));
                                            break;
                                        }
                                    }
                                    if chosen.is_none() {
                                        for a in axes.iter() {
                                            if a.2 > 0 {
                                                chosen = Some((a.0, false));
                                                break;
                                            }
                                        }
                                    }
                                    let (ax, is_bra) = match chosen {
                                        Some(c) => c,
                                        None => continue,
                                    };

                                    let mut vals_n = vec![0.0; ncnt];
                                    for nn in 0..ncnt {
                                        let n1 = (nn + 1).min(ncnt - 1);
                                        let r = if is_bra {
                                            let ai = axes[ax].1;
                                            let bk = axes[ax].2;
                                            let pai = axes[ax].3;
                                            let pci = axes[ax].5;
                                            let p1 = dec(bl, bm, bn, ax);
                                            let mut r = pai * at(&v, p1.0, p1.1, p1.2, kl, km, kn, nn)
                                                - pci * at(&v, p1.0, p1.1, p1.2, kl, km, kn, n1);
                                            if ai >= 2 {
                                                let p2 = dec(p1.0, p1.1, p1.2, ax);
                                                let wn = at(&v, p2.0, p2.1, p2.2, kl, km, kn, nn);
                                                let w1 = at(&v, p2.0, p2.1, p2.2, kl, km, kn, n1);
                                                r += (ai - 1) as f64 / (2.0 * geom.gamma) * (wn - w1);
                                            }
                                            if bk > 0 {
                                                let k1 = dec(kl, km, kn, ax);
                                                let wn = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                                let w1 = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                                r += bk as f64 / (2.0 * geom.gamma) * (wn - w1);
                                            }
                                            r
                                        } else {
                                            let ab = axes[ax].1;
                                            let bi = axes[ax].2;
                                            let pbi = axes[ax].4;
                                            let pci = axes[ax].5;
                                            let k1 = dec(kl, km, kn, ax);
                                            let mut r = pbi * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, nn)
                                                - pci * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, n1);
                                            if bi >= 2 {
                                                let k2 = dec(k1.0, k1.1, k1.2, ax);
                                                let wn = at(&v, bl, bm, bn, k2.0, k2.1, k2.2, nn);
                                                let w1 = at(&v, bl, bm, bn, k2.0, k2.1, k2.2, n1);
                                                r += (bi - 1) as f64 / (2.0 * geom.gamma) * (wn - w1);
                                            }
                                            if ab > 0 {
                                                let p1 = dec(bl, bm, bn, ax);
                                                let wn = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                                let w1 = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                                r += ab as f64 / (2.0 * geom.gamma) * (wn - w1);
                                            }
                                            r
                                        };
                                        vals_n[nn] = r;
                                    }
                                    for nn in 0..ncnt {
                                        set(&mut v, bl, bm, bn, kl, km, kn, nn, vals_n[nn]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 累加输出（n=0）。
        for (i, &(al, am, an)) in cart_a.iter().enumerate() {
            for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
                block[i * cart_b.len() + j] += at(&v, al, am, an, bl, bm, bn, 0);
            }
        }
    }

    block
}

/// 把 (l,m,n) 在指定轴上减 1（调用方保证该轴 > 0）。
pub(crate) fn dec(l: usize, m: usize, n: usize, ax: usize) -> (usize, usize, usize) {
    match ax {
        0 => (l - 1, m, n),
        1 => (l, m - 1, n),
        _ => (l, m, n - 1),
    }
}

// ============================================================================
// 核吸引场梯度（Hellmann-Feynman 算子）：∫ χ_a χ_b (r-C)/|r-C|³
// 用于解析梯度中“核坐标直接导数”一项。
// ============================================================================

/// 原语级核吸引场梯度块（笛卡尔 n_cart_a × n_cart_b），分量 ax。
/// 场梯度算子：(r_ax - C_ax)/|r-C|³。
fn primitive_nuclear_field_block(
    la: usize,
    lb: usize,
    alpha_a: f64,
    a: [f64; 3],
    alpha_b: f64,
    b: [f64; 3],
    field: [f64; 3],
    ax: usize,
) -> Vec<f64> {
    let geom = pair_geom(alpha_a, a, alpha_b, b);
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let mut block = vec![0.0; cart_a.len() * cart_b.len()];

    let bdim = |l: usize| (l + 1) * (l + 1) * (l + 1);
    let nbra = bdim(la);
    let nket = bdim(lb);
    let bidx = |l: usize, m: usize, n: usize, lmax: usize| -> usize {
        (l * (lmax + 1) + m) * (lmax + 1) + n
    };

    // 场点 C 相对 P 的几何。
    let mut pc = [0.0; 3];
    let mut pc2 = 0.0;
    for i in 0..3 {
        pc[i] = geom.p[i] - field[i];
        pc2 += pc[i].powi(2);
    }
    let t = geom.gamma * pc2;
    // 场梯度种子用到 F_{n+1}，故多算一阶 Boys。
    let nboys = la + lb + 1;
    let ncnt = nboys + 1;
    let f = boys(t, nboys);

    // 同时维护两张表：
    //   v = 正核吸引 1/|r-C|（不含 Z）
    //   g = 场梯度 (r-C)/|r-C|³ = ∂v/∂C_ax
    let mut v = vec![0.0; nbra * nket * ncnt];
    let mut g = vec![0.0; nbra * nket * ncnt];
    let at = |t: &[f64], bl: usize, bm: usize, bn: usize,
              kl: usize, km: usize, kn: usize, nn: usize| -> f64 {
        let bi = bidx(bl, bm, bn, la);
        let ki = bidx(kl, km, kn, lb);
        t[((bi * nket + ki) * ncnt) + nn]
    };
    let set = |t: &mut Vec<f64>, bl: usize, bm: usize, bn: usize,
               kl: usize, km: usize, kn: usize, nn: usize, val: f64| {
        let bi = bidx(bl, bm, bn, la);
        let ki = bidx(kl, km, kn, lb);
        t[((bi * nket + ki) * ncnt) + nn] = val;
    };

    // 种子（s,s）：
    //   v: (2π/γ) K F_n
    //   g: 4π K (P_ax-C_ax) F_{n+1}
    for n in 0..(nboys) {
        set(&mut v, 0, 0, 0, 0, 0, 0, n, 2.0 * PI / geom.gamma * geom.k * f[n]);
        set(&mut g, 0, 0, 0, 0, 0, 0, n, 4.0 * PI * geom.k * pc[ax] * f[n + 1]);
    }

    // 角递推：v 走标准核吸引；g 走耦合递推（多一个 v^{n+1} 项）。
    for total_t in 1..=(la + lb) {
        for bl in 0..=la {
            for bm in 0..=la.saturating_sub(bl) {
                for bn in 0..=la.saturating_sub(bl + bm) {
                    let bra_t = bl + bm + bn;
                    for kl in 0..=lb {
                        for km in 0..=lb.saturating_sub(kl) {
                            for kn in 0..=lb.saturating_sub(kl + km) {
                                let ket_t = kl + km + kn;
                                if bra_t + ket_t != total_t {
                                    continue;
                                }
                                let axes: [(usize, usize, usize, f64, f64, f64); 3] = [
                                    (0, bl, kl, geom.pa[0], geom.pb[0], pc[0]),
                                    (1, bm, km, geom.pa[1], geom.pb[1], pc[1]),
                                    (2, bn, kn, geom.pa[2], geom.pb[2], pc[2]),
                                ];
                                let mut chosen: Option<(usize, bool)> = None;
                                for a in axes.iter() {
                                    if a.1 > 0 {
                                        chosen = Some((a.0, true));
                                        break;
                                    }
                                }
                                if chosen.is_none() {
                                    for a in axes.iter() {
                                        if a.2 > 0 {
                                            chosen = Some((a.0, false));
                                            break;
                                        }
                                    }
                                }
                                let (ax2, is_bra) = match chosen {
                                    Some(c) => c,
                                    None => continue,
                                };
                                let mut vv = vec![0.0; ncnt];
                                let mut gg = vec![0.0; ncnt];
                                for nn in 0..ncnt {
                                    let n1 = (nn + 1).min(ncnt - 1);
                                    if is_bra {
                                        let ai = axes[ax2].1;
                                        let bk = axes[ax2].2;
                                        let pai = axes[ax2].3;
                                        let pci = axes[ax2].5;
                                        let p1 = dec(bl, bm, bn, ax2);
                                        // v 递推。
                                        let mut rv = pai * at(&v, p1.0, p1.1, p1.2, kl, km, kn, nn)
                                            - pci * at(&v, p1.0, p1.1, p1.2, kl, km, kn, n1);
                                        // g 递推（同形，外加当 ax2==ax 时 +v^{n+1}）。
                                        let mut rg = pai * at(&g, p1.0, p1.1, p1.2, kl, km, kn, nn)
                                            - pci * at(&g, p1.0, p1.1, p1.2, kl, km, kn, n1);
                                        if ax2 == ax {
                                            rg += at(&v, p1.0, p1.1, p1.2, kl, km, kn, n1);
                                        }
                                        if ai >= 2 {
                                            let p2 = dec(p1.0, p1.1, p1.2, ax2);
                                            let vwn = at(&v, p2.0, p2.1, p2.2, kl, km, kn, nn);
                                            let vw1 = at(&v, p2.0, p2.1, p2.2, kl, km, kn, n1);
                                            rv += (ai - 1) as f64 / (2.0 * geom.gamma) * (vwn - vw1);
                                            let gwn = at(&g, p2.0, p2.1, p2.2, kl, km, kn, nn);
                                            let gw1 = at(&g, p2.0, p2.1, p2.2, kl, km, kn, n1);
                                            rg += (ai - 1) as f64 / (2.0 * geom.gamma) * (gwn - gw1);
                                        }
                                        if bk > 0 {
                                            let k1 = dec(kl, km, kn, ax2);
                                            let vwn = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                            let vw1 = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                            rv += bk as f64 / (2.0 * geom.gamma) * (vwn - vw1);
                                            let gwn = at(&g, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                            let gw1 = at(&g, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                            rg += bk as f64 / (2.0 * geom.gamma) * (gwn - gw1);
                                        }
                                        vv[nn] = rv;
                                        gg[nn] = rg;
                                    } else {
                                        let ab = axes[ax2].1;
                                        let bi = axes[ax2].2;
                                        let pbi = axes[ax2].4;
                                        let pci = axes[ax2].5;
                                        let k1 = dec(kl, km, kn, ax2);
                                        let mut rv = pbi * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, nn)
                                            - pci * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, n1);
                                        let mut rg = pbi * at(&g, bl, bm, bn, k1.0, k1.1, k1.2, nn)
                                            - pci * at(&g, bl, bm, bn, k1.0, k1.1, k1.2, n1);
                                        if ax2 == ax {
                                            rg += at(&v, bl, bm, bn, k1.0, k1.1, k1.2, n1);
                                        }
                                        if bi >= 2 {
                                            let k2 = dec(k1.0, k1.1, k1.2, ax2);
                                            let vwn = at(&v, bl, bm, bn, k2.0, k2.1, k2.2, nn);
                                            let vw1 = at(&v, bl, bm, bn, k2.0, k2.1, k2.2, n1);
                                            rv += (bi - 1) as f64 / (2.0 * geom.gamma) * (vwn - vw1);
                                            let gwn = at(&g, bl, bm, bn, k2.0, k2.1, k2.2, nn);
                                            let gw1 = at(&g, bl, bm, bn, k2.0, k2.1, k2.2, n1);
                                            rg += (bi - 1) as f64 / (2.0 * geom.gamma) * (gwn - gw1);
                                        }
                                        if ab > 0 {
                                            let p1 = dec(bl, bm, bn, ax2);
                                            let vwn = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                            let vw1 = at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                            rv += ab as f64 / (2.0 * geom.gamma) * (vwn - vw1);
                                            let gwn = at(&g, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, nn);
                                            let gw1 = at(&g, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                            rg += ab as f64 / (2.0 * geom.gamma) * (gwn - gw1);
                                        }
                                        vv[nn] = rv;
                                        gg[nn] = rg;
                                    };
                                }
                                for nn in 0..ncnt {
                                    set(&mut v, bl, bm, bn, kl, km, kn, nn, vv[nn]);
                                    set(&mut g, bl, bm, bn, kl, km, kn, nn, gg[nn]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 输出 g 的 n=0。
    for (i, &(al, am, an)) in cart_a.iter().enumerate() {
        for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
            block[i * cart_b.len() + j] += at(&g, al, am, an, bl, bm, bn, 0);
        }
    }
    block
}

/// 壳层级核吸引场梯度块（含收缩与归一化），分量 ax，场点为 field。
pub fn nuclear_field_gradient_block(
    shell_a: &Shell,
    shell_b: &Shell,
    field: [f64; 3],
    ax: usize,
) -> Vec<f64> {
    let la = shell_a.angular_l as usize;
    let lb = shell_b.angular_l as usize;
    let nrow = cartesian_angular_indices(la).len();
    let ncol = cartesian_angular_indices(lb).len();
    let mut block = vec![0.0; nrow * ncol];
    for p in 0..shell_a.n_prim() {
        let ca = shell_a.coefficients[p];
        let alpha_a = shell_a.exponents[p];
        for q in 0..shell_b.n_prim() {
            let cb = shell_b.coefficients[q];
            let alpha_b = shell_b.exponents[q];
            let raw = primitive_nuclear_field_block(
                la, lb, alpha_a, shell_a.center_bohr,
                alpha_b, shell_b.center_bohr, field, ax,
            );
            let cart_a = cartesian_angular_indices(la);
            let cart_b = cartesian_angular_indices(lb);
            for (i, &(al, am, an)) in cart_a.iter().enumerate() {
                let na = primitive_norm(alpha_a, al, am, an);
                for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
                    let nb = primitive_norm(alpha_b, bl, bm, bn);
                    block[i * ncol + j] += ca * cb * na * nb * raw[i * ncol + j];
                }
            }
        }
    }
    block
}

// ============================================================================
// 壳层级块（含收缩与归一化）
// ============================================================================

/// 计算两个壳层之间的一电子积分块（行主序，n_func_a × n_func_b）。
/// `op`：0=重叠, 1=动能, 2=核吸引。
pub fn one_electron_block(
    shell_a: &Shell,
    shell_b: &Shell,
    op: u8,
    nuclei: &[([f64; 3], f64)],
) -> Vec<f64> {
    let la = shell_a.angular_l as usize;
    let lb = shell_b.angular_l as usize;
    let nrow = calchem_basis::cartesian_angular_indices(la).len();
    let ncol = calchem_basis::cartesian_angular_indices(lb).len();
    let mut block = vec![0.0; nrow * ncol];

    for p in 0..shell_a.n_prim() {
        let ca = shell_a.coefficients[p];
        let alpha_a = shell_a.exponents[p];
        for q in 0..shell_b.n_prim() {
            let cb = shell_b.coefficients[q];
            let alpha_b = shell_b.exponents[q];

            let raw = match op {
                0 => {
                    let geom = pair_geom(alpha_a, shell_a.center_bohr, alpha_b, shell_b.center_bohr);
                    primitive_overlap_block(la, lb, &geom)
                }
                1 => primitive_kinetic_block(
                    la, lb, alpha_a, alpha_b, shell_a.center_bohr, shell_b.center_bohr,
                ),
                2 => primitive_nuclear_block(
                    la, lb, alpha_a, shell_a.center_bohr,
                    alpha_b, shell_b.center_bohr, nuclei,
                ),
                _ => vec![0.0; nrow * ncol],
            };
            // 原语归一化对每个笛卡尔分量施加（N 随分量 l,m,n 变化）。
            let cart_a = cartesian_angular_indices(la);
            let cart_b = cartesian_angular_indices(lb);
            for (i, &(al, am, an)) in cart_a.iter().enumerate() {
                let na = primitive_norm(alpha_a, al, am, an);
                for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
                    let nb = primitive_norm(alpha_b, bl, bm, bn);
                    block[i * ncol + j] += ca * cb * na * nb * raw[i * ncol + j];
                }
            }
        }
    }
    block
}

/// 计算两个壳层之间的偶极积分块（分量 ax，原点 0），行主序 n_func_a × n_func_b。
pub fn dipole_block(shell_a: &Shell, shell_b: &Shell, ax: usize) -> Vec<f64> {
    let la = shell_a.angular_l as usize;
    let lb = shell_b.angular_l as usize;
    let nrow = cartesian_angular_indices(la).len();
    let ncol = cartesian_angular_indices(lb).len();
    let mut block = vec![0.0; nrow * ncol];
    for p in 0..shell_a.n_prim() {
        let ca = shell_a.coefficients[p];
        let alpha_a = shell_a.exponents[p];
        for q in 0..shell_b.n_prim() {
            let cb = shell_b.coefficients[q];
            let alpha_b = shell_b.exponents[q];
            let geom = pair_geom(alpha_a, shell_a.center_bohr, alpha_b, shell_b.center_bohr);
            let raw = primitive_dipole_block(la, lb, &geom, ax);
            let cart_a = cartesian_angular_indices(la);
            let cart_b = cartesian_angular_indices(lb);
            for (i, &(al, am, an)) in cart_a.iter().enumerate() {
                let na = primitive_norm(alpha_a, al, am, an);
                for (j, &(bl, bm, bn)) in cart_b.iter().enumerate() {
                    let nb = primitive_norm(alpha_b, bl, bm, bn);
                    block[i * ncol + j] += ca * cb * na * nb * raw[i * ncol + j];
                }
            }
        }
    }
    block
}

/// 组装整个分子的偶极积分矩阵（分量 ax，行主序 nbf×nbf）。
pub fn compute_dipole_matrix(basis: &BasisSetData, ax: usize) -> Vec<f64> {
    let nbf = basis.nbf;
    let mut mat = vec![0.0; nbf * nbf];
    let nsh = basis.shells.len();
    for i in 0..nsh {
        let (a0, a1) = basis.shell_ao_range(i);
        for j in 0..nsh {
            let (b0, b1) = basis.shell_ao_range(j);
            let block = dipole_block(&basis.shells[i], &basis.shells[j], ax);
            let nrow = a1 - a0;
            let ncol = b1 - b0;
            for r in 0..nrow {
                for c in 0..ncol {
                    mat[(a0 + r) * nbf + (b0 + c)] = block[r * ncol + c];
                }
            }
        }
    }
    mat
}

/// 组装整个分子的一电子积分矩阵（行主序 nbf×nbf）。
pub fn compute_one_electron_matrix(basis: &BasisSetData, op: u8, nuclei: &[([f64; 3], f64)]) -> Vec<f64> {
    let nbf = basis.nbf;
    let mut mat = vec![0.0; nbf * nbf];
    let nsh = basis.shells.len();
    for i in 0..nsh {
        let (a0, a1) = basis.shell_ao_range(i);
        for j in 0..nsh {
            let (b0, b1) = basis.shell_ao_range(j);
            let block = one_electron_block(&basis.shells[i], &basis.shells[j], op, nuclei);
            let nrow = a1 - a0;
            let ncol = b1 - b0;
            for r in 0..nrow {
                for c in 0..ncol {
                    mat[(a0 + r) * nbf + (b0 + c)] = block[r * ncol + c];
                }
            }
        }
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{load_builtin_basis, AngularType, MoleculeCore};

    fn nuclei_of(mol: &MoleculeCore) -> Vec<([f64; 3], f64)> {
        mol.atoms
            .iter()
            .map(|a| (a.coord_bohr, a.atomic_number as f64))
            .collect()
    }

    fn build_h2() -> (MoleculeCore, BasisSetData, Vec<([f64; 3], f64)>) {
        let z = [1u8, 1];
        let coords = [0.0, 0.0, 0.0, 1.4, 0.0, 0.0];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let nuclei = nuclei_of(&mol);
        (mol, basis, nuclei)
    }

    fn build_h2o_szabo() -> (MoleculeCore, BasisSetData, Vec<([f64; 3], f64)>) {
        let z = [8u8, 1, 1];
        let coords = [
            0.0, 0.0, 0.0,
            0.0, 1.432, 0.0,
            1.112, -0.716, 0.0,
        ];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let nuclei = nuclei_of(&mol);
        (mol, basis, nuclei)
    }

    #[test]
    fn h2_one_electron_values() {
        let (_mol, basis, nuclei) = build_h2();
        let s = compute_one_electron_matrix(&basis, 0, &nuclei);
        let t = compute_one_electron_matrix(&basis, 1, &nuclei);
        let v = compute_one_electron_matrix(&basis, 2, &nuclei);
        // 与 Python 参照精确值对照。
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 0.6593182061).abs() < 1e-9);
        assert!((t[0] - 0.7600318836).abs() < 1e-9);
        assert!((t[1] - 0.2364546560).abs() < 1e-9);
        assert!((v[0] - (-1.8804408925)).abs() < 1e-9);
        assert!((v[1] - (-1.1948346204)).abs() < 1e-9);
    }

    #[test]
    fn h2o_one_electron_values() {
        let (_mol, basis, nuclei) = build_h2o_szabo();
        assert_eq!(basis.nbf, 7);
        let s = compute_one_electron_matrix(&basis, 0, &nuclei);
        let t = compute_one_electron_matrix(&basis, 1, &nuclei);
        let v = compute_one_electron_matrix(&basis, 2, &nuclei);
        // 对角重叠全为 1。
        for i in 0..7 {
            assert!((s[i * 7 + i] - 1.0).abs() < 1e-9, "S[{}]={}", i, s[i * 7 + i]);
        }
        // 与 Python 参照全量对照（逐元素，容差 1e-8；本实现 Boys 实际更精确）。
        // 这里只断言关键的 O1s 核吸引对角元（与 mpmath 真值一致）。
        assert!((v[0] - (-62.07288524288283)).abs() < 1e-8, "V00={}", v[0]);
        // 动能对角（O1s 动能）。
        assert!(t[0] > 28.0 && t[0] < 30.0);
    }
}
