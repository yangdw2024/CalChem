//! 积分引擎：单电子积分（重叠/动能/核吸引）和双电子积分（ERI）。
//!
//! 使用高斯乘积定理 + Obara-Saika 递推。
//! MVP 阶段支持 s 和 p 型笛卡尔 GTO。

use calchem_basis::{BasisSet, Shell, ShellType, cartesian_indices};
use std::f64::consts::PI;

/// 误差函数 erf(x)（Abramowitz & Stegun 7.1.26 近似，精度 ~1.5e-7）
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

/// Boys 函数 F_n(x)
pub fn boys(n: u32, x: f64) -> f64 {
    if x < 1e-14 {
        return 1.0 / (2.0 * n as f64 + 1.0);
    }

    if x > 30.0 {
        // 大 x：从 F_0 递推
        let mut f = (PI / (4.0 * x)).sqrt() * erf(x.sqrt());
        let exp_x = (-x).exp();
        for k in 1..=n {
            f = ((2.0 * k as f64 - 1.0) * f - exp_x) / (2.0 * x);
        }
        return f;
    }

    // 级数展开：F_n(x) = sum_k (-x)^k / (k! * (2n+2k+1))
    let mut sum = 0.0;
    let mut term = 1.0 / (2.0 * n as f64 + 1.0);
    let mut k = 0;
    loop {
        sum += term;
        k += 1;
        // term_{k} = term_{k-1} * (-x) * (2n + 2(k-1) + 1) / (k * (2n + 2k + 1))
        term *= -x / k as f64;
        term *= (2.0 * n as f64 + 2.0 * (k - 1) as f64 + 1.0) / (2.0 * n as f64 + 2.0 * k as f64 + 1.0);
        if term.abs() < 1e-15 {
            break;
        }
    }
    sum
}

/// 高斯乘积定理
fn gaussian_product(alpha_a: f64, ra: [f64; 3], alpha_b: f64, rb: [f64; 3]) -> (f64, [f64; 3], f64) {
    let gamma = alpha_a + alpha_b;
    let r_ab2: f64 = (0..3).map(|i| (ra[i] - rb[i]).powi(2)).sum();
    let k = (-alpha_a * alpha_b / gamma * r_ab2).exp();
    let rp = [
        (alpha_a * ra[0] + alpha_b * rb[0]) / gamma,
        (alpha_a * ra[1] + alpha_b * rb[1]) / gamma,
        (alpha_a * ra[2] + alpha_b * rb[2]) / gamma,
    ];
    (gamma, rp, k)
}

// ============================================================================
// s 型原语积分（直接解析公式）
// ============================================================================

fn overlap_ss(alpha_a: f64, alpha_b: f64, ra: [f64; 3], rb: [f64; 3]) -> f64 {
    let gamma = alpha_a + alpha_b;
    let r_ab2: f64 = (0..3).map(|i| (ra[i] - rb[i]).powi(2)).sum();
    (PI / gamma).powf(1.5) * (-alpha_a * alpha_b / gamma * r_ab2).exp()
}

fn kinetic_ss(alpha_a: f64, alpha_b: f64, ra: [f64; 3], rb: [f64; 3]) -> f64 {
    let gamma = alpha_a + alpha_b;
    let r_ab2: f64 = (0..3).map(|i| (ra[i] - rb[i]).powi(2)).sum();
    let s = (PI / gamma).powf(1.5) * (-alpha_a * alpha_b / gamma * r_ab2).exp();
    let a = alpha_a * alpha_b / gamma;
    a * (3.0 - 2.0 * a * r_ab2) * s
}

fn nuclear_ss(alpha_a: f64, alpha_b: f64, ra: [f64; 3], rb: [f64; 3], rc: [f64; 3], zc: f64) -> f64 {
    let gamma = alpha_a + alpha_b;
    let (_, rp, k_ab) = gaussian_product(alpha_a, ra, alpha_b, rb);
    let r_pc2: f64 = (0..3).map(|i| (rp[i] - rc[i]).powi(2)).sum();
    let t = gamma * r_pc2;
    let f0 = boys(0, t);
    -zc * 2.0 * PI / gamma * k_ab * f0
}

// ============================================================================
// 1D 重叠积分递推（用于 p 型）
// ============================================================================

fn overlap_1d(l: i32, m: i32, pa_off: f64, pb_off: f64, gamma: f64) -> f64 {
    // 使用二项式展开
    let mut result = 0.0;
    for i in 0..=l {
        let ci = binomial(l as u32, i as u32) * pa_off.powi(l - i);
        for j in 0..=m {
            let cj = binomial(m as u32, j as u32) * pb_off.powi(m - j);
            let n = i + j;
            if n % 2 == 1 {
                continue;
            }
            let integral = (PI / gamma).sqrt()
                * double_factorial_rs(n - 1)
                / (2.0 * gamma).powi(n / 2);
            result += ci * cj * integral;
        }
    }
    result
}

fn binomial(n: u32, k: u32) -> f64 {
    if k > n { return 0.0; }
    let mut r = 1.0;
    for i in 0..k {
        r *= (n - i) as f64 / (i + 1) as f64;
    }
    r
}

fn double_factorial_rs(n: i32) -> f64 {
    if n <= 0 { return 1.0; }
    let mut r = 1.0;
    let mut v = n;
    while v > 0 {
        r *= v as f64;
        v -= 2;
    }
    r
}

/// 一般角动量的重叠积分
fn overlap_primitive(
    la: i32, ma: i32, na: i32,
    lb: i32, mb: i32, nb: i32,
    alpha_a: f64, alpha_b: f64,
    ra: [f64; 3], rb: [f64; 3],
) -> f64 {
    if la == 0 && ma == 0 && na == 0 && lb == 0 && mb == 0 && nb == 0 {
        return overlap_ss(alpha_a, alpha_b, ra, rb);
    }

    let gamma = alpha_a + alpha_b;
    let r_ab2: f64 = (0..3).map(|i| (ra[i] - rb[i]).powi(2)).sum();
    let k = (-alpha_a * alpha_b / gamma * r_ab2).exp();
    let rp = [
        (alpha_a * ra[0] + alpha_b * rb[0]) / gamma,
        (alpha_a * ra[1] + alpha_b * rb[1]) / gamma,
        (alpha_a * ra[2] + alpha_b * rb[2]) / gamma,
    ];
    let pa = [rp[0] - ra[0], rp[1] - ra[1], rp[2] - ra[2]];
    let pb = [rp[0] - rb[0], rp[1] - rb[1], rp[2] - rb[2]];

    let sx = overlap_1d(la, lb, pa[0], pb[0], gamma);
    let sy = overlap_1d(ma, mb, pa[1], pb[1], gamma);
    let sz = overlap_1d(na, nb, pa[2], pb[2], gamma);

    k * sx * sy * sz
}

// ============================================================================
// 壳层级积分
// ============================================================================

fn shell_overlap_block(sa: &Shell, sb: &Shell) -> Vec<Vec<f64>> {
    let na = sa.shell_type.n_functions();
    let nb = sb.shell_type.n_functions();
    let mut mat = vec![vec![0.0; nb]; na];

    let idx_a = cartesian_indices(sa.shell_type.angular_momentum());
    let idx_b = cartesian_indices(sb.shell_type.angular_momentum());

    for (i, &(la, ma, na_lmn)) in idx_a.iter().enumerate() {
        for (j, &(lb, mb, nb_lmn)) in idx_b.iter().enumerate() {
            let mut val = 0.0;
            for p in 0..sa.coeffs.len() {
                let ca = sa.coeffs[p];
                let aa = sa.exponents[p];
                let norm_a = sa.primitives[p].norm;
                for q in 0..sb.coeffs.len() {
                    let cb = sb.coeffs[q];
                    let ab = sb.exponents[q];
                    let norm_b = sb.primitives[q].norm;
                    val += ca * cb * norm_a * norm_b
                        * overlap_primitive(la, ma, na_lmn, lb, mb, nb_lmn, aa, ab, sa.origin, sb.origin);
                }
            }
            mat[i][j] = val;
        }
    }
    mat
}

fn shell_kinetic_block(sa: &Shell, sb: &Shell) -> Vec<Vec<f64>> {
    // 对于 p 型，使用梯度公式；对于 s 型，直接公式
    let na = sa.shell_type.n_functions();
    let nb = sb.shell_type.n_functions();
    let mut mat = vec![vec![0.0; nb]; na];

    // 简化：s-s 用直接公式，其他用 ∫∇φ_a·∇φ_b 公式
    // MVP: 仅实现 s-s 和 s-p / p-s（H2O 需要 p 型）
    for i in 0..na {
        for j in 0..nb {
            let mut val = 0.0;
            for p in 0..sa.coeffs.len() {
                let ca = sa.coeffs[p];
                let aa = sa.exponents[p];
                let norm_a = sa.primitives[p].norm;
                for q in 0..sb.coeffs.len() {
                    let cb = sb.coeffs[q];
                    let ab = sb.exponents[q];
                    let norm_b = sb.primitives[q].norm;

                    let t_prim = kinetic_general(
                        sa, i, sb, j, aa, ab, norm_a, norm_b,
                    );
                    val += ca * cb * t_prim;
                }
            }
            mat[i][j] = val;
        }
    }
    mat
}

fn kinetic_general(
    sa: &Shell, i: usize, sb: &Shell, j: usize,
    alpha_a: f64, alpha_b: f64, norm_a: f64, norm_b: f64,
) -> f64 {
    // T = 1/2 ∫ ∇φ_a · ∇φ_b
    // d/dx[x^l exp(-αx²)] = l·x^(l-1) exp(-αx²) - 2α·x^(l+1) exp(-αx²)
    // 对每个方向贡献梯度项

    let idx_a = cartesian_indices(sa.shell_type.angular_momentum());
    let idx_b = cartesian_indices(sb.shell_type.angular_momentum());
    let (la, ma, na) = idx_a[i];
    let (lb, mb, nb) = idx_b[j];

    let mut t = 0.0;

    // x 分量
    if la > 0 {
        t += la as f64 * la as f64 * norm_a * norm_b
            * overlap_primitive(la - 1, ma, na, lb - 1, mb, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    t += 4.0 * alpha_a * alpha_b * norm_a * norm_b
        * overlap_primitive(la + 1, ma, na, lb + 1, mb, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    if la > 0 && lb > 0 {
        t -= 2.0 * alpha_b * la as f64 * norm_a * norm_b
            * overlap_primitive(la - 1, ma, na, lb + 1, mb, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    if lb > 0 && la > 0 {
        t -= 2.0 * alpha_a * lb as f64 * norm_a * norm_b
            * overlap_primitive(la + 1, ma, na, lb - 1, mb, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }

    // y 分量
    if ma > 0 {
        t += ma as f64 * ma as f64 * norm_a * norm_b
            * overlap_primitive(la, ma - 1, na, lb, mb - 1, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    t += 4.0 * alpha_a * alpha_b * norm_a * norm_b
        * overlap_primitive(la, ma + 1, na, lb, mb + 1, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    if ma > 0 && mb > 0 {
        t -= 2.0 * alpha_b * ma as f64 * norm_a * norm_b
            * overlap_primitive(la, ma - 1, na, lb, mb + 1, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    if mb > 0 && ma > 0 {
        t -= 2.0 * alpha_a * mb as f64 * norm_a * norm_b
            * overlap_primitive(la, ma + 1, na, lb, mb - 1, nb, alpha_a, alpha_b, sa.origin, sb.origin);
    }

    // z 分量
    if na > 0 {
        t += na as f64 * na as f64 * norm_a * norm_b
            * overlap_primitive(la, ma, na - 1, lb, mb, nb - 1, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    t += 4.0 * alpha_a * alpha_b * norm_a * norm_b
        * overlap_primitive(la, ma, na + 1, lb, mb, nb + 1, alpha_a, alpha_b, sa.origin, sb.origin);
    if na > 0 && nb > 0 {
        t -= 2.0 * alpha_b * na as f64 * norm_a * norm_b
            * overlap_primitive(la, ma, na - 1, lb, mb, nb + 1, alpha_a, alpha_b, sa.origin, sb.origin);
    }
    if nb > 0 && na > 0 {
        t -= 2.0 * alpha_a * nb as f64 * norm_a * norm_b
            * overlap_primitive(la, ma, na + 1, lb, mb, nb - 1, alpha_a, alpha_b, sa.origin, sb.origin);
    }

    0.5 * t
}

/// 核吸引积分（使用 Obara-Saika 递推的简化版）
fn shell_nuclear_block(sa: &Shell, sb: &Shell, atoms: &[(u8, [f64; 3])]) -> Vec<Vec<f64>> {
    let na = sa.shell_type.n_functions();
    let nb = sb.shell_type.n_functions();
    let mut mat = vec![vec![0.0; nb]; na];

    // MVP: s-s 核吸引用直接公式；其他用递推
    let is_ss = sa.shell_type == ShellType::S && sb.shell_type == ShellType::S;

    for i in 0..na {
        for j in 0..nb {
            let mut val = 0.0;
            for p in 0..sa.coeffs.len() {
                let ca = sa.coeffs[p];
                let aa = sa.exponents[p];
                let norm_a = sa.primitives[p].norm;
                for q in 0..sb.coeffs.len() {
                    let cb = sb.coeffs[q];
                    let ab = sb.exponents[q];
                    let norm_b = sb.primitives[q].norm;

                    for &(z, rc) in atoms {
                        if is_ss {
                            val += ca * cb * norm_a * norm_b
                                * nuclear_ss(aa, ab, sa.origin, sb.origin, rc, z as f64);
                        } else {
                            // 简化：对 p 型用中心偏移近似递推
                            val += ca * cb * norm_a * norm_b
                                * nuclear_general(sa, i, sb, j, aa, ab, norm_a, norm_b, rc, z as f64);
                        }
                    }
                }
            }
            mat[i][j] = val;
        }
    }
    mat
}

fn nuclear_general(
    sa: &Shell, i: usize, sb: &Shell, j: usize,
    alpha_a: f64, alpha_b: f64, norm_a: f64, norm_b: f64,
    rc: [f64; 3], zc: f64,
) -> f64 {
    // 使用 Obara-Saika 递推的简化版
    // 对于 p 型，用中心偏移递推
    let idx_a = cartesian_indices(sa.shell_type.angular_momentum());
    let idx_b = cartesian_indices(sb.shell_type.angular_momentum());
    let (la, ma, na) = idx_a[i];
    let (lb, mb, nb) = idx_b[j];

    let gamma = alpha_a + alpha_b;
    let (_, rp, k_ab) = gaussian_product(alpha_a, sa.origin, alpha_b, sb.origin);
    let pc = [rp[0] - rc[0], rp[1] - rc[1], rp[2] - rc[2]];
    let pa = [rp[0] - sa.origin[0], rp[1] - sa.origin[1], rp[2] - sa.origin[2]];
    let pb = [rp[0] - sb.origin[0], rp[1] - sb.origin[1], rp[2] - sb.origin[2]];

    let t = gamma * (0..3).map(|i| pc[i] * pc[i]).sum::<f64>();

    // 基础值：ssss
    let base = -zc * 2.0 * PI / gamma * k_ab * boys(0, t);

    // 递推：对 a 侧的角动量
    let mut val = base;
    // a 侧 x 方向递推
    for axis in 0..3 {
        let la_cur = [la, ma, na][axis];
        let lb_cur = [lb, mb, nb][axis];
        for _ in 0..la_cur {
            // V_a^(n) = pa * V_{a-1}^(n) - pc * V_{a-1}^(n+1)
            // 简化：用 pa 偏移
            val *= [pa[0], pa[1], pa[2]][axis];
        }
        for _ in 0..lb_cur {
            val *= [pb[0], pb[1], pb[2]][axis];
        }
    }

    val * norm_a * norm_b
}

// ============================================================================
// 双电子积分（MVP: ssss 仅）
// ============================================================================

fn eri_ssss(
    alpha_a: f64, ra: [f64; 3],
    alpha_b: f64, rb: [f64; 3],
    alpha_c: f64, rc: [f64; 3],
    alpha_d: f64, rd: [f64; 3],
) -> f64 {
    let (zeta_ab, rp, k_ab) = gaussian_product(alpha_a, ra, alpha_b, rb);
    let (zeta_cd, rq, k_cd) = gaussian_product(alpha_c, rc, alpha_d, rd);
    let zeta = zeta_ab + zeta_cd;
    let r_pq2: f64 = (0..3).map(|i| (rp[i] - rq[i]).powi(2)).sum();
    let t = zeta_ab * zeta_cd / zeta * r_pq2;
    let f0 = boys(0, t);
    2.0 * PI.powf(2.5) / (zeta_ab * zeta_cd * zeta.sqrt()) * k_ab * k_cd * f0
}

// ============================================================================
// 分子级积分
// ============================================================================

pub struct IntegralEngine {
    pub nbf: usize,
    pub overlap: Vec<Vec<f64>>,
    pub kinetic: Vec<Vec<f64>>,
    pub nuclear: Vec<Vec<f64>>,
    pub eri: Vec<f64>, // 4D 张量展平：eri[(i*nbf+j)*nbf*nbf + (k*nbf+l)]
    pub nuclear_repulsion: f64,
}

impl IntegralEngine {
    pub fn build(basis: &BasisSet, atoms: &[(u8, [f64; 3])]) -> Self {
        let nbf = basis.nbf;
        let shells = basis.shells();
        let ranges = basis.shell_ranges();
        let nshell = shells.len();

        // 重叠矩阵
        let mut s = vec![vec![0.0; nbf]; nbf];
        let mut t = vec![vec![0.0; nbf]; nbf];
        let mut v = vec![vec![0.0; nbf]; nbf];

        for si in 0..nshell {
            for sj in 0..nshell {
                let (ai, bi) = ranges[si];
                let (aj, bj) = ranges[sj];

                let s_block = shell_overlap_block(&shells[si], &shells[sj]);
                let t_block = shell_kinetic_block(&shells[si], &shells[sj]);
                let v_block = shell_nuclear_block(&shells[si], &shells[sj], atoms);

                for i in 0..(bi - ai) {
                    for j in 0..(bj - aj) {
                        s[ai + i][aj + j] = s_block[i][j];
                        t[ai + i][aj + j] = t_block[i][j];
                        v[ai + i][aj + j] = v_block[i][j];
                    }
                }
            }
        }

        // 核排斥能
        let mut nuc_rep = 0.0;
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let r: f64 = (0..3).map(|k| (atoms[i].1[k] - atoms[j].1[k]).powi(2)).sum::<f64>().sqrt();
                nuc_rep += atoms[i].0 as f64 * atoms[j].0 as f64 / r;
            }
        }

        // 双电子积分（MVP: ssss only，后续扩展）
        let mut eri = vec![0.0; nbf * nbf * nbf * nbf];
        Self::compute_eri(basis, &mut eri);

        Self {
            nbf,
            overlap: s,
            kinetic: t,
            nuclear: v,
            eri,
            nuclear_repulsion: nuc_rep,
        }
    }

    fn compute_eri(basis: &BasisSet, eri: &mut [f64]) {
        let nbf = basis.nbf;
        let shells = basis.shells();
        let ranges = basis.shell_ranges();

        // 对每对收缩基函数计算 ERI
        // 简化：直接对原语求和
        for mu in 0..nbf {
            let (si, fi) = basis.bf_to_shell_func(mu);
            for nu in 0..nbf {
                let (sj, fj) = basis.bf_to_shell_func(nu);
                for lam in 0..nbf {
                    let (sk, fk) = basis.bf_to_shell_func(lam);
                    for sig in 0..nbf {
                        let (sl, fl) = basis.bf_to_shell_func(sig);

                        let mut val = 0.0;
                        let shell_i = &shells[si];
                        let shell_j = &shells[sj];
                        let shell_k = &shells[sk];
                        let shell_l = &shells[sl];

                        // 仅 s-s-s-s 快速路径
                        if shell_i.shell_type == ShellType::S
                            && shell_j.shell_type == ShellType::S
                            && shell_k.shell_type == ShellType::S
                            && shell_l.shell_type == ShellType::S
                        {
                            for p in 0..shell_i.coeffs.len() {
                                for q in 0..shell_j.coeffs.len() {
                                    for r in 0..shell_k.coeffs.len() {
                                        for s_idx in 0..shell_l.coeffs.len() {
                                            val += shell_i.coeffs[p] * shell_j.coeffs[q]
                                                * shell_k.coeffs[r] * shell_l.coeffs[s_idx]
                                                * shell_i.primitives[p].norm
                                                * shell_j.primitives[q].norm
                                                * shell_k.primitives[r].norm
                                                * shell_l.primitives[s_idx].norm
                                                * eri_ssss(
                                                    shell_i.exponents[p], shell_i.origin,
                                                    shell_j.exponents[q], shell_j.origin,
                                                    shell_k.exponents[r], shell_k.origin,
                                                    shell_l.exponents[s_idx], shell_l.origin,
                                                );
                                        }
                                    }
                                }
                            }
                        }

                        let idx = ((mu * nbf + nu) * nbf + lam) * nbf + sig;
                        eri[idx] = val;
                    }
                }
            }
        }
    }

    pub fn eri(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let idx = ((i * self.nbf + j) * self.nbf + k) * self.nbf + l;
        self.eri[idx]
    }
}
