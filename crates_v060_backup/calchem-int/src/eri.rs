//! 双电子排斥积分（ERI）引擎：Obara-Saika 垂直递推（VRR）+ 水平递推（HRR）。
//!
//! 化学记号 (ab|cd) = ∫∫ φ_a(r1) φ_b(r1) 1/r12 φ_c(r2) φ_d(r2) dr1 dr2。
//!
//! 标准 OS 方案：
//!   1) 种子 (00|00)^(n) 由 Boys F_n 给出。
//!   2) VRR 同时构建 bra 中心 A（角到 La+Lb）与 ket 中心 C（角到 Lc+Ld）。
//!   3) bra HRR 把 A 角转移到 B；ket HRR 把 C 角转移到 D。
//!
//! 本模块为笛卡尔（Cartesian）排布，与 STO-3G 基线一致。

use calchem_basis::{cartesian_angular_indices, BasisSetData, Shell};
use std::f64::consts::PI;

use crate::obara_saika_fast::{boys, primitive_norm};

/// 壳层四元组的几何中间量。
struct EriGeom {
    zeta_ab: f64,
    zeta_cd: f64,
    zeta: f64,
    kab: f64,
    kcd: f64,
    pa: [f64; 3], // P-A
    qc: [f64; 3], // Q-C
    pq: [f64; 3], // P-Q
}

#[allow(clippy::too_many_arguments)]
fn eri_geom(
    alpha_a: f64,
    a: [f64; 3],
    alpha_b: f64,
    b: [f64; 3],
    alpha_c: f64,
    c: [f64; 3],
    alpha_d: f64,
    d: [f64; 3],
) -> EriGeom {
    let zab = alpha_a + alpha_b;
    let zcd = alpha_c + alpha_d;
    let zeta = zab + zcd;
    let mut p = [0.0; 3];
    let mut q = [0.0; 3];
    for i in 0..3 {
        p[i] = (alpha_a * a[i] + alpha_b * b[i]) / zab;
        q[i] = (alpha_c * c[i] + alpha_d * d[i]) / zcd;
    }
    let mut ab2 = 0.0;
    let mut cd2 = 0.0;
    for i in 0..3 {
        ab2 += (a[i] - b[i]).powi(2);
        cd2 += (c[i] - d[i]).powi(2);
    }
    let kab = (-alpha_a * alpha_b / zab * ab2).exp();
    let kcd = (-alpha_c * alpha_d / zcd * cd2).exp();
    let mut pa = [0.0; 3];
    let mut qc = [0.0; 3];
    let mut pq = [0.0; 3];
    for i in 0..3 {
        pa[i] = p[i] - a[i];
        qc[i] = q[i] - c[i];
        pq[i] = p[i] - q[i];
    }
    EriGeom { zeta_ab: zab, zeta_cd: zcd, zeta, kab, kcd, pa, qc, pq }
}

/// 把 (l,m,n) 在指定轴减 1。
fn dec(l: usize, m: usize, n: usize, ax: usize) -> (usize, usize, usize) {
    match ax {
        0 => (l - 1, m, n),
        1 => (l, m - 1, n),
        _ => (l, m, n - 1),
    }
}

/// 一个原语四元组的整个 ERI 块（含 K_ab K_cd）。
/// 返回行主序 n_cart_a × n_cart_b × n_cart_c × n_cart_d。
#[allow(clippy::too_many_arguments)]
fn primitive_eri_block(
    la: usize,
    lb: usize,
    lc: usize,
    ld: usize,
    alpha_a: f64,
    a: [f64; 3],
    alpha_b: f64,
    b: [f64; 3],
    alpha_c: f64,
    c: [f64; 3],
    alpha_d: f64,
    d: [f64; 3],
) -> Vec<f64> {
    let g = eri_geom(alpha_a, a, alpha_b, b, alpha_c, c, alpha_d, d);

    // VRR 构建 bra 到 la+lb，ket 到 lc+ld。
    let lbra = la + lb;
    let lket = lc + ld;
    let nboys = lbra + lket;
    let ncnt = nboys + 1;

    let rho = g.zeta_ab * g.zeta_cd / g.zeta;
    let mut pq2 = 0.0;
    for i in 0..3 {
        pq2 += g.pq[i].powi(2);
    }
    let t = rho * pq2;
    let f = boys(t, nboys);

    let cube = |l: usize| (l + 1) * (l + 1) * (l + 1);
    let nbra = cube(lbra);
    let nket = cube(lket);
    let i3 = |l: usize, m: usize, n: usize, lmax: usize| -> usize {
        (l * (lmax + 1) + m) * (lmax + 1) + n
    };

    // VRR 数组：bra × ket × boys。
    let mut v = vec![0.0; nbra * nket * ncnt];
    let at = |v: &[f64], bl: usize, bm: usize, bn: usize,
              kl: usize, km: usize, kn: usize, nn: usize| -> f64 {
        v[((i3(bl, bm, bn, lbra) * nket + i3(kl, km, kn, lket)) * ncnt) + nn]
    };
    let set = |v: &mut Vec<f64>, bl: usize, bm: usize, bn: usize,
               kl: usize, km: usize, kn: usize, nn: usize, val: f64| {
        v[((i3(bl, bm, bn, lbra) * nket + i3(kl, km, kn, lket)) * ncnt) + nn] = val;
    };

    // 种子：2 π^(5/2)/(zab zcd √ζ) K_ab K_cd F_n。
    let seed = 2.0 * PI.powf(2.5) / (g.zeta_ab * g.zeta_cd * g.zeta.sqrt()) * g.kab * g.kcd;
    for n in 0..ncnt {
        set(&mut v, 0, 0, 0, 0, 0, 0, n, seed * f[n]);
    }

    // VRR：按总角（bra+ket）递推。
    for total_t in 1..=(lbra + lket) {
        for bl in 0..=lbra {
            for bm in 0..=lbra.saturating_sub(bl) {
                for bn in 0..=lbra.saturating_sub(bl + bm) {
                    let bt = bl + bm + bn;
                    for kl in 0..=lket {
                        for km in 0..=lket.saturating_sub(kl) {
                            for kn in 0..=lket.saturating_sub(kl + km) {
                                let kt = kl + km + kn;
                                if bt + kt != total_t {
                                    continue;
                                }
                                // 第一个 bra 非零轴；否则第一个 ket 非零轴。
                                let braxyz = [bl, bm, bn];
                                let ketxyz = [kl, km, kn];
                                let mut ax = 0;
                                let mut is_bra = false;
                                let mut found = false;
                                for k in 0..3 {
                                    if braxyz[k] > 0 {
                                        ax = k;
                                        is_bra = true;
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    for k in 0..3 {
                                        if ketxyz[k] > 0 {
                                            ax = k;
                                            is_bra = false;
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                if !found {
                                    continue;
                                }

                                let mut vals_n = vec![0.0; ncnt];
                                for nn in 0..ncnt {
                                    let n1 = (nn + 1).min(ncnt - 1);
                                    let r = if is_bra {
                                        let ai = braxyz[ax];
                                        let ck = ketxyz[ax];
                                        let pai = g.pa[ax];
                                        let pqi = g.pq[ax];
                                        let p1 = dec(bl, bm, bn, ax);
                                        // 与参照实现一致：PQ 辅助项为减。
                                        let mut r = pai * at(&v, p1.0, p1.1, p1.2, kl, km, kn, nn)
                                            - (g.zeta_cd / g.zeta) * pqi
                                                * at(&v, p1.0, p1.1, p1.2, kl, km, kn, n1);
                                        if ai >= 2 {
                                            let p2 = dec(p1.0, p1.1, p1.2, ax);
                                            r += (ai - 1) as f64 / (2.0 * g.zeta_ab)
                                                * at(&v, p2.0, p2.1, p2.2, kl, km, kn, nn)
                                                - (ai - 1) as f64 * g.zeta_cd
                                                    / (2.0 * g.zeta * g.zeta_ab)
                                                    * at(&v, p2.0, p2.1, p2.2, kl, km, kn, n1);
                                        }
                                        if ck > 0 {
                                            let k1 = dec(kl, km, kn, ax);
                                            r += ck as f64 / (2.0 * g.zeta)
                                                * at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
                                        }
                                        r
                                    } else {
                                        let ab_ = braxyz[ax];
                                        let ci = ketxyz[ax];
                                        let qci = g.qc[ax];
                                        let pqi = g.pq[ax];
                                        let k1 = dec(kl, km, kn, ax);
                                        // 与参照实现一致：ket PQ 辅助项为加。
                                        let mut r = qci * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, nn)
                                            + (g.zeta_ab / g.zeta) * pqi
                                                * at(&v, bl, bm, bn, k1.0, k1.1, k1.2, n1);
                                        if ci >= 2 {
                                            let k2 = dec(k1.0, k1.1, k1.2, ax);
                                            r += (ci - 1) as f64 / (2.0 * g.zeta_cd)
                                                * at(&v, bl, bm, bn, k2.0, k2.1, k2.2, nn)
                                                - (ci - 1) as f64 * g.zeta_ab
                                                    / (2.0 * g.zeta * g.zeta_cd)
                                                    * at(&v, bl, bm, bn, k2.0, k2.1, k2.2, n1);
                                        }
                                        if ab_ > 0 {
                                            let p1 = dec(bl, bm, bn, ax);
                                            r += ab_ as f64 / (2.0 * g.zeta)
                                                * at(&v, p1.0, p1.1, p1.2, k1.0, k1.1, k1.2, n1);
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

    // ------------------------------------------------------------------
    // HRR：先 bra（A→B），再 ket（C→D）。
    // ------------------------------------------------------------------
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let cart_c = cartesian_angular_indices(lc);
    let cart_d = cartesian_angular_indices(ld);
    let na = cart_a.len();
    let nb = cart_b.len();
    let nc = cart_c.len();
    let nd = cart_d.len();

    // bra HRR 数组：A（到 lbra）× B（到 lb）× C（到 lket）。
    let nb_c = cube(lb);
    let nket_c = cube(lket);
    let mut bh = vec![0.0; nbra * nb_c * nket_c];
    let bget = |h: &[f64], al: usize, am: usize, an: usize,
                bl: usize, bm: usize, bn: usize,
                cl: usize, cm: usize, cn: usize| -> f64 {
        h[((i3(al, am, an, lbra) * nb_c + i3(bl, bm, bn, lb)) * nket_c) + i3(cl, cm, cn, lket)]
    };
    let bset = |h: &mut Vec<f64>, al: usize, am: usize, an: usize,
                bl: usize, bm: usize, bn: usize,
                cl: usize, cm: usize, cn: usize, val: f64| {
        h[((i3(al, am, an, lbra) * nb_c + i3(bl, bm, bn, lb)) * nket_c) + i3(cl, cm, cn, lket)] = val;
    };

    // 初始化 B=0。
    for al in 0..=lbra {
        for am in 0..=lbra.saturating_sub(al) {
            for an in 0..=lbra.saturating_sub(al + am) {
                for cl in 0..=lket {
                    for cm in 0..=lket.saturating_sub(cl) {
                        for cn in 0..=lket.saturating_sub(cl + cm) {
                            bset(&mut bh, al, am, an, 0, 0, 0, cl, cm, cn,
                                 at(&v, al, am, an, cl, cm, cn, 0));
                        }
                    }
                }
            }
        }
    }

    // bra HRR 步进 B：(A,B+1|C) = (A+1,B|C) + (A_center-B_center)(A,B|C)。
    for bstep in 1..=lb {
        for al in 0..=lbra.saturating_sub(bstep) {
            for am in 0..=(lbra.saturating_sub(bstep)).saturating_sub(al) {
                for an in 0..=(lbra.saturating_sub(bstep)).saturating_sub(al + am) {
                    for bl in 0..=bstep {
                        for bm in 0..=bstep.saturating_sub(bl) {
                            let bn = bstep - bl - bm;
                            for cl in 0..=lket {
                                for cm in 0..=lket.saturating_sub(cl) {
                                    for cn in 0..=lket.saturating_sub(cl + cm) {
                                        let ax = if bl > 0 {
                                            0
                                        } else if bm > 0 {
                                            1
                                        } else {
                                            2
                                        };
                                        let mut a1 = [al, am, an];
                                        a1[ax] += 1;
                                        let mut b1 = [bl, bm, bn];
                                        b1[ax] -= 1;
                                        let ab_cent = a[ax] - b[ax];
                                        let v1 = bget(&bh, a1[0], a1[1], a1[2],
                                                      b1[0], b1[1], b1[2], cl, cm, cn);
                                        let v2 = bget(&bh, al, am, an,
                                                      b1[0], b1[1], b1[2], cl, cm, cn);
                                        bset(&mut bh, al, am, an, bl, bm, bn, cl, cm, cn,
                                             v1 + ab_cent * v2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ket HRR：对每个输出 bra，把 C 角转移到 D。
    let mut out = vec![0.0; na * nb * nc * nd];
    let kd_c = cube(ld);

    for (ia, &(al, am, an)) in cart_a.iter().enumerate() {
        for (ib, &(bl, bm, bn)) in cart_b.iter().enumerate() {
            let mut kh = vec![0.0; nket_c * kd_c];
            let kget = |h: &[f64], cl: usize, cm: usize, cn: usize,
                        dl: usize, dm: usize, dn: usize| -> f64 {
                h[i3(cl, cm, cn, lket) * kd_c + i3(dl, dm, dn, ld)]
            };
            let kset = |h: &mut Vec<f64>, cl: usize, cm: usize, cn: usize,
                        dl: usize, dm: usize, dn: usize, val: f64| {
                h[i3(cl, cm, cn, lket) * kd_c + i3(dl, dm, dn, ld)] = val;
            };

            // 初始化 D=0。
            for cl in 0..=lket {
                for cm in 0..=lket.saturating_sub(cl) {
                    for cn in 0..=lket.saturating_sub(cl + cm) {
                        kset(&mut kh, cl, cm, cn, 0, 0, 0,
                             bget(&bh, al, am, an, bl, bm, bn, cl, cm, cn));
                    }
                }
            }

            // ket HRR 步进 D：(C,D+1) = (C+1,D) + (C_center-D_center)(C,D)。
            for dstep in 1..=ld {
                for cl in 0..=lket.saturating_sub(dstep) {
                    for cm in 0..=(lket.saturating_sub(dstep)).saturating_sub(cl) {
                        for cn in 0..=(lket.saturating_sub(dstep)).saturating_sub(cl + cm) {
                            for dl in 0..=dstep {
                                for dm in 0..=dstep.saturating_sub(dl) {
                                    let dn = dstep - dl - dm;
                                    let ax = if dl > 0 {
                                        0
                                    } else if dm > 0 {
                                        1
                                    } else {
                                        2
                                    };
                                    let mut c1 = [cl, cm, cn];
                                    c1[ax] += 1;
                                    let mut d1 = [dl, dm, dn];
                                    d1[ax] -= 1;
                                    let cd_cent = c[ax] - d[ax];
                                    let v1 = kget(&kh, c1[0], c1[1], c1[2], d1[0], d1[1], d1[2]);
                                    let v2 = kget(&kh, cl, cm, cn, d1[0], d1[1], d1[2]);
                                    kset(&mut kh, cl, cm, cn, dl, dm, dn, v1 + cd_cent * v2);
                                }
                            }
                        }
                    }
                }
            }

            for (ic, &(cl, cm, cn)) in cart_c.iter().enumerate() {
                for (id, &(dl, dm, dn)) in cart_d.iter().enumerate() {
                    let idx = ((ia * nb + ib) * nc + ic) * nd + id;
                    out[idx] = kget(&kh, cl, cm, cn, dl, dm, dn);
                }
            }
        }
    }

    out
}

// ============================================================================
// 壳层级块（含收缩与归一化）
// ============================================================================

/// 计算四个壳层之间的 ERI 块（行主序 nf_a*nf_b*nf_c*nf_d）。
pub fn eri_block(sh_a: &Shell, sh_b: &Shell, sh_c: &Shell, sh_d: &Shell) -> Vec<f64> {
    let la = sh_a.angular_l as usize;
    let lb = sh_b.angular_l as usize;
    let lc = sh_c.angular_l as usize;
    let ld = sh_d.angular_l as usize;
    let cart_a = cartesian_angular_indices(la);
    let cart_b = cartesian_angular_indices(lb);
    let cart_c = cartesian_angular_indices(lc);
    let cart_d = cartesian_angular_indices(ld);
    let n = (cart_a.len(), cart_b.len(), cart_c.len(), cart_d.len());
    let mut block = vec![0.0; n.0 * n.1 * n.2 * n.3];

    for p in 0..sh_a.n_prim() {
        for q in 0..sh_b.n_prim() {
            for r in 0..sh_c.n_prim() {
                for s in 0..sh_d.n_prim() {
                    let raw = primitive_eri_block(
                        la, lb, lc, ld,
                        sh_a.exponents[p], sh_a.center_bohr,
                        sh_b.exponents[q], sh_b.center_bohr,
                        sh_c.exponents[r], sh_c.center_bohr,
                        sh_d.exponents[s], sh_d.center_bohr,
                    );
                    for (ia, &(alx, amx, anx)) in cart_a.iter().enumerate() {
                        let na = primitive_norm(sh_a.exponents[p], alx, amx, anx);
                        for (ib, &(blx, bmx, bnx)) in cart_b.iter().enumerate() {
                            let nb = primitive_norm(sh_b.exponents[q], blx, bmx, bnx);
                            for (ic, &(clx, cmx, cnx)) in cart_c.iter().enumerate() {
                                let nc = primitive_norm(sh_c.exponents[r], clx, cmx, cnx);
                                for (id, &(dlx, dmx, dnx)) in cart_d.iter().enumerate() {
                                    let nd = primitive_norm(sh_d.exponents[s], dlx, dmx, dnx);
                                    let idx = ((ia * n.1 + ib) * n.2 + ic) * n.3 + id;
                                    block[idx] += sh_a.coefficients[p]
                                        * sh_b.coefficients[q]
                                        * sh_c.coefficients[r]
                                        * sh_d.coefficients[s]
                                        * na * nb * nc * nd
                                        * raw[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    block
}

/// 组装整个分子的 ERI 为行主序 nbf×nbf×nbf×nbf（in-core，仅用于小体系）。
pub fn compute_eri_incore(basis: &BasisSetData) -> Vec<f64> {
    let nbf = basis.nbf;
    let nsh = basis.shells.len();
    let mut eri = vec![0.0; nbf * nbf * nbf * nbf];
    for i in 0..nsh {
        let (a0, a1) = basis.shell_ao_range(i);
        for j in i..nsh {
            let (b0, b1) = basis.shell_ao_range(j);
            for k in 0..nsh {
                let (c0, c1) = basis.shell_ao_range(k);
                for l in k..nsh {
                    let (d0, d1) = basis.shell_ao_range(l);
                    let blk = eri_block(
                        &basis.shells[i],
                        &basis.shells[j],
                        &basis.shells[k],
                        &basis.shells[l],
                    );
                    let dims = (a1 - a0, b1 - b0, c1 - c0, d1 - d0);
                    for ia in 0..dims.0 {
                        let p = a0 + ia;
                        for ib in 0..dims.1 {
                            let qq = b0 + ib;
                            for ic in 0..dims.2 {
                                let rr = c0 + ic;
                                for id in 0..dims.3 {
                                    let ss = d0 + id;
                                    let val = blk[((ia * dims.1 + ib) * dims.2 + ic) * dims.3 + id];
                                    let perms = [
                                        (p, qq, rr, ss),
                                        (qq, p, rr, ss),
                                        (p, qq, ss, rr),
                                        (qq, p, ss, rr),
                                        (rr, ss, p, qq),
                                        (ss, rr, p, qq),
                                        (rr, ss, qq, p),
                                        (ss, rr, qq, p),
                                    ];
                                    for (w, x, y, z) in perms {
                                        eri[((w * nbf + x) * nbf + y) * nbf + z] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eri
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{load_builtin_basis, AngularType, MoleculeCore};

    #[test]
    fn single_prim_p_raw() {
        let z = [0.0f64; 3];
        // (px px|s s)：la=1,lb=1,lc=0,ld=0
        let b1 = primitive_eri_block(1, 1, 0, 0, 1.0, z, 1.0, z, 1.0, z, 1.0, z);
        // 输出 na=3 (p), nb=3, nc=1, nd=1；(px px|s s) 是 ia=0, ib=0
        println!("raw (pxpx|ss)={}", b1[0]);
        // (px px|px px)：la=1,lb=1,lc=1,ld=1；ia=0,ib=0,ic=0,id=0
        let b2 = primitive_eri_block(1, 1, 1, 1, 1.0, z, 1.0, z, 1.0, z, 1.0, z);
        println!("raw (pxpx|pxpx)={}", b2[0]);
        // (px px|py py)：ia=0,ib=0,ic=1,id=1
        println!("raw (pxpx|pypy)={}", b2[((0*3+0)*3+1)*3+1]);
        // (px py|px py)：ia=0,ib=1,ic=0,id=1
        println!("raw (pxpy|pxpy)={}", b2[((0*3+1)*3+0)*3+1]);
    }

    #[test]
    fn h2_eri_values() {
        let z = [1u8, 1];
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let eri = compute_eri_incore(&basis);
        let n = basis.nbf;
        let get = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            eri[((i * n + j) * n + k) * n + l]
        };
        // 与 mpmath 独立真值对照（容差 1e-8，微小差异来自库存系数 8 位精度）。
        // 教材 Szabo 常写的 0.3008 为误记，真值 (12|12)=0.29702854。
        let e1111 = get(0, 0, 0, 0);
        let e1122 = get(0, 0, 1, 1);
        let e1212 = get(0, 1, 0, 1);
        println!("e1111={} e1122={} e1212={}", e1111, e1122, e1212);
        assert!((e1111 - 0.7746059442114873).abs() < 1e-8, "e1111={}", e1111);
        assert!((e1122 - 0.5696759264718783).abs() < 1e-8, "e1122={}", e1122);
        assert!((e1212 - 0.2970285411810491).abs() < 1e-8, "e1212={}", e1212);
    }

    #[test]
    fn h2o_eri_diagonal() {
        // H2O(Szabo)：ERI 对角 (ii|ii) 为正且有限，全张量对称。
        let z = [8u8, 1, 1];
        let coords = [
            0.0, 0.0, 0.0,
            0.0, 1.432, 0.0,
            1.112, -0.716, 0.0,
        ];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let eri = compute_eri_incore(&basis);
        let n = basis.nbf;
        assert_eq!(n, 7);
        // 每个 (ii|ii) > 0。
        for i in 0..n {
            let v = eri[((i * n + i) * n + i) * n + i];
            assert!(v > 0.0, "eri[{}]={}", i, v);
        }
        // 对称性抽查：(ij|kl)=(ji|lk)=(kl|ij)。
        let get = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            eri[((i * n + j) * n + k) * n + l]
        };
        assert!((get(0, 1, 2, 3) - get(1, 0, 3, 2)).abs() < 1e-12);
        assert!((get(0, 1, 2, 3) - get(2, 3, 0, 1)).abs() < 1e-12);
    }
}
