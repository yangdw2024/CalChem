//! Direct / In-Core 库仑(J)与交换(K)矩阵构建。
//!
//! - In-Core：先组装完整 ERI 四阶张量，再与密度矩阵收缩（小体系，最简单可靠）。
//! - Direct：按壳层四元组即时计算 ERI 并收缩，配合 Schwarz 预筛选（大体系）。
//!
//! 闭壳层密度约定：P_mn = 2 Σ_i^occ C_mi C_ni（含因子 2）。
//!   J_mn = Σ_kl P_kl (mn|kl)
//!   K_mn = Σ_kl P_kl (ml|kn)
//!   F = H + J - K
//!
//! `k_scale`：RHF=1.0，纯 GGA=0.0，B3LYP=0.20。

use calchem_basis::BasisSetData;
use calchem_tensor::PackedSymMatrix;

use crate::eri::eri_block;
use crate::screening::SchwarzScreener;

/// 由完整 ERI（行主序 nbf⁴）与密度矩阵构建 J、K（in-core）。
///
/// `k_scale`：交换项缩放（RHF=1.0）。
pub fn build_jk_incore(
    nbf: usize,
    eri: &[f64],
    dm: &PackedSymMatrix,
    k_scale: f64,
) -> (PackedSymMatrix, PackedSymMatrix) {
    let mut j = PackedSymMatrix::zeros(nbf);
    let mut k = PackedSymMatrix::zeros(nbf);
    let e = |i: usize, j: usize, k: usize, l: usize| -> f64 {
        eri[((i * nbf + j) * nbf + k) * nbf + l]
    };
    for m in 0..nbf {
        // 只算上三角（n>=m），避免 (m,n) 与 (n,m) 映射到同一紧凑位置而重复累加。
        for n in m..nbf {
            let mut jj = 0.0;
            let mut kk = 0.0;
            for kk2 in 0..nbf {
                for l in 0..nbf {
                    let p = dm.get(kk2, l);
                    jj += p * e(m, n, kk2, l);
                    // K_mn = Σ_kl P_kl (m k | n l)
                    kk += p * e(m, kk2, n, l);
                }
            }
            j.set(m, n, jj);
            k.set(m, n, k_scale * kk);
        }
    }
    (j, k)
}

/// Direct（壳层四元组即时计算 + Schwarz 筛选）构建 J、K。
///
/// 与 in-core 结果一致，但不存储完整 ERI。这里以正确性为先，遍历全部显著壳层四元组。
pub fn build_jk_direct_parallel(
    basis: &BasisSetData,
    screener: &SchwarzScreener,
    dm: &PackedSymMatrix,
    k_scale: f64,
) -> (PackedSymMatrix, PackedSymMatrix) {
    let nbf = basis.nbf;
    let nsh = basis.shells.len();
    let mut j = PackedSymMatrix::zeros(nbf);
    let mut k = PackedSymMatrix::zeros(nbf);

    // 密度矩阵的全局最大元（用于密度加权 Schwarz 判据）。
    let mut max_dm = 0.0f64;
    for i in 0..nbf {
        for jj in 0..nbf {
            max_dm = max_dm.max(dm.get(i, jj).abs());
        }
    }

    // 正确性优先：遍历全部壳层四元组（不做对称展开），按定义直接收缩。
    //   J_mn = Σ_kl P_kl (mn|kl)
    //   K_mn = Σ_kl P_kl (mk|nl)
    // Schwarz 仅用于跳过完全可忽略的四元组（这里阈值取 0，全部计算）。
    for i in 0..nsh {
        let (a0, a1) = basis.shell_ao_range(i);
        for jj in 0..nsh {
            let (b0, b1) = basis.shell_ao_range(jj);
            for kk in 0..nsh {
                let (c0, c1) = basis.shell_ao_range(kk);
                for l in 0..nsh {
                    let (d0, d1) = basis.shell_ao_range(l);
                    // Schwarz 预筛选：|(ij|kl)| ≤ Q_ij Q_kl。
                    let qij = screener.pair_bound(i, jj);
                    let qkl = screener.pair_bound(kk, l);
                    if qij * qkl * max_dm * 4.0 < 1e-14 {
                        continue;
                    }
                    let blk = eri_block(
                        &basis.shells[i],
                        &basis.shells[jj],
                        &basis.shells[kk],
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
                                    // J_pq += P_rs (pq|rs)；只累加上三角（p<=q），
                                    // 因为 (q,p) 块会映射到同一紧凑位置。
                                    if p <= qq {
                                        j.add_assign(p, qq, dm.get(rr, ss) * val);
                                    }
                                    // K_pr += P_qs (pq|rs)；只累加上三角（p<=r）。
                                    if p <= rr {
                                        k.add_assign(p, rr, k_scale * dm.get(qq, ss) * val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (j, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{load_builtin_basis, AngularType, MoleculeCore};

    /// 对 H2 / Ne / H2O，直接 JK 必须与 in-core JK 完全一致（容差 1e-10）。
    #[test]
    fn direct_matches_incore() {
        let cases: &[(&[u8], &[f64])] = &[
            (&[1, 1], &[0.0, 0.0, 0.0, 0.0, 0.0, 1.4]),
            (&[10], &[0.0, 0.0, 0.0]),
            (&[8, 1, 1], &[0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0]),
        ];
        for (z, coords) in cases {
            let mol = MoleculeCore::new(z, coords, 0, 1);
            let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
            let n = basis.nbf;
            let eri = crate::compute_eri_incore(&basis);
            // 用一个非平凡密度（这里用 SAD 风格的全 1 对角 + 小非对角）。
            let mut dm = PackedSymMatrix::zeros(n);
            for i in 0..n {
                dm.set(i, i, 1.0);
                if i + 1 < n {
                    dm.set(i, i + 1, 0.13);
                }
            }
            let (ji, ki) = build_jk_incore(n, &eri, &dm, 1.0);
            let screener = crate::build_schwarz_screener(&basis, 0.0);
            let (jd, kd) = build_jk_direct_parallel(&basis, &screener, &dm, 1.0);
            let mut maxj: f64 = 0.0;
            let mut maxk: f64 = 0.0;
            for i in 0..n {
                for jj in i..n {
                    maxj = maxj.max((ji.get(i, jj) - jd.get(i, jj)).abs());
                    maxk = maxk.max((ki.get(i, jj) - kd.get(i, jj)).abs());
                }
            }
            println!("z={:?} maxJdiff={} maxKdiff={}", z, maxj, maxk);
            assert!(maxj < 1e-10, "J mismatch z={:?} max={}", z, maxj);
            assert!(maxk < 1e-10, "K mismatch z={:?} max={}", z, maxk);
        }
    }
}
