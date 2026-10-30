//! Schwarz（柯西-施瓦茨）积分预筛选器。
//!
//! 对壳层对 (i,j)，定义上界
//!   Q_ij = √[ max_{μ∈i, ν∈j} (μν|μν) ]
//! 则任意四元组满足 |(μν|λσ)| ≤ Q_μν Q_λσ。

use calchem_basis::BasisSetData;

use crate::eri::eri_block;

/// Schwarz 筛选器。
#[derive(Debug, Clone)]
pub struct SchwarzScreener {
    /// 筛选阈值。
    pub threshold: f64,
    /// 当前密度矩阵的全局最大元（用于密度加权筛选）。
    pub max_dm_global: f64,
    /// 有效壳层对 (壳I, 壳J, Q_IJ)，按 Q_IJ 降序排列。
    pub significant_pairs: Vec<(u32, u32, f64)>,
    /// 壳层对的 Schwarz 上界矩阵（紧凑列主序上三角 n_shell*(n_shell+1)/2）。
    pub shell_bounds: Vec<f64>,
    /// 壳层数量。
    pub n_shell: usize,
}

impl SchwarzScreener {
    /// 空筛选器（仅设定阈值）。
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            max_dm_global: 1.0,
            significant_pairs: Vec::new(),
            shell_bounds: Vec::new(),
            n_shell: 0,
        }
    }

    /// 壳层对 (i,j) 的 Schwarz 上界 Q_ij。
    #[inline]
    pub fn pair_bound(&self, i: usize, j: usize) -> f64 {
        let (r, c) = if i <= j { (i, j) } else { (j, i) };
        self.shell_bounds[c * (c + 1) / 2 + r]
    }
}

/// 构建真实的 Schwarz 筛选器：计算每个壳对的 √(μν|μν) 上界。
pub fn build_schwarz_screener(basis: &BasisSetData, threshold: f64) -> SchwarzScreener {
    let nsh = basis.shells.len();
    let mut bounds = vec![0.0; nsh * (nsh + 1) / 2];
    let mut significant = Vec::new();

    for i in 0..nsh {
        let (a0, a1) = basis.shell_ao_range(i);
        for j in i..nsh {
            let (b0, b1) = basis.shell_ao_range(j);
            // 计算 ERI 块 (i j | i j)。
            let blk = eri_block(&basis.shells[i], &basis.shells[j], &basis.shells[i], &basis.shells[j]);
            let dims = (a1 - a0, b1 - b0);
            // 取 max_{μ∈i,ν∈j} (μν|μν)。
            let mut maxv: f64 = 0.0;
            for ia in 0..dims.0 {
                let mu = a0 + ia;
                for ib in 0..dims.1 {
                    let nu = b0 + ib;
                    // 块内 (μν|μν)：索引 (ia,ib,ia,ib)。
                    let v = blk[((ia * dims.1 + ib) * dims.0 + ia) * dims.1 + ib];
                    // 注意：ket 排布与 bra 相同（i,j,i,j），故 ket 分量索引也是 (ia,ib)。
                    let _ = (mu, nu);
                    maxv = maxv.max(v);
                }
            }
            let q = maxv.max(0.0).sqrt();
            let packed = j * (j + 1) / 2 + i;
            bounds[packed] = q;
            if q >= threshold {
                significant.push((i as u32, j as u32, q));
            }
        }
    }
    significant.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    SchwarzScreener {
        threshold,
        max_dm_global: 1.0,
        significant_pairs: significant,
        shell_bounds: bounds,
        n_shell: nsh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{load_builtin_basis, AngularType, MoleculeCore};

    #[test]
    fn schwarz_h2() {
        let z = [1u8, 1];
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let s = build_schwarz_screener(&basis, 0.0);
        // Q_11 = √(11|11) = √0.7746 = 0.8801。
        let q11 = s.pair_bound(0, 0);
        let q12 = s.pair_bound(0, 1);
        println!("q11={} q12={}", q11, q12);
        assert!((q11 - 0.7746059439198977_f64.sqrt()).abs() < 1e-8);
        // Q_12 = √(12|12) = √0.2970 = 0.5450。
        assert!((q12 - 0.2970285411810491_f64.sqrt()).abs() < 1e-8);
    }
}
