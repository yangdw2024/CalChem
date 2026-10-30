//! Pulay 力（能量加权密度）项。
//!
//! 当基函数随原子核移动时，总能量全微分里的"密度响应" tr[F dP] 无法直接用
//! AO 矩阵的导数表达。利用 MO 正交归一条件 CᵀSC=I 微分，可严格改写为：
//!
//!   tr[F dP] = -tr[W dS]，  W_μν = Σ_i n_i ε_i C_μi C_νi
//!
//! 其中 n_i 为轨道占据数（闭壳层占据轨道 n_i=2），ε_i 为 MO 能量，dS 为
//! 重叠矩阵对核坐标的导数。这一项习惯称为 Pulay 力（Pulay force）。
//!
//! 没有它，解析梯度只会给出"冻结密度偏导"（把密度当作不随几何变化），
//! 与松弛（relaxed）的有限差分能量梯度不一致。

use calchem_scf::ScfOutput;
use calchem_tensor::PackedSymMatrix;

/// 构造能量加权密度矩阵 W：
///
///   W_μν = Σ_i n_i ε_i C_μi C_νi
///
/// 闭壳层中占据轨道 n_i=2，空轨道 n_i=0。
pub fn build_energy_weighted_density(scf: &ScfOutput) -> PackedSymMatrix {
    let n = scf.density.dim;
    let n_occ = scf.n_occ;
    let mut w = PackedSymMatrix::zeros(n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for i in 0..n_occ {
                // 闭壳层占据数为 2。
                s += 2.0
                    * scf.mo_energies[i]
                    * scf.mo_coeff[mu * n + i]
                    * scf.mo_coeff[nu * n + i];
            }
            w.set(mu, nu, s);
        }
    }
    w
}

/// Pulay 重叠项：-tr[W dS] = -Σ_μν W_μν dS_μν。
///
/// 它等于总能量全微分中的密度响应 tr[F dP]。
/// `ds` 为行主序 n×n 的重叠导数矩阵。
pub fn pulay_overlap_term(w: &PackedSymMatrix, ds: &[f64], n: usize) -> f64 {
    let mut t = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            t += w.get(mu, nu) * ds[mu * n + nu];
        }
    }
    -t
}
