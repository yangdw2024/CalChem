//! 模型 Hessian 初始猜测（Fischer–Almlof / 简单对角模型）。
//!
//! 几何优化第一步还没有历史信息，需要一个近似 Hessian（曲率）来给出合理的
//! 步长与方向。这里采用一个稳健的"统一对角 + 平移/转动去刚性"方案：
//! 对每个笛卡尔自由度给一个正的对角曲率 h0，使 RFO 矩阵不存在零/负曲率
//! 造成的异常大步；随后由 RFO 主循环用 BFGS 用真实梯度差逐步修正。
//!
//! 参考：Fischer & Almlöf, J. Phys. Chem. 1992, 96, 9768（模型 Hessian）。

/// 构造初始对角 Hessian（行主序 m×m，m=3·natom），对角为 h0。
///
/// h0 取典型键伸缩力常数量级（Hartree/Bohr²）。
pub fn initial_model_hessian(m: usize, h0: f64) -> Vec<f64> {
    let mut h = vec![0.0; m * m];
    for i in 0..m {
        h[i * m + i] = h0;
    }
    h
}

/// 默认对角曲率（Eh / a0²）。
///
/// 0.5 对 STO-3G 小分子的键伸缩是稳健的数量级；BFGS 会很快把它修正到真值。
pub const DEFAULT_H0: f64 = 0.5;
