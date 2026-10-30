//! 分辨率恒等式 / 密度拟合（RI-J / RI-K / RI-MP2）——后续阶段填充。

use calchem_tensor::PackedSymMatrix;

/// RI 引擎（占位）。
pub struct RiEngine {
    /// 辅助基函数数。
    pub n_aux: usize,
}

impl RiEngine {
    /// 占位：三中心张量收缩 J（真实实现需标准辅助基组）。
    pub fn contract_j(&self, _b_tensor: &[f64], dm: &PackedSymMatrix) -> PackedSymMatrix {
        PackedSymMatrix::zeros(dm.dim)
    }
}
