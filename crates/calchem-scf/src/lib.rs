//! SCF 求解层。

pub struct ScfResult {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy: f64,
}

pub fn execute_scf(
    _atoms: &[u8],
    _coords_bohr: &[f64],
    _charge: i32,
    _multiplicity: usize,
    _method: &str,
    _basis: &str,
) -> Result<ScfResult, String> {
    // 占位：真实 SCF 待实现
    Err("Rust SCF 核心尚未实现，当前使用 Python 参考引擎".into())
}
