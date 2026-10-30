use pyo3::prelude::*;

#[pyclass]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
}

#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g"))]
pub fn run_single_point(
    _py: Python<'_>,
    atomic_numbers: Vec<u8>,
    coords_bohr: Vec<f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
) -> PyResult<PyScfWavefunction> {
    // Rust SCF 尚未实现，返回明确错误
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        format!("Rust SCF 核心尚未实现（method={method}, basis={basis}）。当前请使用 Python 参考引擎 backend='python_ref'。")
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfWavefunction>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    Ok(())
}
