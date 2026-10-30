//! PyO3 零拷贝绑定：单点 / 梯度 / 优化 / 三维网格。
//!
//! 所有重计算均通过 `py.allow_threads` 释放 GIL；网格结果以 Rust `Vec<f32>`
//! 所有权直接移交给 NumPy（零拷贝）。

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

use calchem_grad::compute_analytical_gradient;
use calchem_opt::{run_optimization, OptimizationConfig};
use calchem_prop::{
    population_analysis, eval_density_grid, eval_esp_grid, eval_orbital_grid, Grid3DSpec,
};
use calchem_scf::WavefunctionHandle;

/// 暴露给 Python 的波函数对象（内部持有 Rust 句柄）。
#[pyclass]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub scf_energy_hartree: f64,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub homo_energy_hartree: Option<f64>,
    #[pyo3(get)]
    pub lumo_energy_hartree: Option<f64>,
    #[pyo3(get)]
    pub dipole_debye: [f64; 3],
    /// 内部 Rust 句柄（供后续零拷贝 Cube / 梯度 / 性质）。
    pub(crate) inner: Arc<WavefunctionHandle>,
}

#[pymethods]
impl PyScfWavefunction {
    /// 偶极矩大小（Debye）。
    #[getter]
    fn dipole_magnitude_debye(&self) -> f64 {
        let d = self.dipole_debye;
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Mulliken 电荷（按原子顺序）。
    #[getter]
    fn mulliken_charges(&self) -> Vec<f64> {
        let rep = population_analysis(
            &self.inner.mol,
            &self.inner.basis,
            &self.inner.scf.overlap,
            &self.inner.scf.density,
        );
        rep.mulliken_charges
    }
}

impl PyScfWavefunction {
    fn from_handle(h: WavefunctionHandle) -> Self {
        let rep = population_analysis(&h.mol, &h.basis, &h.scf.overlap, &h.scf.density);
        let nocc = h.scf.n_occ;
        let homo = h.scf.mo_energies.get(nocc.saturating_sub(1)).copied();
        let lumo = h.scf.mo_energies.get(nocc).copied();
        let dipole_debye = rep.dipole_debye;
        PyScfWavefunction {
            converged: h.scf.converged,
            iterations: h.scf.iterations,
            scf_energy_hartree: h.scf.electronic_energy,
            total_energy_hartree: h.scf.total_energy,
            homo_energy_hartree: homo,
            lumo_energy_hartree: lumo,
            dipole_debye,
            inner: Arc::new(h),
        }
    }
}

/// 单点能（RHF）。
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, charge, multiplicity, method, basis, xc=None, use_ri=true, max_threads=8))]
pub fn run_single_point(
    py: Python<'_>,
    atomic_numbers: Vec<u8>,
    coords_bohr: PyReadonlyArray2<'_, f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    use_ri: bool,
    max_threads: usize,
) -> PyResult<PyScfWavefunction> {
    let coords_slice = coords_bohr.as_slice()?.to_vec();
    let handle = py
        .allow_threads(move || {
            calchem_scf::execute_scf_to_handle(
                &atomic_numbers,
                &coords_slice,
                charge,
                multiplicity,
                method,
                basis,
                xc,
                use_ri,
                max_threads,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyScfWavefunction::from_handle(handle))
}

/// 解析梯度：返回 (n_atoms, 3) 数组（Eh/Bohr）。
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, charge, multiplicity, basis))]
pub fn run_gradient<'py>(
    py: Python<'py>,
    atomic_numbers: Vec<u8>,
    coords_bohr: PyReadonlyArray2<'_, f64>,
    charge: i32,
    multiplicity: usize,
    basis: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coords_slice = coords_bohr.as_slice()?.to_vec();
    let grad = py
        .allow_threads(move || {
            let handle = calchem_scf::execute_scf_to_handle(
                &atomic_numbers,
                &coords_slice,
                charge,
                multiplicity,
                "rhf",
                basis,
                None,
                false,
                0,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let g = compute_analytical_gradient(&handle.mol, &handle.basis, &handle.scf);
            let natom = g.len();
            let flat: Vec<f64> = g.iter().flat_map(|v| *v).collect();
            Ok((flat, natom))
        })
        .map_err(|e: PyErr| e)?;
    let (flat, natom) = grad;
    let arr = flat.into_pyarray_bound(py);
    arr.reshape((natom, 3))
}

/// 几何优化：返回优化结果对象。
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, basis="sto-3g", max_steps=100))]
pub fn run_optimize(
    py: Python<'_>,
    atomic_numbers: Vec<u8>,
    coords_bohr: PyReadonlyArray2<'_, f64>,
    basis: &str,
    max_steps: usize,
) -> PyResult<PyOptimizationResult> {
    let coords_slice = coords_bohr.as_slice()?.to_vec();
    let result = py.allow_threads(move || {
        let mut cfg = OptimizationConfig::default();
        cfg.basis = basis.to_string();
        cfg.max_steps = max_steps;
        run_optimization(&atomic_numbers, &coords_slice, &cfg)
    });
    Ok(PyOptimizationResult {
        converged: result.converged,
        steps: result.steps,
        energy: result.energy,
        coords: result.coords,
        message: result.message,
    })
}

/// 暴露给 Python 的优化结果。
#[pyclass]
pub struct PyOptimizationResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub energy: f64,
    #[pyo3(get)]
    pub coords: Vec<f64>,
    #[pyo3(get)]
    pub message: String,
}

/// 零拷贝：指定轨道在规则网格上的值，返回 (nz, ny, nx) 的 f32 数组。
#[pyfunction]
#[pyo3(signature = (wfn, orbital_index, origin_bohr, spacing_bohr, dims))]
pub fn eval_orbital_cube_zero_copy<'py>(
    py: Python<'py>,
    wfn: &PyScfWavefunction,
    orbital_index: usize,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    dims: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let inner = wfn.inner.clone();
    let values = py.allow_threads(move || {
        let mo_vec = inner.scf.mo_column(orbital_index);
        let grid = Grid3DSpec::axis_aligned(origin_bohr, spacing_bohr, dims);
        eval_orbital_grid(&inner.basis, &mo_vec, &grid)
    });
    let arr = values.into_pyarray_bound(py);
    // NumPy 数组轴顺序与网格 (nx,ny,nz) 相反，故 reshape 成 (nz,ny,nx)。
    arr.reshape((dims[2], dims[1], dims[0]))
}

/// 零拷贝：电子密度网格，返回 (nz, ny, nx) 的 f32 数组。
#[pyfunction]
#[pyo3(signature = (wfn, origin_bohr, spacing_bohr, dims))]
pub fn eval_density_cube_zero_copy<'py>(
    py: Python<'py>,
    wfn: &PyScfWavefunction,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    dims: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let inner = wfn.inner.clone();
    let values = py.allow_threads(move || {
        let grid = Grid3DSpec::axis_aligned(origin_bohr, spacing_bohr, dims);
        eval_density_grid(&inner.basis, &inner.scf.density, &grid)
    });
    let arr = values.into_pyarray_bound(py);
    arr.reshape((dims[2], dims[1], dims[0]))
}

/// 零拷贝：静电势网格，返回 (nz, ny, nx) 的 f32 数组。
#[pyfunction]
#[pyo3(signature = (wfn, origin_bohr, spacing_bohr, dims))]
pub fn eval_esp_cube_zero_copy<'py>(
    py: Python<'py>,
    wfn: &PyScfWavefunction,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    dims: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let inner = wfn.inner.clone();
    let values = py.allow_threads(move || {
        let grid = Grid3DSpec::axis_aligned(origin_bohr, spacing_bohr, dims);
        eval_esp_grid(&inner.mol, &inner.basis, &inner.scf.density, &grid)
    });
    let arr = values.into_pyarray_bound(py);
    arr.reshape((dims[2], dims[1], dims[0]))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfWavefunction>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(run_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(eval_orbital_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(eval_density_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(eval_esp_cube_zero_copy, m)?)?;
    Ok(())
}
