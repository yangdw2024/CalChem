use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};

#[pyclass]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub scf_energy_hartree: f64,
}

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
    let coords_slice = coords_bohr.as_slice()?;
    let out = py.allow_threads(move || {
        calchem_scf::execute_scf_pipeline(
            &atomic_numbers, coords_slice, charge, multiplicity, method, basis, xc, use_ri, max_threads
        )
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(PyScfWavefunction {
        converged: out.converged,
        iterations: out.iterations,
        total_energy_hartree: out.total_energy,
        scf_energy_hartree: out.total_energy,
    })
}

#[pyfunction]
pub fn eval_orbital_cube_zero_copy<'py>(
    py: Python<'py>,
    orbital_index: usize,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    dims: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let grid_data = py.allow_threads(move || {
        calchem_prop::eval_mo_on_regular_grid(orbital_index, origin_bohr, spacing_bohr, dims)
    });
    let array_1d = grid_data.into_pyarray_bound(py);
    array_1d.reshape(dims)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfWavefunction>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(eval_orbital_cube_zero_copy, m)?)?;
    Ok(())
}
