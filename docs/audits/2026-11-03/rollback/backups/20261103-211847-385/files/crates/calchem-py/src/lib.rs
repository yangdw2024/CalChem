use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rhf;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub electronic_energy_hartree: f64,
    #[pyo3(get)]
    pub nuclear_repulsion_hartree: f64,
    #[pyo3(get)]
    pub mo_energies: Vec<f64>,
    #[pyo3(get)]
    pub mo_occupations: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g", max_iter=100, conv_energy=1e-10, conv_density=1e-8))]
#[allow(clippy::too_many_arguments)]
pub fn run_single_point(
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
) -> PyResult<PyScfWavefunction> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        let atom = Atom::from_bohr(s, *pos).map_err(|e| PyValueError::new_err(e.to_string()))?;
        atoms.push(atom);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let bs =
        BasisSet::from_name(basis, &mol).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let m = method.trim().to_ascii_lowercase();
    if m != "rhf" && m != "hf" {
        return Err(PyValueError::new_err(format!(
            "Unsupported method in Rust Stage 2 core: {method}"
        )));
    }

    let opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        ..Default::default()
    };

    let wf = rhf(&mol, &bs, &opts).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        mo_energies: wf.mo_energies,
        mo_occupations: wf.mo_occupations,
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfWavefunction>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    Ok(())
}
