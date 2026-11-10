use calchem_basis::BasisSet;
use calchem_model::{Atom, CalcError, Molecule, ScfOptions};
use calchem_scf::rhf;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// Distinct Python exception types so callers can tell "the SCF did not converge"
// (a legitimate, reportable outcome) apart from bad input or an internal failure.
create_exception!(_core, ScfNotConvergedError, PyRuntimeError);
create_exception!(_core, RustCoreError, PyRuntimeError);

/// Version of the Rust core reported to Python (keep in sync with crates/*/Cargo.toml).
const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

fn to_py_err(e: CalcError) -> PyErr {
    match e {
        CalcError::NotConverged { .. } => ScfNotConvergedError::new_err(e.to_string()),
        CalcError::InvalidMolecule(_)
        | CalcError::UnsupportedBasis(_)
        | CalcError::UnsupportedMethod(_) => PyValueError::new_err(e.to_string()),
        _ => RustCoreError::new_err(e.to_string()),
    }
}

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
    pub n_ao: usize,
    #[pyo3(get)]
    pub mo_energies: Vec<f64>,
    #[pyo3(get)]
    pub mo_occupations: Vec<f64>,
    /// Row-major AO x MO coefficients, index `mu * n_mo + i`.
    #[pyo3(get)]
    pub mo_coefficients: Vec<f64>,
    /// Spin-summed density P = 2 C_occ C_occ^T, row-major.
    #[pyo3(get)]
    pub density: Vec<f64>,
    #[pyo3(get)]
    pub energy_history: Vec<f64>,
}

#[pymethods]
impl PyScfWavefunction {
    /// Highest occupied MO energy (Eh), or None if there are no occupied orbitals.
    #[getter]
    fn homo_energy_hartree(&self) -> Option<f64> {
        self.mo_energies
            .iter()
            .zip(&self.mo_occupations)
            .filter(|(_, &o)| o > 0.5)
            .map(|(&e, _)| e)
            .fold(None, |acc: Option<f64>, e| Some(acc.map_or(e, |a| a.max(e))))
    }

    /// Lowest unoccupied MO energy (Eh), or None if there are no virtual orbitals.
    #[getter]
    fn lumo_energy_hartree(&self) -> Option<f64> {
        self.mo_energies
            .iter()
            .zip(&self.mo_occupations)
            .filter(|(_, &o)| o <= 0.5)
            .map(|(&e, _)| e)
            .fold(None, |acc: Option<f64>, e| Some(acc.map_or(e, |a| a.min(e))))
    }
}

/// Run a closed-shell RHF single point.
///
/// `symbols`: element symbols; `geometry_bohr`: N x 3 coordinates in Bohr.
/// Raises `ScfNotConvergedError` when the SCF does not converge within `max_iter`,
/// `ValueError` for invalid input / unsupported basis or method.
#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g", max_iter=100, conv_energy=1e-10, conv_density=1e-8, use_diis=true, diis_max_vecs=8, damping=0.0))]
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
    use_diis: bool,
    diis_max_vecs: usize,
    damping: f64,
) -> PyResult<PyScfWavefunction> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be >= 1"));
    }
    if !(conv_energy > 0.0 && conv_density > 0.0) {
        return Err(PyValueError::new_err("convergence thresholds must be positive"));
    }
    if !(0.0..1.0).contains(&damping) {
        return Err(PyValueError::new_err("damping must be in [0, 1)"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        if !pos.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!("non-finite coordinate for atom {s}")));
        }
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;

    let m = method.trim().to_ascii_lowercase();
    if m != "rhf" && m != "hf" {
        return Err(PyValueError::new_err(format!(
            "Unsupported method in Rust core: {method}"
        )));
    }

    let opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        use_diis,
        diis_max_vecs,
        damping,
        ..Default::default()
    };

    let wf = rhf(&mol, &bs, &opts).map_err(to_py_err)?;
    Ok(PyScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        n_ao: wf.n_ao,
        mo_energies: wf.mo_energies,
        mo_occupations: wf.mo_occupations,
        mo_coefficients: wf.mo_coefficients,
        density: wf.density_alpha,
        energy_history: wf.diagnostics.energy_history,
    })
}

/// Basis set names embedded in the Rust core.
#[pyfunction]
pub fn supported_basis_sets() -> Vec<&'static str> {
    vec!["sto-3g", "6-31g"]
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add("ScfNotConvergedError", m.py().get_type_bound::<ScfNotConvergedError>())?;
    m.add("RustCoreError", m.py().get_type_bound::<RustCoreError>())?;
    m.add_class::<PyScfWavefunction>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(supported_basis_sets, m)?)?;
    Ok(())
}
