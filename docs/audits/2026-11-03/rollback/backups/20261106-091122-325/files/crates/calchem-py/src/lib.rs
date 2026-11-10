use calchem_basis::BasisSet;
use calchem_grad::{rhf_gradient, uhf_gradient};
use calchem_model::{symbol_from_atomic_number, Atom, CalcError, Molecule, ScfOptions};
use calchem_opt::{optimize_geometry_with_options, OptOptions, Reference};
use calchem_prop::{compute_dipole, compute_orbital_grid};
use calchem_scf::{rhf, uhf};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
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
    /// Spin-summed total density `P = P_alpha + P_beta` (or `2 C_occ C_occ^T` in RHF), row-major.
    #[pyo3(get)]
    pub density: Vec<f64>,
    #[pyo3(get)]
    pub energy_history: Vec<f64>,
    #[pyo3(get)]
    pub is_uhf: bool,
    #[pyo3(get)]
    pub s_squared: Option<f64>,
    #[pyo3(get)]
    pub n_alpha: usize,
    #[pyo3(get)]
    pub n_beta: usize,
    #[pyo3(get)]
    pub mo_energies_alpha: Option<Vec<f64>>,
    #[pyo3(get)]
    pub mo_energies_beta: Option<Vec<f64>>,
    pub symbols: Vec<String>,
    pub geometry_bohr: Vec<[f64; 3]>,
    pub basis_name: String,
    pub charge: i32,
    pub multiplicity: usize,
    pub dipole: Option<[f64; 3]>,
    pub dipole_mag: Option<f64>,
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

    #[getter]
    fn dipole_debye(&self) -> Option<[f64; 3]> {
        self.dipole
    }

    #[getter]
    fn dipole_magnitude_debye(&self) -> Option<f64> {
        self.dipole_mag
    }
}

#[pyclass]
pub struct PyOptResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub coords: Vec<f64>,
    #[pyo3(get)]
    pub energy: f64,
    #[pyo3(get)]
    pub steps: usize,
}

fn parse_atoms(
    atomic_numbers_or_symbols: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
) -> PyResult<Vec<Atom>> {
    let raw_coords: Vec<[f64; 3]> = if let Ok(py_arr) = coords.extract::<Vec<[f64; 3]>>() {
        py_arr
    } else if let Ok(flat) = coords.extract::<Vec<f64>>() {
        if flat.len() % 3 != 0 {
            return Err(PyValueError::new_err(
                "coords array length must be multiple of 3",
            ));
        }
        let mut v = Vec::with_capacity(flat.len() / 3);
        for chunk in flat.chunks_exact(3) {
            v.push([chunk[0], chunk[1], chunk[2]]);
        }
        v
    } else if let Ok(nested) = coords.extract::<Vec<Vec<f64>>>() {
        let mut v = Vec::with_capacity(nested.len());
        for row in nested {
            if row.len() != 3 {
                return Err(PyValueError::new_err(
                    "each coordinate must have 3 components",
                ));
            }
            v.push([row[0], row[1], row[2]]);
        }
        v
    } else {
        return Err(PyValueError::new_err(
            "failed to parse coords into 3D positions",
        ));
    };

    let n = raw_coords.len();

    let mut atoms = Vec::with_capacity(n);
    if let Ok(nums) = atomic_numbers_or_symbols.extract::<Vec<u8>>() {
        if nums.len() != n {
            return Err(PyValueError::new_err(
                "atomic_numbers and coords length mismatch",
            ));
        }
        for (z, pos) in nums.iter().zip(&raw_coords) {
            atoms.push(Atom {
                symbol: symbol_from_atomic_number(*z).map_err(to_py_err)?,
                z: *z,
                position_bohr: *pos,
            });
        }
    } else if let Ok(nums) = atomic_numbers_or_symbols.extract::<Vec<i64>>() {
        if nums.len() != n {
            return Err(PyValueError::new_err(
                "atomic_numbers and coords length mismatch",
            ));
        }
        for (z, pos) in nums.iter().zip(&raw_coords) {
            let z_u8 = *z as u8;
            atoms.push(Atom {
                symbol: symbol_from_atomic_number(z_u8).map_err(to_py_err)?,
                z: z_u8,
                position_bohr: *pos,
            });
        }
    } else if let Ok(syms) = atomic_numbers_or_symbols.extract::<Vec<String>>() {
        if syms.len() != n {
            return Err(PyValueError::new_err("symbols and coords length mismatch"));
        }
        for (s, pos) in syms.iter().zip(&raw_coords) {
            atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
        }
    } else {
        return Err(PyValueError::new_err("failed to parse atomic_numbers_or_symbols"));
    }

    Ok(atoms)
}

fn resolve_use_uhf(method: &str, multiplicity: usize) -> PyResult<bool> {
    match method.trim().to_ascii_lowercase().as_str() {
        "rhf" => Ok(false),
        "uhf" => Ok(true),
        "hf" | "auto" => Ok(multiplicity != 1),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported method in Rust core: {method}"
        ))),
    }
}

/// Run an RHF (closed-shell) or UHF (open-shell) single-point calculation.
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
    let use_uhf = resolve_use_uhf(method, multiplicity)?;

    let opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        use_diis,
        diis_max_vecs,
        damping,
        ..Default::default()
    };

    if use_uhf {
        let wf = uhf(&mol, &bs, &opts).map_err(to_py_err)?;
        let total_density = wf.total_density();
        let (dipole, dipole_mag) = compute_dipole(&mol, &bs, &total_density);
        let n = wf.n_ao;

        let mut mo_energies = Vec::with_capacity(2 * n);
        mo_energies.extend_from_slice(&wf.mo_energies_alpha);
        mo_energies.extend_from_slice(&wf.mo_energies_beta);

        let mut mo_occupations = vec![0.0; 2 * n];
        for o in mo_occupations.iter_mut().take(wf.n_alpha) {
            *o = 1.0;
        }
        for o in mo_occupations[n..].iter_mut().take(wf.n_beta) {
            *o = 1.0;
        }

        // Row-major (n_ao, 2 * n_ao): columns [0..n] are alpha MOs, [n..2n] are beta MOs.
        let mut mo_coefficients = Vec::with_capacity(n * 2 * n);
        for mu in 0..n {
            mo_coefficients.extend_from_slice(&wf.mo_coefficients_alpha[mu * n..(mu + 1) * n]);
            mo_coefficients.extend_from_slice(&wf.mo_coefficients_beta[mu * n..(mu + 1) * n]);
        }

        Ok(PyScfWavefunction {
            converged: wf.converged,
            iterations: wf.iterations,
            total_energy_hartree: wf.total_energy_hartree,
            electronic_energy_hartree: wf.electronic_energy_hartree,
            nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
            n_ao: wf.n_ao,
            mo_energies,
            mo_occupations,
            mo_coefficients,
            density: total_density,
            energy_history: wf.diagnostics.energy_history,
            is_uhf: true,
            s_squared: Some(wf.s_squared),
            n_alpha: wf.n_alpha,
            n_beta: wf.n_beta,
            mo_energies_alpha: Some(wf.mo_energies_alpha),
            mo_energies_beta: Some(wf.mo_energies_beta),
            symbols,
            geometry_bohr,
            basis_name: basis.to_string(),
            charge,
            multiplicity,
            dipole: Some(dipole),
            dipole_mag: Some(dipole_mag),
        })
    } else {
        let wf = rhf(&mol, &bs, &opts).map_err(to_py_err)?;
        let (dipole, dipole_mag) = compute_dipole(&mol, &bs, &wf.density_alpha);
        let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 0.5).count();

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
            is_uhf: false,
            s_squared: Some(0.0),
            n_alpha: n_occ,
            n_beta: n_occ,
            mo_energies_alpha: None,
            mo_energies_beta: None,
            symbols,
            geometry_bohr,
            basis_name: basis.to_string(),
            charge,
            multiplicity,
            dipole: Some(dipole),
            dipole_mag: Some(dipole_mag),
        })
    }
}

/// Computes the analytical RHF or UHF energy gradient with respect to Cartesian coordinates.
///
/// Returns a 2D numpy array `(N, 3)` in units of Eh/Bohr.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords, charge=0, multiplicity=1, basis="sto-3g", method="auto"))]
pub fn run_gradient<'py>(
    py: Python<'py>,
    atomic_numbers: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    charge: i32,
    multiplicity: usize,
    basis: &str,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let n = atoms.len();
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let use_uhf = resolve_use_uhf(method, multiplicity)?;

    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-11;
    opts.conv_density = 1e-10;

    let grad = if use_uhf {
        let wf = uhf(&mol, &bs, &opts).map_err(to_py_err)?;
        uhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
    } else {
        let wf = rhf(&mol, &bs, &opts).map_err(to_py_err)?;
        rhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
    };

    let mut flat = Vec::with_capacity(n * 3);
    for atom_g in grad {
        flat.extend_from_slice(&atom_g);
    }

    let arr1 = PyArray1::from_vec_bound(py, flat);
    let arr2 = arr1.reshape([n, 3])?;
    Ok(arr2)
}

/// Optimizes molecular geometry using Rust BFGS and analytical RHF/UHF gradients.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords, basis="sto-3g", max_steps=100, charge=0, multiplicity=1, method="auto"))]
pub fn run_optimize(
    atomic_numbers: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
    basis: &str,
    max_steps: usize,
    charge: i32,
    multiplicity: usize,
    method: &str,
) -> PyResult<PyOptResult> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = match method.trim().to_ascii_lowercase().as_str() {
        "rhf" => Reference::Rhf,
        "uhf" => Reference::Uhf,
        "hf" | "auto" => Reference::Auto,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported method in Rust optimizer: {method}"
            )))
        }
    };

    let mut opts = OptOptions::default();
    opts.max_steps = max_steps;
    opts.max_force = 3e-5;
    opts.rms_force = 2e-5;
    opts.max_displacement = 1e-3;
    opts.reference = reference;

    let res = optimize_geometry_with_options(&mol, basis, &opts).map_err(to_py_err)?;

    let mut flat_coords = Vec::with_capacity(res.molecule.atoms.len() * 3);
    for atom in &res.molecule.atoms {
        flat_coords.extend_from_slice(&atom.position_bohr);
    }

    Ok(PyOptResult {
        converged: res.converged,
        coords: flat_coords,
        energy: res.energy_hartree,
        steps: res.steps,
    })
}

/// Evaluates molecular orbital on a 3D grid and returns a numpy 3D array of shape `(nz, ny, nx)`.
#[pyfunction]
pub fn eval_orbital_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    mo_idx: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut atoms = Vec::with_capacity(wf.symbols.len());
    for (s, pos) in wf.symbols.iter().zip(&wf.geometry_bohr) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, wf.charge, wf.multiplicity);
    let bs = BasisSet::from_name(&wf.basis_name, &mol).map_err(to_py_err)?;

    let n_mo = wf.mo_occupations.len();
    let grid_vec = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        origin,
        spacing,
        counts,
    )
    .map_err(to_py_err)?;

    let arr1 = PyArray1::from_vec_bound(py, grid_vec);
    let arr3 = arr1.reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// Basis set names embedded in the Rust core.
#[pyfunction]
pub fn supported_basis_sets() -> Vec<&'static str> {
    BasisSet::SUPPORTED_NAMES.to_vec()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add("ScfNotConvergedError", m.py().get_type_bound::<ScfNotConvergedError>())?;
    m.add("RustCoreError", m.py().get_type_bound::<RustCoreError>())?;
    m.add_class::<PyScfWavefunction>()?;
    m.add_class::<PyOptResult>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(run_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(eval_orbital_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(supported_basis_sets, m)?)?;
    Ok(())
}
