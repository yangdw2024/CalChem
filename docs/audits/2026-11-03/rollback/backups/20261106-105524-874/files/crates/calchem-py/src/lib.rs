use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient, uhf_gradient, uks_gradient};
use calchem_model::{
    symbol_from_atomic_number, Atom, CalcError, Molecule, ScfOptions, BOHR_TO_ANGSTROM,
};
use calchem_opt::{optimize_geometry_with_options, OptOptions, Reference};
use calchem_prop::{
    auto_grid_for_molecule, compute_density_grid, compute_dipole, compute_orbital_grid,
    encode_dual_mesh_binary, extract_dual_isosurface, extract_isosurface, format_gaussian_cube,
    parse_gaussian_cube, DualOrbitalMesh, IsosurfaceMesh,
};
use calchem_scf::{rhf, rks, uhf, uks};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
    /// Spin density `P_alpha - P_beta` for UHF wavefunctions (`None` for closed-shell RHF).
    #[pyo3(get)]
    pub spin_density: Option<Vec<f64>>,
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
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub basis_name: String,
    #[pyo3(get)]
    pub charge: i32,
    #[pyo3(get)]
    pub multiplicity: usize,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    pub dipole: Option<[f64; 3]>,
    pub dipole_mag: Option<f64>,
}

impl PyScfWavefunction {
    fn build_mol_and_basis(&self) -> PyResult<(Molecule, BasisSet)> {
        let mut atoms = Vec::with_capacity(self.symbols.len());
        for (s, pos) in self.symbols.iter().zip(&self.geometry_bohr) {
            atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
        }
        let mol = Molecule::new(atoms, self.charge, self.multiplicity);
        let bs = BasisSet::from_name(&self.basis_name, &mol).map_err(to_py_err)?;
        Ok((mol, bs))
    }

    fn spin_channel_for_mo(&self, mo_idx: usize) -> u32 {
        if self.is_uhf && mo_idx >= self.n_ao {
            1
        } else {
            0
        }
    }
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

/// Zero-copy single-lobe triangle mesh backed by Rust-allocated `float32` / `uint32` NumPy buffers.
#[pyclass]
pub struct PyIsosurfaceMesh {
    #[pyo3(get)]
    pub isovalue: f32,
    #[pyo3(get)]
    pub n_vertices: usize,
    #[pyo3(get)]
    pub n_triangles: usize,
    vertices_arr: Py<PyArray2<f32>>,
    normals_arr: Py<PyArray2<f32>>,
    faces_arr: Py<PyArray2<u32>>,
}

impl PyIsosurfaceMesh {
    fn from_rust(py: Python<'_>, mesh: IsosurfaceMesh) -> PyResult<Self> {
        let n_v = mesh.n_vertices();
        let n_t = mesh.n_triangles();
        let v_arr = PyArray1::from_vec_bound(py, mesh.vertices)
            .reshape([n_v, 3])?
            .unbind();
        let n_arr = PyArray1::from_vec_bound(py, mesh.normals)
            .reshape([n_v, 3])?
            .unbind();
        let f_arr = PyArray1::from_vec_bound(py, mesh.indices)
            .reshape([n_t, 3])?
            .unbind();
        Ok(Self {
            isovalue: mesh.isovalue,
            n_vertices: n_v,
            n_triangles: n_t,
            vertices_arr: v_arr,
            normals_arr: n_arr,
            faces_arr: f_arr,
        })
    }

    fn to_rust(&self, py: Python<'_>) -> PyResult<IsosurfaceMesh> {
        let v = self.vertices_arr.bind(py).readonly();
        let n = self.normals_arr.bind(py).readonly();
        let f = self.faces_arr.bind(py).readonly();
        Ok(IsosurfaceMesh {
            isovalue: self.isovalue,
            vertices: v.as_slice()?.to_vec(),
            normals: n.as_slice()?.to_vec(),
            indices: f.as_slice()?.to_vec(),
        })
    }
}

#[pymethods]
impl PyIsosurfaceMesh {
    /// `(n_vertices, 3)` `float32` vertex position buffer (zero-copy handle).
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.vertices_arr.bind(py).clone()
    }

    /// `(n_vertices, 3)` `float32` unit outward normal buffer (zero-copy handle).
    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.normals_arr.bind(py).clone()
    }

    /// `(n_triangles, 3)` `uint32` CCW triangle index buffer (zero-copy handle).
    #[getter]
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        self.faces_arr.bind(py).clone()
    }

    /// Alias of `faces` (`(n_triangles, 3)` `uint32`).
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        self.faces_arr.bind(py).clone()
    }
}

/// Dual-phase (`+isovalue` and `-isovalue`) molecular orbital / spin-density mesh with binary wire serialization.
#[pyclass]
pub struct PyDualOrbitalMesh {
    #[pyo3(get)]
    pub isovalue: f32,
    #[pyo3(get)]
    pub units_angstrom: bool,
    #[pyo3(get)]
    pub mo_index: i32,
    #[pyo3(get)]
    pub spin_channel: u32,
    #[pyo3(get)]
    pub positive: Py<PyIsosurfaceMesh>,
    #[pyo3(get)]
    pub negative: Py<PyIsosurfaceMesh>,
}

impl PyDualOrbitalMesh {
    fn from_rust(
        py: Python<'_>,
        dual: DualOrbitalMesh,
        mo_index: i32,
        spin_channel: u32,
    ) -> PyResult<Self> {
        let pos = Py::new(py, PyIsosurfaceMesh::from_rust(py, dual.positive)?)?;
        let neg = Py::new(py, PyIsosurfaceMesh::from_rust(py, dual.negative)?)?;
        Ok(Self {
            isovalue: dual.isovalue,
            units_angstrom: dual.units_angstrom,
            mo_index,
            spin_channel,
            positive: pos,
            negative: neg,
        })
    }
}

#[pymethods]
impl PyDualOrbitalMesh {
    /// Serializes both positive and negative lobes into a 64-byte-aligned `CCMSH\x01` binary packet.
    fn to_binary_packet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let pos_ref = self.positive.borrow(py);
        let neg_ref = self.negative.borrow(py);
        let dual = DualOrbitalMesh {
            isovalue: self.isovalue,
            positive: pos_ref.to_rust(py)?,
            negative: neg_ref.to_rust(py)?,
            units_angstrom: self.units_angstrom,
        };
        let encoded = encode_dual_mesh_binary(&dual, self.mo_index, self.spin_channel);
        Ok(PyBytes::new_bound(py, &encoded))
    }
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
        return Err(PyValueError::new_err(
            "failed to parse atomic_numbers_or_symbols",
        ));
    }

    Ok(atoms)
}

#[derive(Clone, Copy)]
enum ResolvedScfKind {
    Rhf,
    Uhf,
    Rks(XcFunctional),
    Uks(XcFunctional),
}

fn resolve_scf_kind(
    method: &str,
    xc: Option<&str>,
    multiplicity: usize,
) -> PyResult<ResolvedScfKind> {
    let m = method.trim().to_ascii_lowercase();
    let parse_xc = |default_xc: &str| -> PyResult<XcFunctional> {
        XcFunctional::from_name(xc.unwrap_or(default_xc)).map_err(to_py_err)
    };
    match m.as_str() {
        "rhf" => Ok(ResolvedScfKind::Rhf),
        "uhf" => Ok(ResolvedScfKind::Uhf),
        "hf" | "auto" => Ok(if multiplicity != 1 {
            ResolvedScfKind::Uhf
        } else {
            ResolvedScfKind::Rhf
        }),
        "rks" => Ok(ResolvedScfKind::Rks(parse_xc("b3lyp")?)),
        "uks" => Ok(ResolvedScfKind::Uks(parse_xc("b3lyp")?)),
        "dft" => {
            let func = parse_xc("b3lyp")?;
            Ok(if multiplicity != 1 {
                ResolvedScfKind::Uks(func)
            } else {
                ResolvedScfKind::Rks(func)
            })
        }
        other => {
            let func = XcFunctional::from_name(other).map_err(|_| {
                PyValueError::new_err(format!("Unsupported method in Rust core: {method}"))
            })?;
            Ok(if multiplicity != 1 {
                ResolvedScfKind::Uks(func)
            } else {
                ResolvedScfKind::Rks(func)
            })
        }
    }
}

/// Run an RHF, UHF, RKS, or UKS single-point calculation in pure Rust.
#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g", max_iter=100, conv_energy=1e-10, conv_density=1e-8, use_diis=true, diis_max_vecs=8, damping=0.0, xc=None, grid_level="medium"))]
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
    xc: Option<&str>,
    grid_level: &str,
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
        return Err(PyValueError::new_err(
            "convergence thresholds must be positive",
        ));
    }
    if !(0.0..1.0).contains(&damping) {
        return Err(PyValueError::new_err("damping must be in [0, 1)"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        if !pos.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "non-finite coordinate for atom {s}"
            )));
        }
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let gl = GridLevel::from_name(grid_level);

    let opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        use_diis,
        diis_max_vecs,
        damping,
        ..Default::default()
    };

    match scf_kind {
        ResolvedScfKind::Uhf | ResolvedScfKind::Uks(_) => {
            let (wf, xc_name) = match scf_kind {
                ResolvedScfKind::Uhf => (uhf(&mol, &bs, &opts).map_err(to_py_err)?, None),
                ResolvedScfKind::Uks(func) => (
                    uks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?,
                    Some(func.canonical_name().to_string()),
                ),
                _ => unreachable!(),
            };
            let total_density = wf.total_density();
            let spin_density: Vec<f64> = wf
                .density_alpha
                .iter()
                .zip(&wf.density_beta)
                .map(|(a, b)| a - b)
                .collect();
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

            let mut mo_coefficients = Vec::with_capacity(n * 2 * n);
            for mu in 0..n {
                mo_coefficients
                    .extend_from_slice(&wf.mo_coefficients_alpha[mu * n..(mu + 1) * n]);
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
                spin_density: Some(spin_density),
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
                xc_functional: xc_name,
                dipole: Some(dipole),
                dipole_mag: Some(dipole_mag),
            })
        }
        ResolvedScfKind::Rhf | ResolvedScfKind::Rks(_) => {
            let (wf, xc_name) = match scf_kind {
                ResolvedScfKind::Rhf => (rhf(&mol, &bs, &opts).map_err(to_py_err)?, None),
                ResolvedScfKind::Rks(func) => (
                    rks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?,
                    Some(func.canonical_name().to_string()),
                ),
                _ => unreachable!(),
            };
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
                spin_density: None,
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
                xc_functional: xc_name,
                dipole: Some(dipole),
                dipole_mag: Some(dipole_mag),
            })
        }
    }
}

/// Computes the analytical RHF, UHF, RKS, or UKS energy gradient with respect to Cartesian coordinates.
///
/// Returns a 2D numpy array `(N, 3)` in units of Eh/Bohr.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords, charge=0, multiplicity=1, basis="sto-3g", method="auto", xc=None, grid_level="medium"))]
#[allow(clippy::too_many_arguments)]
pub fn run_gradient<'py>(
    py: Python<'py>,
    atomic_numbers: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    charge: i32,
    multiplicity: usize,
    basis: &str,
    method: &str,
    xc: Option<&str>,
    grid_level: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let n = atoms.len();
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let gl = GridLevel::from_name(grid_level);

    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-11;
    opts.conv_density = 1e-10;

    let grad = match scf_kind {
        ResolvedScfKind::Uhf => {
            let wf = uhf(&mol, &bs, &opts).map_err(to_py_err)?;
            uhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
        }
        ResolvedScfKind::Rhf => {
            let wf = rhf(&mol, &bs, &opts).map_err(to_py_err)?;
            rhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
        }
        ResolvedScfKind::Rks(func) => {
            let wf = rks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?;
            rks_gradient(&mol, &bs, func, gl, &wf).map_err(to_py_err)?
        }
        ResolvedScfKind::Uks(func) => {
            let wf = uks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?;
            uks_gradient(&mol, &bs, func, gl, &wf).map_err(to_py_err)?
        }
    };

    let mut flat = Vec::with_capacity(n * 3);
    for atom_g in grad {
        flat.extend_from_slice(&atom_g);
    }

    let arr1 = PyArray1::from_vec_bound(py, flat);
    let arr2 = arr1.reshape([n, 3])?;
    Ok(arr2)
}

/// Optimizes molecular geometry using Rust BFGS and analytical RHF/UHF/RKS/UKS gradients.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords, basis="sto-3g", max_steps=100, charge=0, multiplicity=1, method="auto", xc=None, grid_level="medium"))]
#[allow(clippy::too_many_arguments)]
pub fn run_optimize(
    atomic_numbers: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
    basis: &str,
    max_steps: usize,
    charge: i32,
    multiplicity: usize,
    method: &str,
    xc: Option<&str>,
    grid_level: &str,
) -> PyResult<PyOptResult> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc).map_err(|_| {
        PyValueError::new_err(format!("Unsupported method in Rust optimizer: {method}"))
    })?;

    let mut opts = OptOptions::default();
    opts.max_steps = max_steps;
    opts.max_force = 3e-5;
    opts.rms_force = 2e-5;
    opts.max_displacement = 1e-3;
    opts.reference = reference;
    opts.grid_level = GridLevel::from_name(grid_level);

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

/// Returns `(origin_bohr, spacing_bohr, counts_xyz)` for a padded uniform 3D grid around `wf`.
#[pyfunction]
#[pyo3(signature = (wf, margin_bohr=4.5, spacing_bohr=0.25))]
pub fn auto_orbital_grid_spec(
    wf: &PyScfWavefunction,
    margin_bohr: f64,
    spacing_bohr: f64,
) -> PyResult<([f64; 3], [f64; 3], [usize; 3])> {
    let (mol, _bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    Ok((spec.origin_bohr, spec.spacing_bohr, spec.counts))
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
    let (_mol, bs) = wf.build_mol_and_basis()?;
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

/// Evaluates total electron density (`spin_density=false`) or spin density `P_alpha - P_beta`
/// (`spin_density=true`) on a 3D grid and returns a zero-copy `float32` array `(nz, ny, nx)`.
#[pyfunction]
#[pyo3(signature = (wf, origin, spacing, counts, spin_density=false))]
pub fn eval_density_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
    spin_density: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (_mol, bs) = wf.build_mol_and_basis()?;
    let grid_vec = if spin_density {
        if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, origin, spacing, counts).map_err(to_py_err)?
        } else {
            let total = counts[0]
                .checked_mul(counts[1])
                .and_then(|v| v.checked_mul(counts[2]))
                .ok_or_else(|| PyValueError::new_err("Grid dimensions overflow"))?;
            vec![0.0f32; total]
        }
    } else {
        compute_density_grid(&bs, &wf.density, origin, spacing, counts).map_err(to_py_err)?
    };

    let arr1 = PyArray1::from_vec_bound(py, grid_vec);
    let arr3 = arr1.reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// Extracts dual-phase (`+isovalue`, `-isovalue`) Marching Cubes meshes directly from an existing
/// `(nz, ny, nx)` `float32` volumetric grid (enabling millisecond interactive slider updates).
#[pyfunction]
#[pyo3(signature = (grid, origin_bohr, spacing_bohr, isovalue=0.03, output_angstrom=true, mo_index=-1, spin_channel=0))]
pub fn extract_isosurface_from_grid<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    isovalue: f32,
    output_angstrom: bool,
    mo_index: i32,
    spin_channel: u32,
) -> PyResult<PyDualOrbitalMesh> {
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]]; // (nz, ny, nx) -> [nx, ny, nz]
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    let dual = extract_dual_isosurface(
        slice,
        origin_bohr,
        spacing_bohr,
        counts,
        isovalue,
        output_angstrom,
    )
    .map_err(to_py_err)?;
    PyDualOrbitalMesh::from_rust(py, dual, mo_index, spin_channel)
}

/// Extracts a single-lobe Marching Cubes mesh at `isovalue` from an existing `(nz, ny, nx)` grid.
#[pyfunction]
#[pyo3(signature = (grid, origin_bohr, spacing_bohr, isovalue=0.002, output_angstrom=true))]
pub fn extract_single_isosurface_from_grid<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    isovalue: f32,
    output_angstrom: bool,
) -> PyResult<PyIsosurfaceMesh> {
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]];
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    let scale = if output_angstrom {
        BOHR_TO_ANGSTROM
    } else {
        1.0
    };
    let mesh =
        extract_isosurface(slice, origin_bohr, spacing_bohr, counts, isovalue, scale)
            .map_err(to_py_err)?;
    PyIsosurfaceMesh::from_rust(py, mesh)
}

/// End-to-end Rust parallel molecular orbital grid evaluation + dual-lobe Marching Cubes extraction.
#[pyfunction]
#[pyo3(signature = (wf, mo_idx, isovalue=0.03, spacing_bohr=0.25, margin_bohr=4.5, output_angstrom=true))]
pub fn extract_orbital_isosurface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    mo_idx: usize,
    isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    output_angstrom: bool,
) -> PyResult<PyDualOrbitalMesh> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let n_mo = wf.mo_occupations.len();
    let grid_vec = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )
    .map_err(to_py_err)?;
    let dual = extract_dual_isosurface(
        &grid_vec,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        isovalue,
        output_angstrom,
    )
    .map_err(to_py_err)?;
    PyDualOrbitalMesh::from_rust(py, dual, mo_idx as i32, wf.spin_channel_for_mo(mo_idx))
}

/// End-to-end Rust parallel electron/spin density grid evaluation + Marching Cubes extraction.
#[pyfunction]
#[pyo3(signature = (wf, isovalue=0.002, spacing_bohr=0.25, margin_bohr=4.5, spin_density=false, output_angstrom=true))]
pub fn extract_density_isosurface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    spin_density: bool,
    output_angstrom: bool,
) -> PyResult<PyDualOrbitalMesh> {
    if !isovalue.is_finite() || isovalue <= 0.0 {
        return Err(PyValueError::new_err("isovalue must be finite and > 0"));
    }
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;

    if spin_density {
        let grid_vec = if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, spec.origin_bohr, spec.spacing_bohr, spec.counts)
                .map_err(to_py_err)?
        } else {
            vec![0.0f32; spec.counts[0] * spec.counts[1] * spec.counts[2]]
        };
        let dual = extract_dual_isosurface(
            &grid_vec,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
            isovalue,
            output_angstrom,
        )
        .map_err(to_py_err)?;
        PyDualOrbitalMesh::from_rust(py, dual, -1, 2)
    } else {
        let grid_vec =
            compute_density_grid(&bs, &wf.density, spec.origin_bohr, spec.spacing_bohr, spec.counts)
                .map_err(to_py_err)?;
        let scale = if output_angstrom {
            BOHR_TO_ANGSTROM
        } else {
            1.0
        };
        let pos = extract_isosurface(
            &grid_vec,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
            isovalue,
            scale,
        )
        .map_err(to_py_err)?;
        let empty_neg = IsosurfaceMesh {
            isovalue: -isovalue,
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        };
        let dual = DualOrbitalMesh {
            isovalue,
            positive: pos,
            negative: empty_neg,
            units_angstrom: output_angstrom,
        };
        PyDualOrbitalMesh::from_rust(py, dual, -1, 0)
    }
}

/// Evaluates molecular orbital `mo_idx` and formats a standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, mo_idx, spacing_bohr=0.25, margin_bohr=4.5, title=""))]
pub fn export_orbital_gaussian_cube(
    wf: &PyScfWavefunction,
    mo_idx: usize,
    spacing_bohr: f64,
    margin_bohr: f64,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let n_mo = wf.mo_occupations.len();
    let grid = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )
    .map_err(to_py_err)?;
    let c2 = if title.trim().is_empty() {
        format!("Molecular Orbital {mo_idx} ({})", wf.basis_name)
    } else {
        title.to_string()
    };
    format_gaussian_cube(
        &mol,
        Some(&bs),
        &grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Generated by CalChem Rust Core",
        &c2,
    )
    .map_err(to_py_err)
}

/// Evaluates electron or spin density and formats a standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, spacing_bohr=0.25, margin_bohr=4.5, spin_density=false, title=""))]
pub fn export_density_gaussian_cube(
    wf: &PyScfWavefunction,
    spacing_bohr: f64,
    margin_bohr: f64,
    spin_density: bool,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let grid = if spin_density {
        if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, spec.origin_bohr, spec.spacing_bohr, spec.counts)
                .map_err(to_py_err)?
        } else {
            vec![0.0f32; spec.counts[0] * spec.counts[1] * spec.counts[2]]
        }
    } else {
        compute_density_grid(&bs, &wf.density, spec.origin_bohr, spec.spacing_bohr, spec.counts)
            .map_err(to_py_err)?
    };
    let c2 = if title.trim().is_empty() {
        if spin_density {
            format!("Spin Density ({})", wf.basis_name)
        } else {
            format!("Total Electron Density ({})", wf.basis_name)
        }
    } else {
        title.to_string()
    };
    format_gaussian_cube(
        &mol,
        Some(&bs),
        &grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Generated by CalChem Rust Core",
        &c2,
    )
    .map_err(to_py_err)
}

/// Formats an arbitrary `(nz, ny, nx)` `float32` volumetric grid into Gaussian `.cube` text in Rust.
#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, grid, origin_bohr, spacing_bohr, title="", basis=""))]
pub fn format_grid_gaussian_cube<'py>(
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    title: &str,
    basis: &str,
) -> PyResult<String> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(&geometry_bohr) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, 0, 1);
    let bs_opt = if basis.trim().is_empty() {
        None
    } else {
        Some(BasisSet::from_name(basis, &mol).map_err(to_py_err)?)
    };
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]];
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    format_gaussian_cube(
        &mol,
        bs_opt.as_ref(),
        slice,
        origin_bohr,
        spacing_bohr,
        counts,
        "Generated by CalChem Rust Core",
        title,
    )
    .map_err(to_py_err)
}

/// Parses a standard Gaussian `.cube` string in Rust, returning
/// `(symbols, geometry_bohr, origin_bohr, spacing_bohr, grid_zyx_f32)`.
#[pyfunction]
pub fn parse_gaussian_cube_str<'py>(
    py: Python<'py>,
    cube_text: &str,
) -> PyResult<(
    Vec<String>,
    Vec<[f64; 3]>,
    [f64; 3],
    [f64; 3],
    Bound<'py, PyArray3<f32>>,
)> {
    let (mol, spec, grid_zyx) = parse_gaussian_cube(cube_text).map_err(to_py_err)?;
    let symbols: Vec<String> = mol.atoms.iter().map(|a| a.symbol.clone()).collect();
    let geometry_bohr: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position_bohr).collect();
    let arr1 = PyArray1::from_vec_bound(py, grid_zyx);
    let arr3 = arr1.reshape([spec.counts[2], spec.counts[1], spec.counts[0]])?;
    Ok((
        symbols,
        geometry_bohr,
        spec.origin_bohr,
        spec.spacing_bohr,
        arr3,
    ))
}

/// Basis set names embedded in the Rust core.
#[pyfunction]
pub fn supported_basis_sets() -> Vec<&'static str> {
    BasisSet::SUPPORTED_NAMES.to_vec()
}

/// DFT exchange-correlation functional names supported natively by the Rust Kohn-Sham core.
#[pyfunction]
pub fn supported_dft_functionals() -> Vec<&'static str> {
    XcFunctional::SUPPORTED_NAMES.to_vec()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add(
        "ScfNotConvergedError",
        m.py().get_type_bound::<ScfNotConvergedError>(),
    )?;
    m.add("RustCoreError", m.py().get_type_bound::<RustCoreError>())?;
    m.add_class::<PyScfWavefunction>()?;
    m.add_class::<PyOptResult>()?;
    m.add_class::<PyIsosurfaceMesh>()?;
    m.add_class::<PyDualOrbitalMesh>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(run_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(auto_orbital_grid_spec, m)?)?;
    m.add_function(wrap_pyfunction!(eval_orbital_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(eval_density_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(extract_isosurface_from_grid, m)?)?;
    m.add_function(wrap_pyfunction!(extract_single_isosurface_from_grid, m)?)?;
    m.add_function(wrap_pyfunction!(extract_orbital_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(extract_density_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(export_orbital_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(export_density_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(format_grid_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(parse_gaussian_cube_str, m)?)?;
    m.add_function(wrap_pyfunction!(supported_basis_sets, m)?)?;
    m.add_function(wrap_pyfunction!(supported_dft_functionals, m)?)?;
    Ok(())
}
