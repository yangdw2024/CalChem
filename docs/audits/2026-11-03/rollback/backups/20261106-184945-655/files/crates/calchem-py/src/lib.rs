use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient, uhf_gradient, uks_gradient};
use calchem_model::{
    symbol_from_atomic_number, Atom, CalcError, Molecule, ScfOptions, BOHR_TO_ANGSTROM,
};
use calchem_opt::{
    optimize_geometry_with_options, run_irc as rust_run_irc,
    search_transition_state as rust_search_ts, IrcDirection, IrcOptions, OptOptions, Reference,
    TsOptions,
};
use calchem_prop::{
    auto_grid_for_molecule, compute_density_grid, compute_dipole, compute_esp_at_points,
    compute_esp_grid, compute_esp_surface, compute_orbital_grid, encode_dual_mesh_binary,
    encode_esp_surface_binary, extract_dual_isosurface, extract_isosurface, format_gaussian_cube,
    parse_gaussian_cube, population_analysis, DualOrbitalMesh, EspSurface, GridSpec,
    IsosurfaceMesh,
};
use calchem_scf::{rhf, rks, uhf, uks};
use calchem_tddft::{
    broaden_uv_vis_spectrum, run_tddft as rust_run_tddft, run_tddft_from_scf, ExcitedState,
    OrbitalTransition, SpinManifold, TddftOptions, TddftResult,
};
use calchem_vib::{
    compute_frequencies_and_thermo, FreqOptions, VibMethod,
};
use calchem_solv::{
    run_scrf as rust_run_scrf, BoundaryMatrix, CavityLebedevOrder, CavityOptions,
    CavityPotentialEngine, MolecularCavity, SolvationOptions, Solvent, SolventModel,
    HARTREE_TO_KCAL_MOL,
};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

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

/// Output of a pure Rust Transition State Search (P-RFO).
#[pyclass(name = "TsResult")]
#[derive(Clone)]
pub struct PyTsResult {
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub gradient: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub imaginary_frequency_cm: f64,
    #[pyo3(get)]
    pub transition_vector: Vec<f64>,
}

#[pymethods]
impl PyTsResult {
    fn __repr__(&self) -> String {
        format!(
            "TsResult(converged={}, energy={:.8} Eh, imag_freq={:.1} cm^-1, steps={})",
            self.converged, self.energy_hartree, self.imaginary_frequency_cm, self.steps
        )
    }
}

/// Single point along an Intrinsic Reaction Coordinate (IRC) path.
#[pyclass(name = "IrcPoint")]
#[derive(Clone)]
pub struct PyIrcPoint {
    #[pyo3(get)]
    pub point_index: usize,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub path_coordinate_s: f64,
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub delta_energy_kcal_mol: f64,
    #[pyo3(get)]
    pub gradient_norm: f64,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
}

#[pymethods]
impl PyIrcPoint {
    fn __repr__(&self) -> String {
        format!(
            "IrcPoint(idx={}, dir='{}', s={:.3}, dE={:.2} kcal/mol, g_norm={:.3e})",
            self.point_index, self.direction, self.path_coordinate_s, self.delta_energy_kcal_mol, self.gradient_norm
        )
    }
}

/// One unidirectional branch of an IRC reaction pathway (forward or reverse).
#[pyclass(name = "IrcBranch")]
#[derive(Clone)]
pub struct PyIrcBranch {
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub n_points: usize,
    #[pyo3(get)]
    pub barrier_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_energy_kcal_mol: f64,
    #[pyo3(get)]
    pub points: Vec<PyIrcPoint>,
}

#[pymethods]
impl PyIrcBranch {
    fn __repr__(&self) -> String {
        format!(
            "IrcBranch(dir='{}', n_points={}, barrier={:.2} kcal/mol, dE={:.2} kcal/mol)",
            self.direction, self.n_points, self.barrier_kcal_mol, self.delta_energy_kcal_mol
        )
    }
}

/// Complete IRC result holding the transition state and optional forward/reverse branches.
#[pyclass(name = "IrcResult")]
#[derive(Clone)]
pub struct PyIrcResult {
    #[pyo3(get)]
    pub transition_state: PyIrcPoint,
    #[pyo3(get)]
    pub forward: Option<PyIrcBranch>,
    #[pyo3(get)]
    pub reverse: Option<PyIrcBranch>,
    #[pyo3(get)]
    pub total_points: usize,
}

#[pymethods]
impl PyIrcResult {
    fn __repr__(&self) -> String {
        format!(
            "IrcResult(total_points={}, fwd_barrier={:?}, rev_barrier={:?})",
            self.total_points,
            self.forward.as_ref().map(|b| b.barrier_kcal_mol),
            self.reverse.as_ref().map(|b| b.barrier_kcal_mol)
        )
    }
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

/// Mulliken / Löwdin atomic charges (ECP-consistent `Z_eff`) and UHF/UKS spin populations.
///
/// Returns a dict with keys `mulliken`, `lowdin` (lists of per-atom charges in e) and
/// `mulliken_spin`, `lowdin_spin` (per-atom spin populations, or `None` for closed shells).
#[pyfunction]
pub fn compute_population_analysis<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
) -> PyResult<Bound<'py, PyDict>> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let pop = py
        .allow_threads(|| population_analysis(&mol, &bs, &wf.density, wf.spin_density.as_deref()))
        .map_err(to_py_err)?;
    let d = PyDict::new_bound(py);
    d.set_item("mulliken", pop.mulliken_charges)?;
    d.set_item("lowdin", pop.lowdin_charges)?;
    d.set_item("mulliken_spin", pop.mulliken_spin)?;
    d.set_item("lowdin_spin", pop.lowdin_spin)?;
    Ok(d)
}

/// Exact molecular electrostatic potential (Hartree / e) at `(N, 3)` Bohr points.
#[pyfunction]
pub fn eval_esp_at_points<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    points_bohr: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = points_bohr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("points_bohr must have shape (N, 3)"));
    }
    let flat = points_bohr
        .as_slice()
        .map_err(|_| PyValueError::new_err("points_bohr must be a C-contiguous float64 array"))?;
    let pts: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let (mol, bs) = wf.build_mol_and_basis()?;
    let v = py
        .allow_threads(|| compute_esp_at_points(&mol, &bs, &wf.density, &pts))
        .map_err(to_py_err)?;
    Ok(PyArray1::from_vec_bound(py, v))
}

/// Exact ESP on a uniform grid; returns a zero-copy `float32` array of shape `(nz, ny, nx)`.
#[pyfunction]
pub fn eval_esp_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = GridSpec {
        origin_bohr: origin,
        spacing_bohr: spacing,
        counts,
    };
    let grid = py
        .allow_threads(|| compute_esp_grid(&mol, &bs, &wf.density, &spec))
        .map_err(to_py_err)?;
    let arr3 = PyArray1::from_vec_bound(py, grid).reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// ESP-mapped isodensity surface (vertex-colored mesh) with `CCESP\x01` binary serialization.
#[pyclass]
pub struct PyEspSurface {
    #[pyo3(get)]
    pub density_isovalue: f32,
    #[pyo3(get)]
    pub esp_min: f32,
    #[pyo3(get)]
    pub esp_max: f32,
    #[pyo3(get)]
    pub units_angstrom: bool,
    #[pyo3(get)]
    pub mesh: Py<PyIsosurfaceMesh>,
    esp_arr: Py<PyArray1<f32>>,
}

#[pymethods]
impl PyEspSurface {
    /// `(n_vertices,)` `float32` per-vertex ESP (Hartree / e), zero-copy handle.
    #[getter]
    fn esp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.esp_arr.bind(py).clone()
    }

    /// Serializes the surface into a 64-byte-aligned `CCESP\x01` binary packet.
    fn to_binary_packet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let surf = EspSurface {
            density_isovalue: self.density_isovalue,
            mesh: self.mesh.borrow(py).to_rust(py)?,
            esp: self.esp_arr.bind(py).readonly().as_slice()?.to_vec(),
            esp_min: self.esp_min,
            esp_max: self.esp_max,
            units_angstrom: self.units_angstrom,
        };
        Ok(PyBytes::new_bound(py, &encode_esp_surface_binary(&surf)))
    }
}

/// Builds the `rho = density_isovalue` isodensity surface and maps the exact ESP onto its vertices.
#[pyfunction]
#[pyo3(signature = (wf, density_isovalue=0.001, spacing_bohr=0.25, margin_bohr=5.0, output_angstrom=true))]
pub fn extract_esp_surface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    density_isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    output_angstrom: bool,
) -> PyResult<PyEspSurface> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let surf = py
        .allow_threads(|| {
            compute_esp_surface(
                &mol,
                &bs,
                &wf.density,
                density_isovalue,
                spacing_bohr,
                margin_bohr,
                output_angstrom,
            )
        })
        .map_err(to_py_err)?;
    let mesh = Py::new(py, PyIsosurfaceMesh::from_rust(py, surf.mesh)?)?;
    Ok(PyEspSurface {
        density_isovalue: surf.density_isovalue,
        esp_min: surf.esp_min,
        esp_max: surf.esp_max,
        units_angstrom: surf.units_angstrom,
        mesh,
        esp_arr: PyArray1::from_vec_bound(py, surf.esp).unbind(),
    })
}

/// Evaluates molecular electrostatic potential (ESP) on an automatic 3D grid and formats a
/// standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, spacing_bohr=0.25, margin_bohr=5.0, title=""))]
pub fn export_esp_gaussian_cube(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    spacing_bohr: f64,
    margin_bohr: f64,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let grid = py
        .allow_threads(|| compute_esp_grid(&mol, &bs, &wf.density, &spec))
        .map_err(to_py_err)?;
    let c2 = if title.trim().is_empty() {
        format!("Molecular Electrostatic Potential ({})", wf.basis_name)
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

fn scf_wfn_to_py(
    wf: &calchem_model::ScfWavefunction,
    mol: &Molecule,
    bs: &BasisSet,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    basis_name: String,
    charge: i32,
    multiplicity: usize,
    xc_name: Option<String>,
) -> PyScfWavefunction {
    let (dipole, dipole_mag) = compute_dipole(mol, bs, &wf.density_alpha);
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 0.5).count();
    PyScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        n_ao: wf.n_ao,
        mo_energies: wf.mo_energies.clone(),
        mo_occupations: wf.mo_occupations.clone(),
        mo_coefficients: wf.mo_coefficients.clone(),
        density: wf.density_alpha.clone(),
        spin_density: None,
        energy_history: wf.diagnostics.energy_history.clone(),
        is_uhf: false,
        s_squared: Some(0.0),
        n_alpha: n_occ,
        n_beta: n_occ,
        mo_energies_alpha: None,
        mo_energies_beta: None,
        symbols,
        geometry_bohr,
        basis_name,
        charge,
        multiplicity,
        xc_functional: xc_name,
        dipole: Some(dipole),
        dipole_mag: Some(dipole_mag),
    }
}

/// Single occupied -> virtual canonical MO contribution to an excited state.
#[pyclass]
#[derive(Clone)]
pub struct PyOrbitalTransition {
    #[pyo3(get)]
    pub occ_index: usize,
    #[pyo3(get)]
    pub virt_index: usize,
    #[pyo3(get)]
    pub amplitude_x: f64,
    #[pyo3(get)]
    pub amplitude_y: f64,
    #[pyo3(get)]
    pub weight: f64,
}

#[pymethods]
impl PyOrbitalTransition {
    fn __repr__(&self) -> String {
        format!(
            "OrbitalTransition(occ={}, virt={}, weight={:.4}, X={:.4}, Y={:.4})",
            self.occ_index, self.virt_index, self.weight, self.amplitude_x, self.amplitude_y
        )
    }
}

/// Computed electronic excited state and spectroscopic observables.
#[pyclass]
#[derive(Clone)]
pub struct PyExcitedState {
    #[pyo3(get)]
    pub state_index: usize,
    #[pyo3(get)]
    pub manifold_index: usize,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub multiplicity: usize,
    #[pyo3(get)]
    pub excitation_energy_hartree: f64,
    #[pyo3(get)]
    pub excitation_energy_ev: f64,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub wavelength_nm: f64,
    #[pyo3(get)]
    pub transition_dipole_au: [f64; 3],
    #[pyo3(get)]
    pub oscillator_strength: f64,
    #[pyo3(get)]
    pub dominant_transitions: Vec<PyOrbitalTransition>,
}

#[pymethods]
impl PyExcitedState {
    fn __repr__(&self) -> String {
        format!(
            "ExcitedState(label='{}', mult={}, energy_ev={:.4}, lambda_nm={:.1}, f={:.4})",
            self.label,
            self.multiplicity,
            self.excitation_energy_ev,
            self.wavelength_nm,
            self.oscillator_strength
        )
    }
}

/// Gaussian-broadened UV-Vis molar decadic absorption spectrum.
#[pyclass]
#[derive(Clone)]
pub struct PyUvVisSpectrum {
    #[pyo3(get)]
    pub wavelengths_nm: Vec<f64>,
    #[pyo3(get)]
    pub energies_ev: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_l_mol_cm: Vec<f64>,
    #[pyo3(get)]
    pub fwhm_ev: f64,
}

#[pymethods]
impl PyUvVisSpectrum {
    fn __repr__(&self) -> String {
        format!(
            "UvVisSpectrum(n_points={}, range=[{:.1}, {:.1}] nm, max_eps={:.1})",
            self.wavelengths_nm.len(),
            self.wavelengths_nm.first().copied().unwrap_or(0.0),
            self.wavelengths_nm.last().copied().unwrap_or(0.0),
            self.epsilon_l_mol_cm.iter().copied().fold(0.0f64, f64::max)
        )
    }
}

/// Complete output of a pure Rust CIS / TDHF / TDA-DFT / TD-DFT excited-state calculation.
#[pyclass]
#[derive(Clone)]
pub struct PyTddftResult {
    #[pyo3(get)]
    pub ground_energy_hartree: f64,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    #[pyo3(get)]
    pub tda: bool,
    #[pyo3(get)]
    pub n_occ: usize,
    #[pyo3(get)]
    pub n_virt: usize,
    #[pyo3(get)]
    pub homo_energy_hartree: f64,
    #[pyo3(get)]
    pub lumo_energy_hartree: f64,
    #[pyo3(get)]
    pub ks_gap_ev: f64,
    #[pyo3(get)]
    pub optical_gap_ev: Option<f64>,
    #[pyo3(get)]
    pub lowest_triplet_ev: Option<f64>,
    #[pyo3(get)]
    pub delta_est_ev: Option<f64>,
    #[pyo3(get)]
    pub davidson_iterations_singlet: usize,
    #[pyo3(get)]
    pub davidson_iterations_triplet: usize,
    #[pyo3(get)]
    pub states: Vec<PyExcitedState>,
    #[pyo3(get)]
    pub scf_wavefunction: PyScfWavefunction,
}

#[pymethods]
impl PyTddftResult {
    fn __repr__(&self) -> String {
        format!(
            "TddftResult(tda={}, states={}, E_opt={:?} eV, Delta_EST={:?} eV)",
            self.tda,
            self.states.len(),
            self.optical_gap_ev,
            self.delta_est_ev
        )
    }

    /// Evaluates the Gaussian-broadened UV-Vis spectrum over `[min_nm, max_nm]`.
    #[pyo3(signature = (min_nm=150.0, max_nm=800.0, n_points=500, fwhm_ev=0.33))]
    fn broaden_spectrum(
        &self,
        min_nm: f64,
        max_nm: f64,
        n_points: usize,
        fwhm_ev: f64,
    ) -> PyResult<PyUvVisSpectrum> {
        let rust_states = py_states_to_rust(&self.states);
        let spec = broaden_uv_vis_spectrum(&rust_states, min_nm, max_nm, n_points, fwhm_ev)
            .map_err(to_py_err)?;
        Ok(PyUvVisSpectrum {
            wavelengths_nm: spec.wavelengths_nm,
            energies_ev: spec.energies_ev,
            epsilon_l_mol_cm: spec.epsilon_l_mol_cm,
            fwhm_ev: spec.fwhm_ev,
        })
    }
}

fn py_states_to_rust(states: &[PyExcitedState]) -> Vec<ExcitedState> {
    states
        .iter()
        .map(|s| ExcitedState {
            state_index: s.state_index,
            manifold_index: s.manifold_index,
            label: s.label.clone(),
            multiplicity: s.multiplicity,
            excitation_energy_hartree: s.excitation_energy_hartree,
            excitation_energy_ev: s.excitation_energy_ev,
            total_energy_hartree: s.total_energy_hartree,
            wavelength_nm: s.wavelength_nm,
            transition_dipole_au: s.transition_dipole_au,
            oscillator_strength: s.oscillator_strength,
            dominant_transitions: s
                .dominant_transitions
                .iter()
                .map(|t| OrbitalTransition {
                    occ_index: t.occ_index,
                    virt_index: t.virt_index,
                    amplitude_x: t.amplitude_x,
                    amplitude_y: t.amplitude_y,
                    weight: t.weight,
                })
                .collect(),
        })
        .collect()
}

fn rust_to_py_tddft_result(
    res: TddftResult,
    mol: &Molecule,
    bs: &BasisSet,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    basis_name: String,
    charge: i32,
    multiplicity: usize,
) -> PyTddftResult {
    let py_wfn = scf_wfn_to_py(
        &res.scf_wavefunction,
        mol,
        bs,
        symbols,
        geometry_bohr,
        basis_name,
        charge,
        multiplicity,
        res.xc_functional.clone(),
    );

    let py_states: Vec<PyExcitedState> = res
        .states
        .into_iter()
        .map(|s| PyExcitedState {
            state_index: s.state_index,
            manifold_index: s.manifold_index,
            label: s.label,
            multiplicity: s.multiplicity,
            excitation_energy_hartree: s.excitation_energy_hartree,
            excitation_energy_ev: s.excitation_energy_ev,
            total_energy_hartree: s.total_energy_hartree,
            wavelength_nm: s.wavelength_nm,
            transition_dipole_au: s.transition_dipole_au,
            oscillator_strength: s.oscillator_strength,
            dominant_transitions: s
                .dominant_transitions
                .into_iter()
                .map(|t| PyOrbitalTransition {
                    occ_index: t.occ_index,
                    virt_index: t.virt_index,
                    amplitude_x: t.amplitude_x,
                    amplitude_y: t.amplitude_y,
                    weight: t.weight,
                })
                .collect(),
        })
        .collect();

    PyTddftResult {
        ground_energy_hartree: res.ground_energy_hartree,
        xc_functional: res.xc_functional,
        tda: res.tda,
        n_occ: res.n_occ,
        n_virt: res.n_virt,
        homo_energy_hartree: res.homo_energy_hartree,
        lumo_energy_hartree: res.lumo_energy_hartree,
        ks_gap_ev: res.ks_gap_ev,
        optical_gap_ev: res.optical_gap_ev,
        lowest_triplet_ev: res.lowest_triplet_ev,
        delta_est_ev: res.delta_est_ev,
        davidson_iterations_singlet: res.davidson_iterations_singlet,
        davidson_iterations_triplet: res.davidson_iterations_triplet,
        states: py_states,
        scf_wavefunction: py_wfn,
    }
}

fn resolve_tddft_method_and_xc(
    method: &str,
    xc: Option<&str>,
) -> PyResult<(Option<XcFunctional>, bool)> {
    let m = method.trim().to_ascii_lowercase();
    match m.as_str() {
        "cis" => Ok((None, true)),
        "tdhf" | "rpa_hf" => Ok((None, false)),
        "rhf" | "hf" => {
            if let Some(x) = xc {
                let func = XcFunctional::from_name(x).map_err(to_py_err)?;
                Ok((Some(func), false))
            } else {
                Ok((None, false))
            }
        }
        "rks" | "dft" | "tddft" | "td-dft" | "tda" | "tda-dft" => {
            let func_name = xc.unwrap_or("b3lyp");
            let func = XcFunctional::from_name(func_name).map_err(to_py_err)?;
            let tda_by_name = m.starts_with("tda");
            Ok((Some(func), tda_by_name))
        }
        other => {
            if let Ok(func) = XcFunctional::from_name(other) {
                Ok((Some(func), false))
            } else if let Some(x) = xc {
                let func = XcFunctional::from_name(x).map_err(to_py_err)?;
                Ok((Some(func), false))
            } else {
                Err(PyValueError::new_err(format!(
                    "Unsupported TD-DFT method or functional: '{method}'"
                )))
            }
        }
    }
}

/// Runs a linear-response excited-state calculation (CIS, TDHF, TDA-DFT, TD-DFT) from geometry in pure Rust.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="rks",
    basis="6-31g",
    n_states=5,
    tda=false,
    manifold="singlet",
    xc=None,
    grid_level="medium",
    max_iter=80,
    conv_tol=1e-6,
    force_davidson=false,
    scf_max_iter=100,
    scf_conv_energy=1e-10,
    scf_conv_density=1e-8
))]
#[allow(clippy::too_many_arguments)]
pub fn run_tddft(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    n_states: usize,
    tda: bool,
    manifold: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyTddftResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if n_states == 0 {
        return Err(PyValueError::new_err("n_states must be >= 1"));
    }
    if multiplicity != 1 {
        return Err(PyValueError::new_err(
            "TD-DFT/CIS currently requires a closed-shell reference (multiplicity=1)",
        ));
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
    let (xc_func, tda_by_name) = resolve_tddft_method_and_xc(method, xc)?;
    let spin_manifold = SpinManifold::from_name(manifold).map_err(to_py_err)?;
    let gl = GridLevel::from_name(grid_level);

    let opts = TddftOptions {
        n_states,
        tda: tda || tda_by_name,
        manifold: spin_manifold,
        grid_level: gl,
        max_iter,
        conv_tol,
        force_davidson,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_run_tddft(&mol, &bs, xc_func, &opts))
        .map_err(to_py_err)?;

    Ok(rust_to_py_tddft_result(
        res,
        &mol,
        &bs,
        symbols,
        geometry_bohr,
        basis.to_string(),
        charge,
        multiplicity,
    ))
}

/// Runs a linear-response excited-state calculation atop an already-converged ground-state [`PyScfWavefunction`].
#[pyfunction]
#[pyo3(signature = (
    wf,
    n_states=5,
    tda=false,
    manifold="singlet",
    xc=None,
    grid_level="medium",
    max_iter=80,
    conv_tol=1e-6,
    force_davidson=false
))]
#[allow(clippy::too_many_arguments)]
pub fn run_tddft_from_scf_py(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    n_states: usize,
    tda: bool,
    manifold: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
) -> PyResult<PyTddftResult> {
    if wf.is_uhf || wf.multiplicity != 1 {
        return Err(PyValueError::new_err(
            "TD-DFT/CIS currently requires a closed-shell singlet wavefunction",
        ));
    }
    if n_states == 0 {
        return Err(PyValueError::new_err("n_states must be >= 1"));
    }
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spin_manifold = SpinManifold::from_name(manifold).map_err(to_py_err)?;
    let gl = GridLevel::from_name(grid_level);

    let xc_name = xc.or(wf.xc_functional.as_deref());
    let xc_func = match xc_name {
        Some(x)
            if !x.is_empty()
                && x.to_ascii_lowercase() != "rhf"
                && x.to_ascii_lowercase() != "hf" =>
        {
            Some(XcFunctional::from_name(x).map_err(to_py_err)?)
        }
        _ => None,
    };

    let opts = TddftOptions {
        n_states,
        tda,
        manifold: spin_manifold,
        grid_level: gl,
        max_iter,
        conv_tol,
        force_davidson,
        scf_options: ScfOptions::default(),
    };

    let scf_wfn = calchem_model::ScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        n_ao: wf.n_ao,
        n_mo: wf.n_ao,
        mo_energies: wf.mo_energies.clone(),
        mo_occupations: wf.mo_occupations.clone(),
        mo_coefficients: wf.mo_coefficients.clone(),
        density_alpha: wf.density.clone(),
        density_beta: None,
        fock_alpha: vec![0.0; wf.n_ao * wf.n_ao],
        overlap_matrix: vec![0.0; wf.n_ao * wf.n_ao],
        h_core_matrix: vec![0.0; wf.n_ao * wf.n_ao],
        diagnostics: calchem_model::ScfDiagnostics {
            iterations: wf.iterations,
            energy_history: wf.energy_history.clone(),
            rms_density_history: vec![],
            diis_error_history: vec![],
        },
    };

    let res = py
        .allow_threads(|| run_tddft_from_scf(&mol, &bs, &scf_wfn, xc_func, &opts))
        .map_err(to_py_err)?;

    Ok(rust_to_py_tddft_result(
        res,
        &mol,
        &bs,
        wf.symbols.clone(),
        wf.geometry_bohr.clone(),
        wf.basis_name.clone(),
        wf.charge,
        wf.multiplicity,
    ))
}

/// Generates a Gaussian-broadened UV-Vis absorption spectrum from excited states.
#[pyfunction]
#[pyo3(signature = (states, min_nm=150.0, max_nm=800.0, n_points=500, fwhm_ev=0.33))]
pub fn broaden_uv_vis(
    states: Vec<PyExcitedState>,
    min_nm: f64,
    max_nm: f64,
    n_points: usize,
    fwhm_ev: f64,
) -> PyResult<PyUvVisSpectrum> {
    let rust_states = py_states_to_rust(&states);
    let spec = broaden_uv_vis_spectrum(&rust_states, min_nm, max_nm, n_points, fwhm_ev)
        .map_err(to_py_err)?;
    Ok(PyUvVisSpectrum {
        wavelengths_nm: spec.wavelengths_nm,
        energies_ev: spec.energies_ev,
        epsilon_l_mol_cm: spec.epsilon_l_mol_cm,
        fwhm_ev: spec.fwhm_ev,
    })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNormalMode {
    #[pyo3(get)]
    pub mode_index: usize,
    #[pyo3(get)]
    pub frequency_cm: f64,
    #[pyo3(get)]
    pub is_imaginary: bool,
    #[pyo3(get)]
    pub reduced_mass_amu: f64,
    #[pyo3(get)]
    pub force_constant_m_dyn_a: f64,
    #[pyo3(get)]
    pub ir_intensity_km_mol: f64,
    #[pyo3(get)]
    pub displacement: Vec<[f64; 3]>,
}

#[pymethods]
impl PyNormalMode {
    fn __repr__(&self) -> String {
        format!(
            "PyNormalMode(#{}: freq={:.2} cm^-1, red_mass={:.3} amu, k={:.3} mDyn/A, IR={:.3} km/mol, imag={})",
            self.mode_index,
            self.frequency_cm,
            self.reduced_mass_amu,
            self.force_constant_m_dyn_a,
            self.ir_intensity_km_mol,
            self.is_imaginary,
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyThermochemistry {
    #[pyo3(get)]
    pub temperature_k: f64,
    #[pyo3(get)]
    pub pressure_pa: f64,
    #[pyo3(get)]
    pub zpve_hartree: f64,
    #[pyo3(get)]
    pub zpve_kcal_mol: f64,
    #[pyo3(get)]
    pub e_trans_hartree: f64,
    #[pyo3(get)]
    pub e_rot_hartree: f64,
    #[pyo3(get)]
    pub e_vib_hartree: f64,
    #[pyo3(get)]
    pub e_thermal_hartree: f64,
    #[pyo3(get)]
    pub h_hartree: f64,
    #[pyo3(get)]
    pub g_hartree: f64,
    #[pyo3(get)]
    pub s_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_trans_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_rot_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_vib_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_elec_cal_mol_k: f64,
    #[pyo3(get)]
    pub cv_cal_mol_k: f64,
    #[pyo3(get)]
    pub cp_cal_mol_k: f64,
}

#[pymethods]
impl PyThermochemistry {
    fn __repr__(&self) -> String {
        format!(
            "PyThermochemistry(T={:.2} K, P={:.1} Pa, ZPVE={:.6} Eh, H={:.6} Eh, G={:.6} Eh, S={:.3} cal/(mol*K))",
            self.temperature_k,
            self.pressure_pa,
            self.zpve_hartree,
            self.h_hartree,
            self.g_hartree,
            self.s_cal_mol_k,
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVibrationalAnalysis {
    #[pyo3(get)]
    pub n_vib: usize,
    #[pyo3(get)]
    pub is_linear: bool,
    #[pyo3(get)]
    pub principal_moments_amu_bohr2: [f64; 3],
    #[pyo3(get)]
    pub rotational_constants_ghz: [f64; 3],
    #[pyo3(get)]
    pub trans_rot_residuals: Vec<f64>,
    #[pyo3(get)]
    pub modes: Vec<PyNormalMode>,
}

#[pymethods]
impl PyVibrationalAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "PyVibrationalAnalysis(n_vib={}, linear={}, modes={})",
            self.n_vib,
            self.is_linear,
            self.modes.len(),
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyFrequencyResult {
    #[pyo3(get)]
    pub energy_hartree: f64,
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
    pub method: String,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    #[pyo3(get)]
    pub vibrational: PyVibrationalAnalysis,
    #[pyo3(get)]
    pub thermo: PyThermochemistry,
    #[pyo3(get)]
    pub cartesian_hessian: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub apt: Vec<Vec<f64>>,
}

#[pymethods]
impl PyFrequencyResult {
    #[getter]
    fn frequencies_cm(&self) -> Vec<f64> {
        self.vibrational.modes.iter().map(|m| m.frequency_cm).collect()
    }

    #[getter]
    fn ir_intensities_km_mol(&self) -> Vec<f64> {
        self.vibrational.modes.iter().map(|m| m.ir_intensity_km_mol).collect()
    }

    #[getter]
    fn zpve_hartree(&self) -> f64 {
        self.thermo.zpve_hartree
    }

    #[getter]
    fn enthalpy_hartree(&self) -> f64 {
        self.thermo.h_hartree
    }

    #[getter]
    fn gibbs_free_energy_hartree(&self) -> f64 {
        self.thermo.g_hartree
    }

    #[getter]
    fn entropy_cal_mol_k(&self) -> f64 {
        self.thermo.s_cal_mol_k
    }

    #[getter]
    fn heat_capacity_cv_cal_mol_k(&self) -> f64 {
        self.thermo.cv_cal_mol_k
    }

    #[getter]
    fn heat_capacity_cp_cal_mol_k(&self) -> f64 {
        self.thermo.cp_cal_mol_k
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFrequencyResult(energy={:.8} Eh, n_vib={}, ZPVE={:.6} Eh, G={:.6} Eh)",
            self.energy_hartree,
            self.vibrational.n_vib,
            self.thermo.zpve_hartree,
            self.thermo.g_hartree,
        )
    }
}

/// Runs nuclear Hessian finite-difference vibrational frequency analysis and RRHO thermochemistry.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="rks",
    basis="sto-3g",
    step_bohr=0.001,
    xc=None,
    grid_level="medium",
    temperature_k=298.15,
    pressure_pa=101325.0,
    symmetry_number=None,
    max_iter=100,
    conv_energy=1e-10,
    conv_density=1e-8,
))]
pub fn run_frequency(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    step_bohr: f64,
    xc: Option<&str>,
    grid_level: &str,
    temperature_k: f64,
    pressure_pa: f64,
    symmetry_number: Option<usize>,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
) -> PyResult<PyFrequencyResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if symbols.is_empty() {
        return Err(PyValueError::new_err("molecule cannot be empty"));
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
    let _ = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;

    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let (vib_method, xc_name) = match scf_kind {
        ResolvedScfKind::Rhf => (VibMethod::Rhf, None),
        ResolvedScfKind::Uhf => (VibMethod::Uhf, None),
        ResolvedScfKind::Rks(f) => (VibMethod::Rks(f), Some(f.canonical_name().to_string())),
        ResolvedScfKind::Uks(f) => (VibMethod::Uks(f), Some(f.canonical_name().to_string())),
    };

    let gl = GridLevel::from_name(grid_level);
    let opts = FreqOptions {
        step_bohr,
        grid_level: gl,
        temperature_k,
        pressure_pa,
        symmetry_number,
        scf_options: ScfOptions {
            max_iter,
            conv_energy,
            conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| compute_frequencies_and_thermo(&mol, basis, vib_method, &opts))
        .map_err(to_py_err)?;

    let n_atoms = symbols.len();
    let dim = 3 * n_atoms;

    let py_modes: Vec<PyNormalMode> = res
        .vibrational_analysis
        .modes
        .into_iter()
        .map(|m| {
            let disp: Vec<[f64; 3]> = m
                .displacements
                .chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            PyNormalMode {
                mode_index: m.mode_index,
                frequency_cm: m.frequency_cm,
                is_imaginary: m.is_imaginary,
                reduced_mass_amu: m.reduced_mass_amu,
                force_constant_m_dyn_a: m.force_constant_m_dyn_a,
                ir_intensity_km_mol: m.ir_intensity_km_mol,
                displacement: disp,
            }
        })
        .collect();

    let py_vib = PyVibrationalAnalysis {
        n_vib: res.vibrational_analysis.n_vib,
        is_linear: res.vibrational_analysis.is_linear,
        principal_moments_amu_bohr2: res.vibrational_analysis.moments_of_inertia_amu_bohr2,
        rotational_constants_ghz: res.vibrational_analysis.rotational_constants_ghz,
        trans_rot_residuals: res.vibrational_analysis.trans_rot_residuals,
        modes: py_modes,
    };

    let py_thermo = PyThermochemistry {
        temperature_k: res.thermochemistry.temperature_k,
        pressure_pa: res.thermochemistry.pressure_pa,
        zpve_hartree: res.thermochemistry.zpve_hartree,
        zpve_kcal_mol: res.thermochemistry.zpve_kcal_mol,
        e_trans_hartree: res.thermochemistry.e_trans_hartree,
        e_rot_hartree: res.thermochemistry.e_rot_hartree,
        e_vib_hartree: res.thermochemistry.e_vib_hartree,
        e_thermal_hartree: res.thermochemistry.e_thermal_total_hartree,
        h_hartree: res.thermochemistry.enthalpy_hartree,
        g_hartree: res.thermochemistry.gibbs_free_energy_hartree,
        s_cal_mol_k: res.thermochemistry.s_total_cal_mol_k,
        s_trans_cal_mol_k: res.thermochemistry.s_trans_cal_mol_k,
        s_rot_cal_mol_k: res.thermochemistry.s_rot_cal_mol_k,
        s_vib_cal_mol_k: res.thermochemistry.s_vib_cal_mol_k,
        s_elec_cal_mol_k: res.thermochemistry.s_elec_cal_mol_k,
        cv_cal_mol_k: res.thermochemistry.cv_cal_mol_k,
        cp_cal_mol_k: res.thermochemistry.cp_cal_mol_k,
    };

    let mut h_mat = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            h_mat[i][j] = res.cartesian_hessian.get(i, j);
        }
    }

    let mut apt_mat = vec![vec![0.0f64; dim]; 3];
    for p in 0..3 {
        for j in 0..dim {
            apt_mat[p][j] = res.apt.get(p, j);
        }
    }

    Ok(PyFrequencyResult {
        energy_hartree: res.energy_hartree,
        symbols,
        geometry_bohr,
        basis_name: basis.to_string(),
        charge,
        multiplicity,
        method: method.to_string(),
        xc_functional: xc_name,
        vibrational: py_vib,
        thermo: py_thermo,
        cartesian_hessian: h_mat,
        apt: apt_mat,
    })
}

/// Runs a pure Rust Transition State Search using Partitioned Rational Function Optimization (P-RFO).
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="auto",
    basis="sto-3g",
    xc=None,
    grid_level="medium",
    max_steps=50,
    max_force=3e-4,
    rms_force=1.5e-4,
    max_displacement=1.2e-3,
    trust_radius_bohr=0.1,
    mode_index=None,
    recalc_hessian_interval=0,
    scf_max_iter=100,
    scf_conv_energy=1e-11,
    scf_conv_density=1e-10
))]
#[allow(clippy::too_many_arguments)]
pub fn run_ts_search(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_steps: usize,
    max_force: f64,
    rms_force: f64,
    max_displacement: f64,
    trust_radius_bohr: f64,
    mode_index: Option<usize>,
    recalc_hessian_interval: usize,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyTsResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err("symbols and geometry_bohr length mismatch"));
    }
    if symbols.len() < 2 {
        return Err(PyValueError::new_err("TS search requires at least 2 atoms"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc).map_err(|_| {
        PyValueError::new_err(format!("Unsupported method in Rust TS optimizer: {method}"))
    })?;

    let gl = GridLevel::from_name(grid_level);
    let opts = TsOptions {
        max_steps,
        max_force,
        rms_force,
        max_displacement,
        trust_radius_bohr,
        mode_index,
        recalc_hessian_interval,
        reference,
        grid_level: gl,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_search_ts(&mol, basis, &opts))
        .map_err(to_py_err)?;

    let out_symbols: Vec<String> = res
        .molecule
        .atoms
        .iter()
        .map(|a| symbol_from_atomic_number(a.z).unwrap_or_else(|_| "X".into()))
        .collect();
    let out_coords: Vec<[f64; 3]> = res.molecule.atoms.iter().map(|a| a.position_bohr).collect();

    Ok(PyTsResult {
        symbols: out_symbols,
        geometry_bohr: out_coords,
        energy_hartree: res.energy_hartree,
        gradient: res.gradient,
        steps: res.steps,
        converged: res.converged,
        imaginary_frequency_cm: res.imaginary_frequency_cm,
        transition_vector: res.transition_vector,
    })
}

/// Runs a pure Rust Gonzalez-Schlegel Intrinsic Reaction Coordinate (IRC) integration.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    transition_vector,
    charge=0,
    multiplicity=1,
    method="auto",
    basis="sto-3g",
    xc=None,
    grid_level="medium",
    step_size_bohr=0.1,
    max_points=20,
    direction="both",
    convergence_tol=1e-3,
    max_gs_iter=10,
    scf_max_iter=100,
    scf_conv_energy=1e-11,
    scf_conv_density=1e-10
))]
#[allow(clippy::too_many_arguments)]
pub fn run_irc(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    transition_vector: Vec<f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    grid_level: &str,
    step_size_bohr: f64,
    max_points: usize,
    direction: &str,
    convergence_tol: f64,
    max_gs_iter: usize,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyIrcResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err("symbols and geometry_bohr length mismatch"));
    }
    if transition_vector.len() != 3 * symbols.len() {
        return Err(PyValueError::new_err(format!(
            "transition_vector length {} mismatch with 3 * {} atoms",
            transition_vector.len(),
            symbols.len()
        )));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc).map_err(|_| {
        PyValueError::new_err(format!("Unsupported method in Rust IRC: {method}"))
    })?;

    let irc_dir = match direction.trim().to_ascii_lowercase().as_str() {
        "forward" | "fwd" => IrcDirection::Forward,
        "reverse" | "rev" => IrcDirection::Reverse,
        "both" | "full" => IrcDirection::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "Invalid IRC direction '{other}', expected 'forward', 'reverse', or 'both'"
            )))
        }
    };

    let gl = GridLevel::from_name(grid_level);
    let opts = IrcOptions {
        step_size_bohr,
        max_points,
        direction: irc_dir,
        convergence_tol,
        max_gs_iter,
        reference,
        grid_level: gl,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_run_irc(&mol, basis, &transition_vector, &opts))
        .map_err(to_py_err)?;

    let map_pt = |pt: &calchem_opt::IrcPoint, dir: &str| -> PyIrcPoint {
        let coords: Vec<[f64; 3]> = pt.molecule.atoms.iter().map(|a| a.position_bohr).collect();
        PyIrcPoint {
            point_index: pt.point_index,
            direction: dir.to_string(),
            path_coordinate_s: pt.path_coordinate_s,
            energy_hartree: pt.energy_hartree,
            delta_energy_kcal_mol: pt.delta_energy_kcal_mol,
            gradient_norm: pt.gradient_norm,
            geometry_bohr: coords,
        }
    };

    let map_branch = |br: &calchem_opt::IrcBranch| -> PyIrcBranch {
        let last_de = br
            .points
            .last()
            .map(|p| p.delta_energy_kcal_mol)
            .unwrap_or(0.0);
        PyIrcBranch {
            direction: br.direction.clone(),
            n_points: br.points.len(),
            barrier_kcal_mol: br.barrier_kcal_mol,
            delta_energy_kcal_mol: last_de,
            points: br.points.iter().map(|p| map_pt(p, &br.direction)).collect(),
        }
    };

    let total_pts = 1
        + res.forward.as_ref().map_or(0, |f| f.points.len())
        + res.reverse.as_ref().map_or(0, |r| r.points.len());

    let py_ts = map_pt(&res.transition_state, "ts");
    let py_fwd = res.forward.as_ref().map(map_branch);
    let py_rev = res.reverse.as_ref().map(map_branch);

    Ok(PyIrcResult {
        transition_state: py_ts,
        forward: py_fwd,
        reverse: py_rev,
        total_points: total_pts,
    })
}

#[pyclass(name = "SolvationResult")]
#[derive(Debug, Clone)]
pub struct PySolvationResult {
    #[pyo3(get)]
    pub solvated_energy: f64,
    #[pyo3(get)]
    pub gas_energy: f64,
    #[pyo3(get)]
    pub delta_g_solv: f64,
    #[pyo3(get)]
    pub delta_g_solv_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_g_elec_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_g_cds_kcal_mol: f64,
    #[pyo3(get)]
    pub cavity_surface_area_angstrom2: f64,
    #[pyo3(get)]
    pub total_surface_charge: f64,
    pub surface_charges: Vec<f64>,
    #[pyo3(get)]
    pub tessera_coords_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub scf_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub solvent_name: String,
    #[pyo3(get)]
    pub dielectric_constant: f64,
}

#[pymethods]
impl PySolvationResult {
    #[getter]
    fn surface_charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.surface_charges)
    }

    fn __repr__(&self) -> String {
        format!(
            "SolvationResult(solvent='{}', eps={:.2}, solvated_energy={:.6} Eh, dG_solv={:.2} kcal/mol, dG_elec={:.2} kcal/mol, dG_cds={:.2} kcal/mol, SASA={:.1} A^2)",
            self.solvent_name,
            self.dielectric_constant,
            self.solvated_energy,
            self.delta_g_solv_kcal_mol,
            self.delta_g_elec_kcal_mol,
            self.delta_g_cds_kcal_mol,
            self.cavity_surface_area_angstrom2
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    symbols,
    coords,
    basis_name,
    solvent="water",
    solvent_model="cpcm",
    charge=0,
    multiplicity=1,
    method="b3lyp",
    xc=None,
    scaling_factor=1.20,
    include_cds=true,
    max_iter=100,
    conv_energy=1e-8,
    conv_density=1e-6,
))]
pub fn run_solvation(
    _py: Python<'_>,
    symbols: Vec<String>,
    coords: Vec<[f64; 3]>,
    basis_name: &str,
    solvent: &str,
    solvent_model: &str,
    charge: i32,
    multiplicity: usize,
    method: &str,
    xc: Option<&str>,
    scaling_factor: f64,
    include_cds: bool,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
) -> PyResult<PySolvationResult> {
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(&coords) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis_name, &mol).map_err(to_py_err)?;

    let solv = Solvent::from_name(solvent).ok_or_else(|| {
        PyValueError::new_err(format!("Unsupported solvent: '{solvent}'"))
    })?;

    let model = match solvent_model.to_ascii_lowercase().as_str() {
        "cpcm" | "c-pcm" => SolventModel::Cpcm,
        "cosmo" => SolventModel::Cosmo,
        "iefpcm" | "ief-pcm" => SolventModel::IefPcm,
        "onsager" => SolventModel::Onsager,
        other => return Err(PyValueError::new_err(format!("Unknown solvent model: '{other}'"))),
    };

    let cav_opts = CavityOptions {
        scaling_factor,
        lebedev_order: CavityLebedevOrder::N50,
        probe_radius_angstrom: 0.0,
        exclusion_tol_bohr: 1e-5,
    };

    let solv_opts = SolvationOptions {
        solvent: solv.clone(),
        model,
        cavity_opts: cav_opts,
        renormalize_charge: true,
        include_cds,
    };

    let scf_opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        ..Default::default()
    };

    let xc_func = match xc {
        Some(x) => Some(XcFunctional::from_name(x).map_err(to_py_err)?),
        None => {
            if method.eq_ignore_ascii_case("dft") || method.eq_ignore_ascii_case("b3lyp") {
                Some(XcFunctional::B3lyp)
            } else {
                None
            }
        }
    };

    let res = rust_run_scrf(&mol, &bs, &scf_opts, &solv_opts, method, xc_func)
        .map_err(to_py_err)?;

    Ok(PySolvationResult {
        solvated_energy: res.solvated_energy,
        gas_energy: res.gas_energy,
        delta_g_solv: res.delta_g_solv,
        delta_g_solv_kcal_mol: res.delta_g_solv_kcal_mol,
        delta_g_elec_kcal_mol: res.delta_g_elec_kcal_mol,
        delta_g_cds_kcal_mol: res.delta_g_cds_kcal_mol,
        cavity_surface_area_angstrom2: res.cavity_surface_area_angstrom2,
        total_surface_charge: res.total_surface_charge,
        surface_charges: res.surface_charges,
        tessera_coords_bohr: res.tessera_coords_bohr,
        scf_iterations: res.scf_iterations,
        converged: res.converged,
        solvent_name: solv.name.clone(),
        dielectric_constant: solv.epsilon,
    })
}

#[pyfunction]
#[pyo3(signature = (
    scf_wfn,
    solvent="water",
    solvent_model="cpcm",
    scaling_factor=1.20,
    include_cds=true,
))]
pub fn run_pcm_from_scf_py(
    _py: Python<'_>,
    scf_wfn: &PyScfWavefunction,
    solvent: &str,
    solvent_model: &str,
    scaling_factor: f64,
    include_cds: bool,
) -> PyResult<PySolvationResult> {
    let (mol, bs) = scf_wfn.build_mol_and_basis()?;

    let solv = Solvent::from_name(solvent).ok_or_else(|| {
        PyValueError::new_err(format!("Unsupported solvent: '{solvent}'"))
    })?;

    let model = match solvent_model.to_ascii_lowercase().as_str() {
        "cpcm" | "c-pcm" => SolventModel::Cpcm,
        "cosmo" => SolventModel::Cosmo,
        "iefpcm" | "ief-pcm" => SolventModel::IefPcm,
        "onsager" => SolventModel::Onsager,
        other => return Err(PyValueError::new_err(format!("Unknown solvent model: '{other}'"))),
    };

    let cav_opts = CavityOptions {
        scaling_factor,
        lebedev_order: CavityLebedevOrder::N50,
        probe_radius_angstrom: 0.0,
        exclusion_tol_bohr: 1e-5,
    };

    let solv_opts = SolvationOptions {
        solvent: solv.clone(),
        model,
        cavity_opts: cav_opts,
        renormalize_charge: true,
        include_cds,
    };

    let cavity = MolecularCavity::build(&mol, &solv_opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity).map_err(to_py_err)?;
    let potential_engine = CavityPotentialEngine::build(&mol, &bs, &cavity).map_err(to_py_err)?;

    let net_charge = if solv_opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let n = bs.n_ao;
    let density_mat = calchem_tensor::Matrix2D {
        rows: n,
        cols: n,
        data: scf_wfn.density.clone(),
    };

    let f_eps = solv_opts.model.dielectric_factor(solv_opts.solvent.epsilon);
    let v_solute = potential_engine.compute_solute_potential(&density_mat);
    let q = boundary_matrix
        .solve_surface_charges(&v_solute, f_eps, net_charge)
        .map_err(to_py_err)?;

    let e_elec_hartree: f64 = 0.5 * q.iter().zip(v_solute.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();

    let cds_kcal_mol = if solv_opts.include_cds {
        solv_opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let delta_g_solv_hartree = e_elec_hartree + cds_hartree;
    let solvated_energy = scf_wfn.total_energy_hartree + delta_g_solv_hartree;

    let total_q: f64 = q.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(PySolvationResult {
        solvated_energy,
        gas_energy: scf_wfn.total_energy_hartree,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_elec_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: scf_wfn.iterations,
        converged: scf_wfn.converged,
        solvent_name: solv.name.clone(),
        dielectric_constant: solv.epsilon,
    })
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
    m.add_class::<PyTsResult>()?;
    m.add_class::<PyIrcPoint>()?;
    m.add_class::<PyIrcBranch>()?;
    m.add_class::<PyIrcResult>()?;
    m.add_class::<PySolvationResult>()?;
    m.add_class::<PyIsosurfaceMesh>()?;
    m.add_class::<PyDualOrbitalMesh>()?;
    m.add_class::<PyEspSurface>()?;
    m.add_class::<PyOrbitalTransition>()?;
    m.add_class::<PyExcitedState>()?;
    m.add_class::<PyUvVisSpectrum>()?;
    m.add_class::<PyTddftResult>()?;
    m.add_class::<PyNormalMode>()?;
    m.add_class::<PyThermochemistry>()?;
    m.add_class::<PyVibrationalAnalysis>()?;
    m.add_class::<PyFrequencyResult>()?;
    m.add_function(wrap_pyfunction!(run_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(run_ts_search, m)?)?;
    m.add_function(wrap_pyfunction!(run_irc, m)?)?;
    m.add_function(wrap_pyfunction!(run_solvation, m)?)?;
    m.add_function(wrap_pyfunction!(run_pcm_from_scf_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_tddft, m)?)?;
    m.add_function(wrap_pyfunction!(run_tddft_from_scf_py, m)?)?;
    m.add_function(wrap_pyfunction!(broaden_uv_vis, m)?)?;
    m.add_function(wrap_pyfunction!(compute_population_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(eval_esp_at_points, m)?)?;
    m.add_function(wrap_pyfunction!(eval_esp_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(extract_esp_surface, m)?)?;
    m.add_function(wrap_pyfunction!(export_esp_gaussian_cube, m)?)?;
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
