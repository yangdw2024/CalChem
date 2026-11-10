"""Type stub for the Rust extension calchem._core (crates/calchem-py).

Entry points:
- Closed-shell RHF and open-shell UHF single-point calculations
  (STO-3G, 6-31G, 6-31G(d)/6-31G*, 6-31G(d,p)/6-31G**, LANL2DZ, LANL2DZdp;
   including heavy halogens Br/I and scalar Effective Core Potentials)
- Analytical RHF and UHF energy gradients (including d/f shells and ECP gradients)
- BFGS geometry optimization (neutral and charged/open-shell species)
- Electric dipole moments (valence-consistent with ECP effective core charges)
- Zero-copy 3D orbital & electron/spin density volumetric grids (`float32`, `(nz, ny, nx)`)
- Consistent Marching Cubes dual-phase triangle mesh extraction (`float32`/`uint32`) & `CCMSH\\x01` binary streaming
- Pure Rust Gaussian `.cube` export & parser
- Exact molecular electrostatic potential (points / grids), Mulliken & Löwdin charges,
  and ESP-mapped isodensity surfaces with `CCESP\\x01` binary streaming
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

__version__: str

class ScfNotConvergedError(RuntimeError): ...
class RustCoreError(RuntimeError): ...

class PyScfWavefunction:
    converged: bool
    iterations: int
    total_energy_hartree: float
    electronic_energy_hartree: float
    nuclear_repulsion_hartree: float
    n_ao: int
    mo_energies: List[float]
    mo_occupations: List[float]
    mo_coefficients: List[float]  # row-major, index mu * n_mo + i
    density: List[float]  # spin-summed P, row-major
    spin_density: Optional[List[float]]  # P_alpha - P_beta for UHF, None for RHF
    energy_history: List[float]
    is_uhf: bool
    s_squared: Optional[float]
    n_alpha: int
    n_beta: int
    mo_energies_alpha: Optional[List[float]]
    mo_energies_beta: Optional[List[float]]
    symbols: List[str]
    geometry_bohr: List[Tuple[float, float, float]]
    basis_name: str
    charge: int
    multiplicity: int
    xc_functional: Optional[str]
    @property
    def homo_energy_hartree(self) -> Optional[float]: ...
    @property
    def lumo_energy_hartree(self) -> Optional[float]: ...
    @property
    def dipole_debye(self) -> Optional[Tuple[float, float, float]]: ...
    @property
    def dipole_magnitude_debye(self) -> Optional[float]: ...

class PyOptResult:
    converged: bool
    coords: List[float]
    energy: float
    steps: int

class PyIsosurfaceMesh:
    isovalue: float
    n_vertices: int
    n_triangles: int
    @property
    def vertices(self) -> NDArray[np.float32]: ...
    @property
    def normals(self) -> NDArray[np.float32]: ...
    @property
    def faces(self) -> NDArray[np.uint32]: ...
    @property
    def indices(self) -> NDArray[np.uint32]: ...

class PyDualOrbitalMesh:
    isovalue: float
    units_angstrom: bool
    mo_index: int
    spin_channel: int
    positive: PyIsosurfaceMesh
    negative: PyIsosurfaceMesh
    def to_binary_packet(self) -> bytes: ...

class PyEspSurface:
    density_isovalue: float
    esp_min: float
    esp_max: float
    units_angstrom: bool
    mesh: PyIsosurfaceMesh
    @property
    def esp(self) -> NDArray[np.float32]: ...
    def to_binary_packet(self) -> bytes: ...

def compute_population_analysis(wf: PyScfWavefunction) -> Dict[str, Optional[List[float]]]: ...
def eval_esp_at_points(
    wf: PyScfWavefunction, points_bohr: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def eval_esp_cube_zero_copy(
    wf: PyScfWavefunction,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
) -> NDArray[np.float32]: ...
def extract_esp_surface(
    wf: PyScfWavefunction,
    density_isovalue: float = 0.001,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 5.0,
    output_angstrom: bool = True,
) -> PyEspSurface: ...
def export_esp_gaussian_cube(
    wf: PyScfWavefunction,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 5.0,
    title: str = "",
) -> str: ...

def run_single_point(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rhf",
    basis: str = "sto-3g",
    max_iter: int = 100,
    conv_energy: float = 1e-10,
    conv_density: float = 1e-8,
    use_diis: bool = True,
    diis_max_vecs: int = 8,
    damping: float = 0.0,
    xc: Optional[str] = None,
    grid_level: str = "medium",
) -> PyScfWavefunction: ...

def run_gradient(
    atomic_numbers: Any,
    coords: Any,
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = "sto-3g",
    method: str = "auto",
    xc: Optional[str] = None,
    grid_level: str = "medium",
) -> NDArray[np.float64]: ...

def run_optimize(
    atomic_numbers: Any,
    coords: Any,
    basis: str = "sto-3g",
    max_steps: int = 100,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
    xc: Optional[str] = None,
    grid_level: str = "medium",
) -> PyOptResult: ...

def auto_orbital_grid_spec(
    wf: PyScfWavefunction,
    margin_bohr: float = 4.5,
    spacing_bohr: float = 0.25,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]: ...

def eval_orbital_cube_zero_copy(
    wf: PyScfWavefunction,
    mo_idx: int,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
) -> NDArray[np.float32]: ...

def eval_density_cube_zero_copy(
    wf: PyScfWavefunction,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
    spin_density: bool = False,
) -> NDArray[np.float32]: ...

def extract_isosurface_from_grid(
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    isovalue: float = 0.03,
    output_angstrom: bool = True,
    mo_index: int = -1,
    spin_channel: int = 0,
) -> PyDualOrbitalMesh: ...

def extract_single_isosurface_from_grid(
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    isovalue: float = 0.002,
    output_angstrom: bool = True,
) -> PyIsosurfaceMesh: ...

def extract_orbital_isosurface(
    wf: PyScfWavefunction,
    mo_idx: int,
    isovalue: float = 0.03,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    output_angstrom: bool = True,
) -> PyDualOrbitalMesh: ...

def extract_density_isosurface(
    wf: PyScfWavefunction,
    isovalue: float = 0.002,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    spin_density: bool = False,
    output_angstrom: bool = True,
) -> PyDualOrbitalMesh: ...

def export_orbital_gaussian_cube(
    wf: PyScfWavefunction,
    mo_idx: int,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    title: str = "",
) -> str: ...

def export_density_gaussian_cube(
    wf: PyScfWavefunction,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    spin_density: bool = False,
    title: str = "",
) -> str: ...

def format_grid_gaussian_cube(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    title: str = "",
    basis: str = "",
) -> str: ...

def parse_gaussian_cube_str(
    cube_text: str,
) -> Tuple[
    List[str],
    List[Tuple[float, float, float]],
    Tuple[float, float, float],
    Tuple[float, float, float],
    NDArray[np.float32],
]: ...

def supported_basis_sets() -> List[str]: ...
def supported_dft_functionals() -> List[str]: ...

class PyOrbitalTransition:
    @property
    def occ_index(self) -> int: ...
    @property
    def virt_index(self) -> int: ...
    @property
    def amplitude_x(self) -> float: ...
    @property
    def amplitude_y(self) -> float: ...
    @property
    def weight(self) -> float: ...

class PyExcitedState:
    @property
    def state_index(self) -> int: ...
    @property
    def manifold_index(self) -> int: ...
    @property
    def label(self) -> str: ...
    @property
    def multiplicity(self) -> int: ...
    @property
    def excitation_energy_hartree(self) -> float: ...
    @property
    def excitation_energy_ev(self) -> float: ...
    @property
    def total_energy_hartree(self) -> float: ...
    @property
    def wavelength_nm(self) -> float: ...
    @property
    def transition_dipole_au(self) -> Tuple[float, float, float]: ...
    @property
    def oscillator_strength(self) -> float: ...
    @property
    def dominant_transitions(self) -> List[PyOrbitalTransition]: ...

class PyUvVisSpectrum:
    @property
    def wavelengths_nm(self) -> List[float]: ...
    @property
    def energies_ev(self) -> List[float]: ...
    @property
    def epsilon_l_mol_cm(self) -> List[float]: ...
    @property
    def fwhm_ev(self) -> float: ...

class PyTddftResult:
    @property
    def ground_energy_hartree(self) -> float: ...
    @property
    def xc_functional(self) -> Optional[str]: ...
    @property
    def tda(self) -> bool: ...
    @property
    def n_occ(self) -> int: ...
    @property
    def n_virt(self) -> int: ...
    @property
    def homo_energy_hartree(self) -> float: ...
    @property
    def lumo_energy_hartree(self) -> float: ...
    @property
    def ks_gap_ev(self) -> float: ...
    @property
    def optical_gap_ev(self) -> Optional[float]: ...
    @property
    def lowest_triplet_ev(self) -> Optional[float]: ...
    @property
    def delta_est_ev(self) -> Optional[float]: ...
    @property
    def davidson_iterations_singlet(self) -> int: ...
    @property
    def davidson_iterations_triplet(self) -> int: ...
    @property
    def states(self) -> List[PyExcitedState]: ...
    @property
    def scf_wavefunction(self) -> PyScfWavefunction: ...
    def broaden_spectrum(
        self,
        min_nm: float = 150.0,
        max_nm: float = 800.0,
        n_points: int = 500,
        fwhm_ev: float = 0.33,
    ) -> PyUvVisSpectrum: ...

def run_tddft(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rks",
    basis: str = "6-31g",
    n_states: int = 5,
    tda: bool = False,
    manifold: str = "singlet",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    max_iter: int = 80,
    conv_tol: float = 1e-6,
    force_davidson: bool = False,
    scf_max_iter: int = 100,
    scf_conv_energy: float = 1e-10,
    scf_conv_density: float = 1e-8,
) -> PyTddftResult: ...

def run_tddft_from_scf_py(
    wf: PyScfWavefunction,
    n_states: int = 5,
    tda: bool = False,
    manifold: str = "singlet",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    max_iter: int = 80,
    conv_tol: float = 1e-6,
    force_davidson: bool = False,
) -> PyTddftResult: ...

def broaden_uv_vis(
    states: Sequence[PyExcitedState],
    min_nm: float = 150.0,
    max_nm: float = 800.0,
    n_points: int = 500,
    fwhm_ev: float = 0.33,
) -> PyUvVisSpectrum: ...

class PyNormalMode:
    mode_index: int
    frequency_cm: float
    is_imaginary: bool
    reduced_mass_amu: float
    force_constant_m_dyn_a: float
    ir_intensity_km_mol: float
    displacement: List[Tuple[float, float, float]]

class PyThermochemistry:
    temperature_k: float
    pressure_pa: float
    zpve_hartree: float
    zpve_kcal_mol: float
    e_trans_hartree: float
    e_rot_hartree: float
    e_vib_hartree: float
    e_thermal_hartree: float
    h_hartree: float
    g_hartree: float
    s_cal_mol_k: float
    s_trans_cal_mol_k: float
    s_rot_cal_mol_k: float
    s_vib_cal_mol_k: float
    s_elec_cal_mol_k: float
    cv_cal_mol_k: float
    cp_cal_mol_k: float

class PyVibrationalAnalysis:
    n_vib: int
    is_linear: bool
    principal_moments_amu_bohr2: Tuple[float, float, float]
    rotational_constants_ghz: Tuple[float, float, float]
    trans_rot_residuals: List[float]
    modes: List[PyNormalMode]

class PyFrequencyResult:
    energy_hartree: float
    symbols: List[str]
    geometry_bohr: List[Tuple[float, float, float]]
    basis_name: str
    charge: int
    multiplicity: int
    method: str
    xc_functional: Optional[str]
    vibrational: PyVibrationalAnalysis
    thermo: PyThermochemistry
    cartesian_hessian: List[List[float]]
    apt: List[List[float]]
    @property
    def frequencies_cm(self) -> List[float]: ...
    @property
    def ir_intensities_km_mol(self) -> List[float]: ...
    @property
    def zpve_hartree(self) -> float: ...
    @property
    def enthalpy_hartree(self) -> float: ...
    @property
    def gibbs_free_energy_hartree(self) -> float: ...
    @property
    def entropy_cal_mol_k(self) -> float: ...
    @property
    def heat_capacity_cv_cal_mol_k(self) -> float: ...
    @property
    def heat_capacity_cp_cal_mol_k(self) -> float: ...

def run_frequency(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rks",
    basis: str = "sto-3g",
    step_bohr: float = 0.001,
    xc: Optional[str] = None,
    grid_level: str = "medium",
    temperature_k: float = 298.15,
    pressure_pa: float = 101325.0,
    symmetry_number: Optional[int] = None,
    max_iter: int = 100,
    conv_energy: float = 1e-10,
    conv_density: float = 1e-8,
) -> PyFrequencyResult: ...

