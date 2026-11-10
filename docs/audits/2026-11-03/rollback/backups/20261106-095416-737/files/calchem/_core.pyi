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
"""
from typing import Any, List, Optional, Sequence, Tuple
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
) -> PyScfWavefunction: ...

def run_gradient(
    atomic_numbers: Any,
    coords: Any,
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = "sto-3g",
    method: str = "auto",
) -> NDArray[np.float64]: ...

def run_optimize(
    atomic_numbers: Any,
    coords: Any,
    basis: str = "sto-3g",
    max_steps: int = 100,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
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
