"""Type stub for the Rust extension calchem._core (crates/calchem-py).

Entry points:
- Closed-shell RHF and open-shell UHF single-point calculations
  (STO-3G, 6-31G, 6-31G(d)/6-31G*, 6-31G(d,p)/6-31G**, LANL2DZ, LANL2DZdp;
   including heavy halogens Br/I and scalar Effective Core Potentials)
- Analytical RHF and UHF energy gradients (including d/f shells and ECP gradients)
- BFGS geometry optimization (neutral and charged/open-shell species)
- Electric dipole moments (valence-consistent with ECP effective core charges)
- Zero-copy orbital cube generation
"""
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

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
    energy_history: List[float]
    is_uhf: bool
    s_squared: Optional[float]
    n_alpha: int
    n_beta: int
    mo_energies_alpha: Optional[List[float]]
    mo_energies_beta: Optional[List[float]]
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
) -> np.ndarray: ...

def run_optimize(
    atomic_numbers: Any,
    coords: Any,
    basis: str = "sto-3g",
    max_steps: int = 100,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
) -> PyOptResult: ...

def eval_orbital_cube_zero_copy(
    wf: PyScfWavefunction,
    mo_idx: int,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
) -> np.ndarray: ...

def supported_basis_sets() -> List[str]: ...
