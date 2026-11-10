"""Type stub for the Rust extension calchem._core (crates/calchem-py).

Only what the extension actually exports is declared here.
Stage 3 entry points (analytic gradient, optimisation, orbital cubes) do not exist yet.
"""
from typing import List, Optional, Sequence

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
    @property
    def homo_energy_hartree(self) -> Optional[float]: ...
    @property
    def lumo_energy_hartree(self) -> Optional[float]: ...

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

def supported_basis_sets() -> List[str]: ...
