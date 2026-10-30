from typing import List, Optional
import numpy as np

class PyScfWavefunction:
    converged: bool
    iterations: int
    total_energy_hartree: float
    scf_energy_hartree: float

def run_single_point(
    atomic_numbers: List[int],
    coords_bohr: np.ndarray,
    charge: int,
    multiplicity: int,
    method: str,
    basis: str,
    xc: Optional[str] = None,
    use_ri: bool = True,
    max_threads: int = 8,
) -> PyScfWavefunction: ...

def eval_orbital_cube_zero_copy(
    orbital_index: int,
    origin_bohr: List[float],
    spacing_bohr: List[float],
    dims: List[int],
) -> np.ndarray: ...
