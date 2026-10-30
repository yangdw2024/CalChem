from __future__ import annotations
from typing import Optional
import numpy as np
from ..mol.molecule import Molecule
from ..scf.hf import HFMethod, RHF, UHF
from ..utils.logging_utils import get_logger
from .analytical import compute_hf_gradient

logger = get_logger(__name__)

class HFGradient:
    def __init__(self, hf_method: HFMethod, numerical: bool = False, step: float = 1e-5):
        self.hf_method = hf_method
        self.numerical = numerical
        self.step = step
        self.gradient: Optional[np.ndarray] = None

    def kernel(self) -> np.ndarray:
        if not getattr(self.hf_method, "converged", False):
            self.hf_method.kernel()
        if not self.hf_method.converged:
            raise RuntimeError("SCF did not converge; refusing to compute gradient.")
        if self.numerical:
            self.gradient = self._numerical_gradient_central()
        else:
            self.gradient = compute_hf_gradient(self.hf_method)
        return self.gradient

    def _numerical_gradient_central(self) -> np.ndarray:
        mol = self.hf_method.mol
        coords0 = mol.coordinates.copy()
        n_atoms = mol.natom
        grad = np.zeros((n_atoms, 3), dtype=np.float64)
        basis = getattr(self.hf_method, "basis", "sto-3g")
        is_uhf = isinstance(self.hf_method, UHF)
        h = self.step
        for i in range(n_atoms):
            for j in range(3):
                def _eval(delta: float) -> float:
                    c = coords0.copy()
                    c[i, j] += delta
                    m = Molecule(mol.symbols, c, mol.charge, mol.multiplicity, units="bohr", validate_overlap=False)
                    calc = UHF(m, basis=basis, energy_tol=1e-12, density_tol=1e-10) if is_uhf else RHF(m, basis=basis, energy_tol=1e-12, density_tol=1e-10)
                    return calc.kernel()
                grad[i, j] = (_eval(h) - _eval(-h)) / (2.0 * h)
        return grad
