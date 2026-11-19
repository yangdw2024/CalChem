"""
HF 梯度计算模块。

提供 RHF 和 UHF 的梯度计算，支持数值梯度（中心差分）和解析梯度。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..mol.molecule import Molecule
from ..scf.hf import HFMethod, RHF, UHF
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HFGradient:
    """HF 梯度计算类。"""

    def __init__(self, hf_method: HFMethod, numerical: bool = True, step: float = 1e-5):
        self.hf_method = hf_method
        self.numerical = numerical
        self.step = step
        self.gradient: Optional[np.ndarray] = None

    def kernel(self) -> np.ndarray:
        """计算梯度。"""
        if self.numerical:
            self.gradient = self._numerical_gradient()
        else:
            self.gradient = self._analytical_gradient()

        logger.info(f"HF gradient calculated, |g| = {np.linalg.norm(self.gradient):.6f}")
        return self.gradient

    def _numerical_gradient(self) -> np.ndarray:
        """数值梯度（中心差分）。"""
        mol = self.hf_method.mol
        coords = mol.coordinates.copy()
        n_atoms = mol.natom
        grad = np.zeros((n_atoms, 3))

        basis = getattr(self.hf_method, 'basis_name', 'sto-3g')
        is_rhf = isinstance(self.hf_method, RHF)

        for i in range(n_atoms):
            for j in range(3):
                coords[i, j] += self.step
                mol_plus = Molecule(mol.symbols, coords, mol.charge, mol.multiplicity, units='bohr')
                if is_rhf:
                    calc_plus = RHF(mol_plus, basis=basis, verbose=0)
                else:
                    calc_plus = UHF(mol_plus, basis=basis, verbose=0)
                e_plus = calc_plus.kernel()

                coords[i, j] -= 2 * self.step
                mol_minus = Molecule(mol.symbols, coords, mol.charge, mol.multiplicity, units='bohr')
                if is_rhf:
                    calc_minus = RHF(mol_minus, basis=basis, verbose=0)
                else:
                    calc_minus = UHF(mol_minus, basis=basis, verbose=0)
                e_minus = calc_minus.kernel()

                coords[i, j] += self.step
                grad[i, j] = (e_plus - e_minus) / (2 * self.step)

        return grad

    def _analytical_gradient(self) -> np.ndarray:
        """解析梯度。"""
        from .analytical import compute_hf_gradient
        return compute_hf_gradient(self.hf_method)

    def __repr__(self) -> str:
        return f"HFGradient(hf_method={self.hf_method})"
