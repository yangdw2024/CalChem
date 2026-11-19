"""
DFT 梯度计算模块。

提供 RKS 和 UKS 的梯度计算，使用数值梯度（中心差分）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..mol.molecule import Molecule
from ..dft.dft import RKS, UKS
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class DFTGradient:
    """DFT 梯度计算类。"""

    def __init__(self, dft_method, step: float = 1e-5):
        self.dft_method = dft_method
        self.step = step
        self.gradient: Optional[np.ndarray] = None

    def kernel(self) -> np.ndarray:
        """计算数值梯度（中心差分）。"""
        mol = self.dft_method.mol
        coords = mol.coordinates.copy()
        n_atoms = mol.natom
        grad = np.zeros((n_atoms, 3))

        basis = getattr(self.dft_method, 'basis_name', 'sto-3g')
        xc = getattr(self.dft_method, 'xc', 'pbe')
        is_rks = isinstance(self.dft_method, RKS)

        for i in range(n_atoms):
            for j in range(3):
                coords[i, j] += self.step
                mol_plus = Molecule(mol.symbols, coords, mol.charge, mol.multiplicity, units='bohr')
                if is_rks:
                    calc_plus = RKS(mol_plus, basis=basis, xc=xc, verbose=0)
                else:
                    calc_plus = UKS(mol_plus, basis=basis, xc=xc, verbose=0)
                e_plus = calc_plus.kernel()

                coords[i, j] -= 2 * self.step
                mol_minus = Molecule(mol.symbols, coords, mol.charge, mol.multiplicity, units='bohr')
                if is_rks:
                    calc_minus = RKS(mol_minus, basis=basis, xc=xc, verbose=0)
                else:
                    calc_minus = UKS(mol_minus, basis=basis, xc=xc, verbose=0)
                e_minus = calc_minus.kernel()

                coords[i, j] += self.step
                grad[i, j] = (e_plus - e_minus) / (2 * self.step)

        self.gradient = grad
        logger.info(f"DFT gradient calculated, |g| = {np.linalg.norm(grad):.6f}")
        return grad

    def __repr__(self) -> str:
        return f"DFTGradient(dft_method={self.dft_method})"
