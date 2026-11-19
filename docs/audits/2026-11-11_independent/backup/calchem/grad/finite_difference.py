"""有限差分梯度：中心差分计算数值梯度。

先验证数值梯度正确性，再考虑解析梯度 fast path。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.request import MoleculeSpec
from ..mol.molecule import Molecule


def central_gradient(
    energy_fn: Callable[[MoleculeSpec], float],
    mol_spec: MoleculeSpec,
    step_bohr: float = 0.001,
) -> np.ndarray:
    """中心差分计算能量对坐标的梯度。

    Args:
        energy_fn: 输入 MoleculeSpec 返回能量(Eh)
        mol_spec: 分子（Å）
        step_bohr: 步长（Bohr）

    Returns:
        gradient: shape (natom, 3)，单位 Eh/Bohr
    """
    natom = len(mol_spec.symbols)
    grad = np.zeros((natom, 3))
    coords = np.array(mol_spec.coordinates_angstrom)
    # Å→Bohr 转换
    b2a = 0.529177210903
    coords_bohr = coords / b2a

    for i in range(natom):
        for j in range(3):
            # 正扰动
            coords_plus = coords_bohr.copy()
            coords_plus[i, j] += step_bohr
            spec_plus = MoleculeSpec(
                symbols=mol_spec.symbols,
                coordinates_angstrom=coords_plus * b2a,
                charge=mol_spec.charge,
                multiplicity=mol_spec.multiplicity,
            )
            e_plus = energy_fn(spec_plus)

            # 负扰动
            coords_minus = coords_bohr.copy()
            coords_minus[i, j] -= step_bohr
            spec_minus = MoleculeSpec(
                symbols=mol_spec.symbols,
                coordinates_angstrom=coords_minus * b2a,
                charge=mol_spec.charge,
                multiplicity=mol_spec.multiplicity,
            )
            e_minus = energy_fn(spec_minus)

            grad[i, j] = (e_plus - e_minus) / (2 * step_bohr)

    return grad
