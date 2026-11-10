"""几何优化：BFGS 数值梯度优化器。

先用有限差分梯度 + SciPy minimize 建立可工作的优化路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.request import MoleculeSpec


@dataclass
class OptimizationStep:
    """优化一步的记录。"""
    iteration: int
    energy_hartree: float
    gradient_norm: float
    coordinates_angstrom: list[list[float]]


@dataclass
class OptimizationOutcome:
    """优化结果。"""
    converged: bool
    final_energy_hartree: float
    final_molecule: MoleculeSpec
    steps: list[OptimizationStep] = field(default_factory=list)
    message: str = ""


def optimize_geometry(
    energy_fn: Callable[[MoleculeSpec], float],
    gradient_fn: Callable[[MoleculeSpec], np.ndarray],
    initial_mol: MoleculeSpec,
    max_iter: int = 30,
    ftol: float = 1e-6,
    gtol: float = 1e-4,
    step_bohr: float = 0.001,
) -> OptimizationOutcome:
    """用 BFGS 优化分子几何。

    Args:
        energy_fn: 单点能函数
        gradient_fn: 梯度函数（Eh/Bohr）
        initial_mol: 初始几何
        max_iter: 最大步数
        ftol: 能量收敛阈值
        gtol: 梯度收敛阈值
        step_bohr: 有限差分步长（如果 gradient_fn 用数值）

    Returns:
        OptimizationOutcome
    """
    from scipy.optimize import minimize

    b2a = 0.529177210903
    natom = len(initial_mol.symbols)

    def unpack_coords(x: np.ndarray) -> MoleculeSpec:
        coords_bohr = x.reshape(natom, 3)
        coords_ang = coords_bohr * b2a
        return MoleculeSpec(
            symbols=initial_mol.symbols,
            coordinates_angstrom=coords_ang.tolist(),
            charge=initial_mol.charge,
            multiplicity=initial_mol.multiplicity,
        )

    def objective(x: np.ndarray) -> float:
        return energy_fn(unpack_coords(x))

    def jacobian(x: np.ndarray) -> np.ndarray:
        mol = unpack_coords(x)
        g = gradient_fn(mol)  # Eh/Bohr
        return g.flatten()

    # 初始坐标（Bohr）
    x0 = np.array(initial_mol.coordinates_angstrom).flatten() / b2a

    steps_log: list[OptimizationStep] = []

    def callback(xk):
        e = objective(xk)
        g = jacobian(xk)
        steps_log.append(OptimizationStep(
            iteration=len(steps_log),
            energy_hartree=e,
            gradient_norm=float(np.linalg.norm(g)),
            coordinates_angstrom=xk.reshape(natom, 3).tolist(),
        ))

    result = minimize(
        objective,
        x0,
        jac=jacobian,
        method="BFGS",
        callback=callback,
        options={"maxiter": max_iter, "gtol": gtol},
    )

    final_mol = unpack_coords(result.x)
    converged = result.success

    return OptimizationOutcome(
        converged=converged,
        final_energy_hartree=float(result.fun),
        final_molecule=final_mol,
        steps=steps_log,
        message=result.message,
    )
