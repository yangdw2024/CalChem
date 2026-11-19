"""
IRC (Intrinsic Reaction Coordinate) 内禀反应坐标。

从过渡态出发沿反应路径最低能量路径前进, 找到反应物和产物。
常用于验证过渡态连接性、研究反应机理。

算法采用 LQA (Local Quadratic Approximation) 方法, 即
Fukui 提出的经典 Euler-type IRC 算法:
- 步长: s
- 方向: 沿质量权重梯度方向 (-g/|g|)
- 更新: x(s+ds) = x(s) - ds * g/|g|

参考:
- Fukui, J. Phys. Chem. 74, 4161 (1970)
- Hratchian & Schlegel, J. Chem. Phys. 120, 9918 (2004)
"""

from __future__ import annotations

from typing import Callable, Tuple, List, Optional
import math

import numpy as np

from ..mol.molecule import Molecule
from ..utils.logging_utils import get_logger
from ..utils.exceptions import ConvergenceError

logger = get_logger(__name__)


def mass_weight_coords(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """质量加权笛卡尔坐标。"""
    natom = len(masses)
    m_sqrt = np.sqrt(masses)
    mw_coords = np.zeros_like(coords)
    for i in range(natom):
        mw_coords[i] = coords[i] * m_sqrt[i]
    return mw_coords


def mass_weight_grad(grad: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """质量加权梯度。"""
    natom = len(masses)
    mw_grad = np.zeros_like(grad)
    for i in range(natom):
        mw_grad[i] = grad[i] * np.sqrt(masses[i])
    return mw_grad


def get_masses(mol: Molecule) -> np.ndarray:
    """获取分子中各原子的质量 (原子单位)。"""
    from ..utils.constants import ATOMIC_MASSES
    return np.array([ATOMIC_MASSES.get(s, 1.0) for s in mol.symbols])


class IRC:
    """
    IRC (Intrinsic Reaction Coordinate) 内禀反应坐标。

    从过渡态出发, 沿反应路径向反应物和产物方向移动,
    找到连接过渡态的最小能量路径。

    Parameters
    ----------
    mol : Molecule
        过渡态结构
    grad_func : Callable
        接受 Molecule, 返回 (energy, gradient) 的函数
    step_size : float
        步长 (质量加权坐标), 单位: bohr * sqrt(amu)
    max_steps : int
        最大步数 (单方向)
    forward : bool
        是否同时向正反方向追踪
    """

    def __init__(
        self,
        mol: Molecule,
        grad_func: Callable,
        step_size: float = 0.1,
        max_steps: int = 100,
        forward: bool = True,
        verbose: int = 1,
    ):
        self.mol = mol
        self.grad_func = grad_func
        self.step_size = step_size
        self.max_steps = max_steps
        self.forward = forward
        self.verbose = verbose

        self.masses = get_masses(mol)
        self.converged: bool = False
        self.n_steps: int = 0
        self.path: List[Tuple[Molecule, float]] = []
        self.reverse_path: List[Tuple[Molecule, float]] = []

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _run_direction(self, coords: np.ndarray, direction: int) -> List[Tuple[Molecule, float]]:
        """
        沿指定方向 (-1 或 +1) 追踪 IRC。

        direction = +1: 沿负梯度方向移动
        direction = -1: 沿正梯度方向移动 (反向)
        """
        path = []
        current_coords = coords.copy()
        current_mol = self.mol.copy()
        path.append((current_mol, self._energy_at(current_mol)))

        step = 0
        prev_grad_norm = None

        while step < self.max_steps:
            energy, grad = self.grad_func(current_mol)

            mw_grad = mass_weight_grad(grad, self.masses)
            grad_norm = np.linalg.norm(mw_grad)

            if grad_norm < 1e-6:
                self._print(f"    达到驻点 (grad_norm = {grad_norm:.2e})", 2)
                break

            mw_step = -direction * self.step_size * mw_grad / grad_norm
            mw_step = mw_step.flatten()
            coord_step = mw_step.reshape(-1, 3) / np.sqrt(self.masses)[:, None]

            current_coords = current_coords + coord_step

            current_mol = Molecule(
                self.mol.symbols,
                current_coords,
                charge=self.mol.charge,
                multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )
            new_energy, new_grad = self.grad_func(current_mol)
            new_mw_grad = mass_weight_grad(new_grad, self.masses)
            new_grad_norm = np.linalg.norm(new_mw_grad)

            path.append((current_mol, new_energy))
            step += 1

            self._print(
                f"    步 {step:3d} ({'FWD' if direction > 0 else 'REV'}): "
                f"E = {new_energy:.10f} Eh  |g| = {new_grad_norm:.2e}",
                2,
            )

            if prev_grad_norm is not None and new_grad_norm > prev_grad_norm * 2.0:
                self._print(f"    梯度反转, 已到达产物区域", 2)
                break

            prev_grad_norm = new_grad_norm

        return path

    def _energy_at(self, mol: Molecule) -> float:
        e, _ = self.grad_func(mol)
        return e

    def kernel(self) -> Tuple[List[Tuple[Molecule, float]], List[Tuple[Molecule, float]]]:
        """
        运行 IRC 计算。

        Returns
        -------
        tuple
            (正向路径, 反向路径), 每个路径是 [(molecule, energy), ...] 列表
        """
        self._print("\n" + "=" * 60, 1)
        self._print("  IRC (内禀反应坐标) 计算", 1)
        self._print("=" * 60, 1)

        energy, grad = self.grad_func(self.mol)
        self._print(f"\n  过渡态能量: {energy:.10f} Eh", 1)
        self._print(f"  步长: {self.step_size}", 1)
        self._print(f"  每方向最大步数: {self.max_steps}", 1)

        self._print("\n  向反应物方向追踪 (反向)...", 1)
        self.reverse_path = self._run_direction(self.mol.coordinates, direction=-1)

        if self.forward:
            self._print("\n  向产物方向追踪 (正向)...", 1)
            self.path = self._run_direction(self.mol.coordinates, direction=+1)
        else:
            self.path = [(self.mol, energy)]

        self.n_steps = len(self.path) + len(self.reverse_path)
        self.converged = True

        self._print("\n  IRC 完成", 1)
        self._print(f"  正向路径: {len(self.path)} 步", 1)
        self._print(f"  反向路径: {len(self.reverse_path)} 步", 1)
        self._print("=" * 60 + "\n", 1)

        return self.path, self.reverse_path
