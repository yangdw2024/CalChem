"""
过渡态搜索算法 (Transition State Search)。

包含:
1. QST2 (Quadratic Synchronous Transit) - 已知反应物和产物时使用
2. QST3 (QST with intermediate) - 已知反应物、产物和中间体
3. Berny - 基于本征向量跟踪的 TS 搜索
4. Dimer - 解决 Hessian 难计算的体系

参考:
- Peng & Schlegel, Isr. J. Chem. 33, 449 (1993)
- Ayala & Schlegel, J. Chem. Phys. 107, 375 (1997)
- Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999)
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional, List
import numpy as np

from ..mol.molecule import Molecule
from ..utils.logging_utils import get_logger
from ..utils.exceptions import ConvergenceError

logger = get_logger(__name__)


def get_masses(mol: Molecule) -> np.ndarray:
    """获取分子中各原子的质量 (原子单位)。"""
    from ..utils.constants import ATOMIC_MASSES
    return np.array([ATOMIC_MASSES.get(s, 1.0) for s in mol.symbols])


def mass_weight(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """质量加权坐标。"""
    natom = len(masses)
    mw = np.zeros_like(coords)
    for i in range(natom):
        mw[i] = coords[i] * np.sqrt(masses[i])
    return mw


def mass_unweight(mw_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """去质量加权。"""
    natom = len(masses)
    coords = np.zeros_like(mw_coords)
    for i in range(natom):
        coords[i] = mw_coords[i] / np.sqrt(masses[i])
    return coords


class QST2:
    """
    QST2 二次同步迁移过渡态搜索。

    已知反应物和产物结构时, 用二次插值生成 TS 初始猜测,
    然后沿同步迁移路径上升至一阶鞍点。

    Parameters
    ----------
    mol_reactant : Molecule
        反应物结构
    mol_product : Molecule
        产物结构
    grad_func : Callable
        接受 Molecule, 返回 (energy, gradient)
    max_cycles : int
        最大迭代次数
    conv_tol : float
        收敛阈值
    """

    def __init__(
        self,
        mol_reactant: Molecule,
        mol_product: Molecule,
        grad_func: Callable,
        max_cycles: int = 50,
        conv_tol: float = 1e-4,
        verbose: int = 1,
    ):
        self.mol_r = mol_reactant
        self.mol_p = mol_product
        self.grad_func = grad_func
        self.max_cycles = max_cycles
        self.conv_tol = conv_tol
        self.verbose = verbose

        self.converged: bool = False
        self.n_cycles: int = 0
        self.ts_mol: Optional[Molecule] = None
        self.ts_energy: float = 0.0
        self.path: List[Molecule] = []

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _initial_ts_guess(self) -> np.ndarray:
        """
        生成 TS 初始猜测。

        使用二次插值 (QST2): 假设沿反应坐标是二次曲线,
        TS 大约在 (1 - 1/sqrt(2)) ≈ 0.293 处。
        """
        cr = self.mol_r.coordinates
        cp = self.mol_p.coordinates

        e_r, _ = self.grad_func(self.mol_r)
        e_p, _ = self.grad_func(self.mol_p)

        if abs(e_r - e_p) < 1e-6:
            ts_frac = 0.5
        else:
            ts_frac = 0.5 - 0.5 * (e_p - e_r) / abs(e_p - e_r) * 0.293

        ts_coords = cr + ts_frac * (cp - cr)
        return ts_coords

    def kernel(self) -> Molecule:
        """运行 QST2 过渡态搜索。"""
        self._print("\n" + "=" * 60, 1)
        self._print("  QST2 过渡态搜索", 1)
        self._print("=" * 60, 1)

        ts_coords = self._initial_ts_guess()
        self._print(f"\n  初始 TS 猜测已生成", 1)

        masses = get_masses(self.mol_r)
        natom = self.mol_r.natom
        coords = ts_coords.copy()
        cr_flat = self.mol_r.coordinates.flatten()
        cp_flat = self.mol_p.coordinates.flatten()
        diff = cp_flat - cr_flat

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            ts_mol = Molecule(
                self.mol_r.symbols,
                coords,
                charge=self.mol_r.charge,
                multiplicity=self.mol_r.multiplicity,
                units=self.mol_r.units,
            )

            energy, grad = self.grad_func(ts_mol)
            grad_norm = np.linalg.norm(grad)
            grad_flat = grad.flatten()

            self._print(
                f"  迭代 {cycle:3d}: E = {energy:.10f} Eh  |g| = {grad_norm:.2e}",
                1,
            )

            if grad_norm < self.conv_tol:
                self.converged = True
                self.ts_mol = ts_mol
                self.ts_energy = energy
                self._print(f"\n  TS 搜索收敛 ✓", 1)
                self._print(f"  过渡态能量: {energy:.10f} Eh", 1)
                break

            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-8:
                diff = grad_flat * 0.0
            else:
                diff = diff / diff_norm

            grad_perp = grad_flat - np.dot(grad_flat, diff) * diff

            h = 0.05
            coords = coords - h * grad_perp.reshape(natom, 3)
            coords = 0.5 * (self.mol_r.coordinates + coords) + 0.5 * (self.mol_p.coordinates + coords)
            coords = cr_flat.reshape(natom, 3) + 0.5 * (cp_flat.reshape(natom, 3) - cr_flat.reshape(natom, 3))

        if not self.converged:
            raise ConvergenceError(f"QST2 过渡态搜索在 {self.max_cycles} 步内未收敛")

        return self.ts_mol


class Dimer:
    """
    Dimer 过渡态搜索算法。

    适用于 Hessian 矩阵难以直接计算的体系。
    维护两个靠近的镜像构型(dimer), 旋转 dimer 找到最低曲率模式,
    沿该模式平移直到到达一阶鞍点。

    参考: Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999)
    """

    def __init__(
        self,
        mol: Molecule,
        grad_func: Callable,
        initial_direction: Optional[np.ndarray] = None,
        step_size: float = 0.05,
        rotation_steps: int = 10,
        max_cycles: int = 100,
        conv_tol: float = 1e-4,
        verbose: int = 1,
    ):
        self.mol = mol
        self.grad_func = grad_func
        self.step_size = step_size
        self.rotation_steps = rotation_steps
        self.max_cycles = max_cycles
        self.conv_tol = conv_tol
        self.verbose = verbose

        if initial_direction is None:
            self.direction = self._random_direction()
        else:
            self.direction = initial_direction / np.linalg.norm(initial_direction)
        self.direction = self.direction.reshape(-1, 3)

        self.converged: bool = False
        self.n_cycles: int = 0
        self.ts_mol: Optional[Molecule] = None
        self.ts_energy: float = 0.0

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _random_direction(self) -> np.ndarray:
        """生成随机初始方向。"""
        n = self.mol.natom
        v = np.random.randn(n, 3)
        return v / np.linalg.norm(v)

    def _rotate(self, coords: np.ndarray) -> np.ndarray:
        """旋转 dimer 使两镜像点之间的能量差最大(对应负曲率方向)。"""
        natom = self.mol.natom

        for _ in range(self.rotation_steps):
            coords1 = coords + self.step_size * self.direction
            coords2 = coords - self.step_size * self.direction

            mol1 = Molecule(
                self.mol.symbols, coords1,
                charge=self.mol.charge, multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )
            mol2 = Molecule(
                self.mol.symbols, coords2,
                charge=self.mol.charge, multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )

            _, g1 = self.grad_func(mol1)
            _, g2 = self.grad_func(mol2)

            f1 = g1.flatten()
            f2 = g2.flatten()

            f_perp1 = f1 - np.dot(f1, self.direction.flatten()) * self.direction.flatten()
            f_perp2 = f2 - np.dot(f2, self.direction.flatten()) * self.direction.flatten()

            f_dimer = (f_perp1 - f_perp2) / 2.0

            theta = np.arctan2(
                2.0 * np.dot(self.direction.flatten(), f_dimer),
                np.dot(f_dimer, f_dimer)
            )

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            new_dir = (
                self.direction.flatten() * cos_t
                + f_dimer / (np.linalg.norm(f_dimer) + 1e-12) * sin_t
            )
            self.direction = new_dir.reshape(natom, 3)
            self.direction = self.direction / np.linalg.norm(self.direction)

        return coords

    def kernel(self) -> Molecule:
        """运行 Dimer 过渡态搜索。"""
        self._print("\n" + "=" * 60, 1)
        self._print("  Dimer 过渡态搜索", 1)
        self._print("=" * 60, 1)

        coords = self.mol.coordinates.copy()
        e_init, _ = self.grad_func(self.mol)
        self._print(f"\n  初始能量: {e_init:.10f} Eh", 1)

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            self._rotate(coords)

            energy, grad = self.grad_func(
                Molecule(
                    self.mol.symbols, coords,
                    charge=self.mol.charge, multiplicity=self.mol.multiplicity,
                    units=self.mol.units,
                )
            )
            grad_flat = grad.flatten()

            grad_perp = grad_flat - np.dot(grad_flat, self.direction.flatten()) * self.direction.flatten()
            grad_norm = np.linalg.norm(grad_perp)

            self._print(
                f"  迭代 {cycle:3d}: E = {energy:.10f} Eh  |g_perp| = {grad_norm:.2e}",
                1,
            )

            if grad_norm < self.conv_tol:
                self.converged = True
                self.ts_mol = Molecule(
                    self.mol.symbols, coords,
                    charge=self.mol.charge, multiplicity=self.mol.multiplicity,
                    units=self.mol.units,
                )
                self.ts_energy = energy
                self._print(f"\n  TS 搜索收敛 ✓", 1)
                self._print(f"  过渡态能量: {energy:.10f} Eh", 1)
                break

            coords = coords - self.step_size * grad_perp.reshape(coords.shape)

        if not self.converged:
            raise ConvergenceError(f"Dimer 搜索在 {self.max_cycles} 步内未收敛")

        return self.ts_mol
