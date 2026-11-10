"""
Grimme D3 色散校正。

实现 Grimme DFT-D3 色散校正 (zero damping)。
参考: Grimme et al., J. Chem. Phys. 132, 154104 (2010)

E_disp = -Σ_{AB} Σ_{n=6,8} s_n * C_n^AB / R_AB^n * f_damp(R_AB)
"""

from __future__ import annotations

from typing import Dict, Tuple
import math
import numpy as np


# D3 零阻尼参数 (s6, s8, a1, a2)
# 常见泛函的 D3 参数 (zero damping)
D3_ZERO_PARAMS = {
    'b3lyp': (1.0, 1.703, 0.3981, 4.4211),
    'pbe': (1.0, 0.949, 0.3908, 3.9251),
    'blyp': (1.0, 1.682, 0.4298, 4.2359),
    'b88': (1.0, 1.580, 0.4422, 3.9060),
    'bp86': (1.0, 1.137, 0.3847, 4.0642),
    'pw91': (1.0, 0.812, 0.3679, 3.8303),
    'tpss': (1.0, 1.106, 0.3840, 4.0522),
    'b97': (1.0, 1.079, 0.3830, 3.8988),
    'pbe0': (1.0, 0.901, 0.3754, 4.0633),
    'b2plyp': (0.64, 1.027, 0.3065, 4.4821),
    'hf': (1.0, 1.155, 0.3385, 2.8830),
}


# C6 系数 (原子对的平均 C6, J*nm^6/mol)
# 参考: Grimme et al., J. Chem. Phys. 132, 154104 (2010)
# 注意: 这些是元素平均 C6 系数 (简化版本)
# 完整的 D3 会根据原子的配位数和杂化状态调整 C6
C6_ELEMENT = {
    'H': 6.50,
    'He': 1.62,
    'Li': 73.2,
    'Be': 65.8,
    'B': 99.5,
    'C': 105.0,
    'N': 87.9,
    'O': 66.1,
    'F': 46.2,
    'Ne': 28.6,
    'Na': 151.0,
    'Mg': 138.0,
    'Al': 236.0,
    'Si': 242.0,
    'P': 231.0,
    'S': 196.0,
    'Cl': 161.0,
    'Ar': 119.0,
    'K': 269.0,
    'Ca': 249.0,
    'Br': 249.0,
    'I': 385.0,
}


# 原子范德华半径 (Bohr)
# 用于阻尼函数
VDW_RADII = {
    'H': 2.02,
    'He': 1.58,
    'Li': 3.36,
    'Be': 3.20,
    'B': 3.02,
    'C': 2.91,
    'N': 2.82,
    'O': 2.75,
    'F': 2.68,
    'Ne': 2.54,
    'Na': 3.70,
    'Mg': 3.62,
    'Al': 3.68,
    'Si': 3.62,
    'P': 3.55,
    'S': 3.48,
    'Cl': 3.42,
    'Ar': 3.33,
    'K': 4.23,
    'Ca': 4.08,
    'Br': 3.55,
    'I': 3.78,
}


def get_c6_ab(sym_a: str, sym_b: str) -> float:
    """
    计算原子对的 C6 色散系数。
    
    使用组合规则: C6_AB = 2 * C6_A * C6_B / (C6_A + C6_B)
    单位: J*nm^6/mol -> 转换为 Eh*bohr^6

    转换因子: 1 J*nm^6/mol = 1.507706999e-3 Eh*a0^6
    """
    c6_a = C6_ELEMENT.get(sym_a, 2.0)
    c6_b = C6_ELEMENT.get(sym_b, 2.0)

    if c6_a == 0 or c6_b == 0:
        return 0.0

    c6_ab = 2.0 * c6_a * c6_b / (c6_a + c6_b)

    conv = 1.507706999e-3
    return c6_ab * conv


def get_c8_ab(c6_ab: float, sym_a: str, sym_b: str) -> float:
    """
    从 C6 估计 C8 系数。
    
    使用近似: C8 ≈ 3 * C6 * sqrt(R_A^vdW * R_B^vdW)
    参考: Grimme 2004, 2010
    """
    r_a = VDW_RADII.get(sym_a, 3.0)
    r_b = VDW_RADII.get(sym_b, 3.0)
    r_ab = math.sqrt(r_a * r_b)
    return 3.0 * c6_ab * r_ab * r_ab


def compute_d3_energy(
    symbols: list,
    coordinates: np.ndarray,
    functional: str = 'pbe',
    s6: float = None,
    s8: float = None,
    a1: float = None,
    a2: float = None,
    damping: str = 'zero',
) -> float:
    """
    计算 D3 色散能。默认使用纯 Rust 原生高性能引擎。
    """
    if s6 is None and s8 is None and a1 is None and a2 is None:
        try:
            from .. import _core as rust_core
            coords_list = [list(r) for r in coordinates]
            res = rust_core.run_d3_dispersion(list(symbols), coords_list, functional, damping)
            return res.energy_hartree
        except Exception:
            pass

    func_lower = functional.lower()
    if s6 is None or s8 is None or a1 is None or a2 is None:
        params = D3_ZERO_PARAMS.get(func_lower, D3_ZERO_PARAMS['pbe'])
        s6 = s6 if s6 is not None else params[0]
        s8 = s8 if s8 is not None else params[1]
        a1 = a1 if a1 is not None else params[2]
        a2 = a2 if a2 is not None else params[3]

    n_atoms = len(symbols)
    e_disp = 0.0

    for i in range(n_atoms):
        sym_i = symbols[i]
        r_i = coordinates[i]
        for j in range(i + 1, n_atoms):
            sym_j = symbols[j]
            r_j = coordinates[j]

            rij = np.linalg.norm(r_i - r_j)
            if rij < 0.1:
                continue

            c6 = get_c6_ab(sym_i, sym_j)
            c8 = get_c8_ab(c6, sym_i, sym_j)

            if c6 == 0 or c8 == 0:
                continue

            r0 = math.sqrt(c8 / c6)

            if r0 < 0.1:
                continue

            ratio6 = (rij / r0) ** 6
            ratio8 = (rij / r0) ** 8

            f6 = 1.0 / (1.0 + 6.0 / ratio6)
            f8 = 1.0 / (1.0 + 6.0 / ratio8)

            e6 = -s6 * c6 / (rij ** 6) * f6
            e8 = -s8 * c8 / (rij ** 8) * f8

            e_disp += e6 + e8

    return e_disp


def compute_d3_gradient(
    symbols: list,
    coordinates: np.ndarray,
    functional: str = 'pbe',
    s6: float = None,
    s8: float = None,
    a1: float = None,
    a2: float = None,
    damping: str = 'zero',
) -> np.ndarray:
    """
    计算 D3 色散能的梯度。优先使用纯 Rust 原生解析核梯度。
    """
    if s6 is None and s8 is None and a1 is None and a2 is None:
        try:
            from .. import _core as rust_core
            coords_list = [list(r) for r in coordinates]
            res = rust_core.run_d3_dispersion(list(symbols), coords_list, functional, damping)
            return np.array(res.gradient, dtype=np.float64)
        except Exception:
            pass

    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3))
    step = 1e-5

    e0 = compute_d3_energy(symbols, coordinates, functional, s6, s8, a1, a2, damping)

    for i in range(n_atoms):
        for d in range(3):
            coords_plus = coordinates.copy()
            coords_plus[i, d] += step
            e_plus = compute_d3_energy(symbols, coords_plus, functional, s6, s8, a1, a2, damping)

            coords_minus = coordinates.copy()
            coords_minus[i, d] -= step
            e_minus = compute_d3_energy(symbols, coords_minus, functional, s6, s8, a1, a2, damping)

            grad[i, d] = (e_plus - e_minus) / (2.0 * step)

    return grad
