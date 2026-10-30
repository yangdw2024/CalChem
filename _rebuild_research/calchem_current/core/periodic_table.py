"""元素周期表：原子序数、质量、常见同位素。"""

from __future__ import annotations

# 元素符号 → 原子序数
ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}

# 原子质量 (amu)，取最常见同位素
ATOMIC_MASS: dict[str, float] = {
    "H": 1.007825, "He": 4.002603, "Li": 7.016004, "Be": 9.012182,
    "B": 11.009305, "C": 12.000000, "N": 14.003074, "O": 15.994915,
    "F": 18.998403, "Ne": 19.992440, "Na": 22.989769, "Mg": 23.985042,
    "Al": 26.981539, "Si": 27.976927, "P": 30.973762, "S": 31.972071,
    "Cl": 34.968853, "Ar": 39.962383, "K": 38.963707, "Ca": 39.962591,
    "Fe": 55.934937, "Cu": 62.929597, "Zn": 63.929142, "Br": 78.918338,
    "I": 126.904473, "Au": 196.966569, "Pb": 207.976653,
}

# 范德华半径 (Å)，用于 GUI 键判断
VDW_RADII: dict[str, float] = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
    "Fe": 2.04, "Cu": 1.32, "Zn": 2.00, "Br": 1.85, "I": 1.98,
    "Au": 1.66, "Pb": 2.02,
}


def get_atomic_number(symbol: str) -> int:
    """获取元素原子序数。未知元素抛 KeyError。"""
    return ATOMIC_NUMBER[symbol.capitalize()]


def get_atomic_mass(symbol: str) -> float:
    """获取元素原子质量。"""
    return ATOMIC_MASS.get(symbol.capitalize(), 0.0)


def get_vdw_radius(symbol: str) -> float:
    """获取范德华半径。"""
    return VDW_RADII.get(symbol.capitalize(), 2.0)


def is_valid_element(symbol: str) -> bool:
    return symbol.capitalize() in ATOMIC_NUMBER


def total_electrons(symbols: list[str], charge: int) -> int:
    """计算总电子数。"""
    z = sum(ATOMIC_NUMBER.get(s.capitalize(), 0) for s in symbols)
    return z - charge
