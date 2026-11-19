"""外部程序输出解析器基类。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ParsedResult:
    """解析后的外部程序结果。"""
    normal_termination: bool = False
    return_code: int = 0
    scf_converged: bool = False
    scf_iterations: Optional[int] = None
    scf_energy_hartree: Optional[float] = None
    total_energy_hartree: Optional[float] = None
    correlation_energy_hartree: Optional[float] = None
    method_actual: Optional[str] = None
    dipole_magnitude_debye: Optional[float] = None
    dipole_debye: Optional[list[float]] = None
    homo_energy_hartree: Optional[float] = None
    lumo_energy_hartree: Optional[float] = None
    optimized_coordinates_angstrom: Optional[list[list[float]]] = None
    optimization_converged: bool = False
    frequencies_cm: Optional[list[float]] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_last_scf_done(text: str) -> Optional[float]:
    """找到最后一个 SCF Done 能量。"""
    matches = re.findall(r"SCF Done:\s*([A-Za-z]+)\s*=\s*([-\d.]+)", text)
    if matches:
        return float(matches[-1][1])
    return None


def find_final_energy(text: str, method: str) -> Optional[float]:
    """根据方法找最终能量（不是 SCF 能）。"""
    m = method.lower()
    # MP2 total energy
    if m == "mp2":
        matches = re.findall(r"MP2\s+=\s*([-\d.]+)", text)
        if matches:
            return float(matches[-1])
    # CCSD(T) total
    if m in ("ccsd", "ccsd(t)"):
        matches = re.findall(r"(?:CCSD\(T\)|CCSD)\s+=\s*([-\d.]+)", text)
        if matches:
            return float(matches[-1])
    return None
