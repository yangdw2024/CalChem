"""
CalChem SCF 模块。

包含:
- RHF/UHF/ROHF 求解器
- DIIS/ADIIS/EDIIS 收敛加速
- 收敛判据检查
"""

from __future__ import annotations

from .hf import HF, HFMethod, RHF, UHF
from .diis import DIIS, ADIIS, EDIIS, ConvergenceChecker

__all__ = [
    'RHF',
    'UHF',
    'DIIS',
    'ADIIS',
    'EDIIS',
    'ConvergenceChecker',
]