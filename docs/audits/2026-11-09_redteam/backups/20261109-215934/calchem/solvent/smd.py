"""SMD (Solvation Model based on Density) 溶剂模型模块。

纯 Rust 原生高性能 C-PCM / SMD 自洽反应场与全溶剂化能求解器。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .pcm import PCMSolver, PCM


class SMD:
    """SMD 溶剂化模型接口。"""

    def __init__(self, method_or_mf: Any, solvent: str = "water"):
        self.mf = method_or_mf
        self.solvent = solvent.lower()
        self.e_solv_kcal_mol: float = 0.0
        self.pcm = PCM(self.mf, solvent=self.solvent, solvent_model="cpcm", include_cds=True)

    def kernel(self) -> float:
        """执行 SMD 溶剂化计算，返回 Delta G_solv (kcal/mol)。"""
        self.e_solv_kcal_mol = self.pcm.kernel()
        return self.e_solv_kcal_mol

    @property
    def e_solv(self) -> float:
        """Delta G_solv (kcal/mol)。"""
        if self.e_solv_kcal_mol == 0.0 and self.pcm.solver is None:
            self.kernel()
        return self.e_solv_kcal_mol

    def __repr__(self) -> str:
        return f"SMD(solvent='{self.solvent}', delta_g_solv={self.e_solv:.2f} kcal/mol)"
