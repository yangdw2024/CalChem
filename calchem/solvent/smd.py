"""SMD (Solvation Model based on Density; Marenich-Cramer-Truhlar 2009) 溶剂模型模块。"""

from __future__ import annotations

from typing import Any

from .pcm import PCM


class SMD:
    """SMD 溶剂化模型接口（IEF-PCM 静电 + Marenich 2009 CDS 非静电项）。"""

    def __init__(self, method_or_mf: Any, solvent: str = "water"):
        self.mf = method_or_mf
        self.solvent = solvent.lower()
        self.e_solv_kcal_mol: float = 0.0
        self.pcm = PCM(self.mf, solvent=self.solvent, solvent_model="smd", include_cds=True)

    def kernel(self, **kwargs: Any) -> float:
        self.e_solv_kcal_mol = self.pcm.kernel(**kwargs)
        return self.e_solv_kcal_mol

    @property
    def e_solv(self) -> float:
        """Delta G_solv (kcal/mol)。"""
        if self.e_solv_kcal_mol == 0.0 and self.pcm.solver is None:
            self.kernel()
        return self.e_solv_kcal_mol

    def __repr__(self) -> str:
        return f"SMD(solvent='{self.solvent}', e_solv_kcal_mol={self.e_solv_kcal_mol:.4f})"
