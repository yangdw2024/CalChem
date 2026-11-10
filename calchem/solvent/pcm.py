"""PCM (Polarizable Continuum Model) 溶剂模型模块。

纯 Rust 原生高性能 C-PCM / COSMO / IEF-PCM 自洽反应场与溶剂化自由能求解器。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .. import _core
from ..core.units import HARTREE_TO_KCAL_MOL


class PCMSolver:
    """纯 Rust 驱动的 PCM / COSMO 溶剂化求解器。"""

    def __init__(
        self,
        mol: Any,
        basis_set: Any = None,
        solvent: str = "water",
        solvent_model: str = "cpcm",
        scaling_factor: float = 1.20,
        include_cds: bool = True,
    ):
        self.mol = mol
        self.basis_name = str(basis_set.name if hasattr(basis_set, "name") else (basis_set or "sto-3g"))
        self.solvent = solvent.lower()
        self.solvent_model = solvent_model.lower()
        self.scaling_factor = float(scaling_factor)
        self.include_cds = bool(include_cds)
        self.result: Optional[_core.SolvationResult] = None

    def kernel(self, method: str = "b3lyp", xc: Optional[str] = "b3lyp") -> float:
        """执行溶剂化计算并返回溶剂化自由能 (kcal/mol)。"""
        symbols = list(self.mol.symbols)
        if hasattr(self.mol, "coordinates_bohr"):
            coords = self.mol.coordinates_bohr
        elif hasattr(self.mol, "coordinates"):
            # coordinates are in Angstrom or Bohr: if Angstrom convert to Bohr
            from ..core.units import ANGSTROM_TO_BOHR
            coords = (np.asarray(self.mol.coordinates) * ANGSTROM_TO_BOHR).tolist()
        else:
            coords = [[0.0, 0.0, 0.0] for _ in symbols]

        charge = getattr(self.mol, "charge", 0)
        multiplicity = getattr(self.mol, "multiplicity", 1)

        self.result = _core.run_solvation(
            symbols=symbols,
            coords=coords,
            basis_name=self.basis_name,
            solvent=self.solvent,
            solvent_model=self.solvent_model,
            charge=charge,
            multiplicity=multiplicity,
            method=method,
            xc=xc,
            scaling_factor=self.scaling_factor,
            include_cds=self.include_cds,
        )
        return self.result.delta_g_solv_kcal_mol

    def compute_solvation_energy(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.delta_g_solv_kcal_mol

    @property
    def delta_g_solv_kcal_mol(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.delta_g_solv_kcal_mol

    @property
    def delta_g_elec_kcal_mol(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.delta_g_elec_kcal_mol

    @property
    def delta_g_cds_kcal_mol(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.delta_g_cds_kcal_mol

    @property
    def solvated_energy(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.solvated_energy

    @property
    def gas_energy(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.gas_energy

    @property
    def surface_charges(self) -> np.ndarray:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return np.asarray(self.result.surface_charges)

    @property
    def cavity_surface_area_angstrom2(self) -> float:
        if self.result is None:
            self.kernel()
        assert self.result is not None
        return self.result.cavity_surface_area_angstrom2


class PCM:
    """PCM 溶剂模型接口封装。"""

    def __init__(
        self,
        mf: Any,
        solvent: str = "water",
        solvent_model: str = "cpcm",
        scaling_factor: float = 1.20,
        include_cds: bool = True,
    ):
        self.mf = mf
        self.solvent = solvent
        self.solvent_model = solvent_model
        self.scaling_factor = scaling_factor
        self.include_cds = include_cds
        self.solver: Optional[PCMSolver] = None

    def build(self) -> PCMSolver:
        mol = getattr(self.mf, "mol", self.mf)
        basis = getattr(self.mf, "basis", getattr(self.mf, "basis_set", "sto-3g"))
        self.solver = PCMSolver(
            mol,
            basis_set=basis,
            solvent=self.solvent,
            solvent_model=self.solvent_model,
            scaling_factor=self.scaling_factor,
            include_cds=self.include_cds,
        )
        return self.solver

    def compute_solvation_energy(self) -> float:
        if self.solver is None:
            self.build()
        assert self.solver is not None
        method = getattr(self.mf, "method", "b3lyp")
        xc = getattr(self.mf, "xc", "b3lyp")
        return self.solver.kernel(method=method, xc=xc)

    def kernel(self) -> float:
        return self.compute_solvation_energy()


def pcm(mf: Any, solvent: str = "water", solvent_model: str = "cpcm") -> PCM:
    """便捷 PCM 计算接口。"""
    pcm_obj = PCM(mf, solvent=solvent, solvent_model=solvent_model)
    pcm_obj.kernel()
    return pcm_obj
