"""CalculationRequest: 统一计算请求（Pydantic v2）。

所有后端、CLI、GUI、API 共用此模型。
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .units import ANGSTROM_TO_BOHR

# Schema 版本——Result/Request 兼容时递增
REQUEST_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"

TaskType = Literal["single_point", "optimize", "frequency", "tddft", "cis", "excited_states", "ts", "irc", "solvation"]
BackendName = Literal["native", "gaussian", "orca", "pyscf", "xtb", "rest"]

# 合法元素符号（H-118）
_VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

# 元素原子序数（用于电子数计算）
_ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68,
    "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}


class MoleculeSpec(BaseModel):
    """分子快照（跨后端统一表示，坐标单位 Å）。"""

    symbols: List[str] = Field(..., description="元素符号列表")
    coordinates_angstrom: List[List[float]] = Field(
        ..., description="坐标 [[x,y,z], ...]，单位 Å"
    )
    charge: int = 0
    multiplicity: int = 1

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: List[str]) -> List[str]:
        result = []
        for s in v:
            cap = s.capitalize()
            if cap not in _VALID_ELEMENTS:
                raise ValueError(f"Unknown element symbol: '{s}'")
            result.append(cap)
        if not result:
            raise ValueError("Molecule must have at least one atom")
        return result

    @field_validator("coordinates_angstrom")
    @classmethod
    def _validate_coordinates(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("Coordinates cannot be empty")
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(f"Atom {i}: coordinates must be [x,y,z], got {len(row)} values")
            for j, c in enumerate(row):
                if not isinstance(c, (int, float)) or not math.isfinite(c):
                    raise ValueError(f"Atom {i}, coord {j}: must be finite number, got {c}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self):
        if len(self.symbols) != len(self.coordinates_angstrom):
            raise ValueError(
                f"symbols count ({len(self.symbols)}) != coordinates count ({len(self.coordinates_angstrom)})"
            )
        if self.multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {self.multiplicity}")

        # 电子数 = 核电荷总数 - 电荷
        n_electrons = sum(_ATOMIC_NUMBER.get(s, 0) for s in self.symbols) - self.charge
        if n_electrons < 0:
            raise ValueError(f"Negative electron count: {n_electrons} (charge={self.charge} too high)")

        # 未成对电子数 = multiplicity - 1
        n_unpaired = self.multiplicity - 1
        if n_unpaired > n_electrons:
            raise ValueError(
                f"multiplicity={self.multiplicity} requires {n_unpaired} unpaired electrons, "
                f"but only {n_electrons} electrons"
            )
        # 闭壳层电子数必须偶数
        if n_electrons % 2 == 0 and n_unpaired % 2 != 0:
            raise ValueError(
                f"Even electron count ({n_electrons}) cannot have odd unpaired ({n_unpaired})"
            )
        if n_electrons % 2 == 1 and n_unpaired % 2 != 1:
            raise ValueError(
                f"Odd electron count ({n_electrons}) requires odd unpaired, got {n_unpaired}"
            )
        return self

    @property
    def coordinates_bohr(self) -> List[List[float]]:
        return [[c * ANGSTROM_TO_BOHR for c in row] for row in self.coordinates_angstrom]

    @property
    def natom(self) -> int:
        return len(self.symbols)


class ResourceLimits(BaseModel):
    """资源限制。全局 heavy_job_concurrency 固定为 1，此处仅控制单任务资源。"""

    max_cpu_cores: int = Field(default=8, ge=1, le=64)
    max_memory_mb: int = Field(default=12000, ge=512)
    timeout_seconds: int = Field(default=3600, ge=10)


class CalculationRequest(BaseModel):
    """统一计算请求。"""

    schema_version: str = REQUEST_SCHEMA_VERSION
    job_id: Optional[str] = None
    project_id: Optional[str] = None
    task: TaskType
    molecule: MoleculeSpec
    backend: BackendName
    method: str = Field(..., description="如 rhf, uhf, b3lyp, wb97xd, mp2, ccsd(t)")
    basis: str = Field(..., description="如 sto-3g, def2-svp, 6-31g(d)")
    xc: Optional[str] = Field(None, description="DFT 泛函（method=dft 时使用）")
    solvent: Optional[str] = Field(None, description="连续介质溶剂名称，如 water, chlorobenzene, thf, toluene")
    solvent_model: Optional[str] = Field("cpcm", description="溶剂模型，如 cpcm, cosmo, iefpcm")
    options: Dict = Field(default_factory=dict, description="后端特定选项")
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
