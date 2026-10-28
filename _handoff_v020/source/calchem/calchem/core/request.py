"""CalculationRequest: 统一计算请求（Pydantic v2）。

所有后端、CLI、GUI、API 共用此模型。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .units import ANGSTROM_TO_BOHR

# Schema 版本——Result/Request 兼容时递增
REQUEST_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"

TaskType = Literal["single_point", "optimize", "frequency"]
BackendName = Literal["native", "gaussian", "orca", "pyscf", "xtb", "rest"]


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
    def _capitalize(cls, v: List[str]) -> List[str]:
        return [s.capitalize() for s in v]

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
    options: Dict = Field(default_factory=dict, description="后端特定选项")
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
