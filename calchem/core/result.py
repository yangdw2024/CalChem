"""Result: 统一计算结果（Pydantic v2）。

所有后端返回此结构。不可用字段为 None，绝不用 0 填充。
能量单位 Hartree；坐标单位 Å；偶极 Debye；梯度 Eh/Bohr；频率 cm⁻¹。
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .request import MoleculeSpec, RESULT_SCHEMA_VERSION

JobStatus = Literal[
    "pending", "running", "succeeded", "failed", "not_converged", "cancelled"
]


class OrbitalInfo(BaseModel):
    """单个轨道信息。"""

    index: int
    energy_hartree: float
    occupation: float


class Result(BaseModel):
    """统一计算结果。"""

    schema_version: str = RESULT_SCHEMA_VERSION
    job_id: Optional[str] = None
    project_id: Optional[str] = None

    # 后端与版本
    backend: str
    backend_version: Optional[str] = None

    # 任务描述
    task: str
    method: str
    basis: str
    xc: Optional[str] = None

    # 分子快照（使用时几何）
    molecule: MoleculeSpec

    # 状态与计时
    status: JobStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    walltime_seconds: Optional[float] = None

    # 能量（Hartree）—— 不可用为 None
    scf_energy_hartree: Optional[float] = None
    mp2_correction_hartree: Optional[float] = None
    ccsd_correction_hartree: Optional[float] = None
    ccsd_t_correction_hartree: Optional[float] = None
    total_energy_hartree: Optional[float] = None

    # SCF 信息
    scf_converged: Optional[bool] = None
    scf_iterations: Optional[int] = None

    # 轨道
    homo_energy_hartree: Optional[float] = None
    lumo_energy_hartree: Optional[float] = None
    orbitals: List[OrbitalInfo] = Field(default_factory=list)

    # 性质
    dipole_debye: Optional[List[float]] = None
    dipole_magnitude_debye: Optional[float] = None
    gradients: Optional[List[List[float]]] = None
    frequencies_cm: Optional[List[float]] = None

    # 优化后的坐标（若 task=optimize）
    optimized_coordinates_angstrom: Optional[List[List[float]]] = None

    # 错误与警告
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # 可追溯文件
    raw_log_path: Optional[str] = None
    input_file_path: Optional[str] = None
    run_directory: Optional[str] = None

    # 后端特定原始数据（cclib 对象等不直接序列化）
    extra: Dict = Field(default_factory=dict)
