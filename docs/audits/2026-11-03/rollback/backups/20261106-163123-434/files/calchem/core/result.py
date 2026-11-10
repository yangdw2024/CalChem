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


class OrbitalTransitionInfo(BaseModel):
    """激发态轨道对跃迁贡献。"""

    occ_index: int
    virt_index: int
    amplitude_x: float
    amplitude_y: float = 0.0
    weight: float


class ExcitedStateInfo(BaseModel):
    """电子激发态信息。"""

    state_index: int
    manifold_index: int
    label: str
    multiplicity: int
    excitation_energy_hartree: float
    excitation_energy_ev: float
    total_energy_hartree: float
    wavelength_nm: float
    transition_dipole_au: List[float] = Field(default_factory=list)
    oscillator_strength: float
    dominant_transitions: List[OrbitalTransitionInfo] = Field(default_factory=list)


class VibrationalModeInfo(BaseModel):
    """分子振动简正模式信息。"""

    mode_index: int
    frequency_cm: float
    is_imaginary: bool = False
    reduced_mass_amu: float
    force_constant_m_dyn_a: float
    ir_intensity_km_mol: float
    displacement: List[List[float]] = Field(default_factory=list)


class IrcPointInfo(BaseModel):
    """IRC 反应路径单点信息。"""

    point_index: int
    direction: Literal["ts", "forward", "reverse"]
    path_coordinate_s: float
    energy_hartree: float
    delta_energy_kcal_mol: float
    gradient_norm: float
    geometry_bohr: List[List[float]] = Field(default_factory=list)


class IrcBranchInfo(BaseModel):
    """IRC 反应路径单向分支信息。"""

    direction: Literal["forward", "reverse"]
    n_points: int
    barrier_kcal_mol: float
    delta_energy_kcal_mol: float
    points: List[IrcPointInfo] = Field(default_factory=list)


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
    actual_method: str | None = None
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
    mulliken_charges: Optional[List[float]] = None
    lowdin_charges: Optional[List[float]] = None
    mulliken_spin_populations: Optional[List[float]] = None
    lowdin_spin_populations: Optional[List[float]] = None
    gradients: Optional[List[List[float]]] = None
    frequencies_cm: Optional[List[float]] = None
    ir_intensities_km_mol: Optional[List[float]] = None
    zpve_hartree: Optional[float] = None
    zpve_kcal_mol: Optional[float] = None
    thermal_energy_hartree: Optional[float] = None
    enthalpy_hartree: Optional[float] = None
    gibbs_free_energy_hartree: Optional[float] = None
    entropy_cal_mol_k: Optional[float] = None
    heat_capacity_cv_cal_mol_k: Optional[float] = None
    heat_capacity_cp_cal_mol_k: Optional[float] = None
    vibrational_modes: List[VibrationalModeInfo] = Field(default_factory=list)

    # 激发态与光谱（若 task 为 tddft / cis / excited_states）
    optical_gap_ev: Optional[float] = None
    lowest_triplet_ev: Optional[float] = None
    delta_est_ev: Optional[float] = None
    excited_states: List[ExcitedStateInfo] = Field(default_factory=list)

    # 优化后的坐标（若 task=optimize）
    optimized_coordinates_angstrom: Optional[List[List[float]]] = None

    # 过渡态与反应路径（若 task 为 ts / irc）
    is_transition_state: Optional[bool] = None
    imaginary_frequency_cm: Optional[float] = None
    irc_forward_barrier_kcal_mol: Optional[float] = None
    irc_reverse_barrier_kcal_mol: Optional[float] = None
    reaction_energy_kcal_mol: Optional[float] = None
    irc_path: List[IrcPointInfo] = Field(default_factory=list)

    # 错误与警告
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # 可追溯文件
    raw_log_path: Optional[str] = None
    input_file_path: Optional[str] = None
    run_directory: Optional[str] = None

    # 后端特定原始数据（cclib 对象等不直接序列化）
    extra: Dict = Field(default_factory=dict)
