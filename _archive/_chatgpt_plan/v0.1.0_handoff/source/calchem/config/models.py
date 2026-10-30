"""CalChem 配置模型。"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class BackendPaths(BaseModel):
    """外部程序路径配置。"""
    gaussian_exe: str = r"D:\G16W\g16.exe"
    orca_exe: str = r"E:\ORCA\software\ORCA6.1.1\orca.exe"
    xtb_exe: str = ""
    pyscf_auto_detect: bool = True
    rest_endpoint: str = ""


class ResourceConfig(BaseModel):
    """资源限制。"""
    max_cpu_cores: int = 8
    max_memory_mb: int = 12000
    timeout_seconds: int = 600
    heavy_job_concurrency: int = 1  # 固定 1


class AppConfig(BaseModel):
    """CalChem 全局配置。"""
    data_root: str = r"D:\CalChemData"
    run_root: str = ""
    log_root: str = ""
    default_backend: str = "native"
    default_method: str = "rhf"
    default_basis: str = "sto-3g"
    default_xc: str = "b3lyp"

    backends: BackendPaths = Field(default_factory=BackendPaths)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)

    log_level: str = "INFO"

    def resolved_run_root(self) -> Path:
        if self.run_root:
            return Path(self.run_root)
        return Path(self.data_root) / "runs"

    def resolved_log_root(self) -> Path:
        if self.log_root:
            return Path(self.log_root)
        return Path(self.data_root) / "logs"
