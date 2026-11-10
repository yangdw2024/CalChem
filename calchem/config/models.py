"""CalChem 配置模型。"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


def get_default_data_root() -> str:
    """动态获取默认数据根目录（跨平台自适应，避免硬编码）。"""
    env = os.environ.get("CALCHEM_DATA") or os.environ.get("CALCHEM_DATA_DIR")
    if env:
        return env
    d_dir = Path("D:/CalChemData")
    if d_dir.exists():
        return str(d_dir)
    return str(Path.home() / ".calchem" / "data")


class BackendPaths(BaseModel):
    """外部程序路径配置。默认空——必须由用户配置或配置文件提供。"""
    gaussian_exe: str = ""
    orca_exe: str = ""
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
    data_root: str = Field(default_factory=get_default_data_root)
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
