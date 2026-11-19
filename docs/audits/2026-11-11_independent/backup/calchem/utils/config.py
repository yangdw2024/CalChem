import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    max_memory: int = Field(default=4000, description="Maximum memory in MB")
    num_threads: int = Field(default=1, description="Number of threads")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    gpu_id: int = Field(default=0, description="GPU device ID")
    scf_max_cycles: int = Field(default=100, description="Maximum SCF cycles")
    scf_convergence: float = Field(default=1e-8, description="SCF convergence threshold")
    opt_max_cycles: int = Field(default=100, description="Maximum optimization cycles")
    opt_convergence: float = Field(default=1e-3, description="Optimization convergence threshold")
    log_level: str = Field(default="INFO", description="Logging level")
    basis_set_dir: Optional[str] = Field(default=None, description="Basis set directory")
    scratch_dir: str = Field(default="./scratch", description="Scratch directory")

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        if config_path is None:
            config_path = os.environ.get("CALCHEM_CONFIG", None)

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
            return cls(**config_data)

        return cls()

    def save(self, config_path: str) -> None:
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


_default_config: Optional[Config] = None


def get_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = Config.load()
    return _default_config


def set_config(config: Config) -> None:
    global _default_config
    _default_config = config
