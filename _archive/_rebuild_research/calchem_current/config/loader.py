"""配置加载器：从文件/环境变量/自动检测构建 AppConfig。

优先级：CLI 参数 > 项目配置 > 用户配置 > 自动检测 > 默认值。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .models import AppConfig, BackendPaths, ResourceConfig


# 本机已知的常见安装路径（用于自动检测）
_KNOWN_GAUSSIAN_PATHS = [
    r"D:\G16W\g16.exe",
    r"C:\G16W\g16.exe",
    r"D:\Gaussian\g16.exe",
]
_KNOWN_ORCA_PATHS = [
    r"E:\ORCA\software\ORCA6.1.1\orca.exe",
    r"D:\ORCA\orca.exe",
    r"C:\ORCA\orca.exe",
]
_KNOWN_XTB_PATHS = [
    r"D:\xtb\bin\xtb.exe",
    r"C:\xtb\bin\xtb.exe",
]


def _find_first_existing(paths: list[str]) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    return ""


def auto_detect_paths() -> BackendPaths:
    """自动检测本机已知路径的外部程序。"""
    return BackendPaths(
        gaussian_exe=_find_first_existing(_KNOWN_GAUSSIAN_PATHS),
        orca_exe=_find_first_existing(_KNOWN_ORCA_PATHS),
        xtb_exe=_find_first_existing(_KNOWN_XTB_PATHS),
    )


def get_user_config_path() -> Path:
    """用户配置文件路径（非 C 盘优先）。"""
    # 环境变量优先
    env = os.environ.get("CALCHEM_CONFIG")
    if env:
        return Path(env)
    # 用户目录
    return Path.home() / ".calchem" / "config.json"


def load_config(
    config_path: Optional[str] = None,
    auto_detect: bool = True,
) -> AppConfig:
    """加载配置。

    1. 读配置文件（如果存在）
    2. 自动检测已知路径
    3. 环境变量覆盖
    """
    cfg = AppConfig()

    # 自动检测
    if auto_detect:
        detected = auto_detect_paths()
        cfg.backends = detected

    # 读配置文件
    path = Path(config_path) if config_path else get_user_config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            file_cfg = AppConfig.model_validate(data)
            # 文件配置覆盖自动检测
            if file_cfg.backends.gaussian_exe:
                cfg.backends.gaussian_exe = file_cfg.backends.gaussian_exe
            if file_cfg.backends.orca_exe:
                cfg.backends.orca_exe = file_cfg.backends.orca_exe
            if file_cfg.backends.xtb_exe:
                cfg.backends.xtb_exe = file_cfg.backends.xtb_exe
            if file_cfg.data_root:
                cfg.data_root = file_cfg.data_root
        except (json.JSONDecodeError, ValueError):
            pass

    # 环境变量覆盖
    if os.environ.get("CALCHEM_DATA"):
        cfg.data_root = os.environ["CALCHEM_DATA"]
    if os.environ.get("CALCHEM_GAUSSIAN"):
        cfg.backends.gaussian_exe = os.environ["CALCHEM_GAUSSIAN"]
    if os.environ.get("CALCHEM_ORCA"):
        cfg.backends.orca_exe = os.environ["CALCHEM_ORCA"]

    return cfg


def save_config(cfg: AppConfig, path: Optional[Path] = None) -> Path:
    """保存配置到用户配置文件。"""
    path = path or get_user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return path
