"""配置加载器：从文件/环境变量/自动检测构建 AppConfig。

优先级：CLI 参数 > 项目配置 > 用户配置 > 自动检测 > 默认值。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from .models import AppConfig, BackendPaths, ResourceConfig


# 常见默认安装候选路径（作为系统 PATH 与环境变量之后的后备）
_KNOWN_GAUSSIAN_PATHS = [
    r"D:\G16W\g16.exe",
    r"C:\G16W\g16.exe",
    r"D:\Gaussian\g16.exe",
]
_KNOWN_ORCA_PATHS = [
    r"E:\ORCA\software\ORCA6.1.1\orca.exe",
    r"D:\ORCA\orca.exe",
    r"C:\ORCA\orca.exe",
    "/usr/bin/orca",
    "/usr/local/bin/orca",
]
_KNOWN_XTB_PATHS = [
    r"D:\xtb\bin\xtb.exe",
    r"C:\xtb\bin\xtb.exe",
    "/usr/bin/xtb",
    "/usr/local/bin/xtb",
]


def _detect_executable(
    env_vars: list[str],
    command_names: list[str],
    candidate_paths: list[str],
) -> str:
    """动态探测外部可执行程序路径。

    优先级：
    1. 指定环境变量（如 CALCHEM_GAUSSIAN_EXE, GAUSSIAN_EXE, G16_EXE）
    2. 系统 PATH 查找（如 shutil.which("g16")）
    3. 常见默认安装路径（作为后备）
    """
    for ev in env_vars:
        val = os.environ.get(ev)
        if val and Path(val).is_file():
            return val

    for cmd in command_names:
        which = shutil.which(cmd)
        if which:
            return which

    for p in candidate_paths:
        if p and Path(p).is_file():
            return p

    return ""


def _find_first_existing(paths: list[str]) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    return ""


def auto_detect_paths() -> BackendPaths:
    """自动检测外部程序路径（环境变量优先，其次 PATH，再次常见路径）。"""
    return BackendPaths(
        gaussian_exe=_detect_executable(
            env_vars=["CALCHEM_GAUSSIAN_EXE", "GAUSSIAN_EXE", "G16_EXE"],
            command_names=["g16", "g09", "gaussian"],
            candidate_paths=_KNOWN_GAUSSIAN_PATHS,
        ),
        orca_exe=_detect_executable(
            env_vars=["CALCHEM_ORCA_EXE", "ORCA_EXE"],
            command_names=["orca"],
            candidate_paths=_KNOWN_ORCA_PATHS,
        ),
        xtb_exe=_detect_executable(
            env_vars=["CALCHEM_XTB_EXE", "XTB_EXE"],
            command_names=["xtb"],
            candidate_paths=_KNOWN_XTB_PATHS,
        ),
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
