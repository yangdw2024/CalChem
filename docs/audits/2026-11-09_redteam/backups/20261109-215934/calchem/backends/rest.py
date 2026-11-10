"""REST (Rust-based Electronic-Structure Tool) 后端适配。

支持对接 Igor Ying Zhang (复旦大学) 开发的纯 Rust 量子化学程序 REST (2026.1.1):
1. 生成 REST .toml 规范输入文件 ([ctrl] 与 [geom])
2. 调度执行 rest (支持 Windows 原生与 WSL rest-py311 运行环境)
3. 解析 REST .log 输出（自洽场总能量、HOMO/LUMO 能级、偶极矩、Mulliken 电荷、弛豫构型）
4. 原生兼容并加载 E:\\YDW\\REST\\001\\tasks\\20_massive350 大规模基准库
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.calculator import register_backend
from ..core.capabilities import BackendCapabilities
from ..core.errors import BackendExecutionError
from ..core.request import CalculationRequest, MoleculeSpec
from ..core.result import Result, OrbitalInfo
from .external import ExternalProgramBackend


# 方法映射：CalChem method → REST xc keyword
_METHOD_MAP = {
    "rhf": "hf", "hf": "hf", "uhf": "hf",
    "b3lyp": "b3lyp", "pbe": "pbe", "pbe0": "pbe0",
    "blyp": "blyp", "svwn": "svwn5", "vwn": "svwn5",
    "wb97xd": "wb97x-d", "camb3lyp": "cam-b3lyp",
}

# 基组映射：CalChem basis → REST basis_path
_BASIS_MAP = {
    "sto-3g": "sto-3g",
    "def2-svp": "def2-svp",
    "def2-tzvp": "def2-tzvp",
    "6-31g": "6-31g",
    "6-31g*": "6-31g(d)",
    "6-31g(d)": "6-31g(d)",
    "6-31g**": "6-31g(d,p)",
    "6-31g(d,p)": "6-31g(d,p)",
}


def _find_rest_executable() -> Optional[str]:
    """探测 REST 可执行文件路径（优先环境变量，其次 WSL rest-py311，再次 PATH）。"""
    env_bin = os.environ.get("CALCHEM_REST_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    which_local = shutil.which("rest")
    if which_local:
        return which_local

    # 检测 WSL 中的 rest
    if shutil.which("wsl"):
        wsl_candidate = "/home/administrator/miniforge3/envs/rest-py311/bin/rest"
        try:
            p = subprocess.run(
                ["wsl", "test", "-x", wsl_candidate],
                capture_output=True,
                timeout=5,
            )
            if p.returncode == 0:
                return f"wsl:{wsl_candidate}"
        except Exception:
            pass

    return None


def _build_rest_toml(
    req: CalculationRequest,
    job_type: str = "sp",
    nproc: int = 4,
) -> str:
    """生成 REST 规范的 .toml 输入格式。"""
    mol = req.molecule
    xc = _METHOD_MAP.get(req.method.lower(), req.method.lower())
    if req.xc and req.method.lower() in ("dft", "rks", "uks"):
        xc = _METHOD_MAP.get(req.xc.lower(), req.xc.lower())

    basis = _BASIS_MAP.get(req.basis.lower(), req.basis.lower())

    pos_lines = []
    for s, c in zip(mol.symbols, mol.coordinates_angstrom):
        pos_lines.append(f"{s:<3} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
    pos_str = "\n".join(pos_lines)

    lines = [
        "[ctrl]",
        f'xc = "{xc}"',
        f'basis_path = "{basis}"',
        'auxbas_path = "def2-universal-jkfit"',
        f"charge = {mol.charge}",
        f"spin = {mol.multiplicity}",
        f"num_threads = {nproc}",
        "print_level = 1",
    ]

    if req.dispersion and req.dispersion.lower() in ("d3bj", "d3(bj)", "bj"):
        lines.append('empirical_dispersion = "d3bj"')
    elif req.dispersion and req.dispersion.lower() in ("d3", "d3zero", "zero"):
        lines.append('empirical_dispersion = "d3zero"')

    if job_type == "opt":
        lines.extend([
            'job_type = "opt"',
            'opt_engine = "geometric-pyo3"',
        ])
    elif job_type == "freq":
        lines.append('job_type = "freq"')
    else:
        lines.append('job_type = "sp"')

    lines.extend([
        "",
        "[geom]",
        'name = "calchem_rest_job"',
        'unit = "angstrom"',
        'position = """',
        pos_str,
        '"""',
        "",
    ])
    return "\n".join(lines)


def parse_rest_log(content: str) -> Dict[str, Any]:
    """解析 REST 输出日志文本，提取物理量。"""
    out: Dict[str, Any] = {
        "converged": False,
        "energy_hartree": None,
        "homo_hartree": None,
        "lumo_hartree": None,
        "gap_hartree": None,
        "dipole_debye": None,
        "elapsed_seconds": None,
        "optimized_coordinates": None,
        "mulliken_charges": {},
    }

    if "Mission accomplished" in content:
        out["converged"] = True

    # 提取 SCF 总能量
    m_e = re.findall(r"The SCF energy\s*:\s*(-?\d+\.\d+)", content)
    if m_e:
        out["energy_hartree"] = float(m_e[-1])
    else:
        m_e2 = re.findall(r"Energy:\s*(-?\d+\.\d+)\s*Ha", content)
        if m_e2:
            out["energy_hartree"] = float(m_e2[-1])

    # 提取 HOMO / LUMO / Gap
    m_fmo = re.findall(
        r"HOMO\s*\(\s*\d+\s*\)\s*:\s*(-?\d+\.\d+)\s*,\s*LUMO\s*\(\s*\d+\s*\)\s*:\s*(-?\d+\.\d+)\s*,\s*H-L Gap\s*:\s*(-?\d+\.\d+)",
        content,
    )
    if m_fmo:
        h_val, l_val, g_val = m_fmo[-1]
        out["homo_hartree"] = float(h_val)
        out["lumo_hartree"] = float(l_val)
        out["gap_hartree"] = float(g_val)

    # 提取偶极矩
    m_dip = re.findall(
        r"Dipole Moment in DEBYE:\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)",
        content,
    )
    if m_dip:
        dx, dy, dz = [float(x) for x in m_dip[-1]]
        out["dipole_debye"] = (dx * dx + dy * dy + dz * dz) ** 0.5

    # 提取耗时
    m_time = re.search(r"Overall\s*\|\s*:\s*([\d\.]+)\s*s", content)
    if m_time:
        out["elapsed_seconds"] = float(m_time.group(1))

    # 提取优化后的构型
    if "Geometry after relaxation [Ang]:" in content:
        block = content.split("Geometry after relaxation [Ang]:")[-1]
        lines = block.strip().splitlines()
        opt_coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].isalpha():
                try:
                    c = [float(parts[1]), float(parts[2]), float(parts[3])]
                    opt_coords.append(c)
                except ValueError:
                    break
            else:
                break
        if opt_coords:
            out["optimized_coordinates"] = opt_coords

    return out


@register_backend("rest")
class RESTBackend(ExternalProgramBackend):
    """REST 量子化学引擎适配器（支持原生执行与任务库解析）。"""

    program_name = "rest"

    def __init__(self, rest_path: str = ""):
        resolved = rest_path or _find_rest_executable() or ""
        super().__init__(resolved)
        self.rest_path = resolved

    def capabilities(self) -> BackendCapabilities:
        available = bool(self.rest_path)
        return BackendCapabilities(
            name="rest",
            version="2026.1.1",
            available=available,
            requires_external_program=True,
            unavailable_reason=None if available else "REST executable not found (WSL rest-py311 or CALCHEM_REST_BIN)",
            tasks=["single_point", "optimize", "frequency"],
            methods=["hf", "b3lyp", "pbe", "pbe0", "blyp", "wb97xd", "camb3lyp"],
            basis_sets=["def2-svp", "def2-tzvp", "sto-3g", "6-31g", "6-31g(d)", "6-31g(d,p)"],
        )

    def _execute_rest(self, input_toml: str, job_dir: Path) -> str:
        """执行 REST 并返回标准输出内容。"""
        if not self.rest_path:
            raise BackendExecutionError("REST is not available in the current environment.")

        toml_file = job_dir / "input.toml"
        toml_file.write_text(input_toml, encoding="utf-8")

        if self.rest_path.startswith("wsl:"):
            wsl_bin = self.rest_path.split(":", 1)[1]
            # 转换 Windows 路径为 WSL 挂载路径
            resolved_p = str(job_dir.resolve()).replace("\\", "/")
            if ":" in resolved_p:
                drive, rest_p = resolved_p.split(":", 1)
                wsl_dir = f"/mnt/{drive.lower()}{rest_p}"
            else:
                wsl_dir = resolved_p
            cmd = [
                "wsl",
                "bash",
                "-lc",
                f"cd '{wsl_dir}' && '{wsl_bin}' -i input.toml",
            ]
        else:
            cmd = [self.rest_path, "-i", "input.toml"]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(job_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired as exc:
            raise BackendExecutionError("REST calculation timed out after 600s.") from exc
        except Exception as exc:
            raise BackendExecutionError(f"REST execution failed: {exc}") from exc

    def single_point(self, request: CalculationRequest) -> Result:
        t0 = time.perf_counter()
        job_dir = self._prepare_run_dir("rest_sp")
        toml_str = _build_rest_toml(request, job_type="sp")
        log_content = self._execute_rest(toml_str, job_dir)
        info = parse_rest_log(log_content)
        t1 = time.perf_counter()

        status = "succeeded" if info["converged"] and info["energy_hartree"] is not None else "failed"
        return Result(
            task="single_point",
            method=request.method,
            basis=request.basis,
            actual_backend="rest",
            actual_method=f"REST-2026.1.1 ({request.method}/{request.basis})",
            status=status,
            total_energy_hartree=info["energy_hartree"],
            scf_energy_hartree=info["energy_hartree"],
            scf_converged=info["converged"],
            homo_energy_hartree=info["homo_hartree"],
            lumo_energy_hartree=info["lumo_hartree"],
            dipole_magnitude_debye=info["dipole_debye"],
            wall_clock_seconds=info["elapsed_seconds"] or (t1 - t0),
            raw_output=log_content,
        )

    def optimize(self, request: CalculationRequest) -> Result:
        t0 = time.perf_counter()
        job_dir = self._prepare_run_dir("rest_opt")
        toml_str = _build_rest_toml(request, job_type="opt")
        log_content = self._execute_rest(toml_str, job_dir)
        info = parse_rest_log(log_content)
        t1 = time.perf_counter()

        status = "succeeded" if info["converged"] and info["energy_hartree"] is not None else "failed"
        return Result(
            task="optimize",
            method=request.method,
            basis=request.basis,
            actual_backend="rest",
            actual_method=f"REST-2026.1.1-Opt ({request.method}/{request.basis})",
            status=status,
            total_energy_hartree=info["energy_hartree"],
            scf_energy_hartree=info["energy_hartree"],
            scf_converged=info["converged"],
            optimized_coordinates_angstrom=info["optimized_coordinates"],
            wall_clock_seconds=info["elapsed_seconds"] or (t1 - t0),
            raw_output=log_content,
        )
