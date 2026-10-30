"""Gaussian 16 后端适配。

写 .gjf → 调 g16 → 解析 .log。
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from ..core.calculator import register_backend
from ..core.errors import BackendExecutionError, ConvergenceError
from ..core.request import CalculationRequest, MoleculeSpec
from ..core.result import Result
from .external import ExternalProgramBackend


# 方法映射：CalChem method → Gaussian route
_METHOD_MAP = {
    "rhf": "RHF", "hf": "RHF", "uhf": "UHF", "rohf": "ROHF",
    "b3lyp": "B3LYP", "pbe": "PBEPBE", "pbe0": "PBE1PBE",
    "blyp": "BLYP", "tpss": "TPSS", "wb97xd": "wB97XD",
    "mp2": "MP2", "mp4": "MP4", "ccsd": "CCSD",
    "ccsd(t)": "CCSD(T)", "ccsdt": "CCSD(T)",
}


def _build_gjf(req: CalculationRequest, memory_mb: int, nproc: int) -> str:
    """根据 CalculationRequest 生成 Gaussian gjf 输入。"""
    method = _METHOD_MAP.get(req.method.lower(), req.method.upper())
    if req.xc and req.method.lower() in ("dft", "rks", "uks"):
        method = req.xc.upper()

    basis = req.basis.upper()
    mol = req.molecule

    lines = [
        f"%mem={memory_mb}MB",
        f"%nprocshared={nproc}",
        f"%chk=calchem.chk",
        f"# {method}/{basis} SP",
        "",
        f"CalChem auto-generated: {req.method}/{req.basis}",
        "",
        f"{mol.charge} {mol.multiplicity}",
    ]
    for sym, xyz in zip(mol.symbols, mol.coordinates_angstrom):
        lines.append(f"{sym:2s}  {xyz[0]:14.8f}  {xyz[1]:14.8f}  {xyz[2]:14.8f}")
    lines.extend(["", "", ""])
    return "\n".join(lines)


def _parse_g16_log(log_text: str) -> dict:
    """解析 Gaussian 16 输出日志。提取关键能量和收敛信息。"""
    result = {
        "normal_termination": "Normal termination" in log_text,
        "scf_converged": "Stationary point found" in log_text or "SCF Done" in log_text,
        "energy_hartree": None,
        "homo_energy": None,
        "lumo_energy": None,
        "scf_iterations": None,
        "dipole": None,
        "error": None,
    }

    # 总能量
    m = re.search(r"SCF Done:\s*E\([^)]+\)\s*=\s*([-\d.]+)", log_text)
    if m:
        result["energy_hartree"] = float(m.group(1))
    else:
        # MP2/CCSD 能量行
        m = re.search(r"E\(MP2\)\s*=\s*([-\d.]+)", log_text)
        if m:
            result["energy_hartree"] = float(m.group(1))

    # SCF 迭代次数
    m = re.search(r"iterations\s*=\s*(\d+)", log_text)
    if m:
        result["scf_iterations"] = int(m.group(1))

    # 偶极矩
    m = re.search(r"Dipole moment[\s\S]+?X=\s*([-\d.]+)\s+Y=\s*([-\d.]+)\s+Z=\s*([-\d.]+)", log_text)
    if m:
        result["dipole"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    if "Error termination" in log_text:
        err_line = [l for l in log_text.split("\n") if "Error" in l][:3]
        result["error"] = "; ".join(err_line)

    return result


@register_backend("gaussian")
class GaussianBackend(ExternalProgramBackend):
    """Gaussian 16 后端。"""

    program_name = "gaussian"

    def __init__(self, executable: str = r"D:\G16W\g16.exe"):
        super().__init__(executable)

    def capabilities(self):
        caps = super().capabilities()
        caps.tasks = ["single_point"]
        caps.methods = list(_METHOD_MAP.keys())
        caps.basis_sets = ["sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-311g(d,p)",
                           "cc-pvdz", "cc-pvtz", "def2-svp", "def2-tzvp"]
        return caps

    def single_point(self, request: CalculationRequest) -> Result:
        from ..core.request import MoleculeSpec

        started = datetime.now()
        caps = self.capabilities()
        if not caps.available:
            raise BackendExecutionError(f"Gaussian not available: {caps.unavailable_reason}")

        # 创建运行目录
        job_id = request.job_id or f"g16_{int(time.time())}"
        tmp_base = Path(os.environ.get("CALCHEM_RUN_DIR", tempfile.gettempdir())) / "calchem_jobs"
        run_dir = self._make_run_dir(tmp_base, job_id)

        # 写 gjf
        gjf = _build_gjf(
            request,
            memory_mb=request.resources.max_memory_mb,
            nproc=request.resources.max_cpu_cores,
        )
        gjf_path = run_dir / "input.gjf"
        gjf_path.write_text(gjf)

        # 运行 g16
        g16_dir = Path(self.executable).parent
        env = os.environ.copy()
        env["g16root"] = str(g16_dir)
        env["GAUSS_EXEDIR"] = str(g16_dir)
        env["GAUSS_SCRDIR"] = str(run_dir)
        if str(g16_dir) not in env.get("PATH", ""):
            env["PATH"] = str(g16_dir) + os.pathsep + env.get("PATH", "")

        cmd = [self.executable, gjf_path.name]
        try:
            proc = subprocess.run(
                cmd, cwd=str(run_dir), capture_output=True, text=True,
                timeout=request.resources.timeout_seconds, shell=False, env=env,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            raise BackendExecutionError(
                f"Gaussian timed out after {request.resources.timeout_seconds}s"
            )

        # 读日志（Gaussian 输出 .out 不是 .log）
        log_path = run_dir / "input.out"
        if not log_path.exists():
            log_path = run_dir / "input.log"
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(errors="replace")
        elif stdout:
            log_text = stdout

        parsed = _parse_g16_log(log_text)
        walltime = (datetime.now() - started).total_seconds()

        errors = []
        if not parsed["normal_termination"]:
            errors.append(f"Gaussian did not terminate normally (rc={returncode})")
        if parsed["error"]:
            errors.append(parsed["error"])
        if parsed["energy_hartree"] is None:
            errors.append("Could not extract energy from Gaussian log")

        status = "succeeded" if parsed["normal_termination"] and parsed["energy_hartree"] is not None else "failed"

        return Result(
            backend="gaussian",
            backend_version="G16",
            task=request.task,
            method=request.method,
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=parsed["energy_hartree"],
            total_energy_hartree=parsed["energy_hartree"],
            scf_converged=parsed["scf_converged"],
            scf_iterations=parsed["scf_iterations"],
            dipole_debye=parsed["dipole"],
            errors=errors,
            raw_log_path=str(log_path) if log_path.exists() else None,
            input_file_path=str(gjf_path),
            run_directory=str(run_dir),
        )
