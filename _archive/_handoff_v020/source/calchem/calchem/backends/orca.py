"""ORCA 后端适配。

写 .inp + .xyz → 调 orca → 解析 stdout/.out。
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
from ..core.errors import BackendExecutionError
from ..core.request import CalculationRequest
from ..core.result import Result
from .external import ExternalProgramBackend


# CalChem method → ORCA method keyword
_METHOD_MAP = {
    "rhf": "HF", "hf": "HF", "uhf": "UHF", "rohf": "ROHF",
    "b3lyp": "B3LYP", "pbe": "PBE", "pbe0": "PBE0",
    "blyp": "BLYP", "tpss": "TPSS", "wb97xd": "wB97X-D4",
    "mp2": "MP2", "ccsd": "CCSD", "ccsd(t)": "CCSD(T)",
}

# CalChem basis → ORCA basis keyword
_BASIS_MAP = {
    "sto-3g": "STO-3G",
    "3-21g": "3-21G",
    "6-31g": "6-31G",
    "6-31g(d)": "6-31G*",
    "6-31g(d,p)": "6-31G**",
    "6-311g(d,p)": "6-311G**",
    "def2-svp": "def2-SVP",
    "def2-tzvp": "def2-TZVP",
    "cc-pvdz": "cc-pVDZ",
}


def _build_orca_inp(req: CalculationRequest, nproc: int) -> str:
    """生成 ORCA .inp 文件（基于手动验证可用格式）。"""
    method = _METHOD_MAP.get(req.method.lower(), req.method.upper())
    if req.xc and req.method.lower() in ("dft", "rks", "uks"):
        method = req.xc.upper()

    basis = _BASIS_MAP.get(req.basis.lower(), req.basis)
    mol = req.molecule

    lines = [
        f"! {method} {basis} SP",
        f"%pal nprocs {nproc} end",
        "",
        f"* xyzfile {mol.charge} {mol.multiplicity} input.xyz",
        "",
    ]
    return "\n".join(lines)


def _parse_orca_output(text: str) -> dict:
    """解析 ORCA 输出（stdout 或 .out 文件内容）。"""
    result = {
        "normal_termination": "ORCA TERMINATED NORMALLY" in text,
        "energy_hartree": None,
        "scf_converged": "SCF CONVERGED" in text,
        "scf_iterations": None,
        "dipole_magnitude": None,
        "error": None,
    }

    m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", text)
    if m:
        result["energy_hartree"] = float(m.group(1))

    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)", text)
    if m:
        result["scf_iterations"] = int(m.group(1))

    # 偶极矩：只存 magnitude，不伪造矢量分量
    dipole_mag = None
    m = re.search(r"Magnitude \(Debye\)\s*:\s*([-\d.]+)", text)
    if m:
        dipole_mag = float(m.group(1))

    # 错误检测：只匹配真正的 ERROR 行（排除 SCF 参数中的 "DIIS Error" 等）
    if "ERROR" in text.upper():
        err_lines = [
            l.strip() for l in text.split("\n")
            if "ERROR" in l.upper()
            and "DIIS" not in l.upper()
            and "TolErr" not in l.upper()
            and "WARNING" not in l.upper()
            and "Startup" not in l.upper()
        ][:3]
        if err_lines:
            result["error"] = "; ".join(err_lines)

    return result


@register_backend("orca")
class ORCABackend(ExternalProgramBackend):
    """ORCA 后端。"""

    program_name = "orca"

    def __init__(self, executable: str = r"E:\ORCA\software\ORCA6.1.1\orca.exe"):
        super().__init__(executable)

    def capabilities(self):
        caps = super().capabilities()
        caps.tasks = ["single_point"]
        caps.methods = list(_METHOD_MAP.keys())
        caps.basis_sets = list(_BASIS_MAP.keys())
        return caps

    def single_point(self, request: CalculationRequest) -> Result:
        started = datetime.now()
        caps = self.capabilities()
        if not caps.available:
            raise BackendExecutionError(f"ORCA not available: {caps.unavailable_reason}")

        job_id = request.job_id or f"orca_{int(time.time())}"
        tmp_base = Path(os.environ.get("CALCHEM_RUN_DIR", tempfile.gettempdir())) / "calchem_jobs"
        run_dir = self._make_run_dir(tmp_base, job_id)

        # 写 xyz（ORCA 格式：natoms / comment / atoms）
        mol = request.molecule
        xyz_lines = [str(mol.natom), "CalChem"]
        for sym, xyz in zip(mol.symbols, mol.coordinates_angstrom):
            xyz_lines.append(f"{sym:2s}  {xyz[0]:14.8f}  {xyz[1]:14.8f}  {xyz[2]:14.8f}")
        (run_dir / "input.xyz").write_text("\n".join(xyz_lines) + "\n")

        # 写 inp
        inp = _build_orca_inp(request, request.resources.max_cpu_cores)
        (run_dir / "input.inp").write_text(inp)

        # 运行 ORCA（cwd 设为 run_dir，参数列表不用 shell）
        cmd = [self.executable, "input.inp"]
        try:
            proc = subprocess.run(
                cmd, cwd=str(run_dir), capture_output=True, text=True,
                timeout=request.resources.timeout_seconds, shell=False,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            raise BackendExecutionError(f"ORCA timed out after {request.resources.timeout_seconds}s")

        # ORCA 输出：优先用 stdout（ORCA 6.x 主要输出到 stdout），
        # 同时检查 input.out 文件
        out_path = run_dir / "input.out"
        out_text = stdout
        if out_path.exists():
            out_text = out_path.read_text(errors="replace") + "\n" + stdout

        parsed = _parse_orca_output(out_text)
        walltime = (datetime.now() - started).total_seconds()

        errors = []
        if not parsed["normal_termination"]:
            errors.append(f"ORCA did not terminate normally (rc={returncode})")
        if parsed["error"]:
            errors.append(parsed["error"])
        if parsed["energy_hartree"] is None:
            errors.append("Could not extract FINAL SINGLE POINT ENERGY")

        status = "succeeded" if parsed["normal_termination"] and parsed["energy_hartree"] is not None else "failed"

        return Result(
            backend="orca",
            backend_version="ORCA 6.1.1",
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
            dipole_magnitude_debye=dipole_mag,
            errors=errors,
            raw_log_path=str(out_path) if out_path.exists() else None,
            input_file_path=str(run_dir / "input.inp"),
            run_directory=str(run_dir),
        )
