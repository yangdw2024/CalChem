"""ORCA 后端适配。

写 .inp → 调 orca → 解析 .out。
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from ..core.calculator import register_backend
from ..core.errors import BackendExecutionError
from ..core.request import CalculationRequest
from ..core.result import Result
from .external import ExternalProgramBackend


# CalChem method → ORCA method block
_METHOD_MAP = {
    "rhf": "HF", "hf": "HF", "uhf": "UHF",
    "b3lyp": "B3LYP", "pbe": "PBE", "pbe0": "PBE0",
    "blyp": "BLYP", "tpss": "TPSS", "wb97xd": "wB97X-D4",
    "mp2": "MP2", "ccsd": "CCSD", "ccsd(t)": "CCSD(T)",
}


def _build_orca_inp(req: CalculationRequest, nproc: int) -> str:
    """生成 ORCA .inp 文件。"""
    method = _METHOD_MAP.get(req.method.lower(), req.method.upper())
    if req.xc and req.method.lower() in ("dft", "rks", "uks"):
        method = req.xc.upper()

    basis = req.basis.upper().replace("(", "").replace(")", "").replace("+", "+")
    mol = req.molecule

    lines = [
        f"! {method} {basis} SP",
        f"! NOPRINT",
        f"%pal nprocs {nproc} end",
        "",
        f"* xyzfile {mol.charge} {mol.multiplicity} input.xyz",
    ]
    return "\n".join(lines) + "\n"


def _parse_orca_out(out_text: str) -> dict:
    """解析 ORCA 输出。"""
    result = {
        "normal_termination": "ORCA TERMINATED NORMALLY" in out_text,
        "energy_hartree": None,
        "scf_converged": False,
        "scf_iterations": None,
        "homo": None, "lumo": None,
        "dipole": None,
        "error": None,
    }

    # 总能量
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", out_text)
    if m:
        result["energy_hartree"] = float(m.group(1))

    # SCF Converged
    if "TOTAL SCF ENERGY" in out_text or "SCF CONVERGED" in out_text.upper():
        result["scf_converged"] = True

    # ORCA 版本错误
    if "ERROR" in out_text.upper():
        err_lines = [l.strip() for l in out_text.split("\n") if "ERROR" in l.upper()][:3]
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
        caps.basis_sets = ["sto-3g", "def2-svp", "def2-tzvp", "6-31g", "6-31g(d)", "cc-pvdz"]
        return caps

    def single_point(self, request: CalculationRequest) -> Result:
        started = datetime.now()
        caps = self.capabilities()
        if not caps.available:
            raise BackendExecutionError(f"ORCA not available: {caps.unavailable_reason}")

        job_id = request.job_id or f"orca_{int(time.time())}"
        tmp_base = Path(os.environ.get("CALCHEM_RUN_DIR", tempfile.gettempdir())) / "calchem_jobs"
        run_dir = self._make_run_dir(tmp_base, job_id)

        # 写 xyz 和 inp
        mol = request.molecule
        xyz_lines = [str(mol.natom), "CalChem auto-generated"]
        for sym, xyz in zip(mol.symbols, mol.coordinates_angstrom):
            xyz_lines.append(f"{sym:2s}  {xyz[0]:14.8f}  {xyz[1]:14.8f}  {xyz[2]:14.8f}")
        (run_dir / "input.xyz").write_text("\n".join(xyz_lines))

        inp = _build_orca_inp(request, request.resources.max_cpu_cores)
        (run_dir / "input.inp").write_text(inp)

        # 运行 ORCA
        cmd = [self.executable, str(run_dir / "input.inp")]
        returncode, stdout, stderr = self._run_command(
            cmd, run_dir, timeout=request.resources.timeout_seconds
        )

        out_path = run_dir / "input.out"
        out_text = out_path.read_text(errors="replace") if out_path.exists() else stdout

        parsed = _parse_orca_out(out_text)
        walltime = (datetime.now() - started).total_seconds()

        errors = []
        if not parsed["normal_termination"]:
            errors.append(f"ORCA did not terminate normally (rc={returncode})")
        if parsed["error"]:
            errors.append(parsed["error"])

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
            errors=errors,
            raw_log_path=str(out_path),
            input_file_path=str(run_dir / "input.inp"),
            run_directory=str(run_dir),
        )
