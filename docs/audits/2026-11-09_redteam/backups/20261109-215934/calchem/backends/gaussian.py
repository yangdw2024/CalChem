"""Gaussian 16 后端适配。

写 .gjf → 调 g16 → 解析 .log。

解析约定（2026-11-03 修复 Q01/Q02）：
- SCF 能量取日志中 **最后一次** ``SCF Done``（优化任务的首个 SCF 是初始几何）。
- Post-HF 总能量取归档块（``1\\1\\...\\@``）中的 MP2/MP4SDTQ/CCSD/CCSD(T) 字段，
  找不到时视为失败，**绝不**用 SCF 能量顶替。
- 频率、优化几何、偶极、HOMO/LUMO 均取最后一次出现的块。
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path

from ..core.calculator import register_backend
from ..core.errors import BackendExecutionError
from ..core.request import CalculationRequest
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

# 相关方法 → 归档块中总能量字段名（按优先级）
_POST_HF_ARCHIVE_KEYS = {
    "mp2": ["MP2"],
    "mp4": ["MP4SDTQ", "MP4SDQ", "MP4D"],
    "ccsd": ["CCSD"],
    "ccsd(t)": ["CCSD(T)"],
    "ccsdt": ["CCSD(T)"],
}

_FLOAT = r"[-+]?\d+\.\d*(?:[DEde][-+]?\d+)?"


def _to_float(text: str) -> float:
    """Gaussian 使用 Fortran D 指数（-0.11D+01）。"""
    return float(text.replace("D", "E").replace("d", "e"))


def _build_gjf(req: CalculationRequest, memory_mb: int, nproc: int, extra_keyword: str = "") -> str:
    """根据 CalculationRequest 生成 Gaussian gjf 输入。"""
    method = _METHOD_MAP.get(req.method.lower(), req.method.upper())
    if req.xc and req.method.lower() in ("dft", "rks", "uks"):
        method = req.xc.upper()

    basis = req.basis.upper()
    mol = req.molecule
    route_keyword = extra_keyword if extra_keyword else "SP"

    lines = [
        f"%mem={memory_mb}MB",
        f"%nprocshared={nproc}",
        "%chk=calchem.chk",
        f"# {method}/{basis} {route_keyword}",
        "",
        f"CalChem auto-generated: {req.method}/{req.basis}",
        "",
        f"{mol.charge} {mol.multiplicity}",
    ]
    for sym, xyz in zip(mol.symbols, mol.coordinates_angstrom):
        lines.append(f"{sym:2s}  {xyz[0]:14.8f}  {xyz[1]:14.8f}  {xyz[2]:14.8f}")
    lines.extend(["", "", ""])
    return "\n".join(lines)


def _parse_archive(log_text: str) -> dict[str, float]:
    """解析最后一个归档块中的 ``KEY=value`` 能量字段。"""
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("1\\1\\") or line.strip().startswith("1|1|"):
            start = i
    if start is None:
        return {}
    buf = []
    for line in lines[start:]:
        buf.append(line.strip())
        if "@" in line:
            break
    archive = "".join(buf)
    sep = "|" if archive.startswith("1|1|") else "\\"
    out: dict[str, float] = {}
    for field in archive.split(sep):
        if "=" not in field:
            continue
        key, _, val = field.partition("=")
        try:
            out[key.strip()] = float(val.split(",")[0])
        except ValueError:
            continue
    return out


def _parse_orbital_energies(log_text: str) -> tuple[list[float], list[float]]:
    """返回最后一次布居分析中的 (alpha 占据, alpha 空) 轨道能。"""
    occ: list[float] = []
    virt: list[float] = []
    in_block = False
    for line in log_text.splitlines():
        is_occ = "Alpha  occ. eigenvalues" in line
        is_virt = "Alpha virt. eigenvalues" in line
        if is_occ or is_virt:
            if not in_block:
                occ, virt = [], []
                in_block = True
            values = [float(x) for x in re.findall(r"-?\d+\.\d+", line.split("--", 1)[-1])]
            (occ if is_occ else virt).extend(values)
        elif in_block and "Beta" not in line:
            in_block = False
    return occ, virt


def _parse_g16_log(log_text: str, method: str = "") -> dict:
    """解析 Gaussian 16 输出日志。提取关键能量和收敛信息。"""
    result = {
        "normal_termination": "Normal termination" in log_text,
        "scf_converged": None,
        "scf_energy_hartree": None,
        "energy_hartree": None,
        "correlation_energy_hartree": None,
        "homo_energy": None,
        "lumo_energy": None,
        "scf_iterations": None,
        "dipole": None,
        "dipole_magnitude": None,
        "opt_converged": "Stationary point found" in log_text,
        "error": None,
    }

    scf = re.findall(
        rf"SCF Done:\s*E\([^)]+\)\s*=\s*({_FLOAT})(?:\s+A\.U\.\s+after\s+(\d+)\s+cycles)?",
        log_text,
    )
    if scf:
        energy_txt, cycles = scf[-1]
        result["scf_energy_hartree"] = _to_float(energy_txt)
        if cycles:
            result["scf_iterations"] = int(cycles)
        result["scf_converged"] = "Convergence failure" not in log_text
    elif "Convergence failure" in log_text:
        result["scf_converged"] = False

    m_key = method.lower()
    if m_key in _POST_HF_ARCHIVE_KEYS:
        archive = _parse_archive(log_text)
        total = None
        for key in _POST_HF_ARCHIVE_KEYS[m_key]:
            if key in archive:
                total = archive[key]
                break
        if total is None and m_key == "mp2":
            mp2 = re.findall(rf"EUMP2\s*=\s*({_FLOAT})", log_text)
            if mp2:
                total = _to_float(mp2[-1])
        if total is None and m_key in ("ccsd(t)", "ccsdt"):
            cc = re.findall(rf"CCSD\(T\)\s*=\s*({_FLOAT})", log_text)
            if cc:
                total = _to_float(cc[-1])
        result["energy_hartree"] = total
        if total is not None and result["scf_energy_hartree"] is not None:
            result["correlation_energy_hartree"] = total - result["scf_energy_hartree"]
    else:
        result["energy_hartree"] = result["scf_energy_hartree"]

    occ, virt = _parse_orbital_energies(log_text)
    if occ:
        result["homo_energy"] = max(occ)
    if virt:
        result["lumo_energy"] = min(virt)

    dip = re.findall(
        rf"X=\s*({_FLOAT})\s+Y=\s*({_FLOAT})\s+Z=\s*({_FLOAT})\s+Tot=\s*({_FLOAT})", log_text
    )
    if dip:
        x, y, z, tot = dip[-1]
        result["dipole"] = [float(x), float(y), float(z)]
        result["dipole_magnitude"] = float(tot)

    if "Error termination" in log_text:
        err_line = [ln.strip() for ln in log_text.split("\n") if "Error" in ln][:3]
        result["error"] = "; ".join(err_line)

    return result


@register_backend("gaussian")
class GaussianBackend(ExternalProgramBackend):
    """Gaussian 16 后端。"""

    program_name = "gaussian"

    def __init__(self, executable: str = ""):
        super().__init__(executable)

    def capabilities(self):
        caps = super().capabilities()
        caps.tasks = ["single_point", "optimize", "frequency"]
        caps.methods = list(_METHOD_MAP.keys())
        caps.basis_sets = ["sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-311g(d,p)",
                           "cc-pvdz", "cc-pvtz", "def2-svp", "def2-tzvp"]
        return caps

    def single_point(self, request: CalculationRequest) -> Result:
        return self._run_gaussian_task(request, keyword="SP")

    def optimize(self, request: CalculationRequest) -> Result:
        """Gaussian 几何优化。"""
        return self._run_gaussian_task(request, keyword="Opt")

    def frequency(self, request: CalculationRequest) -> Result:
        """Gaussian 频率计算。"""
        return self._run_gaussian_task(request, keyword="Freq")

    def _run_gaussian_task(self, request: CalculationRequest, keyword: str) -> Result:
        """运行 Gaussian SP/Opt/Freq 任务并构造统一 Result。"""
        started = datetime.now()
        caps = self.capabilities()
        if not caps.available:
            raise BackendExecutionError(f"Gaussian not available: {caps.unavailable_reason}")

        job_id = request.job_id or f"g16_{keyword}_{int(time.time())}"
        run_dir = self._make_run_dir(self.default_run_base(), job_id)

        gjf = _build_gjf(
            request,
            memory_mb=request.resources.max_memory_mb,
            nproc=request.resources.max_cpu_cores,
            extra_keyword="" if keyword == "SP" else keyword,
        )
        gjf_path = run_dir / "input.gjf"
        gjf_path.write_text(gjf)

        g16_dir = Path(self.executable).parent
        env = os.environ.copy()
        env["g16root"] = str(g16_dir)
        env["GAUSS_EXEDIR"] = str(g16_dir)
        env["GAUSS_SCRDIR"] = str(run_dir)
        if str(g16_dir) not in env.get("PATH", ""):
            env["PATH"] = str(g16_dir) + os.pathsep + env.get("PATH", "")

        # _run_command 负责超时后结束整个进程树
        returncode, stdout, _stderr = self._run_command(
            [self.executable, gjf_path.name], run_dir,
            timeout=request.resources.timeout_seconds, env=env,
        )

        # 读日志（Gaussian Windows 版输出 .out，Linux 版 .log）
        log_path = run_dir / "input.out"
        if not log_path.exists():
            log_path = run_dir / "input.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else (stdout or "")

        return self._build_result(request, keyword, log_text, returncode, started,
                                  log_path, gjf_path, run_dir)

    def _build_result(self, request, keyword, log_text, returncode, started,
                      log_path: Path, gjf_path: Path, run_dir: Path) -> Result:
        parsed = _parse_g16_log(log_text, request.method)
        walltime = (datetime.now() - started).total_seconds()

        errors: list[str] = []
        warnings: list[str] = []
        if not parsed["normal_termination"]:
            errors.append(f"Gaussian {keyword} did not terminate normally (rc={returncode})")
        if parsed["error"]:
            errors.append(parsed["error"])
        if parsed["energy_hartree"] is None:
            if request.method.lower() in _POST_HF_ARCHIVE_KEYS:
                errors.append(f"Could not extract {request.method.upper()} total energy "
                              "(SCF energy is not substituted)")
            else:
                errors.append("Could not extract energy from Gaussian log")
        if parsed["scf_converged"] is False:
            errors.append("Gaussian SCF convergence failure")

        optimized_coords = None
        opt_not_converged = False
        if keyword == "Opt":
            optimized_coords = _parse_final_geometry(log_text)
            if optimized_coords is None:
                errors.append("Could not parse optimized geometry")
            if not parsed["opt_converged"]:
                opt_not_converged = True
                warnings.append("Gaussian optimization did not report 'Stationary point found'")

        frequencies = None
        if keyword == "Freq":
            frequencies = _parse_frequencies(log_text)
            if frequencies is None:
                errors.append("Could not parse frequencies")

        if errors:
            status = "failed"
        elif opt_not_converged:
            status = "not_converged"
        else:
            status = "succeeded"

        corr = parsed["correlation_energy_hartree"]
        m = request.method.lower()
        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="gaussian",
            backend_version="G16",
            task=request.task,
            method=request.method,
            actual_method=_METHOD_MAP.get(m, request.method.upper()),
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=parsed["scf_energy_hartree"],
            mp2_correction_hartree=corr if m == "mp2" else None,
            ccsd_correction_hartree=corr if m == "ccsd" else None,
            total_energy_hartree=parsed["energy_hartree"],
            scf_converged=parsed["scf_converged"],
            scf_iterations=parsed["scf_iterations"],
            homo_energy_hartree=parsed["homo_energy"],
            lumo_energy_hartree=parsed["lumo_energy"],
            dipole_debye=parsed["dipole"],
            dipole_magnitude_debye=parsed["dipole_magnitude"],
            optimized_coordinates_angstrom=optimized_coords,
            frequencies_cm=frequencies,
            errors=errors,
            warnings=warnings,
            raw_log_path=str(log_path) if log_path.exists() else None,
            input_file_path=str(gjf_path),
            run_directory=str(run_dir),
            extra={"correlation_energy_hartree": corr} if corr is not None else {},
        )


def _parse_orientation_blocks(log_text: str, header: str) -> list[list[list[float]]]:
    blocks = []
    lines = log_text.split("\n")
    i = 0
    while i < len(lines):
        if header in lines[i]:
            block_lines = []
            j = i + 1
            while j < len(lines) and "----" not in lines[j]:
                j += 1
            j += 1  # 跳过 -----
            while j < len(lines) and "----" not in lines[j]:
                j += 1  # 跳过表头
            j += 1
            while j < len(lines) and "----" not in lines[j]:
                parts = lines[j].split()
                if len(parts) >= 6:
                    block_lines.append([float(parts[3]), float(parts[4]), float(parts[5])])
                j += 1
            if block_lines:
                blocks.append(block_lines)
            i = j
        else:
            i += 1
    return blocks


def _parse_final_geometry(log_text: str) -> list[list[float]] | None:
    """提取最后一次几何。优先 Input orientation（保持用户坐标系），否则 Standard orientation。"""
    for header in ("Input orientation:", "Standard orientation:"):
        blocks = _parse_orientation_blocks(log_text, header)
        if blocks:
            return blocks[-1]
    return None


def _parse_frequencies(log_text: str) -> list[float] | None:
    """从 Gaussian 日志提取振动频率（cm^-1）。"""
    freqs = re.findall(r"Frequencies\s*--\s*((?:-?\d+\.\d+[ \t]*)+)", log_text)
    if not freqs:
        return None
    result = []
    for line in freqs:
        result.extend(float(x) for x in line.split())
    return result if result else None
