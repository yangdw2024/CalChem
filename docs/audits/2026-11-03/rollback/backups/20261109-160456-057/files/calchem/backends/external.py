"""外部程序后端基类：run directory、subprocess、stdout/stderr 捕获。"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..core.calculator import Calculator
from ..core.capabilities import BackendCapabilities
from ..core.errors import BackendExecutionError
from ..core.request import CalculationRequest


class ExternalProgramBackend(Calculator):
    """所有外部量子化学程序（Gaussian/ORCA/xTB/REST）的基类。"""

    #: 子类必须设置：可执行文件路径
    executable: str = ""
    #: 子类必须设置：程序名
    program_name: str = "external"

    def __init__(self, executable: Optional[str] = None):
        if executable:
            self.executable = executable
        self._version: Optional[str] = None

    def _detect_version(self) -> Optional[str]:
        """运行 --version 或类似命令获取版本。子类可覆盖。"""
        return None

    def _check_available(self) -> tuple[bool, Optional[str]]:
        """检查可执行文件是否存在。"""
        if not self.executable:
            return False, "executable path not configured"
        p = Path(self.executable)
        if not p.exists():
            return False, f"executable not found: {self.executable}"
        return True, None

    def _run_command(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int = 3600,
        env: Optional[dict] = None,
    ) -> tuple[int, str, str]:
        """执行外部命令，返回 (returncode, stdout, stderr)。

        超时后 kill 整个进程树（Windows 上用 taskkill /T）。
        """
        import os
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            shell=False,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            # Windows: kill 整个进程树
            try:
                pid = proc.pid
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True, timeout=10)
                else:
                    proc.kill()
            except Exception:
                proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            raise BackendExecutionError(
                f"{self.program_name} timed out after {timeout}s"
            )
        except FileNotFoundError:
            raise BackendExecutionError(
                f"{self.program_name} executable not found: {self.executable}"
            )

    def _make_run_dir(self, base_dir: Path, job_id: str) -> Path:
        """创建独立的运行目录。"""
        run_dir = base_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def default_run_base() -> Path:
        """外部程序运行根目录。

        优先级：CALCHEM_RUN_DIR > CALCHEM_DATA/external_runs > D:\\CalChemData\\external_runs。
        不再回退到系统临时目录（默认在 C 盘）。
        """
        env_run = os.environ.get("CALCHEM_RUN_DIR")
        if env_run:
            return Path(env_run) / "calchem_jobs"
        data = os.environ.get("CALCHEM_DATA") or r"D:\CalChemData"
        return Path(data) / "external_runs"

    def capabilities(self) -> BackendCapabilities:
        available, reason = self._check_available()
        return BackendCapabilities(
            name=self.program_name,
            version=self._detect_version() if available else None,
            executable_path=self.executable,
            requires_external_program=True,
            available=available,
            unavailable_reason=reason,
        )
