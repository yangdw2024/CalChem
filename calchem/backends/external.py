"""外部程序后端基类：run directory、subprocess、stdout/stderr 捕获。"""

from __future__ import annotations

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
    ) -> tuple[int, str, str]:
        """执行外部命令，返回 (returncode, stdout, stderr)。

        不使用 shell=True，cmd 是参数列表。
        """
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
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
