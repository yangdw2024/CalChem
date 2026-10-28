"""TaskService: CLI/GUI/API 共用的统一任务入口。

所有计算都通过 TaskService.run() 或 TaskService.submit() 执行。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.calculator import get_backend_class
from ..core.errors import CalChemError
from ..core.request import CalculationRequest
from ..core.result import Result
from ..config.models import AppConfig
from ..jobs.models import Job
from ..jobs.queue import JobQueue


class TaskService:
    """统一任务服务。验证请求、选择后端、执行、保存结果。"""

    def __init__(self, data_root: Optional[str] = None, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.data_root = Path(
            data_root
            or self.config.data_root
            or os.environ.get("CALCHEM_DATA")
            or r"D:\CalChemData"
        )
        self.run_root = self.data_root / "runs"
        self.projects_root = self.data_root / "projects"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.queue = JobQueue()

    def _create_backend(self, name: str):
        """根据 AppConfig 创建后端实例，注入 executable 路径。"""
        backend_cls = get_backend_class(name)
        if name == "gaussian":
            return backend_cls(executable=self.config.backends.gaussian_exe)
        elif name == "orca":
            return backend_cls(executable=self.config.backends.orca_exe)
        elif name == "xtb":
            return backend_cls(executable=self.config.backends.xtb_exe)
        else:
            return backend_cls()

    def run(self, request: CalculationRequest) -> Result:
        """同步执行一个计算请求（阻塞直到完成）。"""
        backend = self._create_backend(request.backend)

        try:
            result = backend.run(request)
        except CalChemError as e:
            result = Result(
                backend=request.backend,
                task=request.task,
                method=request.method,
                basis=request.basis,
                molecule=request.molecule,
                status="failed",
                started_at=datetime.now(),
                finished_at=datetime.now(),
                errors=[str(e)],
            )

        return result

    def submit(self, request: CalculationRequest) -> Job:
        """提交到队列（异步）。返回 Job 对象。"""
        request_json = request.model_dump_json()
        return self.queue.submit(request_json, request.project_id)

    def list_backends(self):
        """列出所有后端能力。"""
        from ..core.calculator import list_backend_capabilities
        return list_backend_capabilities()

    def save_result(self, result: Result, job_id: Optional[str] = None) -> Path:
        """把 Result 保存到磁盘。"""
        job_id = job_id or result.job_id or f"result_{int(datetime.now().timestamp())}"
        job_dir = self.run_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        result_path = job_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2))
        return result_path

    def run_and_save(self, request: CalculationRequest) -> Result:
        """执行计算并保存结果。"""
        result = self.run(request)
        self.save_result(result, request.job_id)
        return result
