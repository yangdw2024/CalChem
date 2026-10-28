"""项目文件 IO：.calchem 目录格式。"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..core.request import CalculationRequest
from ..core.result import Result


PROJECT_SCHEMA_VERSION = "1.0"


class ProjectStore:
    """.calchem 项目目录管理。

    格式：
    water.calchem/
      project.json
      molecules/*.xyz
      jobs/<job-id>/request.json, result.json, manifest.json, input.*, program.log
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.project_file = self.path / "project.json"

    def create(self, name: str, description: str = "") -> dict:
        """创建新项目目录。"""
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "molecules").mkdir(exist_ok=True)
        (self.path / "jobs").mkdir(exist_ok=True)
        project = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": f"proj_{uuid.uuid4().hex[:8]}",
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "jobs": [],
        }
        self._save_project(project)
        return project

    def load(self) -> dict:
        if not self.project_file.exists():
            raise FileNotFoundError(f"项目不存在: {self.project_file}")
        return json.loads(self.project_file.read_text(encoding="utf-8"))

    def save_request(self, request: CalculationRequest, job_id: str) -> Path:
        """保存请求到 jobs/<job-id>/request.json。"""
        job_dir = self.path / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        req_path = job_dir / "request.json"
        req_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return req_path

    def save_result(self, result: Result, job_id: str) -> Path:
        """保存结果和 manifest。"""
        job_dir = self.path / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        result_path = job_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        # manifest
        manifest = {
            "job_id": job_id,
            "calchem_version": "0.1.0",
            "schema_version": PROJECT_SCHEMA_VERSION,
            "backend": result.backend,
            "backend_version": result.backend_version,
            "method": result.method,
            "basis": result.basis,
            "status": result.status,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "finished_at": result.finished_at.isoformat() if result.finished_at else None,
            "walltime_seconds": result.walltime_seconds,
            "scf_energy_hartree": result.scf_energy_hartree,
            "run_directory": result.run_directory,
            "raw_log_path": result.raw_log_path,
            "input_file_path": result.input_file_path,
        }
        (job_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 更新 project.json
        proj = self.load()
        if job_id not in proj["jobs"]:
            proj["jobs"].append(job_id)
        proj["updated_at"] = datetime.now().isoformat()
        self._save_project(proj)

        return result_path

    def list_jobs(self) -> List[dict]:
        """列出所有 job 的 manifest。"""
        proj = self.load()
        results = []
        for job_id in proj.get("jobs", []):
            manifest_path = self.path / "jobs" / job_id / "manifest.json"
            if manifest_path.exists():
                results.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        return results

    def _save_project(self, proj: dict):
        self.project_file.write_text(
            json.dumps(proj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
