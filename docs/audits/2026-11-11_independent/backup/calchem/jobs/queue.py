"""单并发 heavy job 队列。"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .models import Job, JobState


class JobQueue:
    """单并发作业队列。同时只允许一个 heavy job 运行。"""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running_job_id: Optional[str] = None

    def submit(self, request_json: str, project_id: Optional[str] = None) -> Job:
        """提交一个作业到队列。"""
        job_id = f"job_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        job = Job(job_id=job_id, request_json=request_json, project_id=project_id)
        with self._lock:
            self._jobs[job_id] = job
            job.transition_to(JobState.QUEUED)
        return job

    def get_next(self) -> Optional[Job]:
        """获取下一个要运行的作业（FIFO）。"""
        with self._lock:
            if self._running_job_id is not None:
                return None
            queued = [
                j for j in self._jobs.values()
                if j.state == JobState.QUEUED
            ]
            if not queued:
                return None
            queued.sort(key=lambda j: j.created_at)
            job = queued[0]
            job.transition_to(JobState.RUNNING)
            self._running_job_id = job.job_id
            return job

    def complete(self, job_id: str, result_json: Optional[str] = None,
                 error: Optional[str] = None, success: bool = True,
                 not_converged: bool = False):
        """标记作业完成。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.result_json = result_json
            job.error = error
            if job.state == JobState.CANCEL_REQUESTED:
                job.transition_to(JobState.CANCELLED)
            elif not_converged:
                job.transition_to(JobState.NOT_CONVERGED)
            elif success:
                job.transition_to(JobState.SUCCEEDED)
            else:
                job.transition_to(JobState.FAILED)
            if self._running_job_id == job_id:
                self._running_job_id = None

    def cancel(self, job_id: str) -> bool:
        """请求取消一个作业。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state == JobState.QUEUED:
                return job.transition_to(JobState.CANCELLED)
            elif job.state == JobState.RUNNING:
                return job.transition_to(JobState.CANCEL_REQUESTED)
            return False

    def list_jobs(self, project_id: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]
        return jobs

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    @property
    def running_job_id(self) -> Optional[str]:
        return self._running_job_id
