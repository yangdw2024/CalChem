"""JobDispatcher: 真正消费队列的单并发调度器。

所有入口（CLI/GUI/API）都通过 dispatcher 运行 heavy job，
保证同一进程内最多一个 heavy 任务。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .models import Job, JobState
from .queue import JobQueue
from ..core.request import CalculationRequest
from ..core.result import Result

# 避免循环导入：TaskService 用 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..api.service import TaskService


class JobDispatcher:
    """单并发作业调度器。独立线程消费队列。"""

    def __init__(self, service, queue: Optional[JobQueue] = None):
        self.service = service
        self.queue = queue if queue is not None else getattr(service, "queue", JobQueue())
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_job: Optional[Job] = None
        self._callbacks: list[Callable[[Job], None]] = []

    def add_callback(self, cb: Callable[[Job], None]):
        self._callbacks.append(cb)

    def submit(self, request: CalculationRequest) -> Job:
        """提交任务到队列。"""
        job = self.queue.submit(request.model_dump_json(), request.project_id)
        self._ensure_running()
        return job

    def run_now(self, request: CalculationRequest) -> Result:
        """同步执行（绕过队列，用于 CLI 即时运行）。

        仍然通过 _lock 保证不与后台 worker 并发。
        """
        with self._lock:
            return self.service.run(request)

    def _ensure_running(self):
        if not self._running and not self._stop_event.is_set():
            self._running = True
            self._worker_thread = threading.Thread(target=self._work_loop, daemon=True)
            self._worker_thread.start()

    def _work_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                job = self.queue.get_next()

            if job is None:
                time.sleep(0.5)
                continue

            self._current_job = job
            try:
                from ..core.request import CalculationRequest
                req = CalculationRequest.model_validate_json(job.request_json)
                result = self.service.run(req)
                self.service.save_result(result, job.job_id)
                self.queue.complete(job.job_id, result.model_dump_json(),
                                    success=(result.status == "succeeded"),
                                    not_converged=(result.status == "not_converged"))
            except Exception as e:
                self.queue.complete(job.job_id, error=str(e), success=False)
            finally:
                self._current_job = None
                for cb in self._callbacks:
                    try:
                        cb(job)
                    except Exception:
                        pass

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    @property
    def current_job(self) -> Optional[Job]:
        return self._current_job
