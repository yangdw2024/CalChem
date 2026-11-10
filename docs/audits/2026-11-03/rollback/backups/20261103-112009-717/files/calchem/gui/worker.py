"""Qt worker thread: 在后台执行 TaskService.run()，通过信号回传。"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest


class CalcWorker(QObject):
    """后台计算 worker。moveToThread 后在子线程运行。"""

    finished = pyqtSignal(str)   # JSON result
    failed = pyqtSignal(str, str)  # error_type, message
    log = pyqtSignal(str)

    def __init__(self, service: TaskService, request: CalculationRequest):
        super().__init__()
        self.service = service
        self.request = request
        self._cancelled = False

    def run(self):
        self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 开始计算: {self.request.backend}/{self.request.method}/{self.request.basis}")
        try:
            result = self.service.run(self.request)
            self.service.save_result(result, self.request.job_id)
            if self._cancelled:
                result.status = "cancelled"
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 计算完成: {result.status}")
            self.finished.emit(result.model_dump_json())
        except Exception as e:
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 错误: {e}")
            self.failed.emit(type(e).__name__, f"{e}\n{traceback.format_exc()}")

    def cancel(self):
        self._cancelled = True


def start_calc_thread(service: TaskService, request: CalculationRequest,
                      on_finished, on_failed=None, on_log=None):
    """启动一个计算线程，返回 (thread, worker)。"""
    thread = QThread()
    worker = CalcWorker(service, request)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(on_finished)
    if on_failed:
        worker.failed.connect(on_failed)
    if on_log:
        worker.log.connect(on_log)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker
