"""Job 状态机模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_CONVERGED = "not_converged"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


# 合法状态转移
_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.PENDING: {JobState.QUEUED, JobState.CANCELLED},
    JobState.QUEUED: {JobState.RUNNING, JobState.CANCELLED},
    JobState.RUNNING: {
        JobState.SUCCEEDED, JobState.FAILED,
        JobState.NOT_CONVERGED, JobState.CANCEL_REQUESTED,
    },
    JobState.CANCEL_REQUESTED: {JobState.CANCELLED, JobState.FAILED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.NOT_CONVERGED: {JobState.RUNNING},  # 允许重试
    JobState.CANCELLED: set(),
}


class Job(BaseModel):
    """一个计算任务。"""

    job_id: str
    state: JobState = JobState.PENDING
    project_id: Optional[str] = None

    # 请求快照（JSON 字符串）
    request_json: str = ""

    # 结果
    result_json: Optional[str] = None
    error: Optional[str] = None

    # 时间
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # 运行目录
    run_directory: Optional[str] = None

    def can_transition_to(self, new_state: JobState) -> bool:
        return new_state in _TRANSITIONS.get(self.state, set())

    def transition_to(self, new_state: JobState) -> bool:
        """尝试状态转移，返回是否成功。"""
        if self.can_transition_to(new_state):
            self.state = new_state
            if new_state == JobState.RUNNING:
                self.started_at = datetime.now()
            if new_state in (JobState.SUCCEEDED, JobState.FAILED,
                              JobState.NOT_CONVERGED, JobState.CANCELLED):
                self.finished_at = datetime.now()
            return True
        return False
