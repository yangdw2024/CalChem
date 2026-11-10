"""Tests for TaskService, JobQueue state transitions, and file persistence."""
import json
import pytest
from pathlib import Path

from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.jobs.models import JobState
from calchem.jobs.queue import JobQueue


_H2_MOL = MoleculeSpec(
    symbols=["H", "H"],
    coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7408480952641999]],
    charge=0,
    multiplicity=1,
)


def test_task_service_respects_data_root(tmp_path: Path):
    """Q07: Explicit or environment data_root must be respected without shadowing."""
    custom_dir = tmp_path / "custom_data"
    service = TaskService(data_root=str(custom_dir))
    assert service.data_root == custom_dir
    assert service.run_root == custom_dir / "runs"


def test_task_service_run_and_save_creates_request_result_manifest(tmp_path: Path):
    """Q09: run_and_save must persist result.json, request.json, and manifest.json with job_id."""
    service = TaskService(data_root=str(tmp_path / "service_data"))
    req = CalculationRequest(
        task="single_point",
        molecule=_H2_MOL,
        backend="native",
        method="rhf",
        basis="sto-3g",
        job_id="test_job_001",
    )
    result = service.run_and_save(req)
    assert result.status == "succeeded"
    assert result.job_id == "test_job_001"

    job_dir = service.run_root / "test_job_001"
    assert (job_dir / "result.json").exists()
    assert (job_dir / "request.json").exists()
    assert (job_dir / "manifest.json").exists()

    manifest_data = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["job_id"] == "test_job_001"
    assert manifest_data["status"] == "succeeded"


def test_job_queue_cancel_transition_on_complete():
    """Q08: Cancel requested while running transitions to CANCELLED upon complete."""
    queue = JobQueue()
    job = queue.submit('{"dummy": true}')
    assert job.state == JobState.QUEUED

    # Start job
    running_job = queue.get_next()
    assert running_job is not None
    assert running_job.state == JobState.RUNNING
    assert queue.running_job_id == running_job.job_id

    # Request cancellation
    cancelled = queue.cancel(running_job.job_id)
    assert cancelled is True
    assert running_job.state == JobState.CANCEL_REQUESTED

    # When runner completes, it must finalize state as CANCELLED (not stuck at cancel_requested)
    queue.complete(running_job.job_id, success=True)
    assert running_job.state == JobState.CANCELLED
    assert queue.running_job_id is None


def test_service_and_dispatcher_share_queue():
    """Q08: TaskService.submit and JobDispatcher must operate on the same queue."""
    service = TaskService()
    assert service.queue is service.dispatcher.queue
