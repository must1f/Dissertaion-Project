"""API tests for training endpoints."""

from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_training_service
from backend.app.schemas.training import (
    TrainingStatus,
    TrainingJobInfo,
    EpochMetrics,
    AVAILABLE_MODELS,
)


client = TestClient(app)

# ---------------------------------------------------------------------------
# Dummy service
# ---------------------------------------------------------------------------

_JOB = TrainingJobInfo(
    job_id="job-001",
    model_type="lstm",
    ticker="^GSPC",
    status=TrainingStatus.RUNNING,
    current_epoch=5,
    total_epochs=100,
    progress_percent=5.0,
    started_at=datetime(2024, 6, 1),
    best_val_loss=0.03,
    config={},
)


class _DummyTrainingService:
    """Minimal training service stub for API tests."""

    def start_training(self, request):
        return "job-001"

    def stop_training(self, job_id):
        return job_id == "job-001"

    def get_job_status(self, job_id):
        if job_id == "job-001":
            return _JOB
        return None

    def get_job_history(self, job_id):
        return {"train_loss": [0.1, 0.08], "val_loss": [0.12, 0.09]}

    def get_epoch_metrics(self, job_id):
        return [
            EpochMetrics(
                epoch=1,
                train_loss=0.1,
                val_loss=0.12,
                learning_rate=0.001,
                epoch_time_seconds=5.0,
            ),
            EpochMetrics(
                epoch=2,
                train_loss=0.08,
                val_loss=0.09,
                learning_rate=0.001,
                epoch_time_seconds=5.0,
            ),
        ]

    def list_jobs(self, status=None, page=1, page_size=20):
        runs = [_JOB] if status is None or status == TrainingStatus.RUNNING else []
        return {
            "runs": runs,
            "total": len(runs),
            "page": page,
            "page_size": page_size,
        }


def _apply():
    app.dependency_overrides[get_training_service] = _DummyTrainingService


def _clear():
    app.dependency_overrides.pop(get_training_service, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_training_mode():
    resp = client.get("/api/training/mode")
    assert resp.status_code == 200
    data = resp.json()
    assert "mode" in data
    assert "using_real_models" in data


def test_start_training():
    _apply()
    resp = client.post(
        "/api/training/start",
        json={
            "model_type": "lstm",
            "ticker": "^GSPC",
            "epochs": 10,
        },
    )
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["job_id"] == "job-001"
    assert "/api/ws/training/" in data["websocket_url"]


def test_stop_training_success():
    _apply()
    resp = client.post("/api/training/stop/job-001")
    _clear()

    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_stop_training_not_found():
    _apply()
    resp = client.post("/api/training/stop/nonexistent")
    _clear()

    assert resp.status_code == 404


def test_get_training_status():
    _apply()
    resp = client.get("/api/training/status/job-001")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["job"]["job_id"] == "job-001"
    assert data["job"]["status"] == "running"


def test_get_training_status_not_found():
    _apply()
    resp = client.get("/api/training/status/nonexistent")
    _clear()

    assert resp.status_code == 404


def test_get_training_history_for_job():
    _apply()
    resp = client.get("/api/training/history/job-001")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-001"
    assert len(data["epochs"]) == 2
    assert data["best_epoch"] == 2
    assert data["best_val_loss"] == 0.09


def test_list_training_runs():
    _apply()
    resp = client.get("/api/training/history")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["runs"][0]["job_id"] == "job-001"


def test_get_active_jobs():
    _apply()
    resp = client.get("/api/training/active")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1


def test_get_available_batch_models():
    resp = client.get("/api/training/batch/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(AVAILABLE_MODELS)
    assert "by_type" in data
