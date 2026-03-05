"""API tests for metrics endpoints."""

from datetime import datetime

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_metrics_service
from backend.app.schemas.metrics import (
    FinancialMetrics,
    MLMetrics,
    PhysicsMetrics,
    ModelMetricsResponse,
    MetricsComparisonResponse,
    LeaderboardResponse,
    LeaderboardEntry,
)
from backend.app.core.exceptions import ModelNotFoundError


client = TestClient(app)

# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------

_FINANCIAL = FinancialMetrics(
    total_return=0.15,
    annual_return=0.12,
    daily_return_mean=0.0005,
    daily_return_std=0.02,
    sharpe_ratio=1.5,
    sortino_ratio=2.0,
    max_drawdown=-0.10,
    win_rate=0.55,
)

_ML = MLMetrics(
    rmse=0.05,
    mae=0.03,
    mape=2.5,
    r2=0.85,
    directional_accuracy=0.62,
)

_PHYSICS = PhysicsMetrics(
    total_physics_loss=0.001,
    gbm_loss=0.0005,
    ou_loss=0.0005,
)

_MODEL_METRICS = ModelMetricsResponse(
    model_key="lstm",
    model_name="LSTM",
    is_pinn=False,
    ml_metrics=_ML,
    financial_metrics=_FINANCIAL,
)


# ---------------------------------------------------------------------------
# Dummy service
# ---------------------------------------------------------------------------


class _DummyMetricsService:
    """Minimal metrics service stub for API tests."""

    def calculate_financial_metrics(self, returns, risk_free_rate=0.02, periods_per_year=252, benchmark_returns=None):
        return _FINANCIAL

    def calculate_ml_metrics(self, y_true, y_pred):
        return _ML

    def get_physics_metrics(self, model_key):
        if model_key == "pinn_gbm":
            return _PHYSICS
        return None

    def get_model_metrics(self, model_key):
        if model_key == "lstm":
            return _MODEL_METRICS
        raise ModelNotFoundError(model_key)

    def compare_models(self, keys):
        return MetricsComparisonResponse(
            models=[_MODEL_METRICS],
            metric_summary={"rmse": {"lstm": 0.05}},
            best_by_metric={"rmse": "lstm"},
            rankings={"rmse": ["lstm"]},
        )

    def load_saved_metrics(self, model_key):
        if model_key == "lstm":
            return {"rmse": 0.05}
        return None

    def get_leaderboard(self, metric="sharpe_ratio", top_n=10):
        return LeaderboardResponse(
            metric=metric,
            generated_at=datetime(2024, 6, 1),
            n_experiments=1,
            entries=[
                LeaderboardEntry(
                    rank=1,
                    experiment_id="exp-001",
                    model_name="LSTM",
                    metric_value=1.5,
                    metric_name=metric,
                    other_metrics={"rmse": 0.05},
                )
            ],
        )


def _apply():
    app.dependency_overrides[get_metrics_service] = _DummyMetricsService


def _clear():
    app.dependency_overrides.pop(get_metrics_service, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_financial_metrics_get():
    _apply()
    resp = client.get("/api/metrics/financial", params={"returns": "0.01,-0.02,0.015"})
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["sharpe_ratio"] == 1.5
    assert data["total_return"] == 0.15


def test_financial_metrics_invalid_format():
    _apply()
    resp = client.get("/api/metrics/financial", params={"returns": "abc,def"})
    _clear()

    assert resp.status_code == 400


def test_financial_metrics_post():
    _apply()
    resp = client.post(
        "/api/metrics/financial",
        json={"returns": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.003, -0.005, 0.01, 0.008]},
    )
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["sharpe_ratio"] == 1.5
    assert data["input_summary"]["n_returns"] == 10


def test_ml_metrics():
    _apply()
    resp = client.get("/api/metrics/ml", params={"y_true": "1.0,2.0,3.0", "y_pred": "1.1,2.1,2.9"})
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["rmse"] == 0.05
    assert data["directional_accuracy"] == 0.62


def test_ml_metrics_length_mismatch():
    _apply()
    resp = client.get("/api/metrics/ml", params={"y_true": "1.0,2.0", "y_pred": "1.1"})
    _clear()

    assert resp.status_code == 400
    assert "same length" in resp.json()["detail"].lower()


def test_physics_metrics_found():
    _apply()
    resp = client.get("/api/metrics/physics/pinn_gbm")
    _clear()

    assert resp.status_code == 200
    assert resp.json()["total_physics_loss"] == 0.001


def test_physics_metrics_not_pinn():
    _apply()
    resp = client.get("/api/metrics/physics/lstm")
    _clear()

    assert resp.status_code == 400


def test_model_metrics():
    _apply()
    resp = client.get("/api/metrics/model/lstm")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_key"] == "lstm"
    assert data["ml_metrics"]["rmse"] == 0.05


def test_model_metrics_not_found():
    _apply()
    resp = client.get("/api/metrics/model/nonexistent")
    _clear()

    assert resp.status_code == 404


def test_compare_metrics():
    _apply()
    resp = client.get("/api/metrics/comparison", params={"model_keys": "lstm"})
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["best_by_metric"]["rmse"] == "lstm"
    assert len(data["models"]) == 1


def test_saved_metrics_found():
    _apply()
    resp = client.get("/api/metrics/saved/lstm")
    _clear()

    assert resp.status_code == 200
    assert resp.json()["rmse"] == 0.05


def test_saved_metrics_not_found():
    _apply()
    resp = client.get("/api/metrics/saved/nonexistent")
    _clear()

    assert resp.status_code == 404


def test_leaderboard():
    _apply()
    resp = client.get("/api/metrics/leaderboard", params={"metric": "sharpe_ratio", "top_n": 5})
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "sharpe_ratio"
    assert data["n_experiments"] == 1
    assert data["entries"][0]["rank"] == 1
