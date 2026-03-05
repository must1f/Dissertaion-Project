"""API tests for prediction endpoints."""

from datetime import datetime

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_prediction_service
from backend.app.schemas.predictions import (
    PredictionResponse,
    PredictionResult,
    PredictionInterval,
    SignalAction,
)
from backend.app.core.exceptions import PredictionError


client = TestClient(app)

# ---------------------------------------------------------------------------
# Dummy service
# ---------------------------------------------------------------------------

_PRED_RESULT = PredictionResult(
    timestamp=datetime(2024, 6, 1),
    ticker="^GSPC",
    model_key="lstm",
    predicted_price=5100.0,
    predicted_return=0.01,
    current_price=5050.0,
    uncertainty_std=0.02,
    prediction_interval=PredictionInterval(lower=4950.0, upper=5200.0, confidence=0.95),
    confidence_score=0.8,
    signal_action=SignalAction.BUY,
    expected_return=0.01,
)


class _DummyPredictionService:
    """Minimal prediction service stub for API tests."""

    def __init__(self):
        self.cache_cleared = None

    def predict(self, request):
        if request.model_key == "broken":
            raise PredictionError("Model failed")
        return PredictionResponse(
            success=True,
            prediction=_PRED_RESULT,
            model_info={"model_key": request.model_key},
            processing_time_ms=42.0,
        )

    def batch_predict(self, tickers, model_key, sequence_length=60, estimate_uncertainty=True):
        return {
            "success": True,
            "predictions": [_PRED_RESULT],
            "failed_tickers": [],
            "total_processing_time_ms": 100.0,
        }

    def get_prediction_history(self, ticker=None, model_key=None, page=1, page_size=50):
        return {
            "predictions": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
        }

    def get_latest_predictions(self, ticker):
        return {"lstm": _PRED_RESULT}

    def clear_cache(self, model_key=None):
        self.cache_cleared = model_key


def _apply(factory=None):
    app.dependency_overrides[get_prediction_service] = factory or _DummyPredictionService


def _clear():
    app.dependency_overrides.pop(get_prediction_service, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_success():
    _apply()
    resp = client.post(
        "/api/predictions/predict",
        json={"ticker": "^GSPC", "model_key": "lstm"},
    )
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["prediction"]["predicted_price"] == 5100.0
    assert data["prediction"]["signal_action"] == "BUY"


def test_predict_error_returns_500():
    _apply()
    resp = client.post(
        "/api/predictions/predict",
        json={"ticker": "^GSPC", "model_key": "broken"},
    )
    _clear()

    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()


def test_prediction_history_empty():
    _apply()
    resp = client.get("/api/predictions/history")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["predictions"] == []
    assert data["total"] == 0


def test_prediction_history_with_pagination():
    _apply()
    resp = client.get(
        "/api/predictions/history",
        params={"page": 2, "page_size": 10},
    )
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert data["page_size"] == 10


def test_latest_predictions():
    _apply()
    resp = client.get("/api/predictions/%5EGSPC/latest")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "^GSPC"
    assert "lstm" in data["predictions"]
    assert data["consensus_signal"] == "BUY"


def test_clear_cache():
    svc = _DummyPredictionService()
    _apply(lambda: svc)
    resp = client.delete("/api/predictions/cache", params={"model_key": "lstm"})
    _clear()

    assert resp.status_code == 200
    assert svc.cache_cleared == "lstm"
    assert "lstm" in resp.json()["message"]


def test_clear_cache_all():
    svc = _DummyPredictionService()
    _apply(lambda: svc)
    resp = client.delete("/api/predictions/cache")
    _clear()

    assert resp.status_code == 200
    assert svc.cache_cleared is None
