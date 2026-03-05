"""API tests for model endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_model_service
from backend.app.schemas.models import ModelInfo, ModelStatus, ModelWeightsInfo
from backend.app.core.exceptions import ModelNotFoundError, ModelNotTrainedError


client = TestClient(app)

# ---------------------------------------------------------------------------
# Dummy service
# ---------------------------------------------------------------------------

_LSTM_INFO = ModelInfo(
    model_key="lstm",
    model_type="lstm",
    display_name="LSTM",
    status=ModelStatus.TRAINED,
    is_pinn=False,
)

_PINN_INFO = ModelInfo(
    model_key="pinn_gbm",
    model_type="pinn_gbm",
    display_name="PINN GBM",
    status=ModelStatus.NOT_TRAINED,
    is_pinn=True,
)


class _DummyModelService:
    """Minimal model service stub for API tests."""

    def get_all_models(self):
        return [_LSTM_INFO, _PINN_INFO]

    def get_trained_models(self):
        return [_LSTM_INFO]

    def get_available_model_types(self):
        return ["lstm", "gru", "pinn_gbm"]

    def get_model_info(self, model_key: str):
        if model_key == "lstm":
            return _LSTM_INFO
        raise ModelNotFoundError(model_key)

    def get_model_weights_info(self, model_key: str):
        if model_key == "lstm":
            return ModelWeightsInfo(
                model_key="lstm",
                total_parameters=100_000,
                trainable_parameters=100_000,
                layer_info=[{"name": "lstm_0", "shape": [64, 64], "parameters": 16_384, "trainable": True}],
            )
        raise ModelNotFoundError(model_key)

    def compare_models(self, keys):
        return {
            "models": [
                {
                    "model_key": "lstm",
                    "display_name": "LSTM",
                    "is_pinn": False,
                    "metrics": {"rmse": 0.05},
                }
            ],
            "metric_names": ["rmse"],
            "best_by_metric": {"rmse": "lstm"},
        }

    def load_model(self, model_key, device=None):
        if model_key == "missing":
            raise ModelNotFoundError(model_key)

    def unload_model(self, model_key):
        pass

    def list_saved_checkpoints(self):
        return [{"name": "lstm_best.pt", "size_mb": 1.5}]

    def rename_checkpoint(self, old_name, new_name):
        if old_name == "missing.pt":
            raise FileNotFoundError(f"Checkpoint {old_name} not found")
        return {"message": f"Renamed {old_name} to {new_name}"}

    def delete_checkpoint(self, name):
        if name == "missing.pt":
            raise FileNotFoundError(f"Checkpoint {name} not found")
        return {"message": f"Deleted {name}"}


def _apply():
    app.dependency_overrides[get_model_service] = _DummyModelService


def _clear():
    app.dependency_overrides.pop(get_model_service, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_models_returns_all():
    _apply()
    resp = client.get("/api/models/")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["trained_count"] == 1
    assert data["pinn_count"] == 1
    assert len(data["models"]) == 2


def test_list_trained_models():
    _apply()
    resp = client.get("/api/models/trained")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["trained_count"] == 1
    keys = [m["model_key"] for m in data["models"]]
    assert keys == ["lstm"]


def test_get_model_types():
    _apply()
    resp = client.get("/api/models/types")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert "model_types" in data
    assert "categories" in data
    assert "baseline" in data["categories"]


def test_get_model_found():
    _apply()
    resp = client.get("/api/models/lstm")
    _clear()

    assert resp.status_code == 200
    assert resp.json()["model_key"] == "lstm"


def test_get_model_not_found():
    _apply()
    resp = client.get("/api/models/nonexistent")
    _clear()

    assert resp.status_code == 404


def test_get_model_weights():
    _apply()
    resp = client.get("/api/models/lstm/weights")
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_parameters"] == 100_000
    assert len(data["layer_info"]) == 1


def test_get_model_weights_not_found():
    _apply()
    resp = client.get("/api/models/nonexistent/weights")
    _clear()

    assert resp.status_code == 404


def test_load_model_success():
    _apply()
    resp = client.post("/api/models/lstm/load")
    _clear()

    assert resp.status_code == 200
    assert "loaded" in resp.json()["message"].lower()


def test_load_model_not_found():
    _apply()
    resp = client.post("/api/models/missing/load")
    _clear()

    assert resp.status_code == 404


def test_unload_model():
    _apply()
    resp = client.post("/api/models/lstm/unload")
    _clear()

    assert resp.status_code == 200
    assert "unloaded" in resp.json()["message"].lower()


def test_compare_models():
    _apply()
    resp = client.get("/api/models/compare", params={"model_keys": "lstm"})
    _clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["metric_names"] == ["rmse"]
    assert len(data["models"]) == 1
