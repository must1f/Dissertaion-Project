import importlib
import sys
import types
import numpy as np
import pytest

from backend.app.config import settings
from backend.app.services import trading_service, model_service, prediction_service
from backend.app.schemas.trading import AgentConfigRequest
from backend.app.schemas.predictions import PredictionRequest
from backend.app.core.exceptions import PredictionError


def _make_agent(monkeypatch) -> trading_service.TradingAgent:
    """Create a trading agent with demo mode disabled."""
    monkeypatch.setattr(settings, "demo_mode", False)
    # Force ML dependencies to be treated as available to bypass init guard
    monkeypatch.setattr(trading_service, "HAS_ML", True)
    monkeypatch.setattr(trading_service.TradingAgent, "_init_ml_components", lambda self: None)
    config = AgentConfigRequest(model_key="lstm", ticker="^GSPC")
    return trading_service.TradingAgent(config)


def test_trading_mock_signal_disabled_without_demo(monkeypatch):
    agent = _make_agent(monkeypatch)
    with pytest.raises(RuntimeError):
        agent._generate_mock_signal()


def test_trading_mock_market_data_disabled_without_demo(monkeypatch):
    agent = _make_agent(monkeypatch)
    with pytest.raises(RuntimeError):
        agent._get_mock_market_data()


def test_model_service_mock_models_disabled_without_demo(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    svc = model_service.ModelService()
    with pytest.raises(RuntimeError):
        svc._get_mock_models()


def test_prediction_errors_when_src_missing_and_demo_off(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(prediction_service, "HAS_SRC", False)

    svc = prediction_service.PredictionService()

    class FakeData:
        def prepare_sequences(self, ticker, sequence_length):
            return np.zeros((1, sequence_length, 1)), None, None

        def get_stock_data(self, ticker):
            obj = types.SimpleNamespace()
            obj.data = [types.SimpleNamespace(close=100.0)]
            return obj

    svc._data_service = FakeData()

    req = PredictionRequest(ticker="^GSPC", model_key="test_model", sequence_length=60)
    with pytest.raises(PredictionError):
        svc.predict(req)


@pytest.mark.skip(reason="generate_analysis_data.py module was deleted")
def test_generate_predictions_requires_allow_synthetic(tmp_path):
    # Stub sklearn to avoid heavy dependency during dynamic import
    dummy_metrics = types.SimpleNamespace(
        mean_squared_error=lambda y_true, y_pred: 0.0,
        mean_absolute_error=lambda y_true, y_pred: 0.0,
        r2_score=lambda y_true, y_pred: 1.0,
    )
    dummy_scipy_stats = types.SimpleNamespace(
        norm=lambda *args, **kwargs: None,
        t=lambda *args, **kwargs: None,
    )
    class DummyLogger:
        def debug(self, *args, **kwargs): ...
        def info(self, *args, **kwargs): ...
        def warning(self, *args, **kwargs): ...
        def error(self, *args, **kwargs): ...
        def exception(self, *args, **kwargs): ...
        def bind(self, *args, **kwargs): return self

    dummy_logger = DummyLogger()

    sys.modules.setdefault("sklearn", types.SimpleNamespace())
    sys.modules["sklearn.metrics"] = dummy_metrics
    scipy_mod = types.SimpleNamespace(stats=dummy_scipy_stats)
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.stats"] = dummy_scipy_stats
    sys.modules["loguru"] = types.SimpleNamespace(logger=dummy_logger)

    ga = importlib.import_module("generate_analysis_data")
    AnalysisConfig = importlib.import_module("src.evaluation.analysis_utils").AnalysisConfig

    config = AnalysisConfig(project_root=tmp_path, results_dir=tmp_path)

    with pytest.raises(RuntimeError):
        ga.generate_predictions_from_results(
            "missing_model",
            n_samples=10,
            config=config,
            allow_synthetic=False,
        )

    preds, acts = ga.generate_predictions_from_results(
        "missing_model",
        n_samples=7,
        config=config,
        allow_synthetic=True,
    )
    assert len(preds) == 7
    assert len(acts) == 7
