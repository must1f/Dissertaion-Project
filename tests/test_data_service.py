"""Unit tests for DataService."""

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytest

from backend.app.services.data_service import DataService
from backend.app.schemas.data import FeatureData, FeaturesResponse
from backend.app.core.exceptions import DataNotFoundError


@pytest.fixture
def sample_price_df():
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100, 200, 300],
        },
        index=dates,
    )


@pytest.fixture(autouse=True)
def disable_src(monkeypatch):
    """Skip heavy src imports for these tests."""
    monkeypatch.setattr("backend.app.services.data_service.HAS_PREPROCESSOR", False)


def test_get_stock_data_caches_results(monkeypatch, sample_price_df):
    service = DataService()
    call_count = {"calls": 0}

    def fake_load(*args, **kwargs):
        call_count["calls"] += 1
        return sample_price_df

    monkeypatch.setattr(service, "_load_stock_data", fake_load)

    first = service.get_stock_data("^GSPC")
    second = service.get_stock_data("^GSPC")

    assert call_count["calls"] == 1  # loaded only once due to cache
    assert second.count == first.count == len(sample_price_df)
    assert second.data[0].close == pytest.approx(1.05)


def test_get_stock_features_filters_available_columns(monkeypatch, sample_price_df):
    # Add a few engineered columns but leave others missing to ensure filtering
    df = sample_price_df.copy()
    df["log_return"] = [0.0, 0.01, 0.02]
    df["macd"] = [0.1, 0.2, 0.3]

    service = DataService()
    monkeypatch.setattr(service, "_load_stock_data", lambda *_, **__: df)

    response = service.get_stock_features("^GSPC")

    assert response.feature_names == ["log_return", "macd"]
    assert all(f.log_return is not None for f in response.features)
    assert all(f.macd is not None for f in response.features)


def test_prepare_sequences_shapes_and_targets(monkeypatch):
    service = DataService()

    # Build deterministic raw data rows for 8 days (prepare_sequences uses _load_stock_data)
    timestamps = pd.date_range("2024-01-01", periods=8, freq="D")
    raw_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0 + i for i in range(8)],
        "volume": [1000000.0] * 8,
        "log_return": [float(i) / 10 for i in range(8)],
        "rolling_volatility_20": [0.2] * 8,
        "momentum_20": [0.3] * 8,
        "rsi_14": [0.4] * 8,
        "macd": [0.5] * 8,
    })

    # Monkeypatch _load_stock_data since prepare_sequences uses that, not get_stock_features
    monkeypatch.setattr(service, "_load_stock_data", lambda *_args, **_kwargs: raw_df)

    seq_len = 5
    sequences, targets, df = service.prepare_sequences(
        "^GSPC",
        sequence_length=seq_len,
        feature_cols=["log_return", "rolling_volatility_20", "momentum_20"],
    )

    expected_samples = len(raw_df) - seq_len
    assert sequences.shape == (expected_samples, seq_len, 3)
    assert targets.shape == (expected_samples,)


def test_prepare_sequences_raises_when_no_features(monkeypatch):
    service = DataService()

    # Return a feature set that lacks the requested column so available_cols is empty
    minimal_feature = FeatureData(timestamp=datetime(2024, 1, 1), ticker="^GSPC")
    features = FeaturesResponse(
        ticker="^GSPC",
        features=[minimal_feature],
        feature_names=[],
        count=1,
    )
    monkeypatch.setattr(service, "get_stock_features", lambda *_args, **_kwargs: features)

    with pytest.raises(DataNotFoundError):
        service.prepare_sequences("^GSPC", feature_cols=["nonexistent_feature"], sequence_length=2)
