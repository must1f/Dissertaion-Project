import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache import CacheManager
from src.data.fetcher import DataFetcher
from src.data.pipeline import build_benchmark_dataset
from src.data.features.returns import add_adjusted_returns
from src.data.quality import run_qa
from src.data.universe import UniverseDefinition
from src.utils.config import DataConfig


def _make_synthetic_prices(n_days: int = 260) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "^VIX", "^TNX", "GC=F", "CL=F"]
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    frames = []
    for idx, ticker in enumerate(tickers):
        drift = 0.0005 + idx * 0.0001
        shocks = rng.normal(loc=drift, scale=0.01, size=n_days).cumsum()
        base = 50 + idx * 5 + shocks * 10
        close = np.exp(base / 100)
        adjusted = close * (1 + rng.normal(scale=0.001, size=n_days))

        frame = pd.DataFrame(
            {
                "time": dates,
                "ticker": ticker,
                "open": adjusted * (1 + rng.normal(scale=0.002, size=n_days)),
                "high": adjusted * (1 + rng.normal(scale=0.003, size=n_days) + 0.002),
                "low": adjusted * (1 - rng.normal(scale=0.003, size=n_days) - 0.002),
                "close": close,
                "volume": np.abs(rng.normal(loc=1e6, scale=5e4, size=n_days)),
                "adjusted_close": adjusted,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_prices():
    return _make_synthetic_prices()


def _make_config(tmp_path: Path, df: pd.DataFrame) -> SimpleNamespace:
    start = pd.to_datetime(df["time"]).min().date().isoformat()
    end = pd.to_datetime(df["time"]).max().date().isoformat()
    data_cfg = DataConfig(
        start_date=start,
        end_date=end,
        interval="1d",
        sequence_length=20,
        forecast_horizon=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        cache_dir="cache",
        force_refresh=False,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(data=data_cfg, data_dir=data_dir, project_root=tmp_path)


def test_cache_roundtrip_hit_miss(tmp_path, synthetic_prices, monkeypatch):
    cfg = _make_config(tmp_path, synthetic_prices)
    universe = UniverseDefinition(
        symbols=list(synthetic_prices["ticker"].unique()),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        interval=cfg.data.interval,
    )

    fetcher = DataFetcher(cfg)
    cache_paths = fetcher.cache.paths(
        universe=universe,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        interval=cfg.data.interval,
        dataset_tag="raw_cache",
    )

    fetcher.cache.save(
        cache_paths,
        synthetic_prices,
        metadata={"cache_key": cache_paths.cache_key()},
        qa_report={"status": "ok"},
    )

    def _fail_fetch(*_, **__):  # pragma: no cover - ensures cache is used
        raise AssertionError("fetch_yfinance should not run when cache is warm")

    monkeypatch.setattr(DataFetcher, "fetch_yfinance", _fail_fetch)

    loaded = fetcher.fetch_multi_asset_cached(
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        interval=cfg.data.interval,
        force_refresh=False,
        dataset_tag="raw_cache",
    )

    assert len(loaded) == len(synthetic_prices)
    assert Path(cache_paths.metadata_path()).exists()
    with open(cache_paths.metadata_path()) as f:
        meta = json.load(f)
    assert meta.get("cache_key") == cache_paths.cache_key()


def test_adjusted_close_consistency():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="B"),
            "ticker": ["TEST"] * 4,
            "adjusted_close": [100, 101, 103, 104],
        }
    )
    result = add_adjusted_returns(df, price_col="adjusted_close", horizons=[1])
    expected = np.log(df["adjusted_close"] / df["adjusted_close"].shift(1))
    pd.testing.assert_series_equal(result["adj_return_1d"], expected, check_names=False)


def test_split_and_scaler_leakage(tmp_path, synthetic_prices, monkeypatch):
    cfg = _make_config(tmp_path, synthetic_prices)

    monkeypatch.setattr(DataFetcher, "fetch_multi_asset_cached", lambda self, **kwargs: synthetic_prices.copy())

    bundle = build_benchmark_dataset(cfg)
    meta = bundle["metadata"]

    assert meta["leakage_report"]["passed"] is True
    assert meta["split_boundaries"]["train"]["end"] < meta["split_boundaries"]["val"]["start"]
    assert Path(meta["scaler_path"]).exists()


def test_sequence_shape_consistency(tmp_path, synthetic_prices, monkeypatch):
    cfg = _make_config(tmp_path, synthetic_prices)

    monkeypatch.setattr(DataFetcher, "fetch_multi_asset_cached", lambda self, **kwargs: synthetic_prices.copy())

    bundle = build_benchmark_dataset(cfg)
    X_train, y_train = bundle["sequences"]["train"]
    feature_count = len(bundle["metadata"]["feature_columns"])

    assert X_train.shape[1] == cfg.data.sequence_length
    assert X_train.shape[2] == feature_count
    assert y_train.shape[0] == X_train.shape[0]


def test_qa_flagging_behavior():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
            "ticker": ["BAD", "BAD", "BAD"],
            "open": [10, -1, 10],
            "high": [10, 10, 10],
            "low": [10, 10, 10],
            "close": [10, 10, -5],
            "volume": [100, 100, 100],
            "adjusted_close": [10, 10, -5],
        }
    )

    report = run_qa(df)
    assert report["duplicates"]["count"] > 0
    assert report["bad_prices"]["count"] > 0
    assert report["coverage"]["rows"] == 3
