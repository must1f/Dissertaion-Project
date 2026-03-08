import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.cache import CacheManager
from src.data.calendar import align_to_calendar, build_master_calendar
from src.data.fetcher import DataFetcher
from src.data.pipeline import build_benchmark_dataset
from src.data.universe import UniverseDefinition, universe_from_config
from src.utils.config import DataConfig


def _synthetic_prices(n_days: int = 120) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "^VIX", "^TNX", "GC=F", "CL=F"]
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    frames = []
    for i, t in enumerate(tickers):
        base = 100 + i * 3 + rng.normal(scale=0.5, size=n_days).cumsum()
        close = np.exp(base / 100)
        adjusted = close * (1 + rng.normal(scale=0.001, size=n_days))
        frame = pd.DataFrame(
            {
                "time": dates,
                "ticker": t,
                "open": adjusted * (1 - 0.001),
                "high": adjusted * (1 + 0.002),
                "low": adjusted * (1 - 0.002),
                "close": close,
                "volume": 1_000_000,
                "adjusted_close": adjusted,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _config(tmp_path: Path, target_type: str) -> SimpleNamespace:
    df = _synthetic_prices(90)
    start = str(pd.to_datetime(df["time"]).min().date())
    end = str(pd.to_datetime(df["time"]).max().date())
    data_cfg = DataConfig(
        start_date=start,
        end_date=end,
        interval="1d",
        target_type=target_type,
        target_symbol="SPY",
        sequence_length=5,
        forecast_horizon=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        cache_dir="cache",
        cache_ttl_days=1,
        force_refresh=False,
        default_forward_fill_limit=1,
        per_symbol_forward_fill_limit={"^VIX": 0, "^TNX": 0},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(data=data_cfg, data_dir=data_dir, project_root=tmp_path)


def test_next_day_return_path(tmp_path, monkeypatch):
    cfg = _config(tmp_path, "next_day_log_return")
    synthetic = _synthetic_prices(90)
    monkeypatch.setattr(DataFetcher, "fetch_multi_asset_cached", lambda self, **_: synthetic.copy())

    bundle = build_benchmark_dataset(cfg)

    assert "sequences" in bundle
    X_train, y_train = bundle["sequences"]["train"]
    assert X_train.shape[0] == y_train.shape[0]

    artifact_dir = Path(bundle["artifact_dir"])
    assert (artifact_dir / "fairness_contract.json").exists()
    assert (artifact_dir / "qa_raw.json").exists()
    assert (artifact_dir / "train_scaled_next_day_log_return.parquet").exists()


def test_realized_vol_path(tmp_path, monkeypatch):
    cfg = _config(tmp_path, "realized_vol")
    cfg.data.target_column = "target_vol"
    cfg.data.target_vol_window = 3
    synthetic = _synthetic_prices(90)
    monkeypatch.setattr(DataFetcher, "fetch_multi_asset_cached", lambda self, **_: synthetic.copy())

    bundle = build_benchmark_dataset(cfg)
    meta = bundle["metadata"]

    assert meta["target_type"] == "realized_vol"
    assert meta["target_vol_window"] == 3
    assert (Path(bundle["artifact_dir"]) / "train_scaled_realized_vol.parquet").exists()


def test_cache_ttl_and_force_refresh(tmp_path, monkeypatch):
    cfg = _config(tmp_path, "next_day_log_return")
    synthetic = _synthetic_prices(10)
    universe = UniverseDefinition(symbols=list(synthetic["ticker"].unique()), start_date=cfg.data.start_date, end_date=cfg.data.end_date, interval="1d")
    cache = CacheManager(tmp_path / "cache")
    paths = cache.paths(universe, cfg.data.start_date, cfg.data.end_date, cfg.data.interval, dataset_tag="raw_cache")
    cache.save(paths, synthetic, metadata={"saved_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()})

    called = {"count": 0}

    def _fake_fetch(*args, **kwargs):
        called["count"] += 1
        return synthetic.copy()

    monkeypatch.setattr(DataFetcher, "fetch_yfinance", _fake_fetch)
    fetcher = DataFetcher(cfg)
    df = fetcher.fetch_multi_asset_cached(start_date=cfg.data.start_date, end_date=cfg.data.end_date, interval=cfg.data.interval, dataset_tag="raw_cache")
    assert called["count"] == 1
    assert len(df) == len(synthetic)

    # Force refresh should call again even if fresh
    cache.save(paths, synthetic, metadata={"saved_at": datetime.now(timezone.utc).isoformat()})
    df2 = fetcher.fetch_multi_asset_cached(start_date=cfg.data.start_date, end_date=cfg.data.end_date, interval=cfg.data.interval, dataset_tag="raw_cache", force_refresh=True)
    assert called["count"] == 2
    assert len(df2) == len(synthetic)


def test_calendar_policy_alignment():
    dates = pd.bdate_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "time": list(dates[:4]) + list(dates[:4]),
        "ticker": ["SPY"] * 4 + ["^VIX"] * 4,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
        "volume": 100,
        "adjusted_close": 1.0,
    })

    cal = build_master_calendar(str(dates.min().date()), str(dates.max().date()))
    aligned, report = align_to_calendar(
        df,
        cal,
        default_forward_fill_limit=1,
        per_symbol_forward_fill_limit={"^VIX": 0},
        return_report=True,
    )

    vix_rows = aligned[aligned["ticker"] == "^VIX"]
    spy_rows = aligned[aligned["ticker"] == "SPY"]
    # SPY fills one gap, VIX should keep NaNs for missing day
    assert spy_rows["adjusted_close"].isna().sum() <= 1
    assert vix_rows["adjusted_close"].isna().sum() >= 1
    assert report["per_ticker"]["^VIX"]["forward_fill_limit"] == 0


def test_symbol_normalization_in_universe():
    cfg = SimpleNamespace(
        universe_name="core_multi_asset",
        tickers=["SPY", "VIX", "TNX"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        interval="1d",
        calendar="NYSE",
        master_calendar_holidays=[],
    )
    uni = universe_from_config(cfg)
    assert "^VIX" in uni.symbols
    assert "^TNX" in uni.symbols
