import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cache import CacheManager
from src.data.calendar import build_master_calendar, align_to_calendar
from src.data.features import (
    add_adjusted_returns,
    add_rolling_volatility,
    add_momentum_trend,
    add_cross_asset_spreads,
)
from src.data.targets import add_next_day_log_return
from src.data.sequence import build_sequences
from src.data.pipeline import build_benchmark_dataset
from src.data.fetcher import DataFetcher
from src.utils.config import get_config, reset_config


def _synthetic_market_df() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=240)
    symbols = ["SPY", "QQQ", "IWM", "VIX", "DGS2", "DGS10", "GLD", "DBC"]

    rows = []
    rng = np.random.default_rng(7)
    for sym in symbols:
        base = 100.0 + rng.normal(0, 1)
        if sym == "VIX":
            base = 20.0
        if sym == "DGS2":
            base = 0.02
        if sym == "DGS10":
            base = 0.03

        level = base
        for d in dates:
            step = rng.normal(0.0002, 0.01)
            if sym in {"DGS2", "DGS10"}:
                step = rng.normal(0.0, 0.001)
                level = max(0.0001, level + step)
            else:
                level = max(0.1, level * np.exp(step))
            rows.append(
                {
                    "time": d,
                    "ticker": sym,
                    "open": level * 0.999,
                    "high": level * 1.002,
                    "low": level * 0.998,
                    "close": level,
                    "volume": 1_000_000.0,
                    "adjusted_close": level,
                }
            )
    return pd.DataFrame(rows)


def test_cache_manager_roundtrip(tmp_path: Path):
    from src.data.universe import UniverseDefinition

    cm = CacheManager(tmp_path / "cache")
    uni = UniverseDefinition(symbols=["SPY", "QQQ"])
    paths = cm.paths(uni, "2020-01-01", "2020-12-31", "1d", dataset_tag="raw")

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "SPY"],
            "close": [100.0, 101.0],
            "adjusted_close": [100.0, 101.0],
        }
    )
    qa = {"coverage": {"rows": 2}}
    cm.save(paths, df, metadata={"source": "unit-test"}, qa_report=qa)

    loaded = cm.load(paths)
    assert loaded is not None
    assert len(loaded) == 2
    assert paths.metadata_path().exists()
    assert paths.qa_path().exists()


def test_calendar_alignment_and_feature_stack():
    raw = _synthetic_market_df()
    cal = build_master_calendar("2020-01-01", "2020-12-31", "NYSE")
    aligned = align_to_calendar(raw, cal, forward_fill_limit=2)

    feats = add_adjusted_returns(aligned, price_col="adjusted_close", horizons=[1, 5])
    feats = add_rolling_volatility(feats, return_col="adj_return_1d", windows=[10, 20])
    feats = add_momentum_trend(feats, return_col="adj_return_1d", windows=[20, 60])
    feats = add_cross_asset_spreads(
        feats,
        pairs=[("SPY", "QQQ"), ("SPY", "IWM")],
        price_col="adjusted_close",
    )
    feats = add_next_day_log_return(feats, price_col="adjusted_close", target_col="target", group_col="ticker")
    spy = feats[feats["ticker"] == "SPY"].dropna(subset=["adj_return_1d", "rolling_vol_20", "target"])

    X, y, tickers = build_sequences(
        spy,
        feature_cols=["adj_return_1d", "rolling_vol_20", "cross_spy_qqq_spread", "cross_spy_iwm_spread"],
        target_col="target",
        sequence_length=20,
        horizon=1,
        group_col="ticker",
    )
    assert len(X) > 0
    assert X.shape[1] == 20
    assert len(y) == len(X)
    assert set(tickers) == {"SPY"}


def test_build_benchmark_dataset_integration(monkeypatch, tmp_path: Path):
    synthetic = _synthetic_market_df()

    def _mock_fetch(self, **kwargs):
        return synthetic.copy()

    monkeypatch.setattr(DataFetcher, "fetch_multi_asset_cached", _mock_fetch)

    reset_config()
    cfg = get_config()
    cfg.project_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.start_date = "2020-01-01"
    cfg.data.end_date = "2020-12-31"
    cfg.data.tickers = ["SPY", "QQQ", "IWM", "VIX", "DGS2", "DGS10", "GLD", "DBC"]
    cfg.data.sequence_length = 20
    cfg.data.forecast_horizon = 1
    cfg.data.target_symbol = "SPY"
    cfg.data.target_type = "next_day_log_return"
    cfg.data.target_column = "target"
    cfg.data.force_refresh = True

    bundle = build_benchmark_dataset(cfg)
    meta = bundle["metadata"]

    assert meta["target_symbol"] == "SPY"
    assert meta["target_type"] == "next_day_log_return"
    assert meta["leakage_report"]["passed"] is True
    assert len(meta["feature_columns"]) >= 8

    artifact_dir = Path(bundle["artifact_dir"])
    assert artifact_dir.exists()
    assert (artifact_dir / "dataset_metadata.json").exists()
    assert (artifact_dir / "leakage_report.json").exists()

    with open(artifact_dir / "dataset_metadata.json") as f:
        saved_meta = json.load(f)
    assert saved_meta["fingerprint"] == meta["fingerprint"]
