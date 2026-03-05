"""
Tests for window-level persistence and aggregation helpers.
"""

import numpy as np

from src.evaluation.window_results import (
    WindowMetrics,
    compute_window_aggregations,
    WindowResultsDatabase,
)


def test_compute_window_aggregations():
    windows = [
        WindowMetrics(
            window_id=1,
            start_date=None,
            end_date=None,
            start_idx=0,
            end_idx=10,
            n_samples=10,
            sharpe=1.0,
            sortino=1.2,
            total_return=0.05,
            annualized_return=0.1,
            volatility=0.2,
            max_drawdown=-0.05,
            calmar_ratio=2.0,
            mse=0.1,
            mae=0.2,
            rmse=0.316,
        ),
        WindowMetrics(
            window_id=2,
            start_date=None,
            end_date=None,
            start_idx=10,
            end_idx=20,
            n_samples=10,
            sharpe=2.0,
            sortino=1.5,
            total_return=0.08,
            annualized_return=0.12,
            volatility=0.25,
            max_drawdown=-0.04,
            calmar_ratio=3.0,
            mse=0.05,
            mae=0.1,
            rmse=0.224,
        ),
    ]

    aggs = compute_window_aggregations(windows)
    assert "sharpe" in aggs
    assert np.isclose(aggs["sharpe"].mean, 1.5)
    assert aggs["sharpe"].count == 2


def test_window_results_db_roundtrip(tmp_path):
    db = WindowResultsDatabase(tmp_path / "wf.db")
    windows = [
        WindowMetrics(
            window_id=1,
            start_date="2024-01-01",
            end_date="2024-01-10",
            start_idx=0,
            end_idx=10,
            n_samples=10,
            sharpe=1.0,
            sortino=1.2,
            total_return=0.05,
            annualized_return=0.1,
            volatility=0.2,
            max_drawdown=-0.05,
            calmar_ratio=2.0,
        )
    ]

    db.save_experiment("exp1", "model", {"foo": "bar"})
    db.save_window_results(windows, "exp1")
    df = db.load_window_results("exp1")

    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == "exp1"
    assert np.isclose(df.iloc[0]["sharpe"], 1.0)
