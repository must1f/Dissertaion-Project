import numpy as np

from src.evaluation.strategy_engine import StrategyEngine


def test_sign_strategy_applies_one_period_lag():
    signals = np.array([1, -1, 1, 0], dtype=float)
    positions = StrategyEngine.sign_strategy(signals)

    # First position must be flat (no prior signal)
    assert positions[0] == 0.0
    # Subsequent positions are previous signals
    np.testing.assert_array_equal(positions[1:], signals[:-1])


def test_compute_net_returns_uses_lagged_positions():
    # Construct positions that would earn return only if lag is applied
    signals = np.array([1, 1, 1], dtype=float)
    positions = StrategyEngine.sign_strategy(signals)
    returns = np.array([0.0, 0.01, -0.02], dtype=float)

    net_returns, stats = StrategyEngine.compute_net_returns(positions, returns, cost_per_unit_turnover=0.0)

    # Position at t is signal from t-1, so first return must be zero exposure
    assert net_returns[0] == 0.0
    # Turnover counted from position changes
    assert stats["pct_days_trading"] >= 0.0
