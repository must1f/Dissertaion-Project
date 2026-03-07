#!/usr/bin/env python3
"""
Comprehensive metric verification script.

Validates core metric formulas, unit conventions, and strategy-return behavior
across:
  - src/evaluation/metrics.py
  - src/evaluation/financial_metrics.py
"""

from __future__ import annotations

import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

# Ensure local `src/` is importable when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-register package stubs so importing src.evaluation.metrics does not execute
# src/evaluation/__init__.py (which imports many optional dependencies).
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
    sys.modules["src"] = src_pkg

if "src.evaluation" not in sys.modules:
    eval_pkg = types.ModuleType("src.evaluation")
    eval_pkg.__path__ = [str(PROJECT_ROOT / "src" / "evaluation")]
    sys.modules["src.evaluation"] = eval_pkg

# Lightweight dependency shims for environments without full scientific stack.
try:
    import pandas  # type: ignore  # noqa: F401
except ImportError:
    pandas_stub = types.ModuleType("pandas")

    class _Series:  # minimal marker for isinstance checks
        pass

    class _DataFrame:  # minimal marker for isinstance/type hints
        pass

    class _Timestamp:
        @staticmethod
        def now():
            import datetime as _dt
            return _dt.datetime.now()

    pandas_stub.Series = _Series
    pandas_stub.DataFrame = _DataFrame
    pandas_stub.Timestamp = _Timestamp
    sys.modules["pandas"] = pandas_stub

try:
    import torch  # type: ignore  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")

    class _Tensor:
        pass

    torch_stub.Tensor = _Tensor
    sys.modules["torch"] = torch_stub

# Minimal pydantic/dotenv stubs so src.utils.config can import in lightweight envs.
try:
    import pydantic  # type: ignore  # noqa: F401
except ImportError:
    pydantic_stub = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

    class _ConfigDict(dict):
        pass

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.Field = _Field
    pydantic_stub.field_validator = _identity_decorator
    pydantic_stub.model_validator = _identity_decorator
    pydantic_stub.ConfigDict = _ConfigDict
    sys.modules["pydantic"] = pydantic_stub

try:
    import dotenv  # type: ignore  # noqa: F401
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")

    def _load_dotenv(*args, **kwargs):
        return False

    dotenv_stub.load_dotenv = _load_dotenv
    sys.modules["dotenv"] = dotenv_stub

# Stub src.utils package/logger to avoid importing heavy optional deps from
# src/utils/__init__.py during metric-module imports.
if "src.utils" not in sys.modules:
    utils_stub = types.ModuleType("src.utils")
    utils_stub.__path__ = []  # mark as package-like
    sys.modules["src.utils"] = utils_stub

if "src.utils.logger" not in sys.modules:
    logger_stub = types.ModuleType("src.utils.logger")

    class _NoOpLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def bind(self, **kwargs):
            return self

    def _get_logger(name=None):
        return _NoOpLogger()

    logger_stub.get_logger = _get_logger
    sys.modules["src.utils.logger"] = logger_stub

from src.constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from src.evaluation.metrics import (
    MetricsCalculator,
    calculate_metrics as calc_metrics_basic,
    calculate_financial_metrics as calc_financial_basic,
)
from src.evaluation.financial_metrics import (
    FinancialMetrics,
    assert_price_scale,
    destandardise_prices,
    compute_strategy_returns,
    validate_metrics,
)
import src.evaluation.financial_metrics as fm_mod


@dataclass
class TestCase:
    name: str
    fn: Callable[[], None]


class MetricTestFailure(AssertionError):
    pass


def _assert_close(a: float, b: float, tol: float = 1e-8, label: str = "") -> None:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > tol:
        raise MetricTestFailure(f"{label} expected {b}, got {a}")


def _assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise MetricTestFailure(msg)


def test_rmse_mae_mape_r2() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5])

    rmse_manual = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mae_manual = np.mean(np.abs(y_true - y_pred))
    mape_manual = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2_manual = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    _assert_close(MetricsCalculator.rmse(y_true, y_pred), rmse_manual, label="RMSE")
    _assert_close(MetricsCalculator.mae(y_true, y_pred), mae_manual, label="MAE")
    _assert_close(MetricsCalculator.mape(y_true, y_pred), mape_manual, tol=1e-7, label="MAPE")
    _assert_close(MetricsCalculator.r2(y_true, y_pred), r2_manual, tol=1e-7, label="R2")


def test_directional_accuracy_basic_module() -> None:
    prices_true = np.array([100, 101, 100, 102], dtype=float)  # diffs + - +
    prices_pred = np.array([100, 102, 101, 103], dtype=float)  # diffs + - +
    da = MetricsCalculator.directional_accuracy(prices_true, prices_pred, are_returns=False)
    _assert_close(da, 1.0, label="Directional accuracy (prices)")

    rets_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)
    rets_pred = np.array([0.02, -0.01, 0.02, -0.03], dtype=float)
    da_ret = MetricsCalculator.directional_accuracy(rets_true, rets_pred, are_returns=True)
    _assert_close(da_ret, 1.0, label="Directional accuracy (returns)")


def test_basic_financial_metrics_known_values() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)

    sharpe = MetricsCalculator.sharpe_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    sortino = MetricsCalculator.sortino_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    _assert_true(-5.0 <= sharpe <= 5.0, "Sharpe out of clipped bounds")
    _assert_true(-10.0 <= sortino <= 10.0, "Sortino out of clipped bounds")

    cum = np.cumprod(1 + np.clip(returns, -0.99, 1.0))
    max_dd = MetricsCalculator.max_drawdown(cum)
    _assert_true(0.0 <= max_dd <= 100.0, "Basic max_drawdown should be positive percent")

    calmar = MetricsCalculator.calmar_ratio(returns, TRADING_DAYS_PER_YEAR)
    _assert_true(-10.0 <= calmar <= 10.0, "Calmar out of clipped bounds")

    wr = MetricsCalculator.win_rate(returns)
    _assert_close(wr, 50.0, label="Basic win_rate")


def test_calculate_metrics_wrapper_units() -> None:
    y_true = np.array([10.0, 11.0, 12.0, 13.0])
    y_pred = np.array([10.1, 10.9, 12.2, 12.8])
    out = calc_metrics_basic(y_true, y_pred, prefix="test_")
    required = {
        "test_rmse",
        "test_mae",
        "test_mape",
        "test_r2",
        "test_directional_accuracy",
        "test_mse",
    }
    _assert_true(required.issubset(out.keys()), "Missing keys from calculate_metrics")
    _assert_true(0.0 <= out["test_directional_accuracy"] <= 100.0, "Directional accuracy should be percent")
    _assert_close(out["test_mse"], out["test_rmse"] ** 2, tol=1e-10, label="MSE == RMSE^2")


def test_calculate_financial_metrics_wrapper_units() -> None:
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03], dtype=float)
    out = calc_financial_basic(returns, prefix="x_")
    required = {
        "x_sharpe_ratio",
        "x_sortino_ratio",
        "x_max_drawdown",
        "x_calmar_ratio",
        "x_win_rate",
        "x_total_return",
        "x_mean_return",
        "x_volatility",
    }
    _assert_true(required.issubset(out.keys()), "Missing keys from calculate_financial_metrics")
    _assert_true(0.0 <= out["x_win_rate"] <= 100.0, "Win rate should be percent in basic financial wrapper")
    _assert_true(0.0 <= out["x_max_drawdown"] <= 100.0, "Max drawdown should be positive percent in basic wrapper")


def test_financial_metrics_directional_and_signal_quality() -> None:
    pred = np.array([100, 101, 100, 102, 103], dtype=float)
    targ = np.array([99, 100, 99, 101, 102], dtype=float)

    da = FinancialMetrics.directional_accuracy(pred, targ, are_returns=False)
    _assert_close(da, 1.0, label="FinancialMetrics directional accuracy")

    ic = FinancialMetrics.information_coefficient(pred, targ, use_returns=True)
    _assert_true(ic > 0.99, "IC should be near +1 for perfectly aligned changes")

    pr = FinancialMetrics.precision_recall(pred, targ, use_returns=True)
    _assert_true({"precision", "recall", "f1_score"}.issubset(pr.keys()), "Missing precision/recall/f1 keys")
    _assert_true(pr["precision"] >= 0.99 and pr["recall"] >= 0.99, "Precision/recall should be near 1")


def test_financial_metrics_drawdown_return_units() -> None:
    returns = np.array([0.1, -0.2, 0.05], dtype=float)
    max_dd = FinancialMetrics.max_drawdown(returns)
    _assert_true(-1.0 <= max_dd <= 0.0, "FinancialMetrics max_drawdown should be negative decimal")

    total_ret = FinancialMetrics.total_return(returns)
    ann_ret = FinancialMetrics.annualized_return(returns)
    _assert_true(-1.0 <= total_ret <= 10.0, "total_return clipped bounds violated")
    _assert_true(-1.0 <= ann_ret <= 5.0, "annualized_return clipped bounds violated")


def test_sharpe_sortino_raw_vs_clipped() -> None:
    returns = np.array([0.12, -0.08, 0.15, -0.10, 0.20, -0.07, 0.11], dtype=float)
    sharpe_raw = FinancialMetrics.sharpe_ratio_raw(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    sharpe_clip = FinancialMetrics.sharpe_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    sortino_raw = FinancialMetrics.sortino_ratio_raw(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    sortino_clip = FinancialMetrics.sortino_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)

    _assert_true(abs(sharpe_clip) <= 5.0, "Clipped sharpe bounds violated")
    _assert_true(abs(sortino_clip) <= 10.0, "Clipped sortino bounds violated")
    _assert_true(abs(sharpe_raw) + 1e-12 >= abs(sharpe_clip), "Raw Sharpe should be >= clipped magnitude")
    _assert_true(abs(sortino_raw) + 1e-12 >= abs(sortino_clip), "Raw Sortino should be >= clipped magnitude")


def test_cumulative_returns_manual() -> None:
    returns = np.array([0.1, -0.1, 0.2], dtype=float)
    got = FinancialMetrics.cumulative_returns(returns)
    expected = np.array([0.1, -0.01, 0.188], dtype=float)
    _assert_true(np.allclose(got, expected, atol=1e-12), f"cumulative_returns mismatch: {got} vs {expected}")


def test_drawdown_duration_simple_path() -> None:
    # cumulative path: 1.0 -> 0.97 -> 0.95 -> 0.96 -> 1.02
    returns = np.array([-0.03, -0.0206185567, 0.0105263158, 0.0625], dtype=float)
    dd_dur = FinancialMetrics.drawdown_duration(returns)
    _assert_true(dd_dur > 0, "Expected positive drawdown duration for path with drawdown spell")


def test_bootstrap_sharpe_ci_and_dsr() -> None:
    rng = np.random.RandomState(7)
    returns = rng.normal(0.0005, 0.01, size=300)

    point, lower, upper = FinancialMetrics.bootstrapped_sharpe_ci(
        returns, confidence=0.95, n_bootstrap=200, seed=42
    )
    _assert_true(np.isfinite(point) and np.isfinite(lower) and np.isfinite(upper), "Sharpe CI values must be finite")
    _assert_true(lower <= upper, "Sharpe CI lower must be <= upper")

    try:
        dsr = FinancialMetrics.deflated_sharpe_ratio(
            sharpe_ratio=point,
            n_trials=10,
            variance_sharpe=float(np.var(returns) * TRADING_DAYS_PER_YEAR),
            skewness=FinancialMetrics.skewness(returns),
            kurtosis=FinancialMetrics.kurtosis(returns, excess=False),
        )
        _assert_true(0.0 <= dsr <= 1.0, "DSR must be in [0,1]")
    except ModuleNotFoundError:
        # SciPy is optional in this environment. Bootstrapped CI is still verified.
        pass


def test_subsample_stability_contract() -> None:
    rng = np.random.RandomState(123)
    returns = rng.normal(0.0003, 0.012, size=300)
    st = FinancialMetrics.subsample_stability(returns, n_subsamples=6, min_subsample_size=30)
    required = {"mean", "std", "min", "max", "stability", "positive_pct"}
    _assert_true(required.issubset(st.keys()), "Missing subsample stability keys")
    _assert_true(0.0 <= st["stability"] <= 1.0, "stability must be in [0,1]")
    _assert_true(0.0 <= st["positive_pct"] <= 1.0, "positive_pct must be in [0,1]")


def test_profit_factor_information_ratio() -> None:
    returns = np.array([0.03, -0.01, 0.02, -0.02], dtype=float)
    pf = FinancialMetrics.profit_factor(returns)
    manual_pf = (0.03 + 0.02) / (0.01 + 0.02)
    _assert_close(pf, manual_pf, tol=1e-10, label="profit_factor")

    bench = np.array([0.01, 0.0, 0.01, -0.01], dtype=float)
    ir = FinancialMetrics.information_ratio(returns, bench)
    active = returns - bench
    manual_ir = np.mean(active) / np.std(active, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    _assert_close(ir, manual_ir, tol=1e-10, label="information_ratio")


def test_price_scale_helpers() -> None:
    z = np.array([-1.0, 0.0, 1.0], dtype=float)
    p = destandardise_prices(z, price_mean=100.0, price_std=5.0)
    expected = np.array([95.0, 100.0, 105.0], dtype=float)
    _assert_true(np.allclose(p, expected, atol=1e-12), "destandardise_prices mismatch")

    ok = assert_price_scale(np.array([80.0, 120.0, 100.0]), raise_error=False)
    _assert_true(ok, "Expected price scale check to pass for realistic prices")
    bad = assert_price_scale(np.array([0.1, -0.2, 0.05]), raise_error=False)
    _assert_true(not bad, "Expected z-score like values to fail price scale check")


def test_compute_strategy_returns_alignment_and_costs() -> None:
    preds = np.array([0.10, -0.10, 0.20], dtype=float)
    actual = np.array([0.01, -0.02, 0.03], dtype=float)

    # are_returns=True, sign mode, no costs:
    # raw_signal = [ 1, -1, 1]
    # positions  = [ 0,  1,-1]  (shifted by 1)
    # returns    = [0*0.01, 1*(-0.02), -1*(0.03)] = [0, -0.02, -0.03]
    out = compute_strategy_returns(
        predictions=preds,
        actual_prices=actual,
        are_returns=True,
        transaction_cost=0.0,
        threshold=0.0,
        sizing_mode="sign",
    )
    expected = np.array([0.0, -0.02, -0.03], dtype=float)
    _assert_true(np.allclose(out, expected, atol=1e-12), f"Strategy alignment mismatch: {out} vs {expected}")

    # With transaction costs, should reduce returns when position changes occur.
    out_cost = compute_strategy_returns(
        predictions=preds,
        actual_prices=actual,
        are_returns=True,
        transaction_cost=0.001,
        threshold=0.0,
        sizing_mode="sign",
    )
    _assert_true(np.all(out_cost <= out + 1e-12), "Transaction costs should not improve per-period returns")


def test_compute_strategy_returns_sizing_modes_and_details() -> None:
    preds = np.array([0.6, 0.4, 0.8, 0.2], dtype=float)
    actual = np.array([0.01, -0.02, 0.015, -0.01], dtype=float)

    out_sign, details_sign = compute_strategy_returns(
        predictions=preds,
        actual_prices=actual,
        are_returns=True,
        sizing_mode="sign",
        return_details=True,
    )
    _assert_true(len(out_sign) == len(actual), "sign mode length mismatch")
    _assert_true({"positions", "position_changes", "predicted_returns", "actual_returns"}.issubset(details_sign.keys()),
                 "return_details missing required keys")

    out_scaled = compute_strategy_returns(
        predictions=preds,
        actual_prices=actual,
        are_returns=True,
        sizing_mode="scaled",
        max_leverage=1.5,
    )
    _assert_true(len(out_scaled) == len(actual), "scaled mode length mismatch")

    out_prob = compute_strategy_returns(
        predictions=preds,  # interpreted as probabilities in prob mode
        actual_prices=actual,
        are_returns=True,
        sizing_mode="prob",
        max_leverage=1.0,
    )
    _assert_true(len(out_prob) == len(actual), "prob mode length mismatch")


def test_compute_strategy_returns_scale_guard() -> None:
    preds = np.array([0.1, 0.2, 0.3], dtype=float)
    actual = np.array([0.1, 0.2, 0.3], dtype=float)
    raised = False
    try:
        compute_strategy_returns(
            predictions=preds,
            actual_prices=actual,
            are_returns=False,
            require_price_scale=True,
            validate_scale=True,
        )
    except ValueError:
        raised = True
    _assert_true(raised, "Expected ValueError when price scale stats are required but missing")


def test_metrics_validation_warnings_and_errors() -> None:
    bad = {
        "max_drawdown": -1.2,  # impossible
        "sharpe_ratio": 6.0,   # suspicious
        "sortino_ratio": 11.0, # suspicious
        "profit_factor": 20.0, # suspicious
    }
    val = validate_metrics(bad)
    _assert_true(not val["is_valid"], "Expected invalid metrics set")
    _assert_true(len(val["errors"]) >= 1, "Expected at least one error")
    _assert_true(len(val["warnings"]) >= 1, "Expected at least one warning")


def test_financial_compute_all_metrics_contract() -> None:
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03, 0.0, 0.01, -0.01], dtype=float)
    preds = np.array([100, 101, 100, 102, 101, 103, 104, 103], dtype=float)
    targs = np.array([100, 100.5, 100.2, 101.2, 101.0, 102.0, 103.0, 102.5], dtype=float)

    out = FinancialMetrics.compute_all_metrics(
        returns=returns,
        predictions=preds,
        targets=targs,
        predictions_are_returns=False,
        validate_price_scale=False,  # for synthetic values
    )

    required = {
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "volatility",
        "profit_factor",
        "directional_accuracy",
        "information_coefficient",
        "precision",
        "recall",
        "f1_score",
    }
    _assert_true(required.issubset(out.keys()), "Missing keys in FinancialMetrics.compute_all_metrics")
    _assert_true(-1.0 <= out["max_drawdown"] <= 0.0, "max_drawdown contract (negative decimal) violated")
    _assert_true(0.0 <= out["directional_accuracy"] <= 1.0, "directional_accuracy should be 0..1")
    _assert_true(0.0 <= out["win_rate"] <= 1.0, "win_rate should be decimal 0..1")


def test_standalone_wrapper_functions() -> None:
    returns = np.array([0.01, -0.01, 0.015, -0.005, 0.02], dtype=float)
    preds = np.array([100, 101, 100.5, 102, 101.5], dtype=float)
    targs = np.array([99.5, 100.8, 100.2, 101.6, 101.0], dtype=float)

    # Standalone function family in financial_metrics.py (module-level API)
    sharpe = fm_mod.calculate_sharpe_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    sortino = fm_mod.calculate_sortino_ratio(returns, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR)
    max_dd = fm_mod.calculate_max_drawdown(returns)
    calmar = fm_mod.calculate_calmar_ratio(returns, TRADING_DAYS_PER_YEAR)

    _assert_true(-5.0 <= sharpe <= 5.0, "standalone sharpe bounds")
    _assert_true(-10.0 <= sortino <= 10.0, "standalone sortino bounds")
    _assert_true(-1.0 <= max_dd <= 0.0, "standalone max_drawdown bounds")
    _assert_true(-10.0 <= calmar <= 10.0, "standalone calmar bounds")

    allm = fm_mod.compute_all_metrics(
        returns=returns,
        predictions=preds,
        targets=targs,
        predictions_are_returns=False,
    )
    _assert_true("deflated_sharpe_ratio" in allm, "standalone compute_all_metrics missing DSR")
    _assert_true(0.0 <= allm["deflated_sharpe_ratio"] <= 1.0, "DSR should be in [0,1]")


def build_tests() -> List[TestCase]:
    return [
        TestCase("RMSE/MAE/MAPE/R2 formulas", test_rmse_mae_mape_r2),
        TestCase("Directional accuracy (basic module)", test_directional_accuracy_basic_module),
        TestCase("Basic financial metrics bounds/units", test_basic_financial_metrics_known_values),
        TestCase("calculate_metrics wrapper keys/units", test_calculate_metrics_wrapper_units),
        TestCase("calculate_financial_metrics wrapper keys/units", test_calculate_financial_metrics_wrapper_units),
        TestCase("Financial signal-quality metrics", test_financial_metrics_directional_and_signal_quality),
        TestCase("Financial drawdown/return unit contracts", test_financial_metrics_drawdown_return_units),
        TestCase("Sharpe/Sortino raw vs clipped", test_sharpe_sortino_raw_vs_clipped),
        TestCase("Cumulative returns exact path", test_cumulative_returns_manual),
        TestCase("Drawdown duration simple path", test_drawdown_duration_simple_path),
        TestCase("Bootstrap Sharpe CI and DSR bounds", test_bootstrap_sharpe_ci_and_dsr),
        TestCase("Subsample stability contract", test_subsample_stability_contract),
        TestCase("Profit factor and information ratio", test_profit_factor_information_ratio),
        TestCase("Price-scale helper functions", test_price_scale_helpers),
        TestCase("Strategy returns alignment and cost impact", test_compute_strategy_returns_alignment_and_costs),
        TestCase("Strategy sizing modes and return_details", test_compute_strategy_returns_sizing_modes_and_details),
        TestCase("Strategy returns price-scale guard", test_compute_strategy_returns_scale_guard),
        TestCase("Metric validation warnings/errors", test_metrics_validation_warnings_and_errors),
        TestCase("Financial compute_all_metrics contract", test_financial_compute_all_metrics_contract),
        TestCase("Standalone financial wrapper API", test_standalone_wrapper_functions),
    ]


def main() -> int:
    tests = build_tests()
    passed = 0
    failed = 0
    start_all = time.perf_counter()

    print("=" * 80)
    print("METRIC VERIFICATION SUITE")
    print("=" * 80)

    for tc in tests:
        t0 = time.perf_counter()
        try:
            tc.fn()
            dt = (time.perf_counter() - t0) * 1000
            print(f"[PASS] {tc.name} ({dt:.2f} ms)")
            passed += 1
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000
            print(f"[FAIL] {tc.name} ({dt:.2f} ms): {exc}")
            failed += 1

    total_ms = (time.perf_counter() - start_all) * 1000
    print("-" * 80)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed} | Runtime: {total_ms:.2f} ms")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
