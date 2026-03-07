#!/usr/bin/env python3
"""
Synthetic metric validation across all canonical model keys.

This validates metric calculation behavior uniformly for every model identity,
using deterministic synthetic price series + model-specific prediction noise.
"""

from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Package stubs to avoid importing heavy optional stacks from __init__.py chains.
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
    sys.modules["src"] = src_pkg

if "src.evaluation" not in sys.modules:
    eval_pkg = types.ModuleType("src.evaluation")
    eval_pkg.__path__ = [str(PROJECT_ROOT / "src" / "evaluation")]
    sys.modules["src.evaluation"] = eval_pkg

# Lightweight dependency shims.
try:
    import pandas  # type: ignore  # noqa: F401
except ImportError:
    pandas_stub = types.ModuleType("pandas")

    class _Series:
        pass

    class _DataFrame:
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

if "src.utils" not in sys.modules:
    utils_stub = types.ModuleType("src.utils")
    utils_stub.__path__ = []
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
from src.evaluation.metrics import calculate_metrics as calc_metrics_basic
from src.evaluation.metrics import calculate_financial_metrics as calc_financial_basic
from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns


CANONICAL_MODEL_KEYS: List[str] = [
    "lstm",
    "gru",
    "bilstm",
    "attention_lstm",
    "transformer",
    "baseline_pinn",
    "gbm",
    "ou",
    "black_scholes",
    "gbm_ou",
    "global",
    "stacked",
    "residual",
    "spectral_pinn",
    "vol_lstm",
    "vol_gru",
    "vol_transformer",
    "vol_pinn",
    "heston_pinn",
    "stacked_vol_pinn",
    "financial_pinn",
    "financial_dp_pinn",
    "financial_dual_phase_pinn",
    "adaptive_dual_phase_pinn",
]


def _synthetic_price_series(n: int = 800, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 100.0 + 0.08 * t
    cyc = 2.5 * np.sin(t / 22.0) + 1.2 * np.sin(t / 7.0)
    noise = rng.normal(0.0, 0.7, size=n)
    prices = trend + cyc + noise
    prices = np.maximum(prices, 1.0)
    return prices.astype(float)


def _model_specific_predictions(model_key: str, targets: np.ndarray) -> np.ndarray:
    # deterministic per-model perturbation
    seed = abs(hash(model_key)) % (2**32)
    rng = np.random.RandomState(seed)

    n = len(targets)
    t = np.arange(n)

    # Vary noise/bias/lag by model key but keep realistic scale.
    base_noise = 0.4 + (seed % 7) * 0.12
    bias = ((seed % 13) - 6) * 0.03
    lag = seed % 3

    pred = targets.copy()
    if lag > 0:
        pred[lag:] = targets[:-lag]

    pred = pred + bias + rng.normal(0.0, base_noise, size=n)
    pred += 0.3 * np.sin(t / (18 + (seed % 5)))
    pred = np.maximum(pred, 1.0)
    return pred.astype(float)


def _contract_checks(ml: Dict[str, float], fbasic: Dict[str, float], fadv: Dict[str, float]) -> List[str]:
    issues: List[str] = []

    for k in ["rmse", "mae", "mape", "r2", "directional_accuracy", "mse"]:
        v = ml.get(k)
        if v is None or not np.isfinite(v):
            issues.append(f"ml.{k} invalid")

    if ml.get("directional_accuracy", -1) < 0 or ml.get("directional_accuracy", 101) > 100:
        issues.append("ml.directional_accuracy out of 0..100")

    if abs(ml.get("mse", 0.0) - ml.get("rmse", 0.0) ** 2) > 1e-6:
        issues.append("ml.mse != rmse^2")

    if fbasic.get("win_rate", -1) < 0 or fbasic.get("win_rate", 101) > 100:
        issues.append("fbasic.win_rate out of 0..100")
    if fbasic.get("max_drawdown", -1) < 0 or fbasic.get("max_drawdown", 101) > 100:
        issues.append("fbasic.max_drawdown out of 0..100")

    if fadv.get("directional_accuracy", -1) < 0 or fadv.get("directional_accuracy", 2) > 1:
        issues.append("fadv.directional_accuracy out of 0..1")
    if fadv.get("win_rate", -1) < 0 or fadv.get("win_rate", 2) > 1:
        issues.append("fadv.win_rate out of 0..1")
    if fadv.get("max_drawdown", -2) < -1 or fadv.get("max_drawdown", 1) > 0:
        issues.append("fadv.max_drawdown out of -1..0")

    critical = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "annualized_return",
        "profit_factor",
        "volatility",
        "information_coefficient",
    ]
    for k in critical:
        v = fadv.get(k)
        if v is None or not np.isfinite(v):
            issues.append(f"fadv.{k} invalid")

    return issues


def evaluate_model_key(model_key: str, targets: np.ndarray) -> Dict[str, object]:
    preds = _model_specific_predictions(model_key, targets)

    ml = calc_metrics_basic(targets, preds, prefix="")

    strategy_returns = compute_strategy_returns(
        predictions=preds,
        actual_prices=targets,
        transaction_cost=0.003,
        are_returns=False,
        require_price_scale=False,
        validate_scale=True,
    )

    fbasic = calc_financial_basic(
        returns=strategy_returns,
        risk_free_rate=RISK_FREE_RATE,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        prefix="",
    )

    fadv = FinancialMetrics.compute_all_metrics(
        returns=strategy_returns,
        predictions=preds,
        targets=targets,
        risk_free_rate=RISK_FREE_RATE,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        predictions_are_returns=False,
        validate_price_scale=False,
        price_scale_context=f"synthetic:{model_key}",
    )

    issues = _contract_checks(ml, fbasic, fadv)

    return {
        "model_key": model_key,
        "n_samples": int(len(targets)),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "ml_metrics": {
            "rmse": float(ml["rmse"]),
            "mae": float(ml["mae"]),
            "mape": float(ml["mape"]),
            "r2": float(ml["r2"]),
            "directional_accuracy_pct": float(ml["directional_accuracy"]),
        },
        "financial_basic": {
            "sharpe_ratio": float(fbasic["sharpe_ratio"]),
            "sortino_ratio": float(fbasic["sortino_ratio"]),
            "max_drawdown_pct_positive": float(fbasic["max_drawdown"]),
            "win_rate_pct": float(fbasic["win_rate"]),
            "total_return_pct": float(fbasic["total_return"]),
        },
        "financial_advanced": {
            "sharpe_ratio": float(fadv["sharpe_ratio"]),
            "sortino_ratio": float(fadv["sortino_ratio"]),
            "max_drawdown_decimal_negative": float(fadv["max_drawdown"]),
            "win_rate_decimal": float(fadv.get("win_rate", 0.0)),
            "directional_accuracy_decimal": float(fadv.get("directional_accuracy", 0.0)),
            "total_return_decimal": float(fadv["total_return"]),
            "annualized_return_decimal": float(fadv["annualized_return"]),
        },
    }


def main() -> int:
    t0 = time.perf_counter()

    targets = _synthetic_price_series(n=800, seed=42)

    print("=" * 80)
    print("SYNTHETIC METRIC TEST: ALL CANONICAL MODELS")
    print("=" * 80)
    print(f"Models: {len(CANONICAL_MODEL_KEYS)}")
    print(f"Samples per model: {len(targets)}")

    results: List[Dict[str, object]] = []
    passed = 0

    for key in CANONICAL_MODEL_KEYS:
        r = evaluate_model_key(key, targets)
        results.append(r)
        if r["status"] == "pass":
            passed += 1
            print(f"[PASS] {key:26s} | R2={r['ml_metrics']['r2']:.4f} | DA={r['ml_metrics']['directional_accuracy_pct']:.2f}% | Sharpe={r['financial_advanced']['sharpe_ratio']:.3f}")
        else:
            print(f"[FAIL] {key:26s} | issues={r['issues']}")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "synthetic_metrics_all_models.json"

    payload = {
        "summary": {
            "total_models": len(CANONICAL_MODEL_KEYS),
            "passed": passed,
            "failed": len(CANONICAL_MODEL_KEYS) - passed,
            "runtime_ms": (time.perf_counter() - t0) * 1000,
            "note": "Synthetic benchmark over model identities. This does not execute actual neural network forward passes in this environment.",
        },
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2))

    print("-" * 80)
    print(f"Passed: {passed}/{len(CANONICAL_MODEL_KEYS)}")
    print(f"Saved: {out_file}")
    print("=" * 80)

    return 0 if passed == len(CANONICAL_MODEL_KEYS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
