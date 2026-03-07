#!/usr/bin/env python3
"""
Forward-pass synthetic metric test for all model families.

Runs actual model forward passes (untrained weights) on synthetic tensors,
then evaluates ML + financial metric contracts on derived synthetic price paths.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_result(payload: Dict[str, Any]) -> None:
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "synthetic_forward_metrics_all_models.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_file}")


def _extract_prediction(output: Any) -> np.ndarray:
    """Normalize different model forward output signatures to 1D numpy array."""
    import torch

    if isinstance(output, tuple):
        # Many models return (pred, hidden) or (pred, logits, aux)
        pred = output[0]
    else:
        pred = output

    if isinstance(pred, torch.Tensor):
        arr = pred.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(pred).reshape(-1)

    return arr.astype(float)


def _to_price_path_from_signal(sig: np.ndarray, start: float = 100.0, scale: float = 0.01) -> np.ndarray:
    """Map arbitrary model signal to bounded synthetic returns, then to a price path."""
    s = np.asarray(sig, dtype=float).reshape(-1)
    s = s - np.mean(s)
    s = s / (np.std(s) + 1e-8)
    ret = np.tanh(s) * scale  # bounded synthetic returns
    prices = np.empty_like(ret)
    prices[0] = start
    for i in range(1, len(ret)):
        prices[i] = max(1e-6, prices[i - 1] * (1.0 + ret[i]))
    return prices


def _contract_issues(ml: Dict[str, float], fbasic: Dict[str, float], fadv: Dict[str, float]) -> List[str]:
    issues: List[str] = []

    if abs(float(ml["mse"]) - float(ml["rmse"]) ** 2) > 1e-6:
        issues.append("ml.mse != rmse^2")
    if not (0 <= ml["directional_accuracy"] <= 100):
        issues.append("ml.directional_accuracy not in 0..100")

    if not (0 <= fbasic["win_rate"] <= 100):
        issues.append("basic.win_rate not in 0..100")
    if not (0 <= fbasic["max_drawdown"] <= 100):
        issues.append("basic.max_drawdown not in 0..100")

    if not (-1 <= fadv["max_drawdown"] <= 0):
        issues.append("adv.max_drawdown not in -1..0")
    if not (0 <= fadv.get("win_rate", 0.0) <= 1):
        issues.append("adv.win_rate not in 0..1")
    if not (0 <= fadv.get("directional_accuracy", 0.0) <= 1):
        issues.append("adv.directional_accuracy not in 0..1")

    for k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return", "volatility"]:
        if not np.isfinite(fadv.get(k, np.nan)):
            issues.append(f"adv.{k} non-finite")

    return issues


def main() -> int:
    t0 = time.perf_counter()

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        msg = f"SKIPPED: torch not available ({exc})"
        print(msg)
        _write_result(
            {
                "summary": {
                    "status": "skipped",
                    "reason": msg,
                    "total_models": 0,
                    "passed": 0,
                    "failed": 0,
                    "runtime_ms": (time.perf_counter() - t0) * 1000,
                },
                "results": [],
            }
        )
        return 0

    from src.evaluation.metrics import calculate_metrics as calc_metrics_basic
    from src.evaluation.metrics import calculate_financial_metrics as calc_financial_basic
    from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns

    from src.models.baseline import LSTMModel, GRUModel, BiLSTMModel, AttentionLSTM
    from src.models.transformer import TransformerModel
    from src.models.pinn import PINNModel
    from src.models.stacked_pinn import StackedPINN, ResidualPINN
    from src.models.volatility import (
        VolatilityLSTM,
        VolatilityGRU,
        VolatilityTransformer,
        VolatilityPINN,
        StackedVolatilityPINN,
    )
    from src.models.financial_dp_pinn import (
        FinancialPINNBase,
        FinancialDualPhasePINN,
        AdaptiveFinancialDualPhasePINN,
    )

    model_builders: Dict[str, Any] = {
        # Baselines
        "lstm": lambda d: LSTMModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "gru": lambda d: GRUModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "bilstm": lambda d: BiLSTMModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "attention_lstm": lambda d: AttentionLSTM(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "transformer": lambda d: TransformerModel(input_dim=d, d_model=32, nhead=4, num_encoder_layers=1, dim_feedforward=64, dropout=0.0, output_dim=1),
        # PINN variants via lambda settings
        "baseline_pinn": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.0, lambda_bs=0.0, lambda_ou=0.0, lambda_langevin=0.0),
        "gbm": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.1, lambda_bs=0.0, lambda_ou=0.0, lambda_langevin=0.0),
        "ou": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.0, lambda_bs=0.0, lambda_ou=0.1, lambda_langevin=0.0),
        "black_scholes": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.0, lambda_bs=0.1, lambda_ou=0.0, lambda_langevin=0.0),
        "gbm_ou": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.05, lambda_bs=0.0, lambda_ou=0.05, lambda_langevin=0.0),
        "global": lambda d: PINNModel(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0, base_model="lstm", lambda_gbm=0.05, lambda_bs=0.03, lambda_ou=0.05, lambda_langevin=0.02),
        "stacked": lambda d: StackedPINN(input_dim=d, encoder_dim=32, lstm_hidden_dim=32, num_encoder_layers=1, num_rnn_layers=1, prediction_hidden_dim=16, dropout=0.0),
        "residual": lambda d: ResidualPINN(input_dim=d, base_model_type="lstm", base_hidden_dim=32, correction_hidden_dim=16, num_base_layers=1, num_correction_layers=1, dropout=0.0),
        # Volatility models
        "vol_lstm": lambda d: VolatilityLSTM(input_dim=d, hidden_dim=32, num_layers=1, dropout=0.0, output_horizon=1),
        "vol_gru": lambda d: VolatilityGRU(input_dim=d, hidden_dim=32, num_layers=1, dropout=0.0, output_horizon=1),
        "vol_transformer": lambda d: VolatilityTransformer(input_dim=d, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.0, output_horizon=1),
        "vol_pinn": lambda d: VolatilityPINN(input_dim=d, hidden_dim=32, num_layers=1, dropout=0.0, output_horizon=1, base_model="lstm"),
        "heston_pinn": lambda d: VolatilityPINN(input_dim=d, hidden_dim=32, num_layers=1, dropout=0.0, output_horizon=1, base_model="lstm", lambda_heston=0.1, enable_heston_constraint=True),
        "stacked_vol_pinn": lambda d: StackedVolatilityPINN(input_dim=d, encoder_dim=16, rnn_hidden_dim=32, num_encoder_layers=1, num_rnn_layers=1, dropout=0.0, output_horizon=1),
        # Financial DP PINN family
        "financial_pinn": lambda d: FinancialPINNBase(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "financial_dp_pinn": lambda d: FinancialDualPhasePINN(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "financial_dual_phase_pinn": lambda d: FinancialDualPhasePINN(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
        "adaptive_dual_phase_pinn": lambda d: AdaptiveFinancialDualPhasePINN(input_dim=d, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.0),
    }

    # Models declared in registry; spectral_pinn excluded here if constructor contract differs.
    model_order = [
        "lstm", "gru", "bilstm", "attention_lstm", "transformer",
        "baseline_pinn", "gbm", "ou", "black_scholes", "gbm_ou", "global",
        "stacked", "residual",
        "vol_lstm", "vol_gru", "vol_transformer", "vol_pinn", "heston_pinn", "stacked_vol_pinn",
        "financial_pinn", "financial_dp_pinn", "financial_dual_phase_pinn", "adaptive_dual_phase_pinn",
    ]

    # Synthetic tensors
    n = 640
    seq_len = 40
    input_dim = 12
    rng = np.random.RandomState(123)

    X = rng.normal(0.0, 1.0, size=(n, seq_len, input_dim)).astype(np.float32)
    # synthetic target signal from last timestep features
    y_sig = 0.03 * X[:, -1, 0] - 0.02 * X[:, -1, 1] + 0.01 * X[:, -1, 2] + rng.normal(0.0, 0.01, size=n)
    target_prices = _to_price_path_from_signal(y_sig, start=100.0, scale=0.01)

    X_t = torch.tensor(X)

    results: List[Dict[str, Any]] = []
    passed = 0

    print("=" * 80)
    print("FORWARD-PASS SYNTHETIC METRIC TEST")
    print("=" * 80)
    print(f"Models: {len(model_order)} | Samples: {n} | Seq: {seq_len} | Features: {input_dim}")

    for key in model_order:
        t_model = time.perf_counter()
        try:
            model = model_builders[key](input_dim)
            model.eval()
            with torch.no_grad():
                out = model(X_t)
            pred_sig = _extract_prediction(out)
            pred_prices = _to_price_path_from_signal(pred_sig, start=100.0, scale=0.01)

            ml = calc_metrics_basic(target_prices, pred_prices, prefix="")

            strat_ret = compute_strategy_returns(
                predictions=pred_prices,
                actual_prices=target_prices,
                transaction_cost=0.003,
                are_returns=False,
                require_price_scale=False,
                validate_scale=True,
            )
            fbasic = calc_financial_basic(strat_ret, prefix="")
            fadv = FinancialMetrics.compute_all_metrics(
                returns=strat_ret,
                predictions=pred_prices,
                targets=target_prices,
                predictions_are_returns=False,
                validate_price_scale=False,
                price_scale_context=f"forward_synth:{key}",
            )

            issues = _contract_issues(ml, fbasic, fadv)
            status = "pass" if not issues else "fail"
            if status == "pass":
                passed += 1

            dt_ms = (time.perf_counter() - t_model) * 1000
            print(f"[{status.upper()}] {key:26s} | R2={ml['r2']:.4f} | DA={ml['directional_accuracy']:.2f}% | Sharpe={fadv['sharpe_ratio']:.3f} | {dt_ms:.1f} ms")

            results.append(
                {
                    "model_key": key,
                    "status": status,
                    "issues": issues,
                    "runtime_ms": dt_ms,
                    "ml_metrics": {
                        "rmse": float(ml["rmse"]),
                        "mae": float(ml["mae"]),
                        "mape": float(ml["mape"]),
                        "r2": float(ml["r2"]),
                        "directional_accuracy_pct": float(ml["directional_accuracy"]),
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
            )
        except Exception as exc:
            dt_ms = (time.perf_counter() - t_model) * 1000
            print(f"[FAIL] {key:26s} | Exception: {exc} | {dt_ms:.1f} ms")
            results.append(
                {
                    "model_key": key,
                    "status": "fail",
                    "issues": [str(exc)],
                    "runtime_ms": dt_ms,
                }
            )

    total = len(model_order)
    failed = total - passed
    payload = {
        "summary": {
            "status": "completed",
            "total_models": total,
            "passed": passed,
            "failed": failed,
            "runtime_ms": (time.perf_counter() - t0) * 1000,
            "note": "True forward-pass synthetic test over supported model classes.",
        },
        "results": results,
    }
    _write_result(payload)

    print("-" * 80)
    print(f"Passed: {passed}/{total}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
