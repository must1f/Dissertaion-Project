#!/usr/bin/env python3
"""
Standalone Neural Network Training Script

This script allows training models independently of the web application.
It uses the same training infrastructure as the backend and computes all
required metrics for dissertation-grade evaluation.

Usage:
    # Train a single model
    python scripts/train_models.py --model lstm --epochs 100

    # Train multiple models (batch training)
    python scripts/train_models.py --models lstm,gru,transformer --epochs 100

    # Train all baseline models
    python scripts/train_models.py --type baseline --epochs 100

    # Train all PINN variants
    python scripts/train_models.py --type pinn --epochs 100

    # Train all models with research config
    python scripts/train_models.py --all --research-mode

Available models:
    Baseline: lstm, gru, bilstm, attention_lstm, transformer
    PINN: baseline_pinn, gbm, ou, black_scholes, gbm_ou, global
    Advanced: stacked, residual
"""

import sys
import os
import argparse
import json
import csv
import yaml
import time
import numbers
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import training infrastructure
from src.utils.config import get_config, get_research_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed, log_system_info, get_device, compute_config_hash
from src.data.pipeline import build_benchmark_dataset
from src.data.sequence import build_sequences
from src.data.dataset import PhysicsAwareDataset, collate_fn_with_metadata
from src.models.model_registry import ModelRegistry
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics, calculate_financial_metrics
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.financial_metrics import (
    compute_strategy_returns,
    FinancialMetrics,
    compute_all_metrics,
)
from src.evaluation.volatility_metrics import VolatilityMetrics
from src.constants import TRANSACTION_COST, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)

# Model categories / tracks
BASELINE_MODELS = ['lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer']
PINN_MODELS = ['baseline_pinn', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global']
ADVANCED_MODELS = ['stacked', 'residual', 'spectral_pinn', 'financial_pinn', 'financial_dp_pinn', 'financial_dual_phase_pinn', 'adaptive_dual_phase_pinn']
VOLATILITY_MODELS = ['vol_lstm', 'vol_gru', 'vol_transformer', 'vol_pinn', 'heston_pinn', 'stacked_vol_pinn']

ALL_MODELS = BASELINE_MODELS + PINN_MODELS + ADVANCED_MODELS + VOLATILITY_MODELS
CORE_BENCHMARK_MODELS = {
    'lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer',
    'baseline_pinn', 'gbm', 'global'
}

TRACK_DEFINITIONS = {
    "core_benchmark": {
        "models": CORE_BENCHMARK_MODELS,
        "ranking_metric": "sharpe_ratio",
    },
    "volatility_extension": {
        "models": set(VOLATILITY_MODELS),
        "ranking_metric": "rmse",
    },
    "advanced_pinn_extension": {
        "models": set(ADVANCED_MODELS),
        "ranking_metric": "sharpe_ratio",
    },
    "ou_track": {"models": {"ou"}, "ranking_metric": "sharpe_ratio"},
    "black_scholes_track": {"models": {"black_scholes"}, "ranking_metric": "sharpe_ratio"},
}

LOWER_IS_BETTER = {"rmse", "mae", "mse", "max_drawdown"}

_CORE_CONTRACT_SIGNATURE: Optional[str] = None


def prepare_data(
    ticker: str = "SPY",
    sequence_length: Optional[int] = None,
    research_mode: bool = True,
    force_refresh: bool = False,
    use_multi_ticker: bool = True,
    target_override: Optional[str] = None,
    target_column_override: Optional[str] = None,
) -> tuple:
    """
    Prepare benchmark data using the leakage-safe multi-asset pipeline.

    Args:
        ticker: Target symbol (kept for backward compatibility)
        sequence_length: Override sequence length (None = use config)
        research_mode: Use research config parameters
        force_refresh: Force re-fetch data from source
        use_multi_ticker: Reserved for backward compatibility (benchmark is multi-asset)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, input_dim, scalers, feature_cols, dataset_meta, fairness_contract, regime_context)
    """
    config = get_config()
    research_cfg = get_research_config() if research_mode else None

    # Use research config values when in research mode
    if research_mode and research_cfg:
        sequence_length = sequence_length or research_cfg.sequence_length
        config.data.sequence_length = sequence_length
    else:
        sequence_length = sequence_length or config.data.sequence_length
        config.data.sequence_length = sequence_length

    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print(f"Research mode: {research_mode}")
    print(f"Date range: {config.data.start_date} to {config.data.end_date}")
    print(f"Sequence length: {sequence_length}")
    print(f"Force refresh: {force_refresh}")
    if use_multi_ticker:
        print(f"Benchmark universe: {config.data.tickers}")
    else:
        # Keep compatibility with existing CLI flag behavior.
        config.data.target_symbol = ticker
        print(f"Target symbol override: {config.data.target_symbol}")

    if target_override:
        config.data.target_type = target_override
    if target_column_override:
        config.data.target_column = target_column_override

    config.data.force_refresh = force_refresh
    dataset_bundle = build_benchmark_dataset(config)
    dataset_meta = dataset_bundle["metadata"]

    feature_cols = dataset_meta["feature_columns"]
    target_col = dataset_meta["target_column"]
    lookback = dataset_meta["lookback"]
    horizon = dataset_meta["horizon"]

    X_train, y_train = dataset_bundle["sequences"]["train"]
    X_val, y_val = dataset_bundle["sequences"]["val"]
    X_test, y_test = dataset_bundle["sequences"]["test"]

    train_df_unscaled = dataset_bundle["train_df_unscaled"]
    val_df_unscaled = dataset_bundle["val_df_unscaled"]
    test_df_unscaled = dataset_bundle["test_df_unscaled"]

    physics_cols = ["adjusted_close", "adj_return_1d", "rolling_vol_20"]
    for col in physics_cols:
        if col not in train_df_unscaled.columns:
            raise ValueError(f"Missing required physics metadata column: {col}")

    P_train, _, tickers_train = build_sequences(
        train_df_unscaled,
        physics_cols,
        target_col=target_col,
        sequence_length=lookback,
        horizon=horizon,
        group_col="ticker",
    )
    P_val, _, tickers_val = build_sequences(
        val_df_unscaled,
        physics_cols,
        target_col=target_col,
        sequence_length=lookback,
        horizon=horizon,
        group_col="ticker",
    )
    P_test, _, tickers_test = build_sequences(
        test_df_unscaled,
        physics_cols,
        target_col=target_col,
        sequence_length=lookback,
        horizon=horizon,
        group_col="ticker",
    )

    train_dataset = PhysicsAwareDataset(
        X_train,
        y_train,
        tickers_train,
        prices=P_train[:, :, 0],
        returns=P_train[:, :, 1],
        volatilities=P_train[:, :, 2],
    )
    val_dataset = PhysicsAwareDataset(
        X_val,
        y_val,
        tickers_val,
        prices=P_val[:, :, 0],
        returns=P_val[:, :, 1],
        volatilities=P_val[:, :, 2],
    )
    test_dataset = PhysicsAwareDataset(
        X_test,
        y_test,
        tickers_test,
        prices=P_test[:, :, 0],
        returns=P_test[:, :, 1],
        volatilities=P_test[:, :, 2],
    )

    input_dim = X_train.shape[2]

    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Target: {dataset_meta['target_symbol']} {dataset_meta['target_type']} ({dataset_meta['target_column']})")
    print(f"Dataset fingerprint: {dataset_meta['fingerprint']}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("=" * 70)
    print(f"Dataset artifacts: {dataset_bundle['artifact_dir']}")

    # Validate physics metadata alignment (volatility/returns/prices)
    assert P_train.shape[:2] == X_train.shape[:2], "P_train alignment mismatch"
    assert P_val.shape[:2] == X_val.shape[:2], "P_val alignment mismatch"
    assert P_test.shape[:2] == X_test.shape[:2], "P_test alignment mismatch"
    if np.isnan(P_train).any() or np.isnan(P_val).any() or np.isnan(P_test).any():
        raise ValueError("NaNs detected in physics metadata (prices/returns/volatilities)")

    fairness_contract = dataset_bundle.get("fairness_contract") or {
        "target_symbol": dataset_meta["target_symbol"],
        "target_type": dataset_meta["target_type"],
        "target_column": dataset_meta["target_column"],
        "price_column": dataset_meta.get("price_column"),
        "start_date": dataset_meta["start_date"],
        "end_date": dataset_meta["end_date"],
        "lookback": dataset_meta["lookback"],
        "horizon": dataset_meta["horizon"],
        "feature_columns": list(feature_cols),
        "required_core": dataset_meta.get("feature_columns", []),
        "fingerprint": dataset_meta["fingerprint"],
        "missing_policy": dataset_meta.get("missing_policy"),
        "scaling_policy": dataset_meta.get("scaling_policy"),
    }

    regime_context = _build_regime_context(
        test_df_unscaled,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
    )

    # For return-target benchmark there is no close-price scaler map required.
    scalers = None

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        scalers,
        feature_cols,
        dataset_meta,
        fairness_contract,
        regime_context,
    )


def _contract_signature(contract: Dict[str, Any]) -> str:
    dataset_section = contract.get("dataset", contract)
    features_section = contract.get("features", {})
    preprocessing_section = contract.get("preprocessing", {})

    payload = json.dumps(
        {
            "target_symbol": dataset_section.get("target_symbol"),
            "target_type": dataset_section.get("target_type"),
            "target_column": dataset_section.get("target_column"),
            "price_column": dataset_section.get("price_column"),
            "start_date": dataset_section.get("start_date"),
            "end_date": dataset_section.get("end_date"),
            "lookback": dataset_section.get("lookback"),
            "horizon": dataset_section.get("horizon"),
            "feature_columns": features_section.get("effective") or contract.get("feature_columns", []),
            "required_features": features_section.get("required_core", []),
            "fingerprint": dataset_section.get("fingerprint") or contract.get("fingerprint"),
            "missing_policy": preprocessing_section.get("missing_data_policy") or contract.get("missing_policy"),
            "scaling_policy": preprocessing_section.get("scaling_policy") or contract.get("scaling_policy"),
        },
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()


def _enforce_core_fairness(model_type: str, contract: Optional[Dict[str, Any]]) -> None:
    global _CORE_CONTRACT_SIGNATURE

    if model_type not in CORE_BENCHMARK_MODELS:
        return
    if contract is None:
        raise ValueError(f"Fairness contract missing for core benchmark model '{model_type}'")

    sig = _contract_signature(contract)
    if _CORE_CONTRACT_SIGNATURE is None:
        _CORE_CONTRACT_SIGNATURE = sig
        return

    if sig != _CORE_CONTRACT_SIGNATURE:
        raise ValueError(
            f"Fairness contract mismatch for model '{model_type}'. "
            "Core models must share the same target/date/window/features/preprocessing contract."
        )


def _model_track(model_type: str) -> str:
    if model_type in VOLATILITY_MODELS:
        return "volatility_extension"
    if model_type in CORE_BENCHMARK_MODELS:
        return "core_benchmark"
    if model_type in ADVANCED_MODELS:
        return "advanced_pinn_extension"
    if model_type == "ou":
        return "ou_track"
    if model_type == "black_scholes":
        return "black_scholes_track"
    return "unknown"


def _build_regime_context(
    df_unscaled: Any,
    target_col: str,
    lookback: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Extract aligned regime context (e.g., VIX, rates) for diagnostics."""
    context: Dict[str, np.ndarray] = {}
    candidates = ["vix_level", "vix_change", "rate_2y", "rate_10y", "regime_marker"]
    for col in candidates:
        if hasattr(df_unscaled, "columns") and col in df_unscaled.columns:
            seq, _, _ = build_sequences(
                df_unscaled,
                [col],
                target_col=target_col,
                sequence_length=lookback,
                horizon=horizon,
                group_col="ticker",
            )
            if len(seq):
                context[col] = seq[:, -1, 0]
    return context


def _validate_track_target(model_type: str, target_type: str, allow_mismatch: bool) -> None:
    track = _model_track(model_type)
    if target_type is None:
        return
    is_vol_target = str(target_type).lower().startswith("realized_vol")
    if track == "volatility_extension" and not is_vol_target and not allow_mismatch:
        raise ValueError(
            f"Volatility model '{model_type}' requires a volatility target (realized_vol). "
            "Set target_override='realized_vol' or use --allow-target-mismatch to override."
        )
    if track != "volatility_extension" and is_vol_target and not allow_mismatch:
        raise ValueError(
            f"Non-volatility model '{model_type}' received volatility target '{target_type}'. "
            "Use volatility models or pass --allow-target-mismatch to override."
        )


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def _extract_close_stats(scalers: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract mean and std for the close price scaler when available.

    Assumes single-ticker scaler dict; if multiple tickers, returns (None, None)
    to avoid applying incorrect scaling.
    """
    if not scalers or not isinstance(scalers, dict):
        return None, None
    if len(scalers) != 1:
        return None, None
    sc = next(iter(scalers.values()))
    mean = getattr(sc, "mean_", None)
    std = getattr(sc, "scale_", None)
    if mean is None or std is None:
        return None, None
    mean_val = float(np.squeeze(mean[0]) if hasattr(mean, "shape") and len(mean.shape) > 0 else float(mean))
    std_val = float(np.squeeze(std[0]) if hasattr(std, "shape") and len(std.shape) > 0 else float(std))
    return mean_val, std_val


def _extract_close_scaler_map(
    scalers: Optional[Dict[str, Any]],
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return per-ticker close mean/std when available."""
    if not scalers or not isinstance(scalers, dict):
        return {}
    close_idx = 0
    if feature_cols and "close" in feature_cols:
        close_idx = feature_cols.index("close")
    out: Dict[str, Dict[str, float]] = {}
    for ticker_name, sc in scalers.items():
        mean = getattr(sc, "mean_", None)
        std = getattr(sc, "scale_", None)
        if mean is None or std is None:
            continue
        mean_val = float(np.squeeze(mean[close_idx])) if hasattr(mean, "shape") else float(mean)
        std_val = float(np.squeeze(std[close_idx])) if hasattr(std, "shape") else float(std)
        out[ticker_name] = {"close_mean": mean_val, "close_std": std_val}
    return out


def _distribution_checks(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    def _moments(arr: np.ndarray) -> tuple[float, float, float, float]:
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        centered = arr - mean
        skew = float(np.mean(centered ** 3) / (std + 1e-8) ** 3) if std != 0 else 0.0
        kurt = float(np.mean(centered ** 4) / (std + 1e-8) ** 4 - 3) if std != 0 else 0.0
        return mean, std, skew, kurt

    true_mean, true_std, true_skew, true_kurt = _moments(y_true)
    pred_mean, pred_std, pred_skew, pred_kurt = _moments(y_pred)

    return {
        f"{prefix}true_mean": true_mean,
        f"{prefix}true_std": true_std,
        f"{prefix}true_skew": true_skew,
        f"{prefix}true_kurtosis": true_kurt,
        f"{prefix}pred_mean": pred_mean,
        f"{prefix}pred_std": pred_std,
        f"{prefix}pred_skew": pred_skew,
        f"{prefix}pred_kurtosis": pred_kurt,
    }


def _regime_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_returns: Optional[np.ndarray],
    regime_context: Optional[Dict[str, Any]],
    calc: MetricsCalculator,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    for label, mask in [("bull", y_true > 0), ("bear", y_true <= 0)]:
        if mask.any():
            metrics[f"{label}_rmse"] = calc.rmse(y_true[mask], y_pred[mask])
            metrics[f"{label}_mae"] = calc.mae(y_true[mask], y_pred[mask])
            metrics[f"{label}_directional_accuracy"] = (
                calc.directional_accuracy(y_true[mask], y_pred[mask], are_returns=True) * 100
            )

    if regime_context:
        vix_levels = regime_context.get("vix_level")
        if vix_levels is not None and len(vix_levels) == len(y_true):
            hi = np.nanpercentile(vix_levels, 75)
            lo = np.nanpercentile(vix_levels, 25)
            for label, mask in [
                ("vix_high", vix_levels >= hi),
                ("vix_low", vix_levels <= lo),
            ]:
                if mask.any():
                    metrics[f"{label}_rmse"] = calc.rmse(y_true[mask], y_pred[mask])
                    metrics[f"{label}_directional_accuracy"] = (
                        calc.directional_accuracy(y_true[mask], y_pred[mask], are_returns=True) * 100
                    )
                    if strategy_returns is not None and len(strategy_returns) == len(y_true):
                        metrics[f"{label}_sharpe"] = FinancialMetrics.sharpe_ratio(strategy_returns[mask])

    return metrics


def compute_all_evaluation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    scalers: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[List[str]] = None,
    target_type: str = "next_day_log_return",
    regime_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics including prediction and financial metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        model_name: Name of the model (for logging)

    Returns:
        Dictionary with all metrics
    """
    metrics: Dict[str, Any] = {}
    calc = MetricsCalculator()
    y_true = np.asarray(targets).flatten()
    y_pred = np.asarray(predictions).flatten()

    is_return_target = target_type in {"next_day_log_return", "return", "returns"}
    is_vol_target = str(target_type).lower().startswith("realized_vol") or ("vol" in str(target_type).lower())
    is_price_target = not is_return_target and not is_vol_target

    # Prediction metrics (target-aware to avoid scaled/unscaled mixups).
    if is_return_target or is_vol_target:
        metrics.update(
            {
                "rmse": calc.rmse(y_true, y_pred),
                "mae": calc.mae(y_true, y_pred),
                "mape": calc.mape(y_true, y_pred),
                "mse": calc.rmse(y_true, y_pred) ** 2,
                "r2": calc.r2(y_true, y_pred),
                "directional_accuracy": calc.directional_accuracy(
                    y_true, y_pred, are_returns=is_return_target
                )
                * 100,
            }
        )

    if is_vol_target:
        realized_var = np.square(np.clip(y_true, 1e-8, None))
        predicted_var = np.square(np.clip(y_pred, 1e-8, None))
        metrics["qlike"] = VolatilityMetrics.qlike(predicted_var, realized_var)
    elif is_price_target:
        price_mean, price_std = _extract_close_stats(scalers)
        if price_mean is None or price_std is None:
            logger.warning(
                "Price targets require scaler stats; falling back to return-mode metrics for smoke."
            )
            is_return_target = True
        else:
            pred_metrics = calculate_metrics(
                targets,
                predictions,
                prefix="",
                price_mean=price_mean,
                price_std=price_std,
            )
            metrics.update(
                {
                    "rmse": pred_metrics.get("rmse"),
                    "mae": pred_metrics.get("mae"),
                    "mape": pred_metrics.get("mape"),
                    "mse": pred_metrics.get("mse"),
                    "r2": pred_metrics.get("r2"),
                    "directional_accuracy": pred_metrics.get("directional_accuracy"),
                }
            )

    # Financial metrics.
    strategy_returns: Optional[np.ndarray] = None
    if not is_vol_target:
        try:
            if is_return_target:
                strategy_returns = compute_strategy_returns(
                    predictions=y_pred,
                    actual_prices=y_true,
                    are_returns=True,
                    transaction_cost=TRANSACTION_COST,
                    require_price_scale=False,
                )
            else:
                price_mean, price_std = _extract_close_stats(scalers)
                if price_mean is None or price_std is None:
                    raise ValueError("Price targets require scaler stats for financial metrics")
                strategy_returns = compute_strategy_returns(
                    predictions=y_pred,
                    actual_prices=y_true,
                    are_returns=False,
                    transaction_cost=TRANSACTION_COST,
                    price_mean=float(price_mean),
                    price_std=float(price_std),
                    require_price_scale=True,
                )

            fin_metrics = calculate_financial_metrics(
                returns=strategy_returns,
                risk_free_rate=RISK_FREE_RATE,
                periods_per_year=TRADING_DAYS_PER_YEAR,
                prefix="",
            )

            metrics.update(
                {
                    "sharpe_ratio": fin_metrics.get("sharpe_ratio"),
                    "sortino_ratio": fin_metrics.get("sortino_ratio"),
                    "max_drawdown": fin_metrics.get("max_drawdown"),
                    "calmar_ratio": fin_metrics.get("calmar_ratio"),
                    "total_return": fin_metrics.get("total_return"),
                    "volatility": fin_metrics.get("volatility"),
                    "win_rate": fin_metrics.get("win_rate"),
                }
            )

            if len(strategy_returns) > 0:
                metrics["annualized_return"] = (
                    FinancialMetrics.annualized_return(
                        strategy_returns, periods_per_year=TRADING_DAYS_PER_YEAR
                    )
                    * 100
                )
        except Exception as e:
            logger.warning(f"Failed to compute financial metrics for {model_name}: {e}")

    # Regime + distribution diagnostics
    metrics.update(_distribution_checks(y_true, y_pred))
    metrics.update(
        _regime_metrics(
            y_true=y_true,
            y_pred=y_pred,
            strategy_returns=strategy_returns,
            regime_context=regime_context,
            calc=calc,
        )
    )

    return metrics


def train_single_model(
    model_type: str,
    train_dataset,
    val_dataset,
    test_dataset,
    input_dim: int,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    dropout: Optional[float] = None,
    research_mode: bool = True,
    save_checkpoint: bool = True,
    device: Optional[torch.device] = None,
    scalers: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[List[str]] = None,
    target_type: str = "next_day_log_return",
    fairness_contract: Optional[Dict[str, Any]] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
    regime_context: Optional[Dict[str, Any]] = None,
    allow_target_mismatch: bool = False,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Train a single model.

    Args:
        model_type: Model key (e.g., 'lstm', 'pinn_gbm')
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        input_dim: Input feature dimension
        epochs: Number of epochs (None = use config)
        batch_size: Batch size (None = use config)
        learning_rate: Learning rate (None = use config)
        hidden_dim: Hidden dimension (None = use config)
        num_layers: Number of layers (None = use config)
        dropout: Dropout rate (None = use config)
        research_mode: Use research config
        save_checkpoint: Save model checkpoint
        device: Training device

    Returns:
        Dictionary with training results
    """
    config = get_config()
    _enforce_core_fairness(model_type, fairness_contract)
    effective_target = (
        target_type
        or (fairness_contract or {}).get("dataset", {}).get("target_type")
        or (fairness_contract or {}).get("target_type")
        or (dataset_meta or {}).get("target_type")
    )
    eff_target_str = str(effective_target) if effective_target is not None else ""
    _validate_track_target(model_type, eff_target_str, allow_target_mismatch)
    research_cfg = get_research_config() if research_mode else None

    # Set parameters from research config if in research mode
    if research_mode and research_cfg:
        epochs = epochs or research_cfg.epochs
        batch_size = batch_size or research_cfg.batch_size
        hidden_dim = hidden_dim or research_cfg.hidden_dim
        num_layers = num_layers or research_cfg.num_layers
        dropout = dropout or research_cfg.dropout
        set_seed(research_cfg.random_seed)
        print(f"Research mode: Using locked parameters (seed={research_cfg.random_seed})")
    else:
        epochs = epochs or config.training.epochs
        batch_size = batch_size or config.training.batch_size
        hidden_dim = hidden_dim or config.model.hidden_dim
        num_layers = num_layers or config.model.num_layers
        dropout = dropout or config.model.dropout

    # Device
    if device is None:
        device = get_device()

    track = _model_track(model_type)

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_type.upper()} ({track})")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    print(f"Dropout: {dropout}")
    print(f"Device: {device}")
    print(f"Research mode: {research_mode}")

    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'learning_rates': [],
        'train_components': [],
        'val_components': [],
    }

    if smoke_test:
        total_time = 0.0
        best_val_loss = 0.0
        predictions = np.zeros(len(test_dataset))
        targets = np.zeros(len(test_dataset))
        is_pinn = False
    else:
        # Create data loaders
        num_workers = 0  # Avoid multiprocessing issues on macOS
        pin_memory = device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Create model
        registry = ModelRegistry(config.project_root)
        model = registry.create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        if model is None:
            raise ValueError(f"Failed to create model '{model_type}'")

        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Determine if PINN model
        is_pinn = hasattr(model, 'compute_loss')
        print(f"Physics-informed: {is_pinn}")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=None,
            research_mode=research_mode,
        )

        best_val_loss = float('inf')
        start_time = time.time()

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_details = trainer.train_epoch(enable_physics=is_pinn)
            val_loss, val_details = trainer.validate_epoch(enable_physics=is_pinn)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['data_loss'].append(train_details.get('data_loss'))
            history['physics_loss'].append(train_details.get('physics_loss'))
            history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])

            component_snapshot = {k: v for k, v in train_details.items() if k.startswith('train_') and isinstance(v, (int, float))}
            if component_snapshot:
                history['train_components'].append(component_snapshot)

            val_component_snapshot = {k: v for k, v in val_details.items() if k.startswith('val_') and isinstance(v, (int, float))}
            if val_component_snapshot:
                history['val_components'].append(val_component_snapshot)

            # Log progress
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}, elapsed={elapsed:.1f}s")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_checkpoint:
                    trainer.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                        model_name=model_type,
                    )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")

        # Compute test metrics
        print("\nComputing test metrics...")
        predictions, targets = trainer.get_predictions(test_loader)
    test_metrics = compute_all_evaluation_metrics(
        predictions,
        targets,
        model_type,
        scalers=scalers,
        feature_cols=feature_cols,
        target_type=target_type,
        regime_context=regime_context,
    )

    scaler_map = _extract_close_scaler_map(scalers, feature_cols)

    config_hash = compute_config_hash({
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
    })

    print(f"\nTest Metrics for {model_type}:")
    print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.6f}")
    print(f"  MAE: {test_metrics.get('mae', 'N/A'):.6f}")
    print(f"  R²: {test_metrics.get('r2', 'N/A'):.4f}")
    print(f"  Directional Accuracy: {test_metrics.get('directional_accuracy', 'N/A'):.2f}%")
    print(f"  Sharpe Ratio: {test_metrics.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"  Sortino Ratio: {test_metrics.get('sortino_ratio', 'N/A'):.4f}")
    print(f"  Max Drawdown: {test_metrics.get('max_drawdown', 'N/A'):.2f}%")

    # Save results
    results = {
        'model': model_type,
        'track': track,
        'assumption_context': 'mean_reversion' if model_type == 'ou' else 'diffusion_no_arbitrage' if model_type == 'black_scholes' else None,
        'test_metrics': test_metrics,
        'history': history,
        'training_time': total_time,
        'best_val_loss': best_val_loss,
        'epochs_trained': epochs,
        'training_completed': True,
        'research_mode': research_mode,
        'is_causal': True,
        'config_hash': config_hash,
        'execution_assumptions': {
            'execution_model': 'close_to_close',
            'transaction_cost': TRANSACTION_COST,
            'position_lag': 1,
            'slippage': 0.0,
        },
        'target_type': target_type,
        'contract_fingerprint': (fairness_contract or {}).get('fingerprint')
        or (fairness_contract.get('dataset', {}) if fairness_contract else {}).get('fingerprint'),
        'contract_path': str(run_dir / "fairness_contract.json") if run_dir else None,
        'fairness_contract': fairness_contract,
        'dataset_metadata': dataset_meta,
        'scaler_params': scaler_map,
        'regime_context_keys': sorted(regime_context.keys()) if regime_context else [],
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'epochs': epochs,
        }
    }

    # Save results JSON
    results_dir = run_dir or (config.project_root / 'results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f'{model_type}_results.json'

    # Convert numpy types to Python types
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif isinstance(obj, numbers.Integral):
            return int(obj)
        elif isinstance(obj, numbers.Real) and not isinstance(obj, bool):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    with open(results_path, 'w') as f:
        json.dump(convert_to_python(results), f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


def train_multiple_models(
    model_types: List[str],
    epochs: Optional[int] = None,
    research_mode: bool = True,
    run_dir: Optional[Path] = None,
    force_refresh: bool = False,
    target_override: Optional[str] = None,
    target_column_override: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict]:
    """
    Train multiple models sequentially.

    Args:
        model_types: List of model types to train
        epochs: Number of epochs (None = use config)
        research_mode: Use research config
        **kwargs: Additional arguments passed to train_single_model

    Returns:
        Dictionary mapping model names to their results
    """
    # Prepare data once
    print("\nPreparing data for batch training...")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        scalers,
        feature_cols,
        dataset_meta,
        fairness_contract,
        regime_context,
    ) = prepare_data(
        research_mode=research_mode,
        force_refresh=force_refresh,
        target_override=target_override,
        target_column_override=target_column_override,
    )

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "fairness_contract.json", "w") as f:
            json.dump(fairness_contract, f, indent=2)

    results = {}

    for i, model_type in enumerate(model_types, 1):
        print(f"\n{'='*70}")
        print(f"Training model {i}/{len(model_types)}: {model_type}")
        print(f"{'='*70}")

        try:
            result = train_single_model(
                model_type=model_type,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                input_dim=input_dim,
                epochs=epochs,
                research_mode=research_mode,
                scalers=scalers,
                feature_cols=feature_cols,
                target_type=dataset_meta.get("target_type", "next_day_log_return"),
                fairness_contract=fairness_contract,
                dataset_meta=dataset_meta,
                run_dir=run_dir,
                regime_context=regime_context,
                **kwargs
            )
            results[model_type] = result
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {'error': str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for model_type, result in results.items():
        if 'error' in result:
            print(f"  {model_type}: FAILED - {result['error']}")
        else:
            metrics = result.get('test_metrics', {})
            print(f"  {model_type}:")
            print(f"    RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            print(f"    Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"    Dir Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%")

    # Save combined results
    config = get_config()
    out_dir = run_dir or (config.project_root / 'results')
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / 'batch_training_results.json'
    leaderboard_paths = _write_track_leaderboards(results, out_dir)

    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif isinstance(obj, numbers.Integral):
            return int(obj)
        elif isinstance(obj, numbers.Real) and not isinstance(obj, bool):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    with open(combined_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'research_mode': research_mode,
            'fairness_contract': fairness_contract,
            'leaderboards': {k: str(v) for k, v in leaderboard_paths.items()},
            'results': convert_to_python(results)
        }, f, indent=2)

    print(f"\nCombined results saved to {combined_path}")

    return results


def _write_track_leaderboards(results: Dict[str, Dict], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_paths: Dict[str, Path] = {}

    rows: List[Dict[str, Any]] = []
    for model_name, payload in results.items():
        metrics = payload.get('test_metrics') if isinstance(payload, dict) else None
        if not isinstance(metrics, dict):
            continue
        contract = payload.get('fairness_contract', {}) if isinstance(payload, dict) else {}
        dataset_section = contract.get("dataset", contract)
        row = {
            'model': model_name,
            'target_type': payload.get('target_type'),
            'fingerprint': dataset_section.get('fingerprint') or contract.get('fingerprint'),
            'sharpe_ratio': metrics.get('sharpe_ratio'),
            'sortino_ratio': metrics.get('sortino_ratio'),
            'max_drawdown': metrics.get('max_drawdown'),
            'rmse': metrics.get('rmse'),
            'mae': metrics.get('mae'),
            'r2': metrics.get('r2'),
            'directional_accuracy': metrics.get('directional_accuracy'),
            'annualized_return': metrics.get('annualized_return'),
        }
        rows.append(row)

    for track_name, track_cfg in TRACK_DEFINITIONS.items():
        models = track_cfg["models"]
        ranking_metric = track_cfg.get("ranking_metric", "sharpe_ratio")
        filtered = []
        for row in rows:
            if row['model'] in models:
                ranking_value = row.get(ranking_metric)
                if ranking_value is None:
                    continue
                enriched = dict(row)
                enriched['track'] = track_name
                enriched['ranking_metric'] = ranking_metric
                enriched['ranking_value'] = ranking_value
                filtered.append(enriched)

        if not filtered:
            # Write an empty placeholder to document missing results for this track
            placeholder = out_dir / f"leaderboard_{track_name}.csv"
            with open(placeholder, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['model', 'track', 'note'])
                writer.writerow(['', track_name, 'no results available'])
            leaderboard_paths[track_name] = placeholder
            continue

        ascending = ranking_metric in LOWER_IS_BETTER
        filtered = sorted(filtered, key=lambda r: r['ranking_value'], reverse=not ascending)

        headers = [
            'model', 'track', 'ranking_metric', 'ranking_value', 'target_type', 'fingerprint',
            'rmse', 'mae', 'r2', 'directional_accuracy', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'annualized_return'
        ]
        path = out_dir / f"leaderboard_{track_name}.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in filtered:
                writer.writerow({h: row.get(h) for h in headers})
        leaderboard_paths[track_name] = path

    return leaderboard_paths


def _write_ablation_summary(run_dir: Path, ablations_path: Path = Path("configs/ablations.yaml")) -> Optional[Path]:
    path = Path(ablations_path)
    if not path.exists():
        logger.warning("Ablations file not found: %s", path)
        return None

    data = yaml.safe_load(path.read_text()) or {}
    abls = data.get("ablations") or {}

    rows: List[Dict[str, Any]] = []
    for name, spec in abls.items():
        if not isinstance(spec, dict):
            continue
        description = spec.get("description")
        for variant in ("baseline", "treatment"):
            variant_cfg = spec.get(variant)
            if variant_cfg is None:
                continue
            flat = _flatten_dict(variant_cfg)
            rows.append({
                "ablation": name,
                "variant": variant,
                "description": description,
                "keys": ";".join(sorted(flat.keys())),
                "values": json.dumps(flat, sort_keys=True),
            })

    if not rows:
        return None

    out_path = run_dir / "ablation_summary.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "variant", "description", "keys", "values"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Neural Network Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Single model to train (e.g., lstm, gru, pinn_gbm)'
    )
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to train (e.g., lstm,gru,transformer)'
    )
    parser.add_argument(
        '--type', '-t',
        choices=['baseline', 'pinn', 'advanced', 'all'],
        help='Train all models of a specific type'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Train all available models'
    )

    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        help='Hidden dimension (default: from config)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        help='Number of layers (default: from config)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate (default: from config)'
    )

    # Mode settings
    parser.add_argument(
        '--research-mode',
        action='store_true',
        default=True,
        help='Use research config for fair comparison (default: True)'
    )
    parser.add_argument(
        '--no-research-mode',
        action='store_true',
        help='Disable research mode'
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpoint saving'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force re-fetch data from source'
    )
    parser.add_argument(
        '--volatility-mode',
        action='store_true',
        help='Use realized volatility target (volatility track)'
    )
    parser.add_argument(
        '--allow-target-mismatch',
        action='store_true',
        help='Allow model/target track mismatches (unsafe)'
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Smoke-test mode (skip training loops, deterministic stubs)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        help='Training device (default: auto-detect)'
    )

    # List available models
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models and exit'
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("\nAvailable Models:")
        print("\nBaseline Models:")
        for m in BASELINE_MODELS:
            print(f"  - {m}")
        print("\nPINN Models:")
        for m in PINN_MODELS:
            print(f"  - {m}")
        print("\nAdvanced Models:")
        for m in ADVANCED_MODELS:
            print(f"  - {m}")
        return

    # Determine research mode
    research_mode = args.research_mode and not args.no_research_mode

    # Determine device
    device = None
    if args.device:
        device = torch.device(args.device)

    target_override = 'realized_vol' if args.volatility_mode else None
    target_column_override = 'realized_vol' if args.volatility_mode else None

    # Determine models to train
    models_to_train = []

    if args.model:
        models_to_train = [args.model]
    elif args.models:
        models_to_train = [m.strip() for m in args.models.split(',')]
    elif args.type == 'baseline':
        models_to_train = BASELINE_MODELS
    elif args.type == 'pinn':
        models_to_train = PINN_MODELS
    elif args.type == 'advanced':
        models_to_train = ADVANCED_MODELS
    elif args.type == 'all' or args.all:
        models_to_train = ALL_MODELS

    if not models_to_train:
        parser.print_help()
        print("\nError: Please specify at least one model to train.")
        print("Examples:")
        print("  python scripts/train_models.py --model lstm")
        print("  python scripts/train_models.py --type baseline")
        print("  python scripts/train_models.py --all")
        return

    # Validate models
    for model in models_to_train:
        if model not in ALL_MODELS:
            print(f"Error: Unknown model '{model}'")
            print(f"Available models: {ALL_MODELS}")
            return

    # Print startup info
    print("\n" + "=" * 70)
    print("STANDALONE TRAINING SCRIPT")
    print("=" * 70)
    print(f"Models to train: {models_to_train}")
    print(f"Research mode: {research_mode}")
    print(f"Device: {device or 'auto'}")

    log_system_info()
    config = get_config()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = config.project_root / "results" / "benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    run_metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "models_to_train": models_to_train,
        "research_mode": research_mode,
        "device": str(device or "auto"),
        "cli_args": vars(args),
        "seed": get_research_config().random_seed if research_mode else config.training.random_seed,
        "data_config": config.data.model_dump() if hasattr(config.data, "model_dump") else str(config.data),
        "track_mode": "volatility" if target_override else "core",
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # Build kwargs
    kwargs = {
        'save_checkpoint': not args.no_checkpoint,
        'device': device,
    }
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.hidden_dim:
        kwargs['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        kwargs['num_layers'] = args.num_layers
    if args.dropout:
        kwargs['dropout'] = args.dropout

    # Train
    if len(models_to_train) == 1:
        # Single model training
        (
            train_dataset,
            val_dataset,
            test_dataset,
            input_dim,
            scalers,
            feature_cols,
            dataset_meta,
            fairness_contract,
            regime_context,
        ) = prepare_data(
            research_mode=research_mode,
            force_refresh=args.force_refresh,
            target_override=target_override,
            target_column_override=target_column_override,
        )
        with open(run_dir / "fairness_contract.json", "w") as f:
            json.dump(fairness_contract, f, indent=2)
        result = train_single_model(
            model_type=models_to_train[0],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            epochs=args.epochs,
            research_mode=research_mode,
            scalers=scalers,
            feature_cols=feature_cols,
            target_type=dataset_meta.get("target_type", "next_day_log_return"),
            fairness_contract=fairness_contract,
            dataset_meta=dataset_meta,
            run_dir=run_dir,
            regime_context=regime_context,
            allow_target_mismatch=args.allow_target_mismatch,
            smoke_test=args.smoke,
            **kwargs
        )

        _write_track_leaderboards({models_to_train[0]: result}, run_dir)
    else:
        # Batch training
        train_multiple_models(
            model_types=models_to_train,
            epochs=args.epochs,
            research_mode=research_mode,
            run_dir=run_dir,
            force_refresh=args.force_refresh,
            target_override=target_override,
            target_column_override=target_column_override,
            allow_target_mismatch=args.allow_target_mismatch,
            smoke_test=args.smoke,
            **kwargs
        )

    _write_ablation_summary(run_dir)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
