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
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import PhysicsAwareDataset, collate_fn_with_metadata
from src.models.model_registry import ModelRegistry
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics, calculate_financial_metrics
from src.evaluation.financial_metrics import (
    compute_strategy_returns,
    FinancialMetrics,
    compute_all_metrics,
)
from src.constants import TRANSACTION_COST, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)

# Model categories
BASELINE_MODELS = ['lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer']
PINN_MODELS = ['baseline_pinn', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global']
ADVANCED_MODELS = ['stacked', 'residual']
ALL_MODELS = BASELINE_MODELS + PINN_MODELS + ADVANCED_MODELS


def prepare_data(
    ticker: str = "^GSPC",
    sequence_length: Optional[int] = None,
    research_mode: bool = True,
    force_refresh: bool = False,
    use_multi_ticker: bool = True,
) -> tuple:
    """
    Prepare research-grade normalized data for training.

    Args:
        ticker: Primary stock ticker (used for single-ticker mode)
        sequence_length: Override sequence length (None = use config)
        research_mode: Use research config parameters
        force_refresh: Force re-fetch data from source
        use_multi_ticker: Train on multiple S&P 500 stocks

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, input_dim, scalers)
    """
    config = get_config()
    research_cfg = get_research_config() if research_mode else None

    # Use research config values when in research mode
    if research_mode and research_cfg:
        if sequence_length is None:
            sequence_length = research_cfg.sequence_length
    else:
        if sequence_length is None:
            sequence_length = config.data.sequence_length

    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()

    # Date range
    start_date = config.data.start_date
    end_date = config.data.end_date

    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    print(f"Research mode: {research_mode}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Sequence length: {sequence_length}")
    print(f"Force refresh: {force_refresh}")

    # Determine tickers
    if use_multi_ticker:
        tickers = config.data.tickers[:10]
        print(f"Multi-ticker mode: Training on {len(tickers)} stocks")
        print(f"  Tickers: {tickers}")
    else:
        tickers = [ticker]
        print(f"Single-ticker mode: {ticker}")

    # Fetch data
    print("Fetching data...")
    df = fetcher.fetch_and_store(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    if df.empty:
        raise ValueError(f"No data fetched for {tickers}")

    # Data coverage validation
    data_start = df['time'].min()
    data_end = df['time'].max()
    if hasattr(data_start, 'tz') and data_start.tz is not None:
        data_start = data_start.tz_localize(None)
        data_end = data_end.tz_localize(None)

    coverage_days = (data_end - data_start).days
    coverage_years = coverage_days / 365.0
    print(f"Data coverage: {data_start.date()} to {data_end.date()} ({coverage_years:.1f} years)")
    print(f"Total rows: {len(df)}")

    # Process features
    df = preprocessor.process_and_store(df)

    # Feature set (research-grade)
    research_features = [
        'close', 'volume',
        'log_return', 'simple_return',
        'rolling_volatility_5', 'rolling_volatility_20',
        'momentum_5', 'momentum_20',
        'rsi_14', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower', 'atr_14',
    ]
    feature_cols = [col for col in research_features if col in df.columns]

    required = ['close', 'log_return']
    missing = [f for f in required if f not in feature_cols]
    if missing:
        raise ValueError(f"Required features missing: {missing}")

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Temporal split
    train_df, val_df, test_df = preprocessor.split_temporal(df)
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Normalization
    for col in feature_cols:
        train_df[col] = train_df[col].astype(np.float64)
        val_df[col] = val_df[col].astype(np.float64)
        test_df[col] = test_df[col].astype(np.float64)

    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    # Apply same scalers to val/test
    val_df_norm = val_df.copy()
    test_df_norm = test_df.copy()

    for ticker_name in val_df['ticker'].unique():
        if ticker_name in scalers:
            val_mask = val_df_norm['ticker'] == ticker_name
            val_df_norm.loc[val_mask, feature_cols] = scalers[ticker_name].transform(
                val_df_norm.loc[val_mask, feature_cols]
            )

    for ticker_name in test_df['ticker'].unique():
        if ticker_name in scalers:
            test_mask = test_df_norm['ticker'] == ticker_name
            test_df_norm.loc[test_mask, feature_cols] = scalers[ticker_name].transform(
                test_df_norm.loc[test_mask, feature_cols]
            )

    # Create sequences
    target_col = 'close'

    X_train, y_train, tickers_train = preprocessor.create_sequences(
        train_df_norm, feature_cols, target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )
    X_val, y_val, tickers_val = preprocessor.create_sequences(
        val_df_norm, feature_cols, target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )
    X_test, y_test, tickers_test = preprocessor.create_sequences(
        test_df_norm, feature_cols, target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )

    print(f"Sequences - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Physics metadata (unnormalized)
    P_train, _, _ = preprocessor.create_sequences(
        train_df, ['close', 'log_return', 'rolling_volatility_20'], target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )
    P_val, _, _ = preprocessor.create_sequences(
        val_df, ['close', 'log_return', 'rolling_volatility_20'], target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )
    P_test, _, _ = preprocessor.create_sequences(
        test_df, ['close', 'log_return', 'rolling_volatility_20'], target_col=target_col,
        sequence_length=sequence_length, forecast_horizon=1
    )

    # Create datasets
    train_dataset = PhysicsAwareDataset(
        X_train, y_train, tickers_train,
        prices=P_train[:, :, 0], returns=P_train[:, :, 1], volatilities=P_train[:, :, 2]
    )
    val_dataset = PhysicsAwareDataset(
        X_val, y_val, tickers_val,
        prices=P_val[:, :, 0], returns=P_val[:, :, 1], volatilities=P_val[:, :, 2]
    )
    test_dataset = PhysicsAwareDataset(
        X_test, y_test, tickers_test,
        prices=P_test[:, :, 0], returns=P_test[:, :, 1], volatilities=P_test[:, :, 2]
    )

    input_dim = X_train.shape[2]

    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"  Input dimension: {input_dim}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("=" * 70)

    # Validate physics metadata alignment (volatility/returns/prices)
    assert P_train.shape[:2] == X_train.shape[:2], "P_train alignment mismatch"
    assert P_val.shape[:2] == X_val.shape[:2], "P_val alignment mismatch"
    assert P_test.shape[:2] == X_test.shape[:2], "P_test alignment mismatch"
    if np.isnan(P_train).any() or np.isnan(P_val).any() or np.isnan(P_test).any():
        raise ValueError("NaNs detected in physics metadata (prices/returns/volatilities)")

    return train_dataset, val_dataset, test_dataset, input_dim, scalers, feature_cols


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


def compute_all_evaluation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    scalers: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[List[str]] = None,
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
    metrics = {}

    price_mean, price_std = _extract_close_stats(scalers)

    # Prediction metrics (de-standardise when stats available)
    pred_metrics = calculate_metrics(
        targets,
        predictions,
        prefix="",
        price_mean=price_mean,
        price_std=price_std
    )
    metrics.update({
        "rmse": pred_metrics.get("rmse"),
        "mae": pred_metrics.get("mae"),
        "mape": pred_metrics.get("mape"),
        "mse": pred_metrics.get("mse"),
        "r2": pred_metrics.get("r2"),
        "directional_accuracy": pred_metrics.get("directional_accuracy"),
    })

    # Financial metrics
    if price_mean is not None and price_std is not None:
        pm = float(price_mean)
        ps = float(price_std)
        try:
            strategy_returns = compute_strategy_returns(
                predictions=predictions,
                actual_prices=targets,
                are_returns=False,  # derive returns from price levels
                transaction_cost=TRANSACTION_COST,
                price_mean=pm,
                price_std=ps,
                require_price_scale=True,
            )

            fin_metrics = calculate_financial_metrics(
                returns=strategy_returns,
                risk_free_rate=RISK_FREE_RATE,
                periods_per_year=TRADING_DAYS_PER_YEAR,
                prefix=""
            )

            metrics.update({
                "sharpe_ratio": fin_metrics.get("sharpe_ratio"),
                "sortino_ratio": fin_metrics.get("sortino_ratio"),
                "max_drawdown": fin_metrics.get("max_drawdown"),
                "calmar_ratio": fin_metrics.get("calmar_ratio"),
                "total_return": fin_metrics.get("total_return"),
                "volatility": fin_metrics.get("volatility"),
                "win_rate": fin_metrics.get("win_rate"),
            })

            # Annualized return
            if len(strategy_returns) > 0:
                metrics["annualized_return"] = FinancialMetrics.annualized_return(
                    strategy_returns, periods_per_year=TRADING_DAYS_PER_YEAR
                ) * 100

        except Exception as e:
            logger.warning(f"Failed to compute financial metrics for {model_name}: {e}")
    else:
        logger.warning("Skipping financial metrics: close price scaler stats unavailable (multi-ticker or missing scaler)")

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

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_type.upper()}")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    print(f"Dropout: {dropout}")
    print(f"Device: {device}")
    print(f"Research mode: {research_mode}")

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

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'learning_rates': [],
        'train_components': [],
        'val_components': [],
    }

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
        'scaler_params': scaler_map,
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'epochs': epochs,
        }
    }

    # Save results JSON
    results_dir = config.project_root / 'results'
    results_dir.mkdir(exist_ok=True)
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
    train_dataset, val_dataset, test_dataset, input_dim, scalers, feature_cols = prepare_data(
        research_mode=research_mode,
    )

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
    combined_path = config.project_root / 'results' / 'batch_training_results.json'

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
            'results': convert_to_python(results)
        }, f, indent=2)

    print(f"\nCombined results saved to {combined_path}")

    return results


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
        train_dataset, val_dataset, test_dataset, input_dim, scalers, feature_cols = prepare_data(
            research_mode=research_mode,
            force_refresh=args.force_refresh,
        )
        train_single_model(
            model_type=models_to_train[0],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            epochs=args.epochs,
            research_mode=research_mode,
            scalers=scalers,
            feature_cols=feature_cols,
            **kwargs
        )
    else:
        # Batch training
        train_multiple_models(
            model_types=models_to_train,
            epochs=args.epochs,
            research_mode=research_mode,
            **kwargs
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
