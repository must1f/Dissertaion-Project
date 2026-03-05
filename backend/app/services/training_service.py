"""Service for model training wrapping src/training/."""

import sys
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from collections import OrderedDict
import uuid
import asyncio
import threading
import time
import platform

import torch
import numpy as np

# Maximum number of jobs to keep in memory (prevents unbounded growth)
MAX_JOBS_IN_MEMORY = 100
MAX_BATCH_JOBS_IN_MEMORY = 50

# ============================================================
# VERBOSE DEBUG MODE
# ============================================================
# Set TRAINING_DEBUG=1 in environment for extra verbose output
DEBUG_MODE = os.environ.get("TRAINING_DEBUG", "1") == "1"


def debug_log(message: str, level: str = "DEBUG"):
    """Print debug message if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import TrainingError
from backend.app.services.model_service import _get_available_device
from backend.app.schemas.training import (
    TrainingRequest,
    TrainingJobInfo,
    TrainingStatus,
    EpochMetrics,
    # Batch training schemas
    BatchTrainingRequest,
    BatchTrainingJobInfo,
    BatchJobStatus,
    ModelTrainingConfig,
    AVAILABLE_MODELS,
)

# Import from existing src/
HAS_SRC = False
SRC_IMPORT_ERROR = None

try:
    from src.training.trainer import Trainer
    from src.data.fetcher import DataFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.dataset import FinancialDataset, PhysicsAwareDataset, collate_fn_with_metadata
    from src.models.model_registry import ModelRegistry
    from src.evaluation.metrics import calculate_metrics, calculate_financial_metrics
    from src.evaluation.financial_metrics import compute_strategy_returns, FinancialMetrics
    from src.evaluation.unified_evaluator import UnifiedModelEvaluator
    from src.evaluation.pipeline import run_pipeline_evaluation
    from src.utils.reproducibility import set_seed, log_system_info, get_device
    from src.constants import TRANSACTION_COST, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
    HAS_SRC = True
    print("[TrainingService] Successfully imported src/ modules - using REAL neural network training")
except ImportError as e:
    SRC_IMPORT_ERROR = str(e)
    Trainer = None
    calculate_metrics = None
    calculate_financial_metrics = None
    print(f"[TrainingService] WARNING: Failed to import src/ modules: {e}")
    print("[TrainingService] Training will use SIMULATED mode (fake losses)")


def get_training_mode_info() -> dict:
    """Get information about the current training mode."""
    return {
        "mode": "real" if HAS_SRC else "simulated",
        "using_real_models": HAS_SRC,
        "import_error": SRC_IMPORT_ERROR,
        "message": (
            "Connected to actual neural network models (LSTM, GRU, Transformer, PINN)"
            if HAS_SRC
            else f"Using simulated training - src/ import failed: {SRC_IMPORT_ERROR}"
        ),
    }


def compute_research_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    are_returns: bool = True
) -> Dict[str, float]:
    """
    Compute research-grade metrics for model evaluation.

    Args:
        predictions: Model predictions (returns or prices)
        targets: Ground truth values
        are_returns: Whether values are returns (True) or prices (False)

    Returns:
        Dictionary with prediction and financial metrics
    """
    import numpy as np

    if not HAS_SRC:
        return {}

    metrics = {}

    # Prediction metrics (RMSE, MAE, MAPE, R², directional accuracy)
    pred_metrics = calculate_metrics(targets, predictions, prefix="")
    metrics.update({
        "rmse": pred_metrics.get("rmse"),
        "mae": pred_metrics.get("mae"),
        "mape": pred_metrics.get("mape"),
        "mse": pred_metrics.get("mse"),
        "r2": pred_metrics.get("r2"),
        "directional_accuracy": pred_metrics.get("directional_accuracy"),
    })

    # Financial metrics (requires computing strategy returns)
    try:
        strategy_returns = compute_strategy_returns(
            predictions=predictions,
            actual_prices=targets,
            are_returns=are_returns,
            transaction_cost=0.001,  # 10 bps
        )

        fin_metrics = calculate_financial_metrics(
            returns=strategy_returns,
            risk_free_rate=0.02,
            periods_per_year=252,
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
                strategy_returns, periods_per_year=252
            ) * 100  # Convert to percentage

    except Exception as e:
        print(f"[METRICS] Warning: Failed to compute financial metrics: {e}")

    return metrics


def compute_financial_evaluation(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    transaction_cost: float = TRANSACTION_COST if HAS_SRC else 0.003,
    risk_free_rate: float = RISK_FREE_RATE if HAS_SRC else 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR if HAS_SRC else 252,
) -> Dict[str, Any]:
    """
    Compute comprehensive financial metrics using UnifiedModelEvaluator.
    """
    if not HAS_SRC:
        return {}

    evaluator = UnifiedModelEvaluator(
        transaction_cost=transaction_cost,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    try:
        return evaluator.evaluate_model(
            predictions=predictions,
            targets=targets,
            model_name=model_name,
            compute_rolling=False,
        )
    except Exception as e:
        debug_log(f"Financial evaluation failed for {model_name}: {e}", "ERROR")
        return {}


def prepare_normalized_data(
    ticker: str,
    sequence_length: Optional[int] = None,  # None = use config/research config
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = True,  # Default True for research-grade freshness
    use_multi_ticker: bool = True,  # Default True to match terminal training
    min_years: float = 9.0,  # Minimum years of data required
    research_mode: bool = True,  # Use research config parameters
):
    """
    Prepare research-grade normalized data for training.

    Matches the terminal training approach (src/training/train.py) with:
    - Comprehensive feature set (prices, technicals, returns)
    - Proper temporal splits (no data leakage)
    - Standard normalization per ticker
    - PhysicsAwareDataset for PINN models
    - Date coverage validation (requires 9+ years by default)

    Args:
        ticker: Primary stock ticker symbol (or index like ^GSPC)
        sequence_length: Number of time steps for input sequences (default 120 for research)
        start_date: Start date for data (YYYY-MM-DD), defaults to 10 years ago
        end_date: End date for data (YYYY-MM-DD), defaults to today
        force_refresh: If True, re-fetch data even if cached (default True for research)
        use_multi_ticker: If True, train on multiple S&P 500 stocks (like terminal)
        min_years: Minimum years of data coverage required (default 9.0)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, input_dim, scalers)
    """
    if not HAS_SRC:
        raise RuntimeError(f"Cannot prepare data: src/ modules not available. Error: {SRC_IMPORT_ERROR}")

    from torch.utils.data import DataLoader
    from datetime import datetime, timedelta
    import numpy as np
    from src.utils.config import get_config, get_research_config

    config = get_config()
    research_config = get_research_config() if research_mode else None
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()

    # ============================================================
    # USE RESEARCH CONFIG PARAMETERS FOR CONSISTENCY WITH TERMINAL
    # ============================================================
    if research_mode and research_config:
        # Use research config sequence_length unless explicitly overridden
        if sequence_length is None:
            sequence_length = research_config.sequence_length
        print(f"[DATA PREP] Research mode: Using locked parameters from ResearchConfig")
        print(f"[DATA PREP]   sequence_length={sequence_length}, batch_size={research_config.batch_size}")
    else:
        # Fall back to regular config
        if sequence_length is None:
            sequence_length = config.data.sequence_length

    # ============================================================
    # RESEARCH-GRADE DATE RANGE (matches terminal: config.data.start_date/end_date)
    # ============================================================
    # Use config dates to ensure consistency with terminal training
    if end_date is None:
        end_date = config.data.end_date  # Uses dynamic date from config
    if start_date is None:
        start_date = config.data.start_date  # Uses dynamic date from config (10 years by default)

    print("=" * 70)
    print("[DATA PREP] RESEARCH-GRADE DATA PREPARATION (matching terminal training)")
    print("=" * 70)
    print(f"[DATA PREP] Research mode: {research_mode}")
    print(f"[DATA PREP] Date range: {start_date} to {end_date}")
    print(f"[DATA PREP] Force refresh: {force_refresh}")
    print(f"[DATA PREP] Sequence length: {sequence_length}")

    # Determine tickers to use
    # Terminal training uses config.data.tickers[:10] for multi-ticker mode
    if use_multi_ticker:
        # Use top 10 S&P 500 stocks like terminal training (src/training/train.py:52)
        tickers = config.data.tickers[:10]
        print(f"[DATA PREP] Multi-ticker mode (like terminal): Training on {len(tickers)} stocks")
        print(f"[DATA PREP]   Tickers: {tickers}")
    else:
        tickers = [ticker]
        print(f"[DATA PREP] Single-ticker mode: {ticker}")

    # ============================================================
    # FETCH DATA (with force_refresh for research freshness)
    # ============================================================
    print(f"[DATA PREP] Fetching data...")
    df = fetcher.fetch_and_store(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    if df.empty:
        raise ValueError(f"No data fetched for {tickers}. Check ticker symbols and date range.")

    # ============================================================
    # COVERAGE VALIDATION (guardrail)
    # ============================================================
    data_start = df['time'].min()
    data_end = df['time'].max()

    # Handle timezone-aware timestamps
    if hasattr(data_start, 'tz') and data_start.tz is not None:
        data_start = data_start.tz_localize(None)
        data_end = data_end.tz_localize(None)

    coverage_days = (data_end - data_start).days
    coverage_years = coverage_days / 365.0

    print(f"[DATA PREP] Data coverage: {data_start.date()} to {data_end.date()} ({coverage_years:.1f} years)")
    print(f"[DATA PREP] Total rows fetched: {len(df)}")
    print(f"[DATA PREP] Rows per ticker: ~{len(df) // len(tickers)}")

    if coverage_years < min_years:
        raise ValueError(
            f"Insufficient data coverage: {coverage_years:.1f} years < {min_years} years required. "
            f"Data spans {data_start.date()} to {data_end.date()}. "
            f"Try force_refresh=True or check your data source."
        )

    # Process features
    df = preprocessor.process_and_store(df)

    # ============================================================
    # RESEARCH-GRADE FEATURE SET (matches terminal training)
    # ============================================================
    # Comprehensive features: prices + technicals + returns
    # This matches src/training/train.py for consistent research results
    research_features = [
        # Price and volume (normalized)
        'close', 'volume',
        # Return-based features (stationary)
        'log_return', 'simple_return',
        # Volatility features
        'rolling_volatility_5', 'rolling_volatility_20',
        # Momentum features
        'momentum_5', 'momentum_20',
        # Technical indicators (if available)
        'rsi_14', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower', 'atr_14',
    ]

    # Filter to available features
    feature_cols = [col for col in research_features if col in df.columns]

    # Ensure minimum required features
    required_features = ['close', 'log_return']
    missing = [f for f in required_features if f not in feature_cols]
    if missing:
        raise ValueError(f"Required features missing: {missing}. Available: {list(df.columns)}")

    print(f"[DATA PREP] Using {len(feature_cols)} research-grade features: {feature_cols}")

    # ============================================================
    # TEMPORAL SPLIT (no data leakage)
    # ============================================================
    train_df, val_df, test_df = preprocessor.split_temporal(df)

    print(f"[DATA PREP] Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"[DATA PREP] Train period: {train_df['time'].min()} to {train_df['time'].max()}")
    print(f"[DATA PREP] Val period: {val_df['time'].min()} to {val_df['time'].max()}")
    print(f"[DATA PREP] Test period: {test_df['time'].min()} to {test_df['time'].max()}")

    # ============================================================
    # NORMALIZATION (fit on train only, apply to val/test)
    # ============================================================
    # Convert feature columns to float64 to avoid dtype issues during normalization
    for col in feature_cols:
        train_df[col] = train_df[col].astype(np.float64)
        val_df[col] = val_df[col].astype(np.float64)
        test_df[col] = test_df[col].astype(np.float64)

    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    # Apply same scalers to validation and test data (no leakage)
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

    # ============================================================
    # CREATE SEQUENCES
    # ============================================================
    # Target: log returns for return prediction (aligns with evaluation)
    target_col = 'log_return'
    print(f"[DATA PREP] Target column: {target_col} (returns-first to match evaluation)")

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

    print(f"[DATA PREP] Sequences - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"[DATA PREP] Total samples: {len(X_train) + len(X_val) + len(X_test)}")

    # Validation set size check - warn if too small for reliable metrics
    min_val_samples = 500
    if len(X_val) < min_val_samples:
        print(f"[DATA PREP] WARNING: Validation set small ({len(X_val)} < {min_val_samples}). "
              f"Metrics may be unreliable. Consider longer date range or fewer features.")

    # Check train/val ratio
    val_ratio = len(X_val) / len(X_train) if len(X_train) > 0 else 0
    if val_ratio < 0.1:
        print(f"[DATA PREP] WARNING: Val/Train ratio low ({val_ratio:.2%}). "
              f"May cause unstable validation metrics.")

    # ============================================================
    # CREATE DATASETS (PhysicsAwareDataset for PINN support)
    # ============================================================
    # Create unnormalized sequences for physics equations
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

    # Use PhysicsAwareDataset which provides metadata for physics constraints
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

    # ============================================================
    # OBSERVABILITY SUMMARY
    # ============================================================
    print("=" * 70)
    print("[DATA PREP] RESEARCH-GRADE DATA SUMMARY")
    print("=" * 70)
    print(f"  Data coverage: {coverage_years:.1f} years ({data_start.date()} to {data_end.date()})")
    print(f"  Tickers: {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    print(f"  Features: {input_dim} ({', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''})")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Batches/epoch (batch_size=32): {len(X_train) // 32}")
    print(f"  Batches/epoch (batch_size=16): {len(X_train) // 16}")
    print("=" * 70)

    return train_dataset, val_dataset, test_dataset, input_dim, scalers


class TrainingJob:
    """Represents a training job."""

    def __init__(
        self,
        job_id: str,
        request: TrainingRequest,
        callback: Optional[Callable] = None,
    ):
        self.job_id = job_id
        self.request = request
        self.callback = callback

        self.status = TrainingStatus.PENDING
        self.current_epoch = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.learning_rates: List[float] = []
        self.train_data_losses: List[Optional[float]] = []
        self.train_physics_losses: List[Optional[float]] = []
        self.physics_params_history: List[Dict[str, float]] = []

        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.saved_model_name: Optional[str] = None

        # Research-grade metrics (computed on test set)
        self.test_metrics: Optional[Dict[str, float]] = None
        self.final_metrics: Optional[Dict[str, Any]] = None

        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    @property
    def info(self) -> TrainingJobInfo:
        """Get job information."""
        elapsed = None
        if self.started_at:
            end = self.completed_at or datetime.now()
            elapsed = (end - self.started_at).total_seconds()

        # Convert test_metrics dict to ResearchMetrics schema if available
        from backend.app.schemas.training import ResearchMetrics
        test_metrics_schema = None
        if self.test_metrics:
            test_metrics_schema = ResearchMetrics(**self.test_metrics)

        return TrainingJobInfo(
            job_id=self.job_id,
            model_type=self.request.model_type,
            ticker=self.request.ticker,
            status=self.status,
            current_epoch=self.current_epoch,
            total_epochs=self.request.epochs,
            progress_percent=(self.current_epoch / self.request.epochs) * 100,
            started_at=self.started_at,
            completed_at=self.completed_at,
            elapsed_seconds=elapsed,
            current_train_loss=self.train_losses[-1] if self.train_losses else None,
            current_val_loss=self.val_losses[-1] if self.val_losses else None,
            best_val_loss=self.best_val_loss if self.best_val_loss < float("inf") else None,
            test_metrics=test_metrics_schema,
            saved_model_name=self.saved_model_name,
            config=self.request.model_dump(),
            final_metrics=self.final_metrics,
        )

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "learning_rate": self.learning_rates,
        }

    def stop(self):
        """Request job to stop."""
        self._stop_flag = True


class TrainingService:
    """Service for training models."""

    def __init__(self):
        """Initialize training service."""
        # Thread-safe job storage with bounded size
        self._jobs: OrderedDict[str, TrainingJob] = OrderedDict()
        self._batch_jobs: OrderedDict[str, "BatchTrainingJob"] = OrderedDict()
        self._jobs_lock = threading.Lock()  # Protects _jobs access
        self._batch_jobs_lock = threading.Lock()  # Protects _batch_jobs access
        self._device = _get_available_device(settings.default_device)
        self._load_persisted_jobs()

    def _load_persisted_jobs(self):
        """Load past training runs from results directory."""
        try:
            results_dir = settings.results_path
            if not results_dir.exists():
                return
            
            import json
            for results_file in results_dir.glob("*_results.json"):
                try:
                    # Skip rigorous evaluation and comparison files
                    if "rigorous" in results_file.name or results_file.parent.name == "pinn_comparison":
                        continue
                        
                    with open(results_file, "r") as f:
                        data = json.load(f)
                    
                    if "training_completed" in data and "history" in data:
                        model_key = data.get("model", results_file.stem.replace("_results", ""))
                        
                        # Create pseudo-request
                        request = TrainingRequest(
                            model_type=model_key,
                            ticker="^GSPC", # Dummy
                            epochs=len(data["history"].get("train_loss", [])),
                        )
                        
                        job_id = f"past_{model_key}"
                        job = TrainingJob(job_id, request)
                        job.status = TrainingStatus.COMPLETED
                        
                        # Populate history
                        history = data["history"]
                        job.train_losses = history.get("train_loss", [])
                        job.val_losses = history.get("val_loss", [])
                        job.learning_rates = history.get("learning_rates", []) or history.get("learning_rate", [])
                        job.train_data_losses = history.get("data_loss", [])
                        job.train_physics_losses = history.get("physics_loss", [])
                        job.current_epoch = len(job.train_losses)
                        
                        if "best_val_loss" in history:
                            job.best_val_loss = history["best_val_loss"]
                        elif job.val_losses:
                            job.best_val_loss = min((v for v in job.val_losses if v is not None), default=float("inf"))
                            
                        # Set times
                        import os
                        from datetime import timedelta
                        mtime = os.path.getmtime(results_file)
                        dt = datetime.fromtimestamp(mtime)
                        job.completed_at = dt
                        job.started_at = dt - timedelta(minutes=5)
                        job.saved_model_name = model_key
                        
                        self._jobs[job_id] = job
                except Exception as e:
                    print(f"Failed to load past job from {results_file}: {e}")
        except Exception as e:
            print(f"Error loading persisted jobs: {e}")

    def start_training(
        self,
        request: TrainingRequest,
        callback: Optional[Callable] = None,
    ) -> str:
        """Start a new training job."""
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(job_id, request, callback)

        with self._jobs_lock:
            # Evict oldest jobs if at capacity
            while len(self._jobs) >= MAX_JOBS_IN_MEMORY:
                oldest_id, oldest_job = next(iter(self._jobs.items()))
                # Only evict completed/failed/stopped jobs
                if oldest_job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED):
                    self._jobs.pop(oldest_id)
                else:
                    break  # Don't evict running jobs
            self._jobs[job_id] = job

        # Start training in background thread
        job._thread = threading.Thread(
            target=self._run_training,
            args=(job,),
            daemon=True,
        )
        job._thread.start()

        return job_id

    def _run_training(self, job: TrainingJob):
        """Run training in background thread."""
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()

        try:
            if not HAS_SRC:
                # Simulate training
                self._simulate_training(job)
            else:
                self._actual_training(job)

            if job._stop_flag:
                job.status = TrainingStatus.STOPPED
            else:
                job.status = TrainingStatus.COMPLETED

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            print(f"Training job {job.job_id} failed: {e}")

        job.completed_at = datetime.now()

    def _simulate_training(self, job: TrainingJob):
        """Simulate training for testing."""
        import random

        for epoch in range(1, job.request.epochs + 1):
            if job._stop_flag:
                break

            # Simulate epoch
            time.sleep(0.5)  # Simulate training time

            train_loss = 0.5 / (epoch + 1) + random.random() * 0.05
            val_loss = 0.6 / (epoch + 1) + random.random() * 0.05

            job.current_epoch = epoch
            job.train_losses.append(train_loss)
            job.val_losses.append(val_loss)
            job.learning_rates.append(job.request.learning_rate * (0.99 ** epoch))
            job.train_data_losses.append(None)
            job.train_physics_losses.append(None)

            if val_loss < job.best_val_loss:
                job.best_val_loss = val_loss

            # Physics params for PINN
            if "pinn" in job.request.model_type.lower():
                job.physics_params_history.append({
                    "theta": 0.5 + random.random() * 0.1,
                    "gamma": 0.1 + random.random() * 0.05,
                    "temperature": 0.01 + random.random() * 0.005,
                })

            # Call callback if provided
            if job.callback:
                job.callback({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": job.best_val_loss,
                })

    def _actual_training(self, job: TrainingJob):
        """Run actual training using src/ modules."""
        from torch.utils.data import DataLoader

        print(f"[REAL TRAINING] Starting actual neural network training for {job.request.model_type}")
        print(f"[REAL TRAINING] Ticker: {job.request.ticker}, Epochs: {job.request.epochs}")

        # Initialize model registry
        registry = ModelRegistry(settings.project_root)

        # Get research mode setting (default True for research-grade training)
        research_mode = getattr(job.request, 'research_mode', True)

        # Determine multi-ticker mode: force multi-ticker in research mode for consistency with terminal
        if research_mode:
            use_multi_ticker = True  # Force multi-ticker in research mode
            print(f"[REAL TRAINING] Research mode: Forcing multi-ticker mode (like terminal training)")
        else:
            use_multi_ticker = getattr(job.request, 'use_multi_ticker', False)

        # Prepare research-grade normalized data (matches terminal training)
        train_dataset, val_dataset, test_dataset, input_dim, scalers = prepare_normalized_data(
            ticker=job.request.ticker,
            sequence_length=job.request.sequence_length if job.request.sequence_length else None,
            start_date=getattr(job.request, 'start_date', None),
            end_date=getattr(job.request, 'end_date', None),
            use_multi_ticker=use_multi_ticker,
            research_mode=research_mode,
        )

        # Store scalers for later inverse transform if needed
        job.scalers = scalers

        # ============================================================
        # USE RESEARCH CONFIG FOR TRAINING PARAMETERS (matches terminal)
        # ============================================================
        from src.utils.config import get_research_config
        research_cfg = get_research_config() if research_mode else None

        # Determine actual training parameters - research config overrides request in research mode
        if research_mode and research_cfg:
            actual_batch_size = research_cfg.batch_size
            actual_hidden_dim = research_cfg.hidden_dim
            actual_num_layers = research_cfg.num_layers
            actual_dropout = research_cfg.dropout
            actual_epochs = research_cfg.epochs
            print(f"[REAL TRAINING] Research mode: Using locked parameters from ResearchConfig")
            print(f"[REAL TRAINING]   batch_size={actual_batch_size}, hidden_dim={actual_hidden_dim}")
            print(f"[REAL TRAINING]   num_layers={actual_num_layers}, dropout={actual_dropout}")
            print(f"[REAL TRAINING]   epochs={actual_epochs}")
        else:
            actual_batch_size = job.request.batch_size
            actual_hidden_dim = job.request.hidden_dim
            actual_num_layers = job.request.num_layers
            actual_dropout = job.request.dropout
            actual_epochs = job.request.epochs

        pin_memory = self._device.type == "cuda"
        num_workers = 0 if platform.system() == 'Darwin' else 2

        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=actual_batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Create model with research config parameters
        model = registry.create_model(
            model_type=job.request.model_type,
            input_dim=input_dim,
            hidden_dim=actual_hidden_dim,
            num_layers=actual_num_layers,
            dropout=actual_dropout,
        )

        if model is None:
            raise ValueError(f"Failed to create model '{job.request.model_type}'")

        print(f"[REAL TRAINING] Created model: {model.__class__.__name__}")
        print(f"[REAL TRAINING] Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer with research mode for fair model comparison
        # Research mode disables early stopping so models train for full epochs
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self._device,
            config=None,  # Uses default config from get_config()
            research_mode=research_mode,  # Disables early stopping for fair comparison
        )

        if research_mode:
            print(f"[REAL TRAINING] Research mode ENABLED - early stopping DISABLED")

        # Training loop - use actual_epochs from research config when in research mode
        for epoch in range(1, actual_epochs + 1):
            if job._stop_flag:
                break

            train_loss, train_details = trainer.train_epoch(
                enable_physics=job.request.enable_physics,
            )
            val_loss, val_details = trainer.validate_epoch(
                enable_physics=job.request.enable_physics,
            )

            job.current_epoch = epoch
            job.train_losses.append(train_loss)
            job.val_losses.append(val_loss)
            
            # Save data and physics losses
            job.train_data_losses.append(train_details.get("train_data_loss"))
            job.train_physics_losses.append(train_details.get("train_physics_loss"))
            
            # Save learning rate
            job.learning_rates.append(trainer.optimizer.param_groups[0]['lr'])

            if val_loss < job.best_val_loss:
                job.best_val_loss = val_loss
                if job.request.save_checkpoints:
                    # Save checkpoint using model_type as the model name
                    trainer.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                        model_name=job.request.model_type,
                    )
                    # Store the model name for later reference
                    job.saved_model_name = job.request.model_type

            # Get physics params if PINN
            if hasattr(model, "get_learned_physics_params"):
                params = model.get_learned_physics_params()
                job.physics_params_history.append(params)

            # Callback
            if job.callback:
                job.callback({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": job.best_val_loss,
                    "train_details": train_details,
                    "val_details": val_details,
                })

            # Early stopping check
            if trainer.early_stopping and trainer.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # Compute research-grade metrics on test set (basic prediction metrics only)
        print("[REAL TRAINING] Computing test metrics...")
        try:
            predictions, targets = trainer.get_predictions(test_loader)
            job.test_metrics = compute_research_metrics(
                predictions=predictions,
                targets=targets,
                are_returns=True,  # We're predicting log returns
            )

            # Sanity-check: flag suspiciously high risk-adjusted metrics
            sharpe = job.test_metrics.get('sharpe_ratio', 0)
            sortino = job.test_metrics.get('sortino_ratio', 0)
            if abs(sharpe) > 3.0 or abs(sortino) > 5.0:
                print(f"[REAL TRAINING] ⚠️  WARNING: Suspicious metrics detected! "
                      f"Sharpe={sharpe:.2f}, Sortino={sortino:.2f}. "
                      f"Values above 3.0/5.0 typically indicate a calculation issue "
                      f"(e.g. normalized returns treated as raw returns).")

            print(f"[REAL TRAINING] Test metrics: RMSE={job.test_metrics.get('rmse', 'N/A'):.4f}, "
                  f"Dir Acc={job.test_metrics.get('directional_accuracy', 'N/A'):.2f}%, "
                  f"Sharpe={job.test_metrics.get('sharpe_ratio', 'N/A'):.2f}")

            # Run full pipeline evaluation (strategy + plots)
            try:
                pipeline_result = run_pipeline_evaluation(
                    predictions=predictions,
                    actual_returns=targets,
                    model_name=job.request.model_type,
                    strategy="sign",
                    transaction_cost=0.001,
                    generate_plots=True,
                    output_dir=f"results/evaluation/{job.request.model_type}",
                )
                # Merge pipeline metrics into test_metrics for API consumption
                if pipeline_result and 'metrics' in pipeline_result:
                    job.test_metrics['pipeline_sharpe'] = pipeline_result['metrics'].get('sharpe_ratio', 0)
                    job.test_metrics['pipeline_sortino'] = pipeline_result['metrics'].get('sortino_ratio', 0)
                    job.test_metrics['trading_stats'] = pipeline_result.get('trading_stats', {})
                    job.test_metrics['plot_paths'] = pipeline_result.get('plot_paths', [])
                print(f"[REAL TRAINING] Pipeline evaluation complete: "
                      f"Sharpe={pipeline_result['metrics'].get('sharpe_ratio', 0):.3f}, "
                      f"plots={len(pipeline_result.get('plot_paths', []))}")
            except Exception as e:
                print(f"[REAL TRAINING] Warning: Pipeline evaluation failed: {e}")

        except Exception as e:
            print(f"[REAL TRAINING] Warning: Failed to compute test metrics: {e}")
            job.test_metrics = {}
        # Defer comprehensive financial metrics to on-demand endpoints (not during training)
        job.final_metrics = None

        # Save results JSON file (same format as terminal training)
        try:
            self._save_model_results(
                model_key=job.request.model_type,
                test_metrics=job.test_metrics,
                training_history=[{
                    'train_loss': tl,
                    'val_loss': vl,
                    'data_loss': dl,
                    'physics_loss': pl,
                    'learning_rate': lr,
                } for tl, vl, dl, pl, lr in zip(
                    job.train_losses,
                    job.val_losses,
                    job.train_data_losses if hasattr(job, 'train_data_losses') else [None] * len(job.train_losses),
                    job.train_physics_losses if hasattr(job, 'train_physics_losses') else [None] * len(job.train_losses),
                    job.learning_rates if hasattr(job, 'learning_rates') and job.learning_rates else [0.001] * len(job.train_losses)
                )],
                trainer_history=trainer.history if trainer else {},
                physics_params=job.physics_params_history[-1] if hasattr(job, 'physics_params_history') and job.physics_params_history else None,
            )
        except Exception as e:
            print(f"[REAL TRAINING] Warning: Failed to save results: {e}")

    def stop_training(self, job_id: str) -> bool:
        """Stop a training job."""
        with self._jobs_lock:
            if job_id not in self._jobs:
                return False
            job = self._jobs[job_id]

        job.stop()
        return True

    def get_job_status(self, job_id: str) -> Optional[TrainingJobInfo]:
        """Get status of a training job."""
        with self._jobs_lock:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
        return job.info

    def get_job_history(self, job_id: str) -> Optional[Dict[str, List[float]]]:
        """Get training history for a job."""
        with self._jobs_lock:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
        return job.get_history()

    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List training jobs."""
        with self._jobs_lock:
            jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by start time
        jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)

        total = len(jobs)
        start = (page - 1) * page_size
        end = start + page_size
        jobs = jobs[start:end]

        return {
            "runs": [j.info for j in jobs],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_epoch_metrics(self, job_id: str) -> List[EpochMetrics]:
        """Get detailed metrics for each epoch."""
        with self._jobs_lock:
            if job_id not in self._jobs:
                return []
            job = self._jobs[job_id]

        metrics = []

        for i in range(len(job.train_losses)):
            physics_params = None
            if i < len(job.physics_params_history):
                physics_params = job.physics_params_history[i]

            metrics.append(EpochMetrics(
                epoch=i + 1,
                train_loss=job.train_losses[i] if i < len(job.train_losses) else 0,
                val_loss=job.val_losses[i] if i < len(job.val_losses) else 0,
                data_loss=job.train_data_losses[i] if hasattr(job, 'train_data_losses') and i < len(job.train_data_losses) else None,
                physics_loss=job.train_physics_losses[i] if hasattr(job, 'train_physics_losses') and i < len(job.train_physics_losses) else None,
                learning_rate=job.learning_rates[i] if hasattr(job, 'learning_rates') and i < len(job.learning_rates) else job.request.learning_rate,
                theta=physics_params.get("theta") if physics_params else None,
                gamma=physics_params.get("gamma") if physics_params else None,
                temperature=physics_params.get("temperature") if physics_params else None,
                epoch_time_seconds=0.5,  # Placeholder
            ))

        return metrics

    # ============== Batch Training Methods ==============

    def start_batch_training(
        self,
        request: BatchTrainingRequest,
        models: List[ModelTrainingConfig],
        callback: Optional[Callable] = None,
    ) -> str:
        """Start batch training of multiple models."""
        with self._batch_jobs_lock:
            # Prevent overlapping batch runs that hammer the server/websocket layer
            active = [
                job_id for job_id, job in self._batch_jobs.items()
                if job.status == TrainingStatus.RUNNING and not job._stop_flag
            ]
            if active:
                raise RuntimeError(f"A batch training job is already running: {', '.join(active)}")

            batch_id = f"batch_{str(uuid.uuid4())[:8]}"

            batch_job = BatchTrainingJob(
                batch_id=batch_id,
                request=request,
                models=models,
                callback=callback,
            )

            # Evict oldest batch jobs if at capacity
            while len(self._batch_jobs) >= MAX_BATCH_JOBS_IN_MEMORY:
                oldest_id, oldest_job = next(iter(self._batch_jobs.items()))
                # Only evict completed/failed/stopped jobs
                if oldest_job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED):
                    self._batch_jobs.pop(oldest_id)
                else:
                    break  # Don't evict running jobs

            self._batch_jobs[batch_id] = batch_job

        # Start batch training in background thread
        batch_job._thread = threading.Thread(
            target=self._run_batch_training,
            args=(batch_job,),
            daemon=True,
        )
        batch_job._thread.start()

        return batch_id

    def _run_batch_training(self, batch_job: "BatchTrainingJob"):
        """Run batch training in background thread."""
        batch_job.status = TrainingStatus.RUNNING
        batch_job.started_at = datetime.now()

        try:
            for i, model_config in enumerate(batch_job.models):
                if batch_job._stop_flag:
                    break

                model_key = model_config.model_key
                model_info = AVAILABLE_MODELS.get(model_key, {})

                # Update current model being trained
                batch_job.current_model = model_key
                batch_job.model_statuses[model_key].status = TrainingStatus.RUNNING
                batch_job.model_statuses[model_key].started_at = datetime.now()

                # Create training request for this model
                epochs = model_config.epochs or batch_job.request.epochs
                learning_rate = model_config.learning_rate or batch_job.request.learning_rate
                batch_size = model_config.batch_size or batch_job.request.batch_size
                hidden_dim = model_config.hidden_dim or batch_job.request.hidden_dim
                num_layers = model_config.num_layers or batch_job.request.num_layers
                dropout = model_config.dropout or batch_job.request.dropout

                # Run training for this model
                self._train_single_model_in_batch(
                    batch_job=batch_job,
                    model_key=model_key,
                    model_info=model_info,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )

                # Update completed count
                if batch_job.model_statuses[model_key].status == TrainingStatus.COMPLETED:
                    batch_job.completed_models += 1
                elif batch_job.model_statuses[model_key].status == TrainingStatus.FAILED:
                    batch_job.failed_models += 1

            # Set overall status
            if batch_job._stop_flag:
                batch_job.status = TrainingStatus.STOPPED
            elif batch_job.failed_models == len(batch_job.models):
                batch_job.status = TrainingStatus.FAILED
            else:
                batch_job.status = TrainingStatus.COMPLETED

        except Exception as e:
            batch_job.status = TrainingStatus.FAILED
            batch_job.error = str(e)
            print(f"Batch training {batch_job.batch_id} failed: {e}")

        batch_job.completed_at = datetime.now()

    def _train_single_model_in_batch(
        self,
        batch_job: "BatchTrainingJob",
        model_key: str,
        model_info: Dict,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Train a single model within a batch job."""
        model_status = batch_job.model_statuses[model_key]
        model_status.total_epochs = epochs
        model_status.learning_rate = learning_rate

        is_pinn = model_info.get("type") == "pinn"

        try:
            if HAS_SRC:
                self._actual_batch_model_training(
                    batch_job=batch_job,
                    model_key=model_key,
                    model_info=model_info,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    is_pinn=is_pinn,
                )
            else:
                self._simulate_batch_model_training(
                    batch_job=batch_job,
                    model_key=model_key,
                    model_info=model_info,
                    epochs=epochs,
                    is_pinn=is_pinn,
                )

            model_status.status = TrainingStatus.COMPLETED
            model_status.completed_at = datetime.now()
            debug_log(f"Model {model_key} training COMPLETED successfully")

        except Exception as e:
            model_status.status = TrainingStatus.FAILED
            model_status.error_message = str(e)
            print(f"Training {model_key} failed: {e}")
            debug_log(f"Training {model_key} FAILED: {e}", "ERROR")
            debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")

    def _actual_batch_model_training(
        self,
        batch_job: "BatchTrainingJob",
        model_key: str,
        model_info: Dict,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        is_pinn: bool,
    ):
        """Run actual training for a model in batch mode."""
        from torch.utils.data import DataLoader
        from src.utils.config import get_research_config

        model_status = batch_job.model_statuses[model_key]
        request = batch_job.request

        debug_log("=" * 60)
        debug_log(f"STARTING BATCH MODEL TRAINING: {model_key}")
        debug_log("=" * 60)
        debug_log(f"  model_key: {model_key}")
        debug_log(f"  is_pinn: {is_pinn}")
        debug_log(f"  request.ticker: {request.ticker}")
        debug_log(f"  epochs (from request): {epochs}")
        debug_log(f"  batch_size (from request): {batch_size}")
        debug_log(f"  hidden_dim (from request): {hidden_dim}")

        # Get research mode setting (default True for research-grade training)
        research_mode = getattr(request, 'research_mode', True)
        research_cfg = get_research_config() if research_mode else None
        debug_log(f"  research_mode: {research_mode}")

        # ============================================================
        # REPRODUCIBILITY: Set random seeds (dissertation-grade requirement)
        # ============================================================
        if research_mode and research_cfg:
            seed = research_cfg.random_seed
            set_seed(seed)
            debug_log(f"  [REPRODUCIBILITY] Random seed set to {seed}")
            debug_log(f"  [REPRODUCIBILITY] cuDNN deterministic=True, benchmark=False")

        # Log system info for first model only (avoid clutter)
        if model_key == list(batch_job.model_statuses.keys())[0]:
            log_system_info()

        # ============================================================
        # USE RESEARCH CONFIG FOR TRAINING PARAMETERS (matches terminal)
        # ============================================================
        if research_mode and research_cfg:
            actual_batch_size = research_cfg.batch_size
            actual_hidden_dim = research_cfg.hidden_dim
            actual_num_layers = research_cfg.num_layers
            actual_dropout = research_cfg.dropout
            actual_epochs = research_cfg.epochs
            print(f"[REAL BATCH TRAINING] Starting {model_key} with ResearchConfig parameters")
            print(f"[REAL BATCH TRAINING]   batch_size={actual_batch_size}, hidden_dim={actual_hidden_dim}")
            print(f"[REAL BATCH TRAINING]   num_layers={actual_num_layers}, epochs={actual_epochs}, PINN={is_pinn}")
        else:
            actual_batch_size = batch_size
            actual_hidden_dim = hidden_dim
            actual_num_layers = num_layers
            actual_dropout = dropout
            actual_epochs = epochs
            print(f"[REAL BATCH TRAINING] Starting {model_key} with request parameters")
            print(f"[REAL BATCH TRAINING]   epochs={epochs}, PINN={is_pinn}")

        # Initialize model registry
        registry = ModelRegistry(settings.project_root)

        # Determine multi-ticker mode: force multi-ticker in research mode for consistency with terminal
        if research_mode:
            use_multi_ticker = True  # Force multi-ticker in research mode
        else:
            use_multi_ticker = getattr(request, 'use_multi_ticker', False)
        debug_log(f"  use_multi_ticker: {use_multi_ticker}")

        # Prepare research-grade normalized data (matches terminal training)
        debug_log("Calling prepare_normalized_data()...")
        try:
            train_dataset, val_dataset, test_dataset, input_dim, scalers = prepare_normalized_data(
                ticker=request.ticker,
                sequence_length=None,  # Let prepare_normalized_data use research config
                start_date=getattr(request, 'start_date', None),
                end_date=getattr(request, 'end_date', None),
                force_refresh=getattr(request, 'force_refresh', True),
                use_multi_ticker=use_multi_ticker,
                research_mode=research_mode,
            )
            debug_log(f"Data prepared successfully:")
            debug_log(f"  train_dataset: {len(train_dataset)} samples")
            debug_log(f"  val_dataset: {len(val_dataset)} samples")
            debug_log(f"  test_dataset: {len(test_dataset)} samples")
            debug_log(f"  input_dim: {input_dim}")
        except Exception as e:
            debug_log(f"ERROR in prepare_normalized_data: {e}", "ERROR")
            debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
            raise

        # ============================================================
        # OPTIMIZED DataLoaders for faster training
        # ============================================================
        import platform
        # Use 0 workers on macOS (spawn issues), 4 on Linux/Windows
        num_workers = 0 if platform.system() == 'Darwin' else 4
        pin_memory = self._device.type == "cuda"

        debug_log(f"Creating DataLoaders (batch={actual_batch_size}, workers={num_workers}, pin_memory={pin_memory})...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=actual_batch_size,
            collate_fn=collate_fn_with_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Create model using model_key with research config parameters
        debug_log(f"Creating model '{model_key}'...")
        debug_log(f"  input_dim: {input_dim}")
        debug_log(f"  hidden_dim: {actual_hidden_dim}")
        debug_log(f"  num_layers: {actual_num_layers}")
        debug_log(f"  dropout: {actual_dropout}")

        try:
            model = registry.create_model(
                model_type=model_key,
                input_dim=input_dim,
                hidden_dim=actual_hidden_dim,
                num_layers=actual_num_layers,
                dropout=actual_dropout,
            )
        except Exception as e:
            debug_log(f"ERROR creating model: {e}", "ERROR")
            debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
            raise

        if model is None:
            debug_log(f"Model creation returned None for '{model_key}'", "ERROR")
            raise ValueError(f"Failed to create model '{model_key}'")

        print(f"[REAL BATCH TRAINING] Created model: {model.__class__.__name__}")
        print(f"[REAL BATCH TRAINING] Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer with research mode for fair model comparison
        # Research mode disables early stopping so all models train for full epochs
        debug_log(f"Creating Trainer for {model_key}...")
        debug_log(f"  train_loader batches: {len(train_loader)}")
        debug_log(f"  val_loader batches: {len(val_loader)}")
        debug_log(f"  test_loader batches: {len(test_loader)}")
        debug_log(f"  device: {self._device}")
        debug_log(f"  research_mode: {research_mode}")

        # Update model status with batch info for real-time progress
        model_status.total_batches = len(train_loader)
        model_status.current_batch = 0

        # Create batch callback for real-time updates
        def batch_progress_callback(batch_idx: int, total_batches: int, batch_loss: float):
            """Update model status with batch-level progress."""
            model_status.current_batch = batch_idx
            model_status.batch_loss = batch_loss
            # Update progress to include batch-level detail
            epoch_progress = (model_status.current_epoch - 1) / model_status.total_epochs if model_status.total_epochs > 0 else 0
            batch_progress = batch_idx / total_batches if total_batches > 0 else 0
            model_status.progress_percent = (epoch_progress + batch_progress / model_status.total_epochs) * 100
            # Debug: log every 200 batches to confirm callback is working (reduced for performance)
            if batch_idx % 200 == 0:
                debug_log(f"[BATCH CALLBACK] {model_key} batch {batch_idx}/{total_batches}, loss={batch_loss:.4f}, progress={model_status.progress_percent:.1f}%")

        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=self._device,
                config=None,  # Uses default config from get_config()
                research_mode=research_mode,  # Disables early stopping for fair comparison
                batch_callback=batch_progress_callback,  # Real-time batch progress updates
            )
            debug_log(f"Trainer created successfully for {model_key}")
        except Exception as e:
            debug_log(f"ERROR creating Trainer: {e}", "ERROR")
            debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
            raise

        if research_mode:
            print(f"[REAL BATCH TRAINING] Research mode ENABLED - early stopping DISABLED for {model_key}")

        # Update model status with actual epochs
        model_status.total_epochs = actual_epochs

        debug_log(f"Starting training loop for {model_key}: {actual_epochs} epochs")

        # Training loop - use actual_epochs from research config
        for epoch in range(1, actual_epochs + 1):
            if batch_job._stop_flag:
                debug_log(f"Training stopped by user at epoch {epoch}")
                model_status.status = TrainingStatus.STOPPED
                return

            debug_log(f"[{model_key}] Epoch {epoch}/{actual_epochs} - Starting train_epoch...")

            # Run actual training epoch
            try:
                train_loss, train_details = trainer.train_epoch(
                    enable_physics=is_pinn,
                )
                debug_log(f"[{model_key}] Epoch {epoch} - train_loss={train_loss:.6f}")
            except Exception as e:
                debug_log(f"ERROR in train_epoch: {e}", "ERROR")
                debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
                raise

            try:
                val_loss, val_details = trainer.validate_epoch(enable_physics=is_pinn)
                debug_log(f"[{model_key}] Epoch {epoch} - val_loss={val_loss:.6f}")
            except Exception as e:
                debug_log(f"ERROR in validate_epoch: {e}", "ERROR")
                debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
                raise

            # Extract physics losses if available
            data_loss = train_details.get("data_loss") if train_details else None
            physics_loss = train_details.get("physics_loss") if train_details else None

            # Update model status
            model_status.current_epoch = epoch
            model_status.train_loss = train_loss
            model_status.val_loss = val_loss
            model_status.data_loss = data_loss
            model_status.physics_loss = physics_loss

            # Research-grade evaluation metrics (validation set)
            # Use actual values from validate_epoch(), fall back to None if unavailable
            if val_details:
                # Debug: Log validation metrics on first epoch
                if epoch == 1:
                    print(f"[METRICS DEBUG] {model_key} val_details keys: {list(val_details.keys())}")
                    print(f"[METRICS DEBUG] {model_key} val_rmse={val_details.get('val_rmse')}, "
                          f"val_mae={val_details.get('val_mae')}, val_r2={val_details.get('val_r2')}")

                model_status.val_rmse = val_details.get("val_rmse")
                model_status.val_mae = val_details.get("val_mae")
                model_status.val_mape = val_details.get("val_mape")
                model_status.val_r2 = val_details.get("val_r2")
                model_status.val_directional_accuracy = val_details.get("val_directional_accuracy")
            else:
                # Fallback only if val_details is completely missing
                model_status.val_rmse = None
                model_status.val_mae = None
                model_status.val_mape = None
                model_status.val_r2 = None
                model_status.val_directional_accuracy = None

            if val_loss < (model_status.best_val_loss or float("inf")):
                model_status.best_val_loss = val_loss
                # Save best checkpoint
                if request.save_checkpoints:
                    # Use model_key as model_name for checkpoint saving
                    trainer.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                        model_name=model_key,
                    )

            # Update progress
            model_status.progress_percent = (epoch / epochs) * 100

            # Store history
            if model_key not in batch_job.histories:
                batch_job.histories[model_key] = []
            batch_job.histories[model_key].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "data_loss": data_loss,
                "physics_loss": physics_loss,
                "learning_rate": model_status.learning_rate,
                "val_rmse": model_status.val_rmse,
                "val_mae": model_status.val_mae,
                "val_mape": model_status.val_mape,
                "val_r2": model_status.val_r2,
                "val_directional_accuracy": model_status.val_directional_accuracy,
            })

            # Calculate overall progress
            self._update_batch_progress(batch_job)

            # Callback for real-time updates - use actual_epochs not epochs
            self._send_batch_callback(batch_job, model_key, model_info, epoch, actual_epochs,
                                      train_loss, val_loss, data_loss, physics_loss)

            # Log epoch completion
            debug_log(f"[{model_key}] Epoch {epoch}/{actual_epochs} completed - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping check
            if trainer.early_stopping and trainer.early_stopping(val_loss):
                print(f"Early stopping {model_key} at epoch {epoch}")
                break

        # Compute research-grade metrics on test set (prediction metrics only)
        print(f"[REAL BATCH TRAINING] Computing test metrics for {model_key}...")
        try:
            from backend.app.schemas.training import ResearchMetrics
            predictions, targets = trainer.get_predictions(test_loader)
            metrics_dict = compute_research_metrics(
                predictions=predictions,
                targets=targets,
                are_returns=True,  # We're predicting log returns
            )
            model_status.test_metrics = ResearchMetrics(**metrics_dict) if metrics_dict else None
            print(f"[REAL BATCH TRAINING] {model_key} metrics: "
                  f"RMSE={metrics_dict.get('rmse', 'N/A'):.4f}, "
                  f"Dir Acc={metrics_dict.get('directional_accuracy', 'N/A'):.2f}%, "
                  f"Sharpe={metrics_dict.get('sharpe_ratio', 'N/A'):.2f}")
        except Exception as e:
            print(f"[REAL BATCH TRAINING] Warning: Failed to compute test metrics for {model_key}: {e}")
            model_status.test_metrics = None
            metrics_dict = {}
        # Defer financial metrics to on-demand endpoints
        model_status.evaluation_metrics = None

        # Save results JSON file (same format as terminal training)
        try:
            self._save_model_results(
                model_key=model_key,
                test_metrics=metrics_dict,
                training_history=batch_job.histories.get(model_key, []),
                trainer_history=trainer.history if trainer else {},
                physics_params=model.get_learned_physics_params() if hasattr(model, "get_learned_physics_params") else None,
            )
        except Exception as e:
            print(f"[REAL BATCH TRAINING] Warning: Failed to save results for {model_key}: {e}")

    def _simulate_batch_model_training(
        self,
        batch_job: "BatchTrainingJob",
        model_key: str,
        model_info: Dict,
        epochs: int,
        is_pinn: bool,
    ):
        """Simulate training for testing when src/ is not available."""
        import random

        model_status = batch_job.model_statuses[model_key]

        for epoch in range(1, epochs + 1):
            if batch_job._stop_flag:
                model_status.status = TrainingStatus.STOPPED
                return

            # Simulate training time
            time.sleep(0.1)

            # Generate realistic losses
            base_train = 0.5 * (0.98 ** epoch) + 0.05
            base_val = 0.55 * (0.97 ** epoch) + 0.06

            train_loss = base_train + random.uniform(-0.02, 0.02)
            val_loss = base_val + random.uniform(-0.03, 0.03)

            # Physics loss decomposition for PINN models
            data_loss = None
            physics_loss = None
            if is_pinn:
                physics_weight = 0.15 + 0.05 * (epoch / epochs)
                data_loss = train_loss * (1 - physics_weight)
                physics_loss = train_loss * physics_weight

            # Update model status
            model_status.current_epoch = epoch
            model_status.train_loss = train_loss
            model_status.val_loss = val_loss
            model_status.data_loss = data_loss
            model_status.physics_loss = physics_loss

            if val_loss < (model_status.best_val_loss or float("inf")):
                model_status.best_val_loss = val_loss

            # Learning rate decay
            if epoch % 20 == 0:
                model_status.learning_rate *= 0.5

            # Update progress
            model_status.progress_percent = (epoch / epochs) * 100

            # Store history
            if model_key not in batch_job.histories:
                batch_job.histories[model_key] = []
            batch_job.histories[model_key].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "data_loss": data_loss,
                "physics_loss": physics_loss,
                "learning_rate": model_status.learning_rate,
                "val_rmse": model_status.val_rmse,
                "val_mae": model_status.val_mae,
                "val_mape": model_status.val_mape,
                "val_r2": model_status.val_r2,
                "val_directional_accuracy": model_status.val_directional_accuracy,
            })

            # Calculate overall progress
            self._update_batch_progress(batch_job)

            # Callback for real-time updates
            self._send_batch_callback(batch_job, model_key, model_info, epoch, epochs,
                                      train_loss, val_loss, data_loss, physics_loss)

    def _save_model_results(
        self,
        model_key: str,
        test_metrics: Dict,
        training_history: List[Dict],
        trainer_history: Dict,
        physics_params: Optional[Dict] = None,
    ):
        """
        Save model results to JSON file (same format as terminal training).

        This ensures trained models have associated results files that can be
        loaded by the dashboard and evaluation tools.

        Args:
            model_key: Model identifier (e.g., 'lstm', 'pinn_gbm')
            test_metrics: Test set metrics (RMSE, MAE, Sharpe, etc.)
            training_history: Per-epoch training history from batch job
            trainer_history: Trainer's internal history dict
        """
        import json
        import numpy as np
        from pathlib import Path

        def convert_to_python_types(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None or isinstance(obj, (int, float, str, bool)):
                return obj
            else:
                return str(obj)

        # Get project root from config
        from src.utils.config import get_config
        config = get_config()
        results_dir = config.project_root / 'results'
        results_dir.mkdir(exist_ok=True)

        # Determine the results filename
        # Model keys from AVAILABLE_MODELS already have correct naming (e.g., 'pinn_gbm', 'lstm')
        results_filename = f'{model_key}_results.json'

        results_path = results_dir / results_filename

        # Build combined history
        combined_history = {
            'train_loss': [h.get('train_loss') for h in training_history],
            'val_loss': [h.get('val_loss') for h in training_history],
            'data_loss': [h.get('data_loss') for h in training_history],
            'physics_loss': [h.get('physics_loss') for h in training_history],
            'learning_rates': [h.get('learning_rate') for h in training_history],
            'epochs': list(range(1, len(training_history) + 1)),
        }

        # Merge with trainer history if available
        if trainer_history:
            for key in ['best_epoch', 'best_val_loss', 'total_epochs_trained', 'research_mode']:
                if key in trainer_history:
                    combined_history[key] = trainer_history[key]

        # Build results object
        results = {
            'model': model_key,
            'test_metrics': convert_to_python_types(test_metrics),
            'history': convert_to_python_types(combined_history),
            'physics_params': convert_to_python_types(physics_params) if physics_params else {},
            'training_completed': True,
        }

        # Save to file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[REAL BATCH TRAINING] Results saved to {results_path}")
        debug_log(f"Results saved to {results_path}")

    def _update_batch_progress(self, batch_job: "BatchTrainingJob"):
        """Update overall batch progress."""
        completed_epochs = sum(
            s.current_epoch for s in batch_job.model_statuses.values()
        )
        total_epochs = sum(
            s.total_epochs for s in batch_job.model_statuses.values()
        )
        batch_job.overall_progress = (completed_epochs / total_epochs) * 100 if total_epochs > 0 else 0

    def _send_batch_callback(
        self,
        batch_job: "BatchTrainingJob",
        model_key: str,
        model_info: Dict,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        data_loss: Optional[float],
        physics_loss: Optional[float],
    ):
        """Send callback for batch training updates."""
        model_status = batch_job.model_statuses[model_key]
        if batch_job.callback:
            batch_job.callback({
                "batch_id": batch_job.batch_id,
                "model_key": model_key,
                "model_name": model_info.get("name", model_key),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": model_status.best_val_loss,
                "val_rmse": model_status.val_rmse,
                "val_mae": model_status.val_mae,
                "val_mape": model_status.val_mape,
                "val_r2": model_status.val_r2,
                "val_directional_accuracy": model_status.val_directional_accuracy,
                "data_loss": data_loss,
                "physics_loss": physics_loss,
                "learning_rate": model_status.learning_rate,
                "overall_progress": batch_job.overall_progress,
                "completed_models": batch_job.completed_models,
                "total_models": len(batch_job.models),
            })

    def stop_batch_training(self, batch_id: str) -> bool:
        """Stop a batch training job."""
        with self._batch_jobs_lock:
            if batch_id not in self._batch_jobs:
                return False
            batch_job = self._batch_jobs[batch_id]

        batch_job._stop_flag = True
        return True

    def get_batch_status(self, batch_id: str) -> Optional[BatchTrainingJobInfo]:
        """Get status of a batch training job."""
        with self._batch_jobs_lock:
            if batch_id not in self._batch_jobs:
                return None
            batch_job = self._batch_jobs[batch_id]
        return batch_job.info

    def get_batch_history(self, batch_id: str) -> Dict[str, List[Dict[str, float]]]:
        """Get training history for all models in a batch."""
        with self._batch_jobs_lock:
            if batch_id not in self._batch_jobs:
                return {}
            batch_job = self._batch_jobs[batch_id]
        return batch_job.histories

    def list_batch_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """List batch training jobs."""
        with self._batch_jobs_lock:
            jobs = list(self._batch_jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by start time
        jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)

        total = len(jobs)
        start = (page - 1) * page_size
        end = start + page_size
        jobs = jobs[start:end]

        return {
            "batches": [j.info for j in jobs],
            "total": total,
            "page": page,
            "page_size": page_size,
        }


class BatchTrainingJob:
    """Represents a batch training job for multiple models."""

    def __init__(
        self,
        batch_id: str,
        request: BatchTrainingRequest,
        models: List[ModelTrainingConfig],
        callback: Optional[Callable] = None,
    ):
        self.batch_id = batch_id
        self.request = request
        self.models = models
        self.callback = callback

        self.status = TrainingStatus.PENDING
        self.current_model: Optional[str] = None
        self.completed_models = 0
        self.failed_models = 0
        self.overall_progress = 0.0

        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None

        # Initialize status for each model
        self.model_statuses: Dict[str, BatchJobStatus] = {}
        for model_config in models:
            model_key = model_config.model_key
            model_info = AVAILABLE_MODELS.get(model_key, {})
            self.model_statuses[model_key] = BatchJobStatus(
                model_key=model_key,
                model_name=model_info.get("name", model_key),
                model_type=model_info.get("type", "unknown"),
                status=TrainingStatus.PENDING,
                total_epochs=model_config.epochs or request.epochs,
            )

        # Training histories per model
        self.histories: Dict[str, List[Dict[str, float]]] = {}

        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    @property
    def info(self) -> BatchTrainingJobInfo:
        """Get batch job information."""
        return BatchTrainingJobInfo(
            batch_id=self.batch_id,
            status=self.status,
            total_models=len(self.models),
            completed_models=self.completed_models,
            failed_models=self.failed_models,
            current_model=self.current_model,
            models=list(self.model_statuses.values()),
            started_at=self.started_at,
            completed_at=self.completed_at,
            overall_progress=self.overall_progress,
            config={
                "epochs": self.request.epochs,
                "batch_size": self.request.batch_size,
                "learning_rate": self.request.learning_rate,
                "hidden_dim": self.request.hidden_dim,
                "num_layers": self.request.num_layers,
                "dropout": self.request.dropout,
            },
        )
