"""
Walk-Forward Validation for Model Robustness Testing

Implements multiple validation strategies:
- Anchored (expanding window)
- Rolling (fixed window)
- Combinatorial Purged Cross-Validation (CPCV)

These methods help assess out-of-sample performance and detect overfitting.

References:
    - Bailey, D. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
      Frontier." SSRN.
    - de Prado, M. L. (2018). "Advances in Financial Machine Learning."
      Wiley.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import warnings

from .financial_metrics import FinancialMetrics
from .window_results import (
    WindowMetrics,
    WindowAggregation,
    WindowResultsDatabase,
    ExtendedWalkForwardResult,
    compute_window_aggregations,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward validation"""
    fold_id: int
    train_start: int  # Index
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int
    train_start_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None


@dataclass
class WalkForwardResult:
    """Results from a single fold"""
    fold_id: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_volatility: float
    test_volatility: float
    overfitting_ratio: float  # train_sharpe / test_sharpe
    metrics: Dict[str, float]


@dataclass
class WalkForwardSummary:
    """Summary of walk-forward validation"""
    n_folds: int
    avg_train_sharpe: float
    avg_test_sharpe: float
    std_test_sharpe: float
    avg_overfitting_ratio: float
    test_sharpe_consistency: float  # Std / Mean
    positive_test_pct: float  # % of folds with positive test Sharpe
    degradation_pct: float  # % of folds where test < train
    folds: List[WalkForwardResult]


class WalkForwardValidator:
    """
    Walk-Forward Validation Framework

    Supports multiple validation strategies:
    - Anchored: Train on all data up to test period (expanding window)
    - Rolling: Fixed training window that rolls forward
    - Combinatorial: All possible train/test combinations with purging
    """

    def __init__(
        self,
        method: str = 'anchored',
        n_folds: int = 5,
        train_ratio: float = 0.8,
        min_train_size: int = 252,  # 1 year minimum
        min_test_size: int = 63,    # ~3 months minimum
        embargo_size: int = 5,       # Gap between train and test
    ):
        """
        Initialize walk-forward validator.

        Args:
            method: 'anchored', 'rolling', or 'combinatorial'
            n_folds: Number of folds
            train_ratio: Ratio of training data (for rolling)
            min_train_size: Minimum training samples
            min_test_size: Minimum test samples
            embargo_size: Gap between train and test (prevents leakage)
        """
        self.method = method
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        self.embargo_size = embargo_size

    def generate_folds(
        self,
        n_samples: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[WalkForwardFold]:
        """
        Generate train/test folds.

        Args:
            n_samples: Total number of samples
            timestamps: Optional timestamps for date tracking

        Returns:
            List of WalkForwardFold objects
        """
        if self.method == 'anchored':
            return self._generate_anchored_folds(n_samples, timestamps)
        elif self.method == 'rolling':
            return self._generate_rolling_folds(n_samples, timestamps)
        elif self.method == 'combinatorial':
            return self._generate_combinatorial_folds(n_samples, timestamps)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_anchored_folds(
        self,
        n_samples: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[WalkForwardFold]:
        """
        Generate anchored (expanding window) folds.

        Train: [0, train_end]
        Test: [train_end + embargo, test_end]
        Each fold expands the training set.
        """
        folds = []

        # Calculate test size per fold
        available = n_samples - self.min_train_size - self.embargo_size
        if available < self.min_test_size * self.n_folds:
            warnings.warn("Not enough data for requested folds")
            actual_folds = max(1, available // self.min_test_size)
        else:
            actual_folds = self.n_folds

        test_size = available // actual_folds

        for i in range(actual_folds):
            train_end = self.min_train_size + i * test_size
            test_start = train_end + self.embargo_size
            test_end = min(test_start + test_size, n_samples)

            folds.append(WalkForwardFold(
                fold_id=i + 1,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_end,
                test_size=test_end - test_start,
                train_start_date=timestamps[0] if timestamps is not None else None,
                test_start_date=timestamps[test_start] if timestamps is not None else None,
            ))

        return folds

    def _generate_rolling_folds(
        self,
        n_samples: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[WalkForwardFold]:
        """
        Generate rolling (fixed window) folds.

        Both train and test windows roll forward.
        """
        folds = []

        # Fixed train size
        train_size = int(n_samples * self.train_ratio / (1 + 1/self.n_folds))
        train_size = max(train_size, self.min_train_size)

        # Test size per fold
        remaining = n_samples - train_size - self.embargo_size
        test_size = remaining // self.n_folds
        test_size = max(test_size, self.min_test_size)

        step = test_size

        for i in range(self.n_folds):
            train_start = i * step
            train_end = train_start + train_size
            test_start = train_end + self.embargo_size
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples or train_end > n_samples:
                break

            folds.append(WalkForwardFold(
                fold_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start,
                train_start_date=timestamps[train_start] if timestamps is not None else None,
                test_start_date=timestamps[test_start] if timestamps is not None else None,
            ))

        return folds

    def _generate_combinatorial_folds(
        self,
        n_samples: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[WalkForwardFold]:
        """
        Generate combinatorial purged cross-validation folds.

        Simplified version: generate multiple train/test splits
        with purging to prevent leakage.
        """
        folds = []

        # Divide into groups
        n_groups = self.n_folds + 1
        group_size = n_samples // n_groups

        for test_group in range(n_groups):
            test_start = test_group * group_size
            test_end = min(test_start + group_size, n_samples)

            # Train on all other groups (with embargo)
            train_indices = []
            for i in range(n_groups):
                if i == test_group:
                    continue
                group_start = i * group_size

                # Apply embargo
                if i == test_group - 1:
                    # Group before test
                    group_end = group_start + group_size - self.embargo_size
                elif i == test_group + 1:
                    # Group after test
                    group_start += self.embargo_size
                    group_end = group_start + group_size

                else:
                    group_end = group_start + group_size

                if group_start < group_end:
                    train_indices.extend(range(group_start, min(group_end, n_samples)))

            if len(train_indices) < self.min_train_size:
                continue

            folds.append(WalkForwardFold(
                fold_id=test_group + 1,
                train_start=min(train_indices),
                train_end=max(train_indices) + 1,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_indices),
                test_size=test_end - test_start,
                train_start_date=timestamps[min(train_indices)] if timestamps is not None else None,
                test_start_date=timestamps[test_start] if timestamps is not None else None,
            ))

        return folds

    def validate(
        self,
        returns: Union[np.ndarray, pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
        compute_additional_metrics: bool = True,
    ) -> WalkForwardSummary:
        """
        Run walk-forward validation.

        Args:
            returns: Return series
            timestamps: Optional timestamps
            compute_additional_metrics: Include extra metrics per fold

        Returns:
            WalkForwardSummary with all results
        """
        if isinstance(returns, pd.Series):
            if timestamps is None:
                timestamps = returns.index
            returns = returns.values

        returns = returns[~np.isnan(returns)]
        n_samples = len(returns)

        # Generate folds
        folds = self.generate_folds(n_samples, timestamps)

        if len(folds) == 0:
            logger.warning("No valid folds generated")
            return WalkForwardSummary(
                n_folds=0,
                avg_train_sharpe=0,
                avg_test_sharpe=0,
                std_test_sharpe=0,
                avg_overfitting_ratio=0,
                test_sharpe_consistency=0,
                positive_test_pct=0,
                degradation_pct=0,
                folds=[],
            )

        # Run validation on each fold
        results = []

        for fold in folds:
            train_returns = returns[fold.train_start:fold.train_end]
            test_returns = returns[fold.test_start:fold.test_end]

            # Calculate metrics
            train_sharpe = FinancialMetrics.sharpe_ratio(train_returns)
            test_sharpe = FinancialMetrics.sharpe_ratio(test_returns)

            train_return = FinancialMetrics.total_return(train_returns)
            test_return = FinancialMetrics.total_return(test_returns)

            train_vol = float(np.std(train_returns, ddof=1) * np.sqrt(252))
            test_vol = float(np.std(test_returns, ddof=1) * np.sqrt(252))

            # Overfitting ratio
            if abs(test_sharpe) > 0.01:
                overfitting = train_sharpe / test_sharpe
            else:
                overfitting = train_sharpe if train_sharpe > 0 else 0

            # Additional metrics
            metrics = {}
            if compute_additional_metrics:
                metrics = {
                    'train_sortino': FinancialMetrics.sortino_ratio(train_returns),
                    'test_sortino': FinancialMetrics.sortino_ratio(test_returns),
                    'train_max_dd': FinancialMetrics.max_drawdown(train_returns),
                    'test_max_dd': FinancialMetrics.max_drawdown(test_returns),
                }

            results.append(WalkForwardResult(
                fold_id=fold.fold_id,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_return=train_return,
                test_return=test_return,
                train_volatility=train_vol,
                test_volatility=test_vol,
                overfitting_ratio=overfitting,
                metrics=metrics,
            ))

        # Compute summary statistics
        train_sharpes = [r.train_sharpe for r in results]
        test_sharpes = [r.test_sharpe for r in results]
        overfitting_ratios = [r.overfitting_ratio for r in results]

        avg_test = np.mean(test_sharpes)
        std_test = np.std(test_sharpes, ddof=1) if len(test_sharpes) > 1 else 0

        return WalkForwardSummary(
            n_folds=len(folds),
            avg_train_sharpe=float(np.mean(train_sharpes)),
            avg_test_sharpe=float(avg_test),
            std_test_sharpe=float(std_test),
            avg_overfitting_ratio=float(np.mean(overfitting_ratios)),
            test_sharpe_consistency=float(std_test / abs(avg_test)) if abs(avg_test) > 0.01 else float('inf'),
            positive_test_pct=float(np.mean([s > 0 for s in test_sharpes])),
            degradation_pct=float(np.mean([r.test_sharpe < r.train_sharpe for r in results])),
            folds=results,
        )


def run_walk_forward_analysis(
    returns: Union[np.ndarray, pd.Series],
    timestamps: Optional[pd.DatetimeIndex] = None,
    methods: List[str] = None,
    n_folds: int = 5,
) -> Dict[str, WalkForwardSummary]:
    """
    Run walk-forward analysis with multiple methods.

    Args:
        returns: Return series
        timestamps: Optional timestamps
        methods: List of methods to use (default: all)
        n_folds: Number of folds

    Returns:
        Dictionary mapping method names to summaries
    """
    if methods is None:
        methods = ['anchored', 'rolling']

    results = {}

    for method in methods:
        validator = WalkForwardValidator(
            method=method,
            n_folds=n_folds,
        )
        results[method] = validator.validate(returns, timestamps)

    return results


def walk_forward_summary_to_dataframe(
    summary: WalkForwardSummary
) -> pd.DataFrame:
    """Convert walk-forward results to DataFrame."""
    records = []

    for fold in summary.folds:
        record = {
            'Fold': fold.fold_id,
            'Train Sharpe': f"{fold.train_sharpe:.2f}",
            'Test Sharpe': f"{fold.test_sharpe:.2f}",
            'Train Return': f"{fold.train_return:.1%}",
            'Test Return': f"{fold.test_return:.1%}",
            'Overfitting': f"{fold.overfitting_ratio:.2f}",
        }
        records.append(record)

    # Add summary row
    records.append({
        'Fold': 'Average',
        'Train Sharpe': f"{summary.avg_train_sharpe:.2f}",
        'Test Sharpe': f"{summary.avg_test_sharpe:.2f}",
        'Train Return': '---',
        'Test Return': '---',
        'Overfitting': f"{summary.avg_overfitting_ratio:.2f}",
    })

    return pd.DataFrame(records)



def detect_window_regime(
    returns: np.ndarray,
    vol_thresholds: Tuple[float, float] = (0.15, 0.25)
) -> str:
    """
    Detect regime for a window based on realized volatility.

    Args:
        returns: Window returns
        vol_thresholds: (low_high, high) annualized vol boundaries

    Returns:
        Regime label
    """
    ann_vol = float(np.std(returns) * np.sqrt(252))

    if ann_vol < vol_thresholds[0]:
        return "low_volatility"
    elif ann_vol < vol_thresholds[1]:
        return "medium_volatility"
    else:
        return "high_volatility"


class ExtendedWalkForwardValidator(WalkForwardValidator):
    """
    Extended walk-forward validator with full window-level tracking
    and persistence capabilities.
    """

    def __init__(
        self,
        method: str = 'anchored',
        n_folds: int = 5,
        train_ratio: float = 0.8,
        min_train_size: int = 252,
        min_test_size: int = 63,
        embargo_size: int = 5,
        db_path: Optional[Union[str, Path]] = None,
        track_regimes: bool = True
    ):
        """
        Initialize extended validator.

        Args:
            method: Validation method
            n_folds: Number of folds
            train_ratio: Training data ratio
            min_train_size: Minimum training samples
            min_test_size: Minimum test samples
            embargo_size: Embargo gap size
            db_path: Optional database path for persistence
            track_regimes: Whether to detect and track regimes
        """
        super().__init__(
            method=method,
            n_folds=n_folds,
            train_ratio=train_ratio,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            embargo_size=embargo_size
        )

        self.db = WindowResultsDatabase(db_path) if db_path else None
        self.track_regimes = track_regimes

    def validate_extended(
        self,
        returns: Union[np.ndarray, pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        model_name: str = "unknown",
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ExtendedWalkForwardResult:
        """
        Run extended validation with full window tracking.

        Args:
            returns: Return series
            timestamps: Optional timestamps
            predictions: Optional model predictions
            actuals: Optional actual values
            model_name: Model name for tracking
            experiment_id: Experiment identifier
            config: Configuration dictionary

        Returns:
            ExtendedWalkForwardResult with all details
        """
        if isinstance(returns, pd.Series):
            if timestamps is None:
                timestamps = returns.index
            returns = returns.values

        returns = returns[~np.isnan(returns)]
        n_samples = len(returns)

        # Generate experiment ID if not provided
        if experiment_id is None:
            experiment_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        config = config or {}

        # Get base summary
        summary = self.validate(returns, timestamps)

        # Generate detailed window metrics
        folds = self.generate_folds(n_samples, timestamps)
        window_metrics = []

        for i, fold in enumerate(folds):
            test_returns = returns[fold.test_start:fold.test_end]

            # Calculate comprehensive metrics
            sharpe = FinancialMetrics.sharpe_ratio(test_returns)
            sortino = FinancialMetrics.sortino_ratio(test_returns)
            total_return = FinancialMetrics.total_return(test_returns)
            ann_return = total_return * (252 / len(test_returns)) if len(test_returns) > 0 else 0
            volatility = float(np.std(test_returns) * np.sqrt(252))
            max_dd = FinancialMetrics.max_drawdown(test_returns)
            calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.01 else 0

            # Prediction metrics
            mse, mae, rmse, mape, dir_acc = None, None, None, None, None
            if predictions is not None and actuals is not None:
                pred_window = predictions[fold.test_start:fold.test_end]
                act_window = actuals[fold.test_start:fold.test_end]
                if len(pred_window) > 0:
                    mse = float(np.mean((pred_window - act_window) ** 2))
                    mae = float(np.mean(np.abs(pred_window - act_window)))
                    rmse = float(np.sqrt(mse))
                    mape = float(np.mean(np.abs((pred_window - act_window) / (act_window + 1e-8))))
                    # Directional accuracy
                    pred_dir = np.sign(np.diff(pred_window))
                    act_dir = np.sign(np.diff(act_window))
                    dir_acc = float(np.mean(pred_dir == act_dir)) if len(pred_dir) > 0 else None

            # Regime detection
            regime = None
            if self.track_regimes:
                regime = detect_window_regime(test_returns)

            window_metrics.append(WindowMetrics(
                window_id=i + 1,
                start_date=str(timestamps[fold.test_start]) if timestamps is not None else None,
                end_date=str(timestamps[fold.test_end - 1]) if timestamps is not None else None,
                start_idx=fold.test_start,
                end_idx=fold.test_end,
                n_samples=fold.test_size,
                sharpe=sharpe,
                sortino=sortino,
                total_return=total_return,
                annualized_return=ann_return,
                volatility=volatility,
                max_drawdown=max_dd,
                calmar_ratio=calmar,
                mse=mse,
                mae=mae,
                rmse=rmse,
                mape=mape,
                directional_accuracy=dir_acc,
                regime=regime,
                avg_volatility=volatility,
                model_name=model_name,
                experiment_id=experiment_id
            ))

        # Compute aggregations
        aggregations = compute_window_aggregations(window_metrics)

        # Compute regime breakdown
        regime_breakdown = {}
        if self.track_regimes:
            regime_groups = {}
            for w in window_metrics:
                if w.regime:
                    if w.regime not in regime_groups:
                        regime_groups[w.regime] = []
                    regime_groups[w.regime].append(w)

            for regime, windows in regime_groups.items():
                regime_breakdown[regime] = {
                    'count': len(windows),
                    'avg_sharpe': np.mean([w.sharpe for w in windows]),
                    'avg_return': np.mean([w.total_return for w in windows]),
                    'avg_drawdown': np.mean([w.max_drawdown for w in windows]),
                }

        # Persist if database configured
        if self.db:
            self.db.save_experiment(experiment_id, model_name, config)
            self.db.save_window_results(window_metrics, experiment_id)
            self.db.save_aggregations(aggregations, experiment_id)
            logger.info(f"Saved results to database: {self.db.db_path}")

        return ExtendedWalkForwardResult(
            summary=summary,
            window_metrics=window_metrics,
            aggregations=aggregations,
            regime_breakdown=regime_breakdown,
            experiment_id=experiment_id,
            model_name=model_name,
            config=config
        )


def extended_results_to_dataframe(
    result: ExtendedWalkForwardResult
) -> pd.DataFrame:
    """Convert extended results to detailed DataFrame."""
    records = []

    for w in result.window_metrics:
        records.append({
            'window_id': w.window_id,
            'start_date': w.start_date,
            'end_date': w.end_date,
            'n_samples': w.n_samples,
            'sharpe': w.sharpe,
            'sortino': w.sortino,
            'total_return': w.total_return,
            'volatility': w.volatility,
            'max_drawdown': w.max_drawdown,
            'regime': w.regime,
            'mse': w.mse,
            'mae': w.mae,
            'directional_accuracy': w.directional_accuracy
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Walk-Forward Validation Demo")
    print("=" * 60)

    # Generate synthetic returns
    np.random.seed(42)
    n_days = 1000

    # Returns with slight positive drift
    returns = np.random.randn(n_days) * 0.02 + 0.0003
    dates = pd.date_range(start='2019-01-01', periods=n_days, freq='B')

    print(f"\nGenerated {n_days} days of returns")
    print(f"Overall Sharpe: {FinancialMetrics.sharpe_ratio(returns):.2f}")

    # Run validation
    print("\n" + "-" * 40)
    print("Running walk-forward validation...")

    results = run_walk_forward_analysis(
        returns=returns,
        timestamps=dates,
        methods=['anchored', 'rolling'],
        n_folds=5,
    )

    for method, summary in results.items():
        print(f"\n{method.upper()} Method:")
        print(f"  Avg Train Sharpe: {summary.avg_train_sharpe:.2f}")
        print(f"  Avg Test Sharpe: {summary.avg_test_sharpe:.2f}")
        print(f"  Test Sharpe Std: {summary.std_test_sharpe:.2f}")
        print(f"  Positive Test %: {summary.positive_test_pct:.1%}")
        print(f"  Degradation %: {summary.degradation_pct:.1%}")

    print("\n" + "-" * 40)
    print("Fold Details (Anchored):")
    df = walk_forward_summary_to_dataframe(results['anchored'])
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Demo complete!")
