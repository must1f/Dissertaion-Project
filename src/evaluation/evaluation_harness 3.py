"""
Unified Evaluation Harness

Single interface to train, validate, and test any model with:
- Consistent splits, seeds, and metrics
- Walk-forward validation support
- Regime-stratified evaluation
- Statistical significance testing
- Results persistence

This is the central orchestrator for all model evaluation.
"""

import numpy as np
import pandas as pd
import torch
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .metrics import MetricsCalculator, calculate_metrics, calculate_financial_metrics
from .metrics_registry import MetricsRegistry
from .leaderboard import create_experiment_entry
from .result_logger import ResultLogger
from .statistical_tests import (
    StatisticalTests,
    ModelComparator,
    BootstrapResult,
    DieboldMarianoResult,
    bootstrap_sharpe_ci
)
from .walk_forward_validation import (
    WalkForwardValidator,
    WalkForwardSummary,
    run_walk_forward_analysis,
    ExtendedWalkForwardValidator,
    ExtendedWalkForwardResult,
)
from .split_manager import SplitManager, SplitConfig, SplitStrategy
from .window_results import WindowResultsDatabase
from ..utils.logger import get_logger
from ..utils.reproducibility import set_seed, get_environment_info

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result from a single evaluation run"""
    model_key: str
    model_name: str
    seed: int
    split_id: int
    timestamp: str

    # ML metrics
    ml_metrics: Dict[str, float]

    # Financial metrics
    financial_metrics: Dict[str, float]

    # Predictions
    n_samples: int
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    errors: Optional[np.ndarray] = None

    # Regime analysis
    regime_metrics: Optional[Dict[str, Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'model_key': self.model_key,
            'model_name': self.model_name,
            'seed': self.seed,
            'split_id': self.split_id,
            'timestamp': self.timestamp,
            'ml_metrics': self.ml_metrics,
            'financial_metrics': self.financial_metrics,
            'n_samples': self.n_samples,
            'regime_metrics': self.regime_metrics
        }
        return result

    def _persist_window_results(
        self,
        returns: np.ndarray,
        predictions: Optional[np.ndarray],
        targets: Optional[np.ndarray],
        timestamps: Optional[pd.DatetimeIndex],
        model_name: str,
        experiment_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Run walk-forward window analysis and persist window-level metrics."""
        validator = ExtendedWalkForwardValidator(
            method='anchored' if self.split_config.strategy == SplitStrategy.EXPANDING else 'rolling',
            n_folds=self.split_config.n_folds,
            min_train_size=self.split_config.min_train_size,
            min_test_size=self.split_config.min_test_size,
            embargo_size=self.split_config.embargo_size,
            db_path=self.window_db.db_path if self.window_db else None,
            track_regimes=True
        )

        result: ExtendedWalkForwardResult = validator.validate_extended(
            returns=returns,
            timestamps=timestamps,
            predictions=predictions,
            actuals=targets,
            model_name=model_name,
            experiment_id=experiment_id,
            config=config,
        )

        if self.window_db:
            self.window_db.save_experiment(experiment_id, model_name, config)
            self.window_db.save_window_results(result.window_metrics, experiment_id)
            self.window_db.save_aggregations(result.aggregations, experiment_id)


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs/seeds"""
    model_key: str
    model_name: str
    n_runs: int
    seeds: List[int]

    # Aggregated metrics (mean, std, CI)
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    metrics_ci_lower: Dict[str, float]
    metrics_ci_upper: Dict[str, float]

    # Individual results
    individual_results: List[EvaluationResult]

    # Statistical tests
    bootstrap_results: Optional[Dict[str, BootstrapResult]] = None


@dataclass
class ComparisonResult:
    """Result of model comparison"""
    models_compared: List[str]
    baseline_model: str
    metric_name: str

    # Pairwise comparisons
    pairwise_tests: Dict[str, Dict[str, Any]]

    # Rankings
    ranking: List[Tuple[str, float]]

    # Best model
    best_model: str
    best_value: float



class MetricsRegistry:
    """
    Centralized registry for all evaluation metrics.

    Ensures consistent metric computation across all models.
    """

    def __init__(
        self,
        transaction_cost: float = 0.003,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.calculator = MetricsCalculator()

    def compute_ml_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard ML metrics"""
        predictions = predictions.flatten()
        targets = targets.flatten()

        return {
            'mse': float(np.mean((predictions - targets) ** 2)),
            'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
            'mae': float(np.mean(np.abs(predictions - targets))),
            'mape': float(np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100),
            'r2': float(self.calculator.r2(targets, predictions)),
            'directional_accuracy': float(self.calculator.directional_accuracy(targets, predictions))
        }

    def compute_financial_metrics(
        self,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Compute financial performance metrics"""
        returns = np.clip(returns, -0.99, 1.0)

        return {
            'sharpe_ratio': float(self.calculator.sharpe_ratio(
                returns, self.risk_free_rate, self.periods_per_year
            )),
            'sortino_ratio': float(self.calculator.sortino_ratio(
                returns, self.risk_free_rate, self.periods_per_year
            )),
            'max_drawdown': float(self.calculator.max_drawdown((1 + returns).cumprod())),
            'calmar_ratio': float(self.calculator.calmar_ratio(returns, self.periods_per_year)),
            'win_rate': float(self.calculator.win_rate(returns)),
            'total_return': float((np.prod(1 + returns) - 1) * 100),
            'volatility': float(np.std(returns) * np.sqrt(self.periods_per_year) * 100)
        }

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute all metrics"""
        ml_metrics = self.compute_ml_metrics(predictions, targets)

        if returns is not None:
            financial_metrics = self.compute_financial_metrics(returns)
            ml_metrics.update(financial_metrics)

        return ml_metrics


class ResultsDatabase:
    """
    SQLite database for persisting evaluation results.

    Enables:
    - Result tracking and versioning
    - Cross-experiment comparison
    - Leaderboard generation
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                description TEXT
            )
        ''')

        # Model runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        ''')

        # Leaderboard table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                rank INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def save_result(self, result: EvaluationResult, experiment_id: Optional[int] = None):
        """Save a single evaluation result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metrics_json = json.dumps({
            **result.ml_metrics,
            **result.financial_metrics
        })

        cursor.execute('''
            INSERT INTO model_runs (experiment_id, model_key, model_name, seed, split_id, timestamp, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (experiment_id, result.model_key, result.model_name, result.seed,
              result.split_id, result.timestamp, metrics_json))

        conn.commit()
        conn.close()

    def get_results(
        self,
        model_key: Optional[str] = None,
        experiment_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve results from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM model_runs WHERE 1=1"
        params = []

        if model_key:
            query += " AND model_key = ?"
            params.append(model_key)

        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'experiment_id': row[1],
                'model_key': row[2],
                'model_name': row[3],
                'seed': row[4],
                'split_id': row[5],
                'timestamp': row[6],
                'metrics': json.loads(row[7])
            })

        return results

    def update_leaderboard(self, metric_name: str):
        """Update leaderboard for a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get best result per model
        cursor.execute(f'''
            SELECT model_key, AVG(json_extract(metrics_json, '$.{metric_name}')) as avg_metric
            FROM model_runs
            GROUP BY model_key
            ORDER BY avg_metric DESC
        ''')

        rows = cursor.fetchall()

        # Clear old leaderboard entries for this metric
        cursor.execute("DELETE FROM leaderboard WHERE metric_name = ?", (metric_name,))

        # Insert new entries
        for rank, (model_key, metric_value) in enumerate(rows, 1):
            cursor.execute('''
                INSERT INTO leaderboard (model_key, metric_name, metric_value, rank, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_key, metric_name, metric_value, rank, datetime.now().isoformat()))

        conn.commit()
        conn.close()


class EvaluationHarness:
    """
    Unified evaluation harness for all models.

    Usage:
        harness = EvaluationHarness(output_dir='outputs/eval')

        # Evaluate single model
        result = harness.evaluate(
            model=trained_model,
            model_key='lstm',
            X_test=X_test,
            y_test=y_test
        )

        # Multi-seed evaluation
        results = harness.evaluate_multi_seed(
            train_func=train_lstm,
            model_key='lstm',
            data=data,
            n_seeds=5
        )

        # Compare models
        comparison = harness.compare_models(
            model_results={'lstm': lstm_results, 'pinn': pinn_results},
            baseline='lstm'
        )
    """

    def __init__(
        self,
        output_dir: Path = Path('outputs/evaluation'),
        split_config: Optional[SplitConfig] = None,
        transaction_cost: float = 0.003,
        risk_free_rate: float = 0.02,
        save_predictions: bool = True,
        use_database: bool = True,
        log_results: bool = True,
        window_db_path: Optional[Path] = None,
        result_logger_path: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.split_config = split_config or SplitConfig()
        # MetricsRegistry now centralises metric implementations
        self.metrics_registry = MetricsRegistry()
        self.statistical_tests = StatisticalTests()
        self.model_comparator = ModelComparator()
        self.save_predictions = save_predictions

        if use_database:
            self.database = ResultsDatabase(self.output_dir / 'results.db')
        else:
            self.database = None

        # Use a separate DB file for the lightweight result logger to avoid schema collisions
        logger_db_path = result_logger_path or (self.output_dir / 'exp_results.db')
        self.result_logger = ResultLogger(
            db_path=logger_db_path,
            artifacts_dir=self.output_dir / 'artifacts'
        ) if log_results else None

        self.window_db = WindowResultsDatabase(window_db_path or (self.output_dir / 'walk_forward_results.db'))

        # Track all results
        self.results: Dict[str, List[EvaluationResult]] = {}

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        model_key: str,
        model_name: Optional[str] = None,
        seed: int = 42,
        split_id: int = 0,
        returns: Optional[np.ndarray] = None,
        compute_bootstrap: bool = True,
        volatility_for_regime: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        persist_windows: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a single model's predictions.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            model_key: Model identifier
            model_name: Display name for model
            seed: Random seed used
            split_id: Split/fold identifier
            returns: Strategy returns (optional, computed from predictions if not provided)
            compute_bootstrap: Whether to compute bootstrap CIs
            volatility_for_regime: Volatility values for regime-stratified evaluation

        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating {model_key} (seed={seed}, split={split_id})")

        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()

        # Validate
        if len(predictions) != len(targets):
            raise ValueError(f"Prediction/target length mismatch: {len(predictions)} vs {len(targets)}")

        # Remove NaN
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Compute ML metrics
        ml_metrics = self.metrics_registry.compute_ml_metrics(predictions, targets)

        # Compute returns if not provided
        if returns is None:
            # Simple directional strategy returns
            pred_direction = np.sign(np.diff(predictions))
            actual_returns = np.diff(targets) / targets[:-1]
            returns = pred_direction * actual_returns

        # Compute financial metrics
        financial_metrics = self.metrics_registry.compute_financial_metrics(returns)

        # Regime-stratified metrics
        regime_metrics = None
        if volatility_for_regime is not None:
            regime_metrics = self._compute_regime_metrics(
                predictions, targets, returns, volatility_for_regime
            )

        # Create result
        result = EvaluationResult(
            model_key=model_key,
            model_name=model_name or model_key,
            seed=seed,
            split_id=split_id,
            timestamp=datetime.now().isoformat(),
            ml_metrics=ml_metrics,
            financial_metrics=financial_metrics,
            n_samples=len(predictions),
            predictions=predictions if self.save_predictions else None,
            targets=targets if self.save_predictions else None,
            errors=predictions - targets if self.save_predictions else None,
            regime_metrics=regime_metrics
        )

        # Store result
        if model_key not in self.results:
            self.results[model_key] = []
        self.results[model_key].append(result)

        # Save to database
        if self.database:
            self.database.save_result(result)

        # Optional: persist walk-forward window metrics if returns are available
        if persist_windows and returns is not None and len(returns) > 0:
            split_cfg = asdict(self.split_config)
            if isinstance(self.split_config.strategy, SplitStrategy):
                split_cfg["strategy"] = self.split_config.strategy.value

            self._persist_window_results(
                returns=np.asarray(returns),
                predictions=predictions if self.save_predictions else None,
                targets=targets if self.save_predictions else None,
                timestamps=timestamps,
                model_name=model_name or model_key,
                experiment_id=result.model_key,
                config={"seed": seed, "split_id": split_id, "split_config": split_cfg},
            )

        # Persist to unified logger (JSON + SQLite schema used by UI)
        if self.result_logger:
            metrics_combined = {**ml_metrics, **financial_metrics}
            experiment_id = f"{model_key}_seed{seed}_split{split_id}_{result.timestamp.replace(':', '').replace('-', '').replace('T', '_')}"
            split_cfg = asdict(self.split_config)
            if isinstance(self.split_config.strategy, SplitStrategy):
                split_cfg["strategy"] = self.split_config.strategy.value
            entry = create_experiment_entry(
                experiment_id=experiment_id,
                model_name=model_name or model_key,
                model_type=model_key,
                metrics=metrics_combined,
                config={
                    "seed": seed,
                    "split_id": split_id,
                    "split_config": split_cfg,
                    "timestamps": bool(timestamps is not None),
                },
                training_info={"seed": seed},
            )
            try:
                stat_tests_payload = None
                if compute_bootstrap:
                    stat_tests_payload = {
                        "bootstrap": True,
                        "ml_metrics": ml_metrics,
                        "financial_metrics": financial_metrics,
                    }
                self.result_logger.log_experiment(entry, config=entry.__dict__, predictions=None, stat_tests=stat_tests_payload)
            except Exception as exc:  # pragma: no cover - logging is best effort
                logger.warning(f"Result logging failed: {exc}")

        # Log key metrics
        logger.info(f"  MSE: {ml_metrics['mse']:.6f}")
        logger.info(f"  MAE: {ml_metrics['mae']:.6f}")
        logger.info(f"  Directional Accuracy: {ml_metrics['directional_accuracy']:.2%}")
        logger.info(f"  Sharpe Ratio: {financial_metrics['sharpe_ratio']:.3f}")

        return result

    def _compute_regime_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        volatility: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by volatility regime"""
        # Define regime thresholds (annualized volatility)
        low_vol_threshold = 0.15
        high_vol_threshold = 0.25

        # Compute annualized volatility
        vol_annualized = volatility * np.sqrt(252)

        # Create regime masks
        low_vol_mask = vol_annualized < low_vol_threshold
        med_vol_mask = (vol_annualized >= low_vol_threshold) & (vol_annualized < high_vol_threshold)
        high_vol_mask = vol_annualized >= high_vol_threshold

        regime_metrics = {}

        for regime_name, mask in [
            ('low_vol', low_vol_mask),
            ('medium_vol', med_vol_mask),
            ('high_vol', high_vol_mask)
        ]:
            if np.sum(mask) > 10:  # Minimum samples
                regime_preds = predictions[mask[:len(predictions)]]
                regime_targets = targets[mask[:len(targets)]]
                regime_returns = returns[mask[:len(returns)]] if len(returns) <= len(mask) else returns

                regime_metrics[regime_name] = {
                    'n_samples': int(np.sum(mask)),
                    'mse': float(np.mean((regime_preds - regime_targets) ** 2)),
                    'mae': float(np.mean(np.abs(regime_preds - regime_targets))),
                    'sharpe': float(self.metrics_registry.calculator.sharpe_ratio(regime_returns))
                }

        return regime_metrics

    def _persist_window_results(
        self,
        returns: np.ndarray,
        predictions: Optional[np.ndarray],
        targets: Optional[np.ndarray],
        timestamps: Optional[pd.DatetimeIndex],
        model_name: str,
        experiment_id: str,
        config: Dict[str, Any],
    ) -> None:
        """Run walk-forward window analysis and persist window-level metrics."""
        validator = ExtendedWalkForwardValidator(
            method='anchored' if self.split_config.strategy == SplitStrategy.EXPANDING else 'rolling',
            n_folds=self.split_config.n_folds,
            min_train_size=self.split_config.min_train_size,
            min_test_size=self.split_config.min_test_size,
            embargo_size=self.split_config.embargo_size,
            db_path=self.window_db.db_path if self.window_db else None,
            track_regimes=True,
        )

        result: ExtendedWalkForwardResult = validator.validate_extended(
            returns=returns,
            timestamps=timestamps,
            predictions=predictions,
            actuals=targets,
            model_name=model_name,
            experiment_id=experiment_id,
            config=config,
        )

        if self.window_db:
            self.window_db.save_experiment(experiment_id, model_name, config)
            self.window_db.save_window_results(result.window_metrics, experiment_id)
            self.window_db.save_aggregations(result.aggregations, experiment_id)

    def evaluate_walk_forward(
        self,
        train_func: Callable,
        data: Dict[str, np.ndarray],
        model_key: str,
        model_name: Optional[str] = None,
        seed: int = 42
    ) -> List[EvaluationResult]:
        """
        Evaluate model using walk-forward validation.

        Args:
            train_func: Function that trains model and returns predictions
                        Signature: train_func(X_train, y_train, X_test) -> predictions
            data: Dict with 'X' and 'y' arrays
            model_key: Model identifier
            model_name: Display name
            seed: Random seed

        Returns:
            List of EvaluationResult for each fold
        """
        logger.info(f"Walk-forward evaluation for {model_key}")

        set_seed(seed)

        X = data['X']
        y = data['y']
        n_samples = len(X)

        # Create splits
        split_manager = SplitManager(self.split_config, seed=seed)
        splits = split_manager.create_walk_forward_splits(n_samples)

        results = []

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"  Fold {fold_id + 1}/{len(splits)}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train and predict
            predictions = train_func(X_train, y_train, X_test)

            # Evaluate
            result = self.evaluate(
                predictions=predictions,
                targets=y_test,
                model_key=model_key,
                model_name=model_name,
                seed=seed,
                split_id=fold_id
            )
            results.append(result)

        return results

    def aggregate_results(
        self,
        model_key: str,
        compute_bootstrap: bool = True,
        n_bootstrap: int = 1000
    ) -> AggregatedResult:
        """
        Aggregate results across multiple runs/seeds.

        Args:
            model_key: Model to aggregate
            compute_bootstrap: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples

        Returns:
            AggregatedResult with statistics
        """
        if model_key not in self.results:
            raise ValueError(f"No results found for {model_key}")

        individual_results = self.results[model_key]
        n_runs = len(individual_results)
        seeds = list(set(r.seed for r in individual_results))

        # Collect all metrics
        all_metrics = {}
        for result in individual_results:
            for metric_name, value in {**result.ml_metrics, **result.financial_metrics}.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Compute statistics
        metrics_mean = {}
        metrics_std = {}
        metrics_ci_lower = {}
        metrics_ci_upper = {}
        bootstrap_results = {}

        for metric_name, values in all_metrics.items():
            arr = np.array(values)
            metrics_mean[metric_name] = float(np.mean(arr))
            metrics_std[metric_name] = float(np.std(arr, ddof=1))

            if compute_bootstrap and len(arr) >= 5:
                boot_result = self.statistical_tests.bootstrap_confidence_interval(
                    arr,
                    metric_func=np.mean,
                    n_bootstrap=n_bootstrap
                )
                metrics_ci_lower[metric_name] = boot_result.ci_lower
                metrics_ci_upper[metric_name] = boot_result.ci_upper
                bootstrap_results[metric_name] = boot_result
            else:
                # Use standard error approximation
                se = metrics_std[metric_name] / np.sqrt(len(arr))
                metrics_ci_lower[metric_name] = metrics_mean[metric_name] - 1.96 * se
                metrics_ci_upper[metric_name] = metrics_mean[metric_name] + 1.96 * se

        return AggregatedResult(
            model_key=model_key,
            model_name=individual_results[0].model_name,
            n_runs=n_runs,
            seeds=seeds,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metrics_ci_lower=metrics_ci_lower,
            metrics_ci_upper=metrics_ci_upper,
            individual_results=individual_results,
            bootstrap_results=bootstrap_results if compute_bootstrap else None
        )

    def compare_models(
        self,
        model_keys: List[str],
        baseline: Optional[str] = None,
        metric_name: str = 'sharpe_ratio'
    ) -> ComparisonResult:
        """
        Compare multiple models with statistical tests.

        Args:
            model_keys: List of model keys to compare
            baseline: Baseline model for comparison
            metric_name: Metric to compare on

        Returns:
            ComparisonResult with statistical tests
        """
        logger.info(f"Comparing models on {metric_name}")

        if baseline is None:
            baseline = model_keys[0]

        # Collect metric values for each model
        model_metrics = {}
        for model_key in model_keys:
            if model_key not in self.results:
                logger.warning(f"No results found for {model_key}")
                continue

            values = []
            for result in self.results[model_key]:
                all_metrics = {**result.ml_metrics, **result.financial_metrics}
                if metric_name in all_metrics:
                    values.append(all_metrics[metric_name])

            if values:
                model_metrics[model_key] = np.array(values)

        # Pairwise comparisons against baseline
        pairwise_tests = {}
        baseline_values = model_metrics.get(baseline)

        if baseline_values is not None:
            for model_key, values in model_metrics.items():
                if model_key == baseline:
                    continue

                # Paired t-test
                t_result = self.statistical_tests.paired_t_test(values, baseline_values)

                pairwise_tests[model_key] = {
                    't_statistic': t_result.test_statistic,
                    'p_value': t_result.p_value,
                    'significance': t_result.significance.value,
                    'effect_size': t_result.effect_size,
                    'mean_diff': t_result.mean_diff
                }

        # Ranking
        ranking = sorted(
            [(k, float(np.mean(v))) for k, v in model_metrics.items()],
            key=lambda x: x[1],
            reverse=True  # Higher is better for most metrics
        )

        best_model = ranking[0][0] if ranking else baseline
        best_value = ranking[0][1] if ranking else 0.0

        return ComparisonResult(
            models_compared=list(model_metrics.keys()),
            baseline_model=baseline,
            metric_name=metric_name,
            pairwise_tests=pairwise_tests,
            ranking=ranking,
            best_model=best_model,
            best_value=best_value
        )

    def generate_leaderboard(
        self,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate leaderboard DataFrame.

        Args:
            metrics: List of metrics to include (default: sharpe, mae, directional_accuracy)

        Returns:
            DataFrame with rankings
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'mae', 'directional_accuracy']

        rows = []

        for model_key in self.results.keys():
            agg = self.aggregate_results(model_key, compute_bootstrap=False)

            row = {
                'Model': agg.model_name,
                'Model_Key': model_key,
                'N_Runs': agg.n_runs
            }

            for metric in metrics:
                if metric in agg.metrics_mean:
                    row[f'{metric}_mean'] = agg.metrics_mean[metric]
                    row[f'{metric}_std'] = agg.metrics_std[metric]

            rows.append(row)

        df = pd.DataFrame(rows)

        # Add rankings
        for metric in metrics:
            col = f'{metric}_mean'
            if col in df.columns:
                # Higher is better for sharpe, lower for MAE
                ascending = metric in ['mse', 'mae', 'rmse', 'mape', 'max_drawdown']
                df[f'{metric}_rank'] = df[col].rank(ascending=ascending, method='min')

        return df.sort_values('sharpe_ratio_rank' if 'sharpe_ratio_rank' in df.columns else 'Model')

    def save_results(self, filename: str = 'evaluation_results.json'):
        """Save all results to JSON"""
        output_path = self.output_dir / filename

        all_results = {}
        for model_key, results in self.results.items():
            all_results[model_key] = [r.to_dict() for r in results]

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def load_results(self, filename: str = 'evaluation_results.json'):
        """Load results from JSON"""
        input_path = self.output_dir / filename

        with open(input_path, 'r') as f:
            all_results = json.load(f)

        for model_key, results in all_results.items():
            self.results[model_key] = []
            for r in results:
                self.results[model_key].append(EvaluationResult(
                    model_key=r['model_key'],
                    model_name=r['model_name'],
                    seed=r['seed'],
                    split_id=r['split_id'],
                    timestamp=r['timestamp'],
                    ml_metrics=r['ml_metrics'],
                    financial_metrics=r['financial_metrics'],
                    n_samples=r['n_samples'],
                    regime_metrics=r.get('regime_metrics')
                ))

        logger.info(f"Results loaded from {input_path}")


# Convenience function for quick evaluation
def evaluate_model(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_key: str,
    output_dir: Path = Path('outputs/evaluation')
) -> Dict[str, float]:
    """
    Quick evaluation of a model.

    Args:
        predictions: Model predictions
        targets: Ground truth
        model_key: Model identifier
        output_dir: Output directory

    Returns:
        Dict of all metrics
    """
    harness = EvaluationHarness(output_dir=output_dir, use_database=False)
    result = harness.evaluate(predictions, targets, model_key)
    return {**result.ml_metrics, **result.financial_metrics}


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Evaluation Harness Demo")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 500
    targets = np.cumsum(np.random.randn(n_samples) * 0.02 + 0.0003)

    # Model 1 predictions (good)
    predictions1 = targets + np.random.randn(n_samples) * 0.01

    # Model 2 predictions (worse)
    predictions2 = targets + np.random.randn(n_samples) * 0.02

    # Create harness
    harness = EvaluationHarness(
        output_dir=Path('/tmp/eval_demo'),
        use_database=False
    )

    # Evaluate models
    print("\nEvaluating LSTM...")
    result1 = harness.evaluate(predictions1, targets, 'lstm', 'LSTM Baseline')

    print("\nEvaluating Transformer...")
    result2 = harness.evaluate(predictions2, targets, 'transformer', 'Transformer')

    # Compare
    print("\n" + "-" * 40)
    print("Model Comparison:")
    comparison = harness.compare_models(['lstm', 'transformer'], baseline='lstm')
    print(f"  Ranking: {comparison.ranking}")
    print(f"  Best model: {comparison.best_model}")

    # Leaderboard
    print("\n" + "-" * 40)
    print("Leaderboard:")
    leaderboard = harness.generate_leaderboard()
    print(leaderboard.to_string(index=False))

    print("\n" + "=" * 60)
