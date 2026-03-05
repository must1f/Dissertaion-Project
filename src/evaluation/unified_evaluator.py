"""
Unified Model Evaluator

Computes comprehensive financial metrics for all neural network models
Ensures consistent evaluation across LSTM, GRU, Transformer, and all PINN variants
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

from .financial_metrics import FinancialMetrics, compute_strategy_returns
from .rolling_metrics import RollingPerformanceAnalyzer
from .walk_forward_validation import WalkForwardValidator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedModelEvaluator:
    """
    Unified evaluator for all model types

    Computes comprehensive financial metrics regardless of model architecture
    """

    def __init__(
        self,
        transaction_cost: float = 0.003,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Args:
            transaction_cost: Transaction cost per trade (default: 0.3% for dissertation realism)
                Accounts for bid-ask spread (0.05-0.15%), slippage (0.05-0.20%), and
                execution costs (0.05-0.10%). Realistic minimum for equity trading.
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (default: 252 trading days)
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def evaluate_model(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        model_name: str = "Model",
        compute_rolling: bool = True,
        rolling_window_size: int = 63,
        are_returns: bool = True,
        strategy_threshold: float = 0.0,
        strategy_mode: str = "sign",
        max_leverage: float = 1.0,
        volatility: np.ndarray | None = None,
        use_walk_forward: bool = False,
        walk_forward_method: str = "anchored",
        walk_forward_folds: int = 5,
        embargo_size: int = 5,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model

        Args:
            predictions: Model predictions
            targets: Actual values
            model_name: Name of the model
            compute_rolling: Whether to compute rolling metrics
            rolling_window_size: Size of rolling window (default: 3 months = 63 days)

        Returns:
            Dict with all metrics
        """
        logger.info(f"=" * 80)
        logger.info(f"COMPREHENSIVE EVALUATION: {model_name}")
        logger.info(f"=" * 80)

        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if len(predictions) == 0:
            logger.warning(f"No valid predictions for {model_name}")
            return {}

        logger.info(f"Valid samples: {len(predictions)}")

        results = {
            'model_name': model_name,
            'n_samples': len(predictions),
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

        # ========== BASIC ML METRICS ==========
        results['ml_metrics'] = self._compute_ml_metrics(predictions, targets)

        # ========== STRATEGY RETURNS ==========
        logger.info("Computing strategy returns with transaction costs...")
        logger.info(f"  Targets interpreted as {'returns' if are_returns else 'prices'}")
        logger.info(f"  Transaction cost: {self.transaction_cost*100:.2f}%")
        strategy_result = compute_strategy_returns(
            predictions,
            targets,
            transaction_cost=self.transaction_cost,
            are_returns=are_returns,
            threshold=strategy_threshold,
            sizing_mode=strategy_mode,
            max_leverage=max_leverage,
            volatility=volatility,
            return_details=True,
        )

        if isinstance(strategy_result, tuple):
            strategy_returns, strategy_details = strategy_result
        else:
            strategy_returns = strategy_result
            strategy_details = None

        # ========== COMPREHENSIVE FINANCIAL METRICS ==========
        logger.info("Computing comprehensive financial metrics...")
        financial_metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=targets,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )

        if strategy_details:
            turnover = float(np.mean(strategy_details["position_changes"]))
            trading_pct = float(np.mean(strategy_details["position_changes"] > 0))
            financial_metrics["average_turnover"] = turnover
            financial_metrics["trading_days_pct"] = trading_pct

        results['financial_metrics'] = financial_metrics

        # ========== ROLLING WINDOW ANALYSIS ==========
        if compute_rolling and len(predictions) >= rolling_window_size * 2:
            logger.info("Computing rolling window performance...")

            try:
                analyzer = RollingPerformanceAnalyzer(
                    window_size=rolling_window_size,
                    step_size=rolling_window_size // 3,  # Overlapping windows
                    min_samples=20
                )

                rolling_results, stability_metrics = analyzer.analyze(
                    predictions=predictions,
                    targets=targets,
                    transaction_cost=self.transaction_cost,
                    risk_free_rate=self.risk_free_rate,
                    periods_per_year=self.periods_per_year
                )

                # Serialize rolling windows for downstream analysis
                serialized_windows = []
                for r in rolling_results:
                    serialized_windows.append({
                        'window_id': r.window_id,
                        'start_idx': r.start_idx,
                        'end_idx': r.end_idx,
                        'n_samples': r.n_samples,
                        **{k: v for k, v in r.metrics.items()}
                    })

                results['rolling_metrics'] = {
                    'n_windows': len(rolling_results),
                    'stability': stability_metrics,
                    'windows': serialized_windows
                }

                logger.info(f"  Rolling windows: {len(rolling_results)}")
                logger.info(f"  Sharpe stability: {stability_metrics.get('sharpe_ratio_cv', 0):.3f} CV")

            except Exception as e:
                logger.warning(f"Rolling analysis failed: {e}")
                results['rolling_metrics'] = None

        # ========== WALK-FORWARD VALIDATION ==========
        if use_walk_forward:
            try:
                wf_results = self._evaluate_walk_forward(
                    predictions=predictions,
                    targets=targets,
                    are_returns=are_returns,
                    strategy_threshold=strategy_threshold,
                    strategy_mode=strategy_mode,
                    max_leverage=max_leverage,
                    volatility=volatility,
                    method=walk_forward_method,
                    n_folds=walk_forward_folds,
                    embargo_size=embargo_size,
                    timestamps=timestamps
                )
                results['walk_forward'] = wf_results
            except Exception as e:
                logger.warning(f"Walk-forward validation failed: {e}")
                results['walk_forward'] = None

        # ========== SUMMARY LOGGING ==========
        self._log_summary(results)

        return results

    def _evaluate_walk_forward(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        are_returns: bool,
        strategy_threshold: float,
        strategy_mode: str,
        max_leverage: float,
        volatility: np.ndarray | None,
        method: str,
        n_folds: int,
        embargo_size: int,
        timestamps: Optional[pd.DatetimeIndex]
    ) -> Dict[str, Any]:
        """
        Perform walk-forward evaluation on predictions/targets using chronological folds.
        """
        validator = WalkForwardValidator(method=method, n_folds=n_folds, embargo_size=embargo_size)
        folds = validator.generate_folds(len(predictions), timestamps=timestamps)

        fold_metrics: List[Dict[str, float]] = []

        for fold in folds:
            test_slice = slice(fold.test_start, fold.test_end)
            preds_fold = predictions[test_slice]
            targets_fold = targets[test_slice]
            if len(preds_fold) == 0:
                continue

            strat_res = compute_strategy_returns(
                preds_fold,
                targets_fold,
                transaction_cost=self.transaction_cost,
                are_returns=are_returns,
                threshold=strategy_threshold,
                sizing_mode=strategy_mode,
                max_leverage=max_leverage,
                volatility=None if volatility is None else volatility[test_slice],
                return_details=True,
            )

            strategy_returns, details = strat_res if isinstance(strat_res, tuple) else (strat_res, None)
            fm = FinancialMetrics.compute_all_metrics(
                returns=strategy_returns,
                predictions=preds_fold,
                targets=targets_fold,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )

            if details:
                fm['average_turnover'] = float(np.mean(details['position_changes']))
                fm['trading_days_pct'] = float(np.mean(details['position_changes'] > 0))

            fm['fold_id'] = fold.fold_id
            fm['test_size'] = fold.test_size
            fold_metrics.append(fm)

        if not fold_metrics:
            return {}

        def _collect(metric: str) -> List[float]:
            return [m.get(metric, 0.0) for m in fold_metrics]

        summary = {
            'n_folds': len(fold_metrics),
            'sharpe_mean': float(np.mean(_collect('sharpe_ratio'))),
            'sharpe_std': float(np.std(_collect('sharpe_ratio'))),
            'sortino_mean': float(np.mean(_collect('sortino_ratio'))),
            'sortino_std': float(np.std(_collect('sortino_ratio'))),
            'calmar_mean': float(np.mean(_collect('calmar_ratio'))),
            'calmar_std': float(np.std(_collect('calmar_ratio'))),
            'diracc_mean': float(np.mean(_collect('directional_accuracy'))),
            'diracc_std': float(np.std(_collect('directional_accuracy'))),
            'max_drawdown_worst': float(min(_collect('max_drawdown'))),
        }

        return {
            'folds': fold_metrics,
            'summary': summary
        }

    def _compute_ml_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute traditional ML metrics"""
        mse = float(np.mean((predictions - targets) ** 2))
        mae = float(np.mean(np.abs(predictions - targets)))
        rmse = float(np.sqrt(mse))

        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # MAPE
        mape = float(np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def _log_summary(self, results: Dict):
        """Log evaluation summary"""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        # ML Metrics
        ml = results.get('ml_metrics', {})
        logger.info(f"\nTraditional ML Metrics:")
        logger.info(f"  MSE:  {ml.get('mse', 0):.6f}")
        logger.info(f"  MAE:  {ml.get('mae', 0):.6f}")
        logger.info(f"  RMSE: {ml.get('rmse', 0):.6f}")
        logger.info(f"  R²:   {ml.get('r2', 0):.4f}")

        # Financial Metrics
        fin = results.get('financial_metrics', {})
        logger.info(f"\nFinancial Performance Metrics:")
        logger.info(f"  Sharpe Ratio:         {fin.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Sortino Ratio:        {fin.get('sortino_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown:         {fin.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"  Calmar Ratio:         {fin.get('calmar_ratio', 0):.3f}")
        logger.info(f"  Annualized Return:    {fin.get('annualized_return', 0)*100:.2f}%")
        logger.info(f"  Directional Accuracy: {fin.get('directional_accuracy', 0)*100:.2f}%")
        logger.info(f"  Profit Factor:        {fin.get('profit_factor', 0):.2f}")
        logger.info(f"  Win Rate:             {fin.get('win_rate', 0)*100:.2f}%")
        logger.info(f"  Information Coef:     {fin.get('information_coefficient', 0):.3f}")

        # Rolling Metrics
        if 'rolling_metrics' in results and results['rolling_metrics']:
            rolling = results['rolling_metrics']['stability']
            logger.info(f"\nRobustness & Stability:")
            logger.info(f"  Rolling Windows:      {results['rolling_metrics']['n_windows']}")
            logger.info(f"  Sharpe CV:            {rolling.get('sharpe_ratio_cv', 0):.3f}")
            logger.info(f"  Sharpe Consistency:   {rolling.get('sharpe_ratio_consistency', 0)*100:.1f}%")
            logger.info(f"  DirAcc Consistency:   {rolling.get('directional_accuracy_consistency', 0)*100:.1f}%")

        logger.info("=" * 80)

    def save_results(
        self,
        results: Dict,
        model_key: str,
        output_dir: Path
    ):
        """
        Save evaluation results to JSON

        Args:
            results: Evaluation results
            model_key: Model identifier (e.g., 'lstm', 'gbm', 'stacked')
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f'{model_key}_results.json'

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_types(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def compare_models(
        self,
        model_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            model_results: Dict of {model_key: results_dict}

        Returns:
            DataFrame with comparison
        """
        logger.info("=" * 80)
        logger.info("MULTI-MODEL COMPARISON")
        logger.info("=" * 80)

        comparison_data = []

        for model_key, results in model_results.items():
            ml = results.get('ml_metrics', {})
            fin = results.get('financial_metrics', {})
            rolling = results.get('rolling_metrics', {})

            row = {
                'Model': results.get('model_name', model_key),
                'Model_Key': model_key,

                # ML Metrics
                'MSE': ml.get('mse', np.nan),
                'MAE': ml.get('mae', np.nan),
                'RMSE': ml.get('rmse', np.nan),
                'R²': ml.get('r2', np.nan),

                # Financial Metrics
                'Sharpe': fin.get('sharpe_ratio', np.nan),
                'Sortino': fin.get('sortino_ratio', np.nan),
                'Max_DD_%': fin.get('max_drawdown', 0) * 100,
                'DD_Duration': fin.get('drawdown_duration', np.nan),
                'Calmar': fin.get('calmar_ratio', np.nan),
                'Annual_Ret_%': fin.get('annualized_return', 0) * 100,
                'Dir_Acc_%': fin.get('directional_accuracy', 0) * 100,
                'Precision': fin.get('precision', np.nan),
                'Recall': fin.get('recall', np.nan),
                'F1': fin.get('f1_score', np.nan),
                'IC': fin.get('information_coefficient', np.nan),
                'Profit_Factor': fin.get('profit_factor', np.nan),
                'Win_Rate_%': fin.get('win_rate', 0) * 100,
                'Volatility_%': fin.get('volatility', 0) * 100,

                # Stability
                'N_Windows': rolling.get('n_windows', 0) if rolling else 0,
                'Sharpe_CV': rolling.get('stability', {}).get('sharpe_ratio_cv', np.nan) if rolling else np.nan,
                'Sharpe_Consistency_%': rolling.get('stability', {}).get('sharpe_ratio_consistency', 0) * 100 if rolling else 0
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Rank by key metrics
        df['Sharpe_Rank'] = df['Sharpe'].rank(ascending=False, method='min')
        df['DD_Rank'] = df['Max_DD_%'].rank(ascending=False, method='min')  # Less negative is better
        df['DirAcc_Rank'] = df['Dir_Acc_%'].rank(ascending=False, method='min')

        # Combined score (lower is better)
        df['Overall_Rank'] = (df['Sharpe_Rank'] + df['DD_Rank'] + df['DirAcc_Rank']) / 3

        df = df.sort_values('Overall_Rank')

        logger.info(f"\nComparison generated for {len(df)} models")
        logger.info(f"Best model (Overall): {df.iloc[0]['Model']}")
        logger.info(f"  Sharpe: {df.iloc[0]['Sharpe']:.3f}")
        logger.info(f"  Max DD: {df.iloc[0]['Max_DD_%']:.2f}%")
        logger.info(f"  Dir Acc: {df.iloc[0]['Dir_Acc_%']:.2f}%")

        return df


# Convenience function
def evaluate_and_save(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_key: str,
    model_name: str,
    output_dir: Path,
    **kwargs
) -> Dict:
    """
    Convenience function to evaluate and save results

    Args:
        predictions: Model predictions
        targets: Actual values
        model_key: Model identifier
        model_name: Model display name
        output_dir: Output directory
        **kwargs: Additional arguments for evaluator

    Returns:
        Evaluation results
    """
    evaluator = UnifiedModelEvaluator(**kwargs)

    results = evaluator.evaluate_model(
        predictions=predictions,
        targets=targets,
        model_name=model_name
    )

    evaluator.save_results(results, model_key, output_dir)

    return results
