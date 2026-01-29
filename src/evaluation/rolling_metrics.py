"""
Rolling Out-of-Sample Performance Analysis

Evaluates model stability and robustness across walk-forward windows
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .financial_metrics import FinancialMetrics, compute_strategy_returns
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RollingMetrics:
    """Results from rolling window evaluation"""
    window_id: int
    start_idx: int
    end_idx: int
    n_samples: int
    metrics: Dict[str, float]
    returns: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray


class RollingPerformanceAnalyzer:
    """
    Analyze model performance across rolling windows

    This addresses overfitting detection and regime sensitivity by
    evaluating metrics across multiple time periods
    """

    def __init__(
        self,
        window_size: int = 63,  # ~3 months
        step_size: int = 21,    # ~1 month
        min_samples: int = 20
    ):
        """
        Args:
            window_size: Number of samples per window
            step_size: Step size for rolling window
            min_samples: Minimum samples required for valid window
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples

    def analyze(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Tuple[List[RollingMetrics], Dict[str, float]]:
        """
        Perform rolling window analysis

        Args:
            predictions: Model predictions
            targets: Actual values
            transaction_cost: Transaction cost per trade
            risk_free_rate: Risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            (rolling_results, stability_metrics)
        """
        logger.info("=" * 80)
        logger.info("ROLLING OUT-OF-SAMPLE PERFORMANCE ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Window size: {self.window_size}, Step size: {self.step_size}")

        predictions = predictions.flatten()
        targets = targets.flatten()

        n_samples = len(predictions)
        rolling_results = []

        # Generate windows
        window_id = 0
        start_idx = 0

        while start_idx + self.window_size <= n_samples:
            end_idx = start_idx + self.window_size

            # Extract window data
            window_preds = predictions[start_idx:end_idx]
            window_targets = targets[start_idx:end_idx]

            if len(window_preds) < self.min_samples:
                break

            # Compute strategy returns for window
            window_returns = compute_strategy_returns(
                window_preds,
                window_targets,
                transaction_cost
            )

            # Compute metrics for window
            metrics = FinancialMetrics.compute_all_metrics(
                returns=window_returns,
                predictions=window_preds,
                targets=window_targets,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year
            )

            # Store window results
            rolling_results.append(RollingMetrics(
                window_id=window_id,
                start_idx=start_idx,
                end_idx=end_idx,
                n_samples=len(window_preds),
                metrics=metrics,
                returns=window_returns,
                predictions=window_preds,
                targets=window_targets
            ))

            window_id += 1
            start_idx += self.step_size

        logger.info(f"Generated {len(rolling_results)} rolling windows")

        # Compute stability metrics
        stability_metrics = self._compute_stability_metrics(rolling_results)

        return rolling_results, stability_metrics

    def _compute_stability_metrics(
        self,
        rolling_results: List[RollingMetrics]
    ) -> Dict[str, float]:
        """
        Compute stability metrics across windows

        Lower standard deviation = more stable performance
        """
        if not rolling_results:
            return {}

        # Extract metric values across windows
        metric_names = rolling_results[0].metrics.keys()

        stability = {}

        for metric_name in metric_names:
            values = [r.metrics[metric_name] for r in rolling_results]
            values = np.array(values)
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            # Mean and stability (std / mean)
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Coefficient of variation (CV)
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-6 else 0.0

            # Min and max
            min_val = np.min(values)
            max_val = np.max(values)

            # Consistency (percentage of windows above certain threshold)
            if metric_name == 'sharpe_ratio':
                consistency = np.mean(values > 1.0)  # Sharpe > 1
            elif metric_name == 'directional_accuracy':
                consistency = np.mean(values > 0.5)  # Accuracy > 50%
            elif metric_name == 'max_drawdown':
                consistency = np.mean(values > -0.2)  # Drawdown < 20%
            else:
                consistency = np.mean(values > 0)

            stability[f'{metric_name}_mean'] = float(mean_val)
            stability[f'{metric_name}_std'] = float(std_val)
            stability[f'{metric_name}_cv'] = float(cv)
            stability[f'{metric_name}_min'] = float(min_val)
            stability[f'{metric_name}_max'] = float(max_val)
            stability[f'{metric_name}_consistency'] = float(consistency)

        logger.info("\nStability Metrics:")
        logger.info(f"  Sharpe Ratio:   {stability.get('sharpe_ratio_mean', 0):.3f} ± {stability.get('sharpe_ratio_std', 0):.3f}")
        logger.info(f"  Dir. Accuracy:  {stability.get('directional_accuracy_mean', 0):.3f} ± {stability.get('directional_accuracy_std', 0):.3f}")
        logger.info(f"  Max Drawdown:   {stability.get('max_drawdown_mean', 0):.3f} ± {stability.get('max_drawdown_std', 0):.3f}")
        logger.info(f"  Consistency:    {stability.get('sharpe_ratio_consistency', 0)*100:.1f}% windows profitable")

        return stability

    def to_dataframe(
        self,
        rolling_results: List[RollingMetrics]
    ) -> pd.DataFrame:
        """Convert rolling results to DataFrame"""
        rows = []

        for result in rolling_results:
            row = {
                'window_id': result.window_id,
                'start_idx': result.start_idx,
                'end_idx': result.end_idx,
                'n_samples': result.n_samples
            }
            row.update(result.metrics)
            rows.append(row)

        return pd.DataFrame(rows)


def compare_model_stability(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    window_size: int = 63,
    step_size: int = 21,
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    """
    Compare stability of multiple models

    Args:
        model_results: Dict of {model_name: (predictions, targets)}
        window_size: Rolling window size
        step_size: Step size
        transaction_cost: Transaction cost

    Returns:
        DataFrame comparing model stability
    """
    logger.info("=" * 80)
    logger.info("MULTI-MODEL STABILITY COMPARISON")
    logger.info("=" * 80)

    analyzer = RollingPerformanceAnalyzer(window_size, step_size)

    comparison_data = []

    for model_name, (predictions, targets) in model_results.items():
        logger.info(f"\nAnalyzing {model_name}...")

        rolling_results, stability = analyzer.analyze(
            predictions=predictions,
            targets=targets,
            transaction_cost=transaction_cost
        )

        # Compile comparison metrics
        row = {
            'Model': model_name,
            'Num_Windows': len(rolling_results),
            'Sharpe_Mean': stability.get('sharpe_ratio_mean', 0),
            'Sharpe_Std': stability.get('sharpe_ratio_std', 0),
            'Sharpe_CV': stability.get('sharpe_ratio_cv', 0),
            'Sharpe_Consistency': stability.get('sharpe_ratio_consistency', 0),
            'DirAcc_Mean': stability.get('directional_accuracy_mean', 0),
            'DirAcc_Std': stability.get('directional_accuracy_std', 0),
            'DirAcc_Consistency': stability.get('directional_accuracy_consistency', 0),
            'MaxDD_Mean': stability.get('max_drawdown_mean', 0),
            'MaxDD_Std': stability.get('max_drawdown_std', 0),
            'MaxDD_Consistency': stability.get('max_drawdown_consistency', 0),
            'Return_Mean': stability.get('annualized_return_mean', 0),
            'Return_Std': stability.get('annualized_return_std', 0)
        }

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Rank models by stability
    # Lower CV = more stable
    df['Stability_Rank'] = df['Sharpe_CV'].rank()

    # Higher consistency = better
    df['Consistency_Rank'] = df['Sharpe_Consistency'].rank(ascending=False)

    # Combined stability score
    df['Stability_Score'] = (df['Stability_Rank'] + df['Consistency_Rank']) / 2

    df = df.sort_values('Stability_Score')

    logger.info("\n" + "=" * 80)
    logger.info("STABILITY RANKING")
    logger.info("=" * 80)
    logger.info(f"\n{df[['Model', 'Sharpe_Mean', 'Sharpe_CV', 'Sharpe_Consistency', 'Stability_Score']].to_string(index=False)}\n")

    return df


def detect_regime_sensitivity(
    rolling_results: List[RollingMetrics],
    threshold: float = 0.3
) -> Dict[str, any]:
    """
    Detect regime sensitivity by analyzing metric variance

    High variance across windows suggests regime-dependent performance

    Args:
        rolling_results: Rolling window results
        threshold: CV threshold for "high sensitivity"

    Returns:
        Dict with regime sensitivity analysis
    """
    if not rolling_results:
        return {}

    metric_names = ['sharpe_ratio', 'directional_accuracy', 'max_drawdown']

    sensitivity_analysis = {}

    for metric in metric_names:
        values = [r.metrics.get(metric, 0) for r in rolling_results]
        values = np.array(values)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            continue

        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-6 else 0.0

        # Determine if regime-sensitive
        is_sensitive = cv > threshold

        sensitivity_analysis[metric] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'cv': float(cv),
            'regime_sensitive': is_sensitive,
            'interpretation': 'High regime sensitivity' if is_sensitive else 'Stable across regimes'
        }

    return sensitivity_analysis
