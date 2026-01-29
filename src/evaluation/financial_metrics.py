"""
Financial Evaluation Metrics for Trading Strategies

Implements:
- Sharpe Ratio
- Maximum Drawdown
- Cumulative PnL
- Directional Accuracy
- Information Ratio
- Sortino Ratio
- Calmar Ratio
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FinancialMetrics:
    """
    Comprehensive financial performance metrics
    """

    @staticmethod
    def sharpe_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe Ratio

        Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (default: 252 trading days)

        Returns:
            Annualized Sharpe ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualized risk-free rate per period
        rf_per_period = risk_free_rate / periods_per_year

        sharpe = (mean_return - rf_per_period) / std_return * np.sqrt(periods_per_year)

        return float(sharpe)

    @staticmethod
    def sortino_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total std)

        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sortino ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        rf_per_period = risk_free_rate / periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < rf_per_period]
        if len(downside_returns) == 0:
            return np.inf if mean_return > rf_per_period else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - rf_per_period) / downside_std * np.sqrt(periods_per_year)

        return float(sortino)

    @staticmethod
    def max_drawdown(
        returns: Union[np.ndarray, pd.Series],
        return_series: bool = False
    ) -> Union[float, tuple]:
        """
        Calculate maximum drawdown

        Args:
            returns: Array of returns
            return_series: If True, return (max_dd, drawdown_series)

        Returns:
            Maximum drawdown (negative value) or (max_dd, drawdown_series)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0 if not return_series else (0.0, np.array([]))

        # Cumulative returns
        cum_returns = np.cumprod(1 + returns)

        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Drawdown series
        drawdown = (cum_returns - running_max) / running_max

        max_dd = np.min(drawdown)

        if return_series:
            return float(max_dd), drawdown
        return float(max_dd)

    @staticmethod
    def calmar_ratio(
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio = Annual Return / |Max Drawdown|

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # Annualized return
        total_return = np.prod(1 + returns) - 1
        n_years = len(returns) / periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Max drawdown
        max_dd = FinancialMetrics.max_drawdown(returns)

        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0

        calmar = annual_return / abs(max_dd)

        return float(calmar)

    @staticmethod
    def cumulative_returns(
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Calculate cumulative returns

        Args:
            returns: Array of returns

        Returns:
            Cumulative returns array
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return np.array([])

        cum_returns = np.cumprod(1 + returns) - 1

        return cum_returns

    @staticmethod
    def total_return(
        returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate total return

        Args:
            returns: Array of returns

        Returns:
            Total return
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        total_ret = np.prod(1 + returns) - 1

        return float(total_ret)

    @staticmethod
    def directional_accuracy(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        are_returns: bool = False,
        threshold: float = 1e-8
    ) -> float:
        """
        Calculate directional accuracy (sign agreement of price changes)

        For price-level inputs: compares direction of predicted vs actual price CHANGES
        For return inputs: compares signs directly

        Args:
            predictions: Predicted prices or returns
            targets: Actual prices or returns
            are_returns: If True, inputs are already returns (compare signs directly)
                        If False, inputs are prices (compare direction of changes)
            threshold: Minimum absolute value to consider non-zero (avoids spurious
                      comparisons when both values are near zero)

        Returns:
            Accuracy (0 to 1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if len(predictions) < 2:
            return 0.0

        if are_returns:
            # Inputs are already returns - compare signs directly
            pred_direction = predictions
            actual_direction = targets
        else:
            # Inputs are price levels - compare direction of CHANGES
            # This is the correct metric for price prediction models
            pred_direction = np.diff(predictions)  # predicted price change
            actual_direction = np.diff(targets)    # actual price change

        # Apply threshold to avoid comparing near-zero values
        # (when both changes are tiny, sign comparison is meaningless)
        significant_mask = (np.abs(actual_direction) > threshold)

        if np.sum(significant_mask) == 0:
            return 0.5  # No significant movements, return random baseline

        pred_significant = pred_direction[significant_mask]
        actual_significant = actual_direction[significant_mask]

        # Sign agreement
        correct = np.sign(pred_significant) == np.sign(actual_significant)
        accuracy = np.mean(correct)

        return float(accuracy)

    @staticmethod
    def information_ratio(
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio

        IR = (mean_active_return) / std(active_return) * sqrt(periods_per_year)

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized information ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values

        # Active returns
        active_returns = returns - benchmark_returns

        active_returns = active_returns[~np.isnan(active_returns)]

        if len(active_returns) == 0:
            return 0.0

        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns, ddof=1)

        if std_active == 0:
            return 0.0

        ir = mean_active / std_active * np.sqrt(periods_per_year)

        return float(ir)

    @staticmethod
    def drawdown_duration(returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate average drawdown duration in periods

        Args:
            returns: Array of returns

        Returns:
            Average drawdown duration
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # Cumulative returns
        cum_returns = np.cumprod(1 + returns)

        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Drawdown series
        drawdown = (cum_returns - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < -0.01  # 1% threshold

        if not np.any(in_drawdown):
            return 0.0

        # Calculate durations
        durations = []
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return float(np.mean(durations)) if durations else 0.0

    @staticmethod
    def profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)

        Args:
            returns: Array of returns

        Returns:
            Profit factor
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    @staticmethod
    def information_coefficient(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Information Coefficient (correlation between predictions and actual returns)

        Args:
            predictions: Predicted returns
            targets: Actual returns

        Returns:
            Information coefficient (-1 to 1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if len(predictions) < 2:
            return 0.0

        # Pearson correlation
        ic = np.corrcoef(predictions, targets)[0, 1]

        return float(ic) if not np.isnan(ic) else 0.0

    @staticmethod
    def precision_recall(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate precision and recall for positive return prediction

        Args:
            predictions: Predicted returns
            targets: Actual returns

        Returns:
            Dict with precision and recall
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if len(predictions) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        # Binary classification: positive return or not
        pred_positive = predictions > 0
        actual_positive = targets > 0

        # True positives, false positives, false negatives
        tp = np.sum(pred_positive & actual_positive)
        fp = np.sum(pred_positive & ~actual_positive)
        fn = np.sum(~pred_positive & actual_positive)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }

    @staticmethod
    def annualized_return(
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized return

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized return
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        total_return = np.prod(1 + returns) - 1
        n_years = len(returns) / periods_per_year

        if n_years <= 0:
            return 0.0

        annual_return = (1 + total_return) ** (1 / n_years) - 1

        return float(annual_return)

    @staticmethod
    def compute_all_metrics(
        returns: Union[np.ndarray, pd.Series],
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        predictions_are_returns: bool = False
    ) -> Dict[str, float]:
        """
        Compute all comprehensive financial metrics

        Args:
            returns: Strategy returns
            predictions: Optional predicted prices or returns (for directional accuracy)
            targets: Optional actual prices or returns (for directional accuracy)
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate
            periods_per_year: Trading periods per year
            predictions_are_returns: If True, predictions/targets are returns
                                    If False (default), they are price levels

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Return metrics
        metrics['total_return'] = FinancialMetrics.total_return(returns)
        metrics['annualized_return'] = FinancialMetrics.annualized_return(returns, periods_per_year)
        metrics['cumulative_return_final'] = FinancialMetrics.cumulative_returns(returns)[-1] if len(returns) > 0 else 0.0

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = FinancialMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        metrics['sortino_ratio'] = FinancialMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year)

        # Drawdown metrics
        max_dd = FinancialMetrics.max_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = FinancialMetrics.drawdown_duration(returns)
        metrics['calmar_ratio'] = FinancialMetrics.calmar_ratio(returns, periods_per_year)

        # Volatility
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = returns
        returns_array = returns_array[~np.isnan(returns_array)]
        metrics['volatility'] = float(np.std(returns_array, ddof=1) * np.sqrt(periods_per_year)) if len(returns_array) > 0 else 0.0

        # Trading viability metrics
        metrics['profit_factor'] = FinancialMetrics.profit_factor(returns)

        # Directional accuracy and signal quality
        if predictions is not None and targets is not None:
            metrics['directional_accuracy'] = FinancialMetrics.directional_accuracy(
                predictions, targets, are_returns=predictions_are_returns
            )
            metrics['information_coefficient'] = FinancialMetrics.information_coefficient(predictions, targets)

            # Precision and recall
            pr_metrics = FinancialMetrics.precision_recall(predictions, targets)
            metrics.update(pr_metrics)

        # Information ratio
        if benchmark_returns is not None:
            metrics['information_ratio'] = FinancialMetrics.information_ratio(returns, benchmark_returns, periods_per_year)

        # Win rate
        if len(returns_array) > 0:
            metrics['win_rate'] = float(np.mean(returns_array > 0))

        return metrics


def compute_strategy_returns(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    transaction_cost: float = 0.001,
    are_returns: bool = False
) -> np.ndarray:
    """
    Compute strategy returns from price predictions

    Converts normalized prices to returns, then computes strategy performance

    Simple strategy: long if prediction shows upward movement, flat otherwise

    Args:
        predictions: Predicted prices (normalized) or returns
        actual_prices: Actual prices (normalized) or actual returns
        transaction_cost: Transaction cost per trade (default: 0.3% for dissertation realism)
        are_returns: If True, inputs are already returns; if False, convert from prices

    Returns:
        Strategy returns array (same length as inputs)

    Note:
        For dissertation purposes, transaction_cost default is 0.3% (not 0.1%)
        to account for:
        - Bid-ask spread: 0.05-0.15%
        - Slippage: 0.05-0.20%
        - Execution costs: 0.05-0.10%
    """
    predictions = predictions.flatten()
    actual_prices = actual_prices.flatten()

    if len(predictions) != len(actual_prices):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) vs actual ({len(actual_prices)})")

    # ===== CONVERT PRICES TO RETURNS =====
    if are_returns:
        # Already returns, use directly
        actual_returns = actual_prices
        predicted_returns = predictions
    else:
        # Convert normalized prices to returns
        # For normalized prices near 0, this approximates: (p[t+1] - p[t]) / p[t]
        # But we need to be careful with edge cases

        # Compute actual returns: change from current to next period
        actual_returns = np.zeros_like(actual_prices)
        for i in range(len(actual_prices) - 1):
            # Avoid division by zero: if price is very close to 0, use small epsilon
            denom = max(abs(actual_prices[i]), 1e-6)
            actual_returns[i] = (actual_prices[i + 1] - actual_prices[i]) / denom
        actual_returns[-1] = actual_returns[-2] if len(actual_returns) > 1 else 0  # Last period

        # Compute predicted returns: direction of price movement
        predicted_returns = np.zeros_like(predictions)
        for i in range(len(predictions) - 1):
            denom = max(abs(predictions[i]), 1e-6)
            predicted_returns[i] = (predictions[i + 1] - predictions[i]) / denom
        predicted_returns[-1] = predicted_returns[-2] if len(predicted_returns) > 1 else 0  # Last period

    # ===== COMPUTE TRADING POSITIONS =====
    # Position: 1 (long) if predicted return is positive, 0 (flat) if negative/zero
    positions = (predicted_returns > 0).astype(float)

    # ===== COMPUTE POSITION CHANGES (TRADES) =====
    # Track when we change from 0→1 or 1→0
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))

    # ===== COMPUTE STRATEGY RETURNS WITH COSTS =====
    # Strategy return = position_held * actual_return - transaction_cost * trades_executed
    # The position held at time t generates return equal to actual_returns[t]
    # Trading cost is incurred when position changes
    strategy_returns = positions * actual_returns - position_changes * transaction_cost

    return strategy_returns
