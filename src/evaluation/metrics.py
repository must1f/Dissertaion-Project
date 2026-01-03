"""
Evaluation metrics for forecasting and trading performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculate various prediction and trading metrics"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of correct directional predictions
        (Did we predict the direction of change correctly?)
        """
        # Calculate returns (change from previous value)
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Percentage of correct directions
        correct = (true_direction == pred_direction).sum()
        total = len(true_direction)

        return (correct / total) * 100 if total > 0 else 0.0

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Sharpe Ratio: (mean return - risk-free rate) / std of returns

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize
        mean_return = np.mean(returns) * periods_per_year
        std_return = np.std(returns) * np.sqrt(periods_per_year)

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Sortino Ratio: Like Sharpe but only considers downside volatility

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Downside returns (negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        # Annualize
        mean_return = np.mean(returns) * periods_per_year
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - risk_free_rate) / downside_std
        return sortino

    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Maximum Drawdown: Largest peak-to-trough decline

        Args:
            cumulative_returns: Array of cumulative returns

        Returns:
            Maximum drawdown (as positive percentage)
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # Return maximum drawdown as positive value
        return abs(np.min(drawdown)) * 100

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calmar Ratio: Annualized return / Maximum drawdown

        Args:
            returns: Array of returns
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Annualized return
        total_return = cumulative_returns[-1] - 1
        n_years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Maximum drawdown
        max_dd = MetricsCalculator.max_drawdown(cumulative_returns) / 100

        if max_dd == 0:
            return 0.0

        return annualized_return / max_dd

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Win rate: Percentage of profitable periods

        Args:
            returns: Array of returns

        Returns:
            Win rate as percentage
        """
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        total = len(returns)

        return (wins / total) * 100


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all prediction metrics

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "train_", "test_")

    Returns:
        Dictionary of metrics
    """
    calc = MetricsCalculator()

    metrics = {
        f"{prefix}rmse": calc.rmse(y_true, y_pred),
        f"{prefix}mae": calc.mae(y_true, y_pred),
        f"{prefix}mape": calc.mape(y_true, y_pred),
        f"{prefix}r2": calc.r2(y_true, y_pred),
        f"{prefix}directional_accuracy": calc.directional_accuracy(y_true, y_pred),
    }

    return metrics


def calculate_financial_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate financial/trading metrics

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        prefix: Prefix for metric names

    Returns:
        Dictionary of financial metrics
    """
    calc = MetricsCalculator()

    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()

    metrics = {
        f"{prefix}sharpe_ratio": calc.sharpe_ratio(returns, risk_free_rate, periods_per_year),
        f"{prefix}sortino_ratio": calc.sortino_ratio(returns, risk_free_rate, periods_per_year),
        f"{prefix}max_drawdown": calc.max_drawdown(cumulative_returns),
        f"{prefix}calmar_ratio": calc.calmar_ratio(returns, periods_per_year),
        f"{prefix}win_rate": calc.win_rate(returns),
        f"{prefix}total_return": (cumulative_returns[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0.0,
        f"{prefix}mean_return": np.mean(returns) * 100,
        f"{prefix}volatility": np.std(returns) * np.sqrt(periods_per_year) * 100,
    }

    return metrics


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy

    Tests H0: errors1 and errors2 have equal forecast accuracy

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        h: Forecast horizon

    Returns:
        Tuple of (test_statistic, p_value)
    """
    # Calculate loss differential
    d = errors1 ** 2 - errors2 ** 2

    # Mean of loss differential
    d_mean = np.mean(d)

    # Variance of loss differential (with Newey-West correction for autocorrelation)
    n = len(d)
    gamma0 = np.var(d, ddof=1)

    # Autocorrelations
    gamma = 0
    for lag in range(1, h):
        gamma += np.corrcoef(d[:-lag], d[lag:])[0, 1] * np.var(d, ddof=1)

    d_var = (gamma0 + 2 * gamma) / n

    # DM test statistic
    if d_var == 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(d_var)

    # P-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def bootstrap_confidence_interval(
    data: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for a metric

    Args:
        data: Data array
        metric_func: Function to calculate metric (takes data, returns scalar)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, (lower_ci, upper_ci))
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Point estimate
    point_estimate = metric_func(data)

    # Bootstrap samples
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_estimates.append(metric_func(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Confidence interval
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    upper_ci = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

    return point_estimate, (lower_ci, upper_ci)
