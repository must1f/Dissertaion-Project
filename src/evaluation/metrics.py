"""
Evaluation metrics for forecasting and trading performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# sklearn is used only for a few basic metrics; provide fallbacks so training
# doesn't fail when the dependency is missing in lightweight environments.
try:  # pragma: no cover - guard for optional dependency
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:  # minimal NumPy replacements
    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))

# scipy is used only for a normal CDF in the Diebold-Mariano test.
try:  # pragma: no cover
    from scipy import stats
except ImportError:
    import math

    class _NormFallback:
        @staticmethod
        def cdf(x: float) -> float:
            # Standard normal CDF via error function
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    class _StatsFallback:
        norm = _NormFallback()

    stats = _StatsFallback()

from ..utils.logger import get_logger
from ..constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

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
    def directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        are_returns: bool = False,
        threshold: float = 1e-8
    ) -> float:
        """
        Percentage of correct directional predictions
        (Did we predict the direction of change correctly?)

        Args:
            y_true: True values (prices or returns)
            y_pred: Predicted values (prices or returns)
            are_returns: If True, inputs are returns (compare signs directly)
                        If False, inputs are prices (compare direction of changes)
            threshold: Minimum absolute change to consider significant

        Returns:
            Directional accuracy as percentage (0 to 100)
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0

        if are_returns:
            # Inputs are already returns - compare signs directly
            true_direction = y_true
            pred_direction = y_pred
        else:
            # Calculate returns (change from previous value)
            true_direction = np.diff(y_true)
            pred_direction = np.diff(y_pred)

        # Apply threshold to filter out insignificant movements
        significant_mask = np.abs(true_direction) > threshold

        if np.sum(significant_mask) == 0:
            # No significant movements; return random-baseline accuracy on 0–1 scale
            return 0.5

        true_significant = np.sign(true_direction[significant_mask])
        pred_significant = np.sign(pred_direction[significant_mask])

        # Percentage of correct directions
        correct = (true_significant == pred_significant).sum()
        total = len(true_significant)

        # FIX: Return 0-1 scale for consistency with financial_metrics.py
        # Dashboard will multiply by 100 for display
        return (correct / total) if total > 0 else 0.0

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Sharpe Ratio: (mean return - risk-free rate) / std of returns

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe ratio (capped at reasonable bounds)
        """
        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns to realistic bounds
        returns = np.clip(returns, -0.99, 1.0)

        std_return = np.std(returns)
        if std_return < 1e-10:
            return 0.0

        # Annualize
        mean_return = np.mean(returns) * periods_per_year
        std_return = std_return * np.sqrt(periods_per_year)

        sharpe = (mean_return - risk_free_rate) / std_return

        # Cap at reasonable bounds
        sharpe = np.clip(sharpe, -5.0, 5.0)

        return float(sharpe)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Sortino Ratio: Like Sharpe but only considers downside volatility

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sortino ratio (capped at reasonable bounds)
        """
        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns to realistic bounds
        returns = np.clip(returns, -0.99, 1.0)

        # Downside returns (negative returns below 0 - standard Sortino)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            # No downside returns: risk-adjusted ratio is undefined
            return 0.0

        # Annualize
        mean_return = np.mean(returns) * periods_per_year
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)

        if downside_std < 1e-10:
            return 0.0

        sortino = (mean_return - risk_free_rate) / downside_std

        # Cap at reasonable bounds
        sortino = np.clip(sortino, -10.0, 10.0)

        return float(sortino)

    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Maximum Drawdown: Largest peak-to-trough decline

        Args:
            cumulative_returns: Array of cumulative returns

        Returns:
            Maximum drawdown (as positive percentage, capped at 100%)
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # CRITICAL FIX: Ensure equity doesn't go negative
        cumulative_returns = np.maximum(cumulative_returns, 1e-10)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # CRITICAL FIX: Cap drawdown at -100% (total loss is the max possible)
        drawdown = np.maximum(drawdown, -1.0)

        # Return maximum drawdown as positive value (capped at 100%)
        max_dd = min(abs(np.min(drawdown)) * 100, 100.0)

        return float(max_dd)

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
            Calmar ratio (capped at reasonable bounds)
        """
        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns to realistic bounds
        returns = np.clip(returns, -0.99, 1.0)

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Annualized return
        total_return = cumulative_returns[-1] - 1
        n_years = len(returns) / periods_per_year

        if n_years <= 0:
            return 0.0

        # Handle edge case for total loss
        if total_return <= -1:
            annualized_return = -1.0
        else:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Maximum drawdown
        max_dd = MetricsCalculator.max_drawdown(cumulative_returns) / 100

        if max_dd < 0.001:  # Less than 0.1% drawdown
            return 0.0

        calmar = annualized_return / max_dd

        # Cap at reasonable bounds
        calmar = np.clip(calmar, -10.0, 10.0)

        return float(calmar)

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
    prefix: str = "",
    price_mean: Optional[float] = None,
    price_std: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate all prediction metrics

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "train_", "test_")
        price_mean: Optional mean used for de-standardising prices
        price_std: Optional std used for de-standardising prices

    Returns:
        Dictionary of metrics
    """
    calc = MetricsCalculator()

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Optionally de-standardise before computing metrics
    if price_mean is not None and price_std is not None:
        y_true_eval = y_true_arr * price_std + price_mean
        y_pred_eval = y_pred_arr * price_std + price_mean
    else:
        y_true_eval = y_true_arr
        y_pred_eval = y_pred_arr

    # FIX: directional_accuracy returns 0-1 scale; convert to % for display
    dir_acc = calc.directional_accuracy(y_true_eval, y_pred_eval)

    metrics = {
        f"{prefix}rmse": calc.rmse(y_true_eval, y_pred_eval),
        f"{prefix}mae": calc.mae(y_true_eval, y_pred_eval),
        f"{prefix}mape": calc.mape(y_true_eval, y_pred_eval),
        f"{prefix}r2": calc.r2(y_true_eval, y_pred_eval),
        f"{prefix}directional_accuracy": dir_acc * 100,  # Convert to percentage for display
        f"{prefix}mse": calc.rmse(y_true_eval, y_pred_eval) ** 2,  # FIX: Add MSE explicitly
    }

    return metrics


def calculate_financial_metrics(
    returns: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
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

    # CRITICAL FIX: Clip returns to realistic bounds first
    returns = np.clip(returns, -0.99, 1.0)

    # Calculate cumulative returns with equity floor
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns = np.maximum(cumulative_returns, 1e-10)

    # Total return (capped at bounds)
    total_ret = (cumulative_returns[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0.0
    # Prevent absurd returns: Cap total_ret strictly at [-100%, 1000%]
    total_ret = float(np.clip(total_ret, -100.0, 1000.0))

    metrics = {
        f"{prefix}sharpe_ratio": calc.sharpe_ratio(returns, risk_free_rate, periods_per_year),
        f"{prefix}sortino_ratio": calc.sortino_ratio(returns, risk_free_rate, periods_per_year),
        f"{prefix}max_drawdown": calc.max_drawdown(cumulative_returns),
        f"{prefix}calmar_ratio": calc.calmar_ratio(returns, periods_per_year),
        f"{prefix}win_rate": calc.win_rate(returns),
        f"{prefix}total_return": total_ret,
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
