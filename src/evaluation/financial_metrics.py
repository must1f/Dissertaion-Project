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
- Skewness & Kurtosis (tail risk)
- Bootstrapped Sharpe CI
- Deflated Sharpe Ratio (Bailey & Prado, 2014)
- Subsample Stability Analysis

References:
    Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting, and
        Non-Normality." JFP.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union, Tuple, List

# SciPy is optional; provide lightweight fallbacks for skew/kurtosis so core
# training can run even if scipy isn't installed in the environment.
try:  # pragma: no cover
    from scipy import stats as scipy_stats
except ImportError:
    class _ScipyStatsFallback:
        @staticmethod
        def skew(x, bias=False):
            x = np.asarray(x)
            x = x[~np.isnan(x)]
            if len(x) < 3:
                return 0.0
            mean = x.mean()
            std = x.std(ddof=0 if bias else 1)
            if std == 0:
                return 0.0
            return float(np.mean(((x - mean) / std) ** 3))

        @staticmethod
        def kurtosis(x, fisher=True, bias=False):
            x = np.asarray(x)
            x = x[~np.isnan(x)]
            if len(x) < 4:
                return 0.0
            mean = x.mean()
            std = x.std(ddof=0 if bias else 1)
            if std == 0:
                return 0.0
            kurt = float(np.mean(((x - mean) / std) ** 4))
            if fisher:
                kurt -= 3.0
            return kurt

    scipy_stats = _ScipyStatsFallback()

from ..utils.logger import get_logger
from ..constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)


# ===== PRICE SCALE VALIDATION =====
# CRITICAL: Financial metrics require de-standardised price levels, not z-scores

def assert_price_scale(
    prices: np.ndarray,
    context: str = "trading metrics",
    min_std_threshold: float = 1.0,
    raise_error: bool = True
) -> bool:
    """
    Validate that prices are de-standardised (not z-scores) before computing financial metrics.

    Using z-scores in trading simulations produces meaningless Sharpe ratios and returns.
    This function provides a fail-fast check to prevent such errors.

    Args:
        prices: Array of price values to validate
        context: Description of where validation is happening (for error messages)
        min_std_threshold: Minimum standard deviation expected for real prices.
                          Z-scores typically have std ~1, real prices have std >> 1.
        raise_error: If True, raise ValueError on failure. If False, return bool.

    Returns:
        True if prices appear to be real price levels, False otherwise.

    Raises:
        ValueError: If prices appear to be z-scores and raise_error=True

    Example:
        >>> # Z-scores will fail
        >>> z_scores = np.array([0.1, -0.2, 0.5, -0.1, 0.3])  # std ~0.26
        >>> assert_price_scale(z_scores)  # Raises ValueError

        >>> # Real prices will pass
        >>> prices = np.array([150.2, 151.5, 149.8, 152.1, 150.9])  # std > 1
        >>> assert_price_scale(prices)  # Returns True
    """
    prices = np.asarray(prices).flatten()
    prices = prices[~np.isnan(prices)]

    if len(prices) < 2:
        if raise_error:
            raise ValueError(f"Insufficient data for {context}: need at least 2 valid prices")
        return False

    price_std = np.std(prices)
    price_mean = np.mean(prices)

    # Z-scores typically have: mean ~0, std ~1
    # Real prices typically have: mean >> 1, std >> 1 (for stocks, ETFs, etc.)
    # We check std primarily since mean could legitimately be near 0 for some instruments

    if price_std < min_std_threshold:
        error_msg = (
            f"Input appears to be z-scores rather than price levels for {context}. "
            f"Price std={price_std:.4f} < threshold={min_std_threshold}. "
            f"De-standardise predictions before computing trading metrics: "
            f"price = z_score * close_std + close_mean"
        )
        if raise_error:
            raise ValueError(error_msg)
        logger.warning(error_msg)
        return False

    return True


def destandardise_prices(
    z_scores: np.ndarray,
    price_mean: float,
    price_std: float
) -> np.ndarray:
    """
    Convert z-score normalised values back to real price levels.

    This MUST be called before computing financial metrics if your model
    outputs normalised predictions.

    Args:
        z_scores: Normalised predictions (z-scores)
        price_mean: Original mean used for normalisation
        price_std: Original standard deviation used for normalisation

    Returns:
        De-standardised prices

    Example:
        >>> # During training, you normalised: z = (price - mean) / std
        >>> # Before metrics, reverse it:
        >>> real_prices = destandardise_prices(predictions, close_mean, close_std)
    """
    return z_scores * price_std + price_mean


class FinancialMetrics:
    """
    Comprehensive financial performance metrics
    """

    @staticmethod
    def sharpe_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized Sharpe Ratio

        Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (default: 252 trading days)

        Returns:
            Annualized Sharpe ratio (capped at reasonable bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns to realistic bounds first
        returns = np.clip(returns, -0.99, 1.0)

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return < 1e-10:
            return 0.0

        # Annualized risk-free rate per period
        rf_per_period = risk_free_rate / periods_per_year

        sharpe = (mean_return - rf_per_period) / std_return * np.sqrt(periods_per_year)

        # Cap at reasonable bounds (Sharpe > 5 or < -5 is suspicious)
        sharpe = np.clip(sharpe, -5.0, 5.0)

        return float(sharpe)

    @staticmethod
    def sharpe_ratio_raw(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized Sharpe Ratio WITHOUT CLIPPING.

        Use this for research/debugging when you need the true unclipped value.
        For display/reporting, use sharpe_ratio() which clips to [-5, 5].

        Returns:
            Annualized Sharpe ratio (unclipped, may be very large or inf)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # Clip returns to realistic bounds first (to avoid numeric overflow)
        returns = np.clip(returns, -0.99, 1.0)

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return < 1e-10:
            return 0.0

        rf_per_period = risk_free_rate / periods_per_year
        sharpe = (mean_return - rf_per_period) / std_return * np.sqrt(periods_per_year)

        # NO CLIPPING - return raw value
        return float(sharpe) if not np.isinf(sharpe) else float(np.sign(sharpe) * 999.0)

    @staticmethod
    def sortino_ratio_raw(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino Ratio WITHOUT CLIPPING.

        Use this for research/debugging when you need the true unclipped value.
        For display/reporting, use sortino_ratio() which clips to [-10, 10].

        Returns:
            Annualized Sortino ratio (unclipped, may be very large or inf)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        rf_per_period = risk_free_rate / periods_per_year

        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std < 1e-10:
            return 0.0

        sortino = (mean_return - rf_per_period) / downside_std * np.sqrt(periods_per_year)

        # NO CLIPPING - return raw value
        return float(sortino) if not np.isinf(sortino) else float(np.sign(sortino) * 999.0)

    @staticmethod
    def sortino_ratio(
        returns: Union[np.ndarray, pd.Series],
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total std)

        Uses standard Sortino definition: downside = returns below target (typically 0)
        NOT returns below risk-free rate (that's a common error)

        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate
            periods_per_year: Trading periods per year
            target_return: Minimum acceptable return per period (default: 0)

        Returns:
            Annualized Sortino ratio (capped at reasonable bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        rf_per_period = risk_free_rate / periods_per_year

        # CRITICAL FIX: Standard Sortino uses returns below target (typically 0)
        # NOT returns below risk-free rate
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            # No downside returns: risk-adjusted ratio is undefined
            # Return 0.0 rather than an inflated fallback value
            return 0.0

        # Downside deviation: std of returns below target
        downside_std = np.std(downside_returns, ddof=1)

        if downside_std < 1e-10:
            return 0.0

        sortino = (mean_return - rf_per_period) / downside_std * np.sqrt(periods_per_year)

        # Cap at reasonable bounds (Sortino > 10 or < -10 is suspicious)
        sortino = np.clip(sortino, -10.0, 10.0)

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
            Maximum drawdown (negative value, capped at -1.0 = -100%) or (max_dd, drawdown_series)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0 if not return_series else (0.0, np.array([]))

        # CRITICAL FIX: Clip individual returns to realistic bounds
        # No single-period return can exceed -100% (total loss) in reality
        # Upper bound of +100% is generous for daily returns
        returns_clipped = np.clip(returns, -0.99, 1.0)

        # Cumulative returns with equity floor
        # Equity cannot go below 0 (you can't lose more than 100%)
        cum_returns = np.cumprod(1 + returns_clipped)
        cum_returns = np.maximum(cum_returns, 1e-10)  # Equity floor (essentially 0)

        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Drawdown series
        drawdown = (cum_returns - running_max) / running_max

        # Cap drawdown at -100% (physically impossible to exceed)
        drawdown = np.maximum(drawdown, -1.0)

        max_dd = np.min(drawdown)

        if return_series:
            return float(max_dd), drawdown
        return float(max_dd)

    @staticmethod
    def calmar_ratio(
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate Calmar Ratio = Annual Return / |Max Drawdown|

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio (capped at reasonable bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns before computing
        returns = np.clip(returns, -0.99, 1.0)

        # Annualized return
        total_return = np.prod(1 + returns) - 1
        n_years = len(returns) / periods_per_year

        if n_years <= 0:
            return 0.0

        # Handle edge cases for annual return calculation
        if total_return <= -1:
            annual_return = -1.0  # Total loss
        else:
            annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Max drawdown
        max_dd = FinancialMetrics.max_drawdown(returns)

        if abs(max_dd) < 0.001:  # Less than 0.1% drawdown
            # Drawdown too small for meaningful risk-adjusted return
            return 0.0

        calmar = annual_return / abs(max_dd)

        # Cap at reasonable bounds (Calmar > 10 is exceptional)
        calmar = np.clip(calmar, -10.0, 10.0)

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
            Cumulative returns array (with equity floor at -100%)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return np.array([])

        # CRITICAL FIX: Clip returns to realistic bounds
        returns = np.clip(returns, -0.99, 1.0)

        cum_returns = np.cumprod(1 + returns) - 1

        # Equity floor: can't go below -100%
        cum_returns = np.maximum(cum_returns, -1.0)

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
            Total return (capped at bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip returns to realistic bounds first
        returns = np.clip(returns, -0.99, 1.0)

        total_ret = np.prod(1 + returns) - 1

        # Cap bounds
        total_ret = np.clip(total_ret, -1.0, 10.0) 

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
        periods_per_year: int = TRADING_DAYS_PER_YEAR
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
            Profit factor (capped at reasonable bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss < 1e-10:
            # No losses: return bounded value instead of inf
            return 10.0 if gross_profit > 0 else 1.0

        pf = gross_profit / gross_loss

        # Cap at reasonable bounds (PF > 10 is exceptional)
        pf = np.clip(pf, 0.0, 10.0)

        return float(pf)

    @staticmethod
    def information_coefficient(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        use_returns: bool = True
    ) -> float:
        """
        Calculate Information Coefficient (correlation between predicted and actual returns)

        Args:
            predictions: Predicted prices or returns
            targets: Actual prices or returns
            use_returns: If True, compute IC on returns (price changes), not levels.
                        This is the standard definition for trading signal quality.

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

        # FIX: Compute IC on returns (changes) for proper trading signal quality
        if use_returns and len(predictions) > 2:
            pred_returns = np.diff(predictions)
            target_returns = np.diff(targets)
            ic = np.corrcoef(pred_returns, target_returns)[0, 1]
        else:
            # Fallback to level correlation
            ic = np.corrcoef(predictions, targets)[0, 1]

        return float(ic) if not np.isnan(ic) else 0.0

    @staticmethod
    def precision_recall(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        use_returns: bool = True
    ) -> Dict[str, float]:
        """
        Calculate precision and recall for positive return prediction

        Args:
            predictions: Predicted prices or returns
            targets: Actual prices or returns
            use_returns: If True, compute on price changes (returns), not absolute values.
                        This is more meaningful for trading signal quality.

        Returns:
            Dict with precision, recall, and f1_score
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
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        # FIX: Use price changes (returns) for classification, not absolute levels
        if use_returns:
            pred_changes = np.diff(predictions)
            actual_changes = np.diff(targets)
            pred_positive = pred_changes > 0
            actual_positive = actual_changes > 0
        else:
            # Fallback to absolute values
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
    def skewness(
        returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate skewness of returns (third standardized moment).

        Negative skewness indicates left tail (large losses) are more extreme.
        Positive skewness indicates right tail (large gains) are more extreme.

        Args:
            returns: Array of returns

        Returns:
            Skewness value (typically between -3 and 3)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) < 3:
            return 0.0

        return float(scipy_stats.skew(returns, bias=False))

    @staticmethod
    def kurtosis(
        returns: Union[np.ndarray, pd.Series],
        excess: bool = True
    ) -> float:
        """
        Calculate kurtosis of returns (fourth standardized moment).

        Excess kurtosis > 0 indicates fat tails (more extreme events).
        Normal distribution has excess kurtosis of 0.

        Args:
            returns: Array of returns
            excess: If True, return excess kurtosis (kurtosis - 3)

        Returns:
            Kurtosis value (excess kurtosis typically between -2 and 10)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) < 4:
            return 0.0

        # scipy.stats.kurtosis with fisher=True returns excess kurtosis
        return float(scipy_stats.kurtosis(returns, fisher=excess, bias=False))

    @staticmethod
    def bootstrapped_sharpe_ci(
        returns: Union[np.ndarray, pd.Series],
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        seed: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrapped confidence interval for Sharpe ratio.

        Uses block bootstrap to preserve autocorrelation structure.

        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            seed: Random seed for reproducibility

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)

        Reference:
            Ledoit, O. & Wolf, M. (2008). "Robust Performance Hypothesis
            Testing with the Sharpe Ratio." JEF.
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
            return (sharpe, sharpe - 1.0, sharpe + 1.0)

        if seed is not None:
            np.random.seed(seed)

        # Block size for block bootstrap (cube root of n is common choice)
        block_size = max(1, int(np.ceil(len(returns) ** (1/3))))
        n = len(returns)

        def compute_sharpe(sample):
            mean_ret = np.mean(sample)
            std_ret = np.std(sample, ddof=1)
            if std_ret < 1e-10:
                return 0.0
            rf_per_period = risk_free_rate / periods_per_year
            return (mean_ret - rf_per_period) / std_ret * np.sqrt(periods_per_year)

        # Point estimate
        point_estimate = compute_sharpe(returns)

        # Bootstrap samples using block bootstrap
        boot_sharpes = []
        for _ in range(n_bootstrap):
            # Generate block starting indices
            n_blocks = int(np.ceil(n / block_size))
            block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

            # Build bootstrap sample
            sample = []
            for start in block_starts:
                sample.extend(returns[start:start + block_size])
            sample = np.array(sample[:n])  # Truncate to original length

            boot_sharpes.append(compute_sharpe(sample))

        boot_sharpes = np.array(boot_sharpes)

        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(boot_sharpes, alpha / 2 * 100)
        upper = np.percentile(boot_sharpes, (1 - alpha / 2) * 100)

        return (float(point_estimate), float(lower), float(upper))

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe_ratio: float,
        n_trials: int,
        variance_sharpe: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado (2014).

        Adjusts Sharpe ratio for multiple testing and non-normality.

        The DSR answers: "What is the probability that the observed Sharpe
        was achieved through skill rather than luck?"

        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_trials: Number of backtests/strategies tested
            variance_sharpe: Variance of Sharpe ratio estimate
            skewness: Skewness of returns (default 0 = normal)
            kurtosis: Kurtosis of returns (default 3 = normal)

        Returns:
            Deflated Sharpe Ratio (probability of skill, 0-1)

        Reference:
            Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe
            Ratio: Correcting for Selection Bias, Backtest Overfitting,
            and Non-Normality." JFP.
        """
        if n_trials < 1:
            n_trials = 1

        # Expected maximum Sharpe under null (all strategies have zero skill)
        # E[max(Z_1, ..., Z_n)] for standard normal Z_i
        from scipy.stats import norm

        # Approximate expected maximum using Euler-Mascheroni constant
        euler_mascheroni = 0.5772156649
        expected_max = (1 - euler_mascheroni) * norm.ppf(1 - 1/n_trials) + \
                       euler_mascheroni * norm.ppf(1 - 1/(n_trials * np.e))

        # Adjust for non-normality
        # SR* = SR / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)
        if variance_sharpe > 0:
            sr_std = np.sqrt(variance_sharpe)
        else:
            sr_std = 1.0

        # Non-normality adjustment factor
        adjustment = 1 - skewness * sharpe_ratio / 3 + (kurtosis - 3) / 4 * (sharpe_ratio ** 2)
        if adjustment > 0:
            adjusted_sharpe = sharpe_ratio / np.sqrt(adjustment)
        else:
            adjusted_sharpe = sharpe_ratio

        # Deflated Sharpe: probability that observed SR exceeds expected max
        dsr = norm.cdf((adjusted_sharpe - expected_max) / sr_std)

        return float(np.clip(dsr, 0.0, 1.0))

    @staticmethod
    def subsample_stability(
        returns: Union[np.ndarray, pd.Series],
        metric_func: callable = None,
        n_subsamples: int = 10,
        min_subsample_size: int = 50,
    ) -> Dict[str, float]:
        """
        Analyze stability of a metric across time subsamples.

        Tests whether performance is consistent across different time periods,
        which helps detect overfitting or regime dependence.

        Args:
            returns: Array of returns
            metric_func: Function to compute metric (default: Sharpe ratio)
            n_subsamples: Number of non-overlapping subsamples
            min_subsample_size: Minimum observations per subsample

        Returns:
            Dictionary with:
            - mean: Mean metric across subsamples
            - std: Standard deviation across subsamples
            - min: Minimum metric
            - max: Maximum metric
            - stability: 1 - (std/mean) clipped to [0, 1]
            - positive_pct: Percentage of subsamples with positive metric
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if metric_func is None:
            metric_func = lambda r: FinancialMetrics.sharpe_ratio(r)

        n = len(returns)
        subsample_size = n // n_subsamples

        if subsample_size < min_subsample_size:
            # Not enough data for requested subsamples
            n_subsamples = max(1, n // min_subsample_size)
            subsample_size = n // n_subsamples

        if n_subsamples < 2:
            metric = metric_func(returns)
            return {
                'mean': metric,
                'std': 0.0,
                'min': metric,
                'max': metric,
                'stability': 1.0,
                'positive_pct': 1.0 if metric > 0 else 0.0,
            }

        metrics = []
        for i in range(n_subsamples):
            start = i * subsample_size
            end = start + subsample_size
            subsample = returns[start:end]
            if len(subsample) >= min_subsample_size:
                metrics.append(metric_func(subsample))

        metrics = np.array(metrics)
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics, ddof=1)

        # Stability: higher is better (consistent performance)
        if abs(mean_metric) > 1e-8:
            stability = 1 - np.clip(std_metric / abs(mean_metric), 0, 1)
        else:
            stability = 0.0 if std_metric > 0 else 1.0

        return {
            'mean': float(mean_metric),
            'std': float(std_metric),
            'min': float(np.min(metrics)),
            'max': float(np.max(metrics)),
            'stability': float(stability),
            'positive_pct': float(np.mean(metrics > 0)),
        }

    @staticmethod
    def annualized_return(
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized return

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized return (capped at reasonable bounds)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # CRITICAL FIX: Clip individual period returns strictly to prevent numeric overflow
        returns = np.clip(returns, -0.99, 1.0)

        total_return = np.prod(1 + returns) - 1

        # Prevent absurd total returns before applying the exponent fraction wrapper limit
        total_return = np.clip(total_return, -1.0, 10.0)

        n_years = len(returns) / periods_per_year

        if n_years <= 0:
            return 0.0

        if total_return <= -1:
            return -1.0

        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_return = np.clip(annual_return, -1.0, 5.0)

        return float(annual_return)

    @staticmethod
    def compute_all_metrics(
        returns: Union[np.ndarray, pd.Series],
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        risk_free_rate: float = RISK_FREE_RATE,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
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
        # Store BOTH raw (unclipped) and display (clipped) versions for research integrity
        sharpe_raw = FinancialMetrics.sharpe_ratio_raw(returns, risk_free_rate, periods_per_year)
        sortino_raw = FinancialMetrics.sortino_ratio_raw(returns, risk_free_rate, periods_per_year)
        sharpe_display = FinancialMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_display = FinancialMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year)

        # Raw values (unclipped) - for research analysis and detecting issues
        metrics['sharpe_ratio_raw'] = sharpe_raw
        metrics['sortino_ratio_raw'] = sortino_raw

        # Display values (clipped to bounds) - for reporting/UI
        metrics['sharpe_ratio'] = sharpe_display
        metrics['sortino_ratio'] = sortino_display
        metrics['sharpe_ratio_display'] = sharpe_display
        metrics['sortino_ratio_display'] = sortino_display

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

        # ===== ADVANCED METRICS =====

        # Higher moments (tail risk)
        metrics['skewness'] = FinancialMetrics.skewness(returns)
        metrics['kurtosis'] = FinancialMetrics.kurtosis(returns)

        # Bootstrapped Sharpe CI (reduced samples for speed)
        sharpe_point, sharpe_lower, sharpe_upper = FinancialMetrics.bootstrapped_sharpe_ci(
            returns, confidence=0.95, n_bootstrap=1000
        )
        metrics['sharpe_ci_lower'] = sharpe_lower
        metrics['sharpe_ci_upper'] = sharpe_upper

        # Subsample stability
        stability = FinancialMetrics.subsample_stability(returns)
        metrics['sharpe_stability'] = stability['stability']
        metrics['sharpe_subsample_std'] = stability['std']
        metrics['positive_subsample_pct'] = stability['positive_pct']

        # FIX: Validate all metrics and replace invalid values
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isinf(value):
                    logger.warning(f"Metric '{key}' is infinite, capping to bounded value")
                    metrics[key] = 10.0 if value > 0 else -10.0
                elif np.isnan(value):
                    logger.warning(f"Metric '{key}' is NaN, setting to 0")
                    metrics[key] = 0.0

        return metrics


def compute_strategy_returns(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    transaction_cost: float = 0.001,
    are_returns: bool = False,
    max_return: float = 0.20,
    min_return: float = -0.20,
    price_mean: float = None,
    price_std: float = None,
    threshold: float = 0.0,
    sizing_mode: str = "sign",
    max_leverage: float = 1.0,
    volatility: np.ndarray | None = None,
    return_details: bool = False,
    validate_scale: bool = True,
) -> np.ndarray:
    """
    Compute strategy returns from price predictions

    Converts prices (optionally de-standardised) to percentage returns, then
    computes a simple long/flat strategy:
        position = 1 if predicted return > 0 else 0

    Args:
        predictions: Predicted prices (normalized) or returns
        actual_prices: Actual prices (normalized) or actual returns
        transaction_cost: Transaction cost per trade (default: 0.1%)
        are_returns: If True, inputs are already returns; if False, convert from prices
        max_return: Maximum allowed single-period return (default: +20%)
        min_return: Minimum allowed single-period return (default: -20%)
        price_mean: Optional mean to de-standardise prices
        price_std: Optional std to de-standardise prices
        threshold: Minimum predicted return magnitude to generate signal
        sizing_mode: Position sizing method ("sign", "scaled", "prob")
        max_leverage: Maximum position size for scaled/prob modes
        volatility: Optional volatility array for scaled sizing
        return_details: If True, return (returns, details_dict)
        validate_scale: If True (default), validate that prices are not z-scores.
                       Set to False only if you're certain inputs are correct.

    Returns:
        Strategy returns array (same length as inputs) or tuple of (returns, details)

    Note:
        For dissertation purposes:
        - Transaction costs account for bid-ask spread, slippage, and execution costs
        - Single-period returns are capped to realistic bounds (±20% daily is extreme)
        - This prevents numerical explosions from normalized price edge cases
    """
    predictions = np.asarray(predictions).flatten()
    actual_prices = np.asarray(actual_prices).flatten()

    if len(predictions) != len(actual_prices):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) vs actual ({len(actual_prices)})")

    # ===== CRITICAL: VALIDATE PRICE SCALE =====
    # Ensure inputs are de-standardised before computing trading metrics
    if validate_scale and not are_returns:
        # If price_mean/price_std provided, data will be de-standardised below
        # Only validate if NOT auto-de-standardising
        if price_mean is None or price_std is None:
            # Check if actual prices look like z-scores
            try:
                assert_price_scale(
                    actual_prices,
                    context="compute_strategy_returns (actual_prices)",
                    raise_error=True
                )
            except ValueError as e:
                logger.error(str(e))
                raise

    # ===== CONVERT PRICES TO RETURNS =====
    if are_returns:
        # Already returns, use directly but clip to realistic bounds
        actual_returns = np.clip(actual_prices, min_return, max_return)
        predicted_returns = np.clip(predictions, min_return, max_return)
    else:
        # Optional de-standardisation for price inputs
        if price_mean is not None and price_std is not None:
            actual_prices = actual_prices * price_std + price_mean
            predictions = predictions * price_std + price_mean

        # Percentage returns: (p_t - p_{t-1}) / p_{t-1}
        actual_returns = np.zeros_like(actual_prices)
        predicted_returns = np.zeros_like(predictions)

        # Avoid division by zero using a small floor
        denom_actual = np.clip(actual_prices[:-1], 1e-8, None)
        denom_pred = np.clip(predictions[:-1], 1e-8, None)

        actual_returns[1:] = (actual_prices[1:] - actual_prices[:-1]) / denom_actual
        predicted_returns[1:] = (predictions[1:] - predictions[:-1]) / denom_pred

        # CRITICAL: Clip returns to realistic bounds
        actual_returns = np.clip(actual_returns, min_return, max_return)
        predicted_returns = np.clip(predicted_returns, min_return, max_return)

    # ===== COMPUTE TRADING POSITIONS =====
    # sizing_mode options:
    #   sign:        {-1, 0, 1} with optional threshold
    #   scaled:      continuous sizing scaled by volatility (or raw prediction) and clipped to max_leverage
    #   prob:        probabilities mapped to [-1, 1]
    predicted_abs = np.abs(predicted_returns)

    if sizing_mode == "sign":
        raw_signal = np.sign(predicted_returns)
        raw_signal[predicted_abs <= threshold] = 0.0
    elif sizing_mode == "scaled":
        if volatility is None:
            scaled = predicted_returns
        else:
            vol = np.asarray(volatility).flatten()
            if len(vol) != len(predicted_returns):
                raise ValueError("volatility length must match predictions for scaled sizing")
            scaled = predicted_returns / (vol + 1e-8)
        scaled = np.clip(scaled, -max_leverage, max_leverage)
        scaled[predicted_abs <= threshold] = 0.0
        raw_signal = scaled
    elif sizing_mode == "prob":
        # predictions are probabilities in [0,1]; map to [-1,1]
        raw_signal = 2 * predicted_returns - 1
        raw_signal[np.abs(raw_signal) <= threshold] = 0.0
        raw_signal = np.clip(raw_signal, -max_leverage, max_leverage)
    else:
        raise ValueError(f"Unknown sizing_mode: {sizing_mode}")

    # CRITICAL: Shift signal by 1 period to prevent look-ahead bias.
    positions = np.zeros_like(raw_signal)
    positions[1:] = raw_signal[:-1]

    # ===== COMPUTE POSITION CHANGES (TRADES) =====
    # Track turnover: absolute change in positions
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))

    # ===== COMPUTE STRATEGY RETURNS WITH COSTS =====
    # Strategy return = position_held * actual_return - transaction_cost * turnover
    strategy_returns = positions * actual_returns - position_changes * transaction_cost

    # Final sanity check: clip strategy returns
    strategy_returns = np.clip(strategy_returns, min_return, max_return)

    # Validate cumulative returns don't overflow
    test_cum = np.cumprod(1 + strategy_returns)
    if np.any(np.isinf(test_cum)) or np.any(np.isnan(test_cum)) or np.any(test_cum > 1e10):
        logger.warning(
            "Cumulative returns overflow detected even after clipping. "
            "This likely indicates a data pipeline issue (e.g. normalized values "
            "being treated as raw returns). Applying emergency bounds."
        )
        strategy_returns = np.clip(strategy_returns, -0.05, 0.05)

    if not return_details:
        return strategy_returns

    details = {
        "positions": positions,
        "position_changes": position_changes,
        "predicted_returns": predicted_returns,
        "actual_returns": actual_returns,
    }

    return strategy_returns, details


def validate_metrics(metrics: Dict[str, float]) -> Dict[str, any]:
    """
    Validate financial metrics and flag any suspicious values

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Dictionary with validation results and warnings
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }

    # Check for impossible values
    max_dd = metrics.get('max_drawdown', 0)
    if max_dd < -1.0:
        validation['errors'].append(f"Max drawdown {max_dd:.2%} exceeds -100% (impossible)")
        validation['is_valid'] = False

    # Check for infinite values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isinf(value):
                validation['errors'].append(f"{key} is infinite")
                validation['is_valid'] = False
            elif np.isnan(value):
                validation['warnings'].append(f"{key} is NaN")

    # Check for suspicious Sharpe/Sortino
    sharpe = metrics.get('sharpe_ratio', 0)
    sortino = metrics.get('sortino_ratio', 0)

    if abs(sharpe) > 5:
        validation['warnings'].append(f"Sharpe {sharpe:.2f} is unusually high (>5 or <-5)")

    if abs(sortino) > 10:
        validation['warnings'].append(f"Sortino {sortino:.2f} is unusually high (>10 or <-10)")

    # Check for inconsistency: high Sortino with large drawdown
    if sortino > 2 and max_dd < -0.5:  # Sortino > 2 with DD > 50%
        validation['warnings'].append(
            f"Inconsistent: Sortino {sortino:.2f} implies controlled downside, "
            f"but Max DD is {max_dd:.2%}"
        )

    # Check annualized return
    ann_ret = metrics.get('annualized_return', 0)
    if ann_ret > 5.0:  # > 500% annual
        validation['warnings'].append(f"Annual return {ann_ret:.2%} is suspiciously high")

    # Check profit factor
    pf = metrics.get('profit_factor', 1)
    if pf > 10:
        validation['warnings'].append(f"Profit factor {pf:.2f} is unusually high")

    # Log warnings
    if validation['warnings']:
        for warning in validation['warnings']:
            logger.warning(f"Metric validation: {warning}")

    if validation['errors']:
        for error in validation['errors']:
            logger.error(f"Metric validation: {error}")

    return validation


# ===== STANDALONE FUNCTIONS =====
# Complete implementations for backward compatibility with backtesting_platform.py

def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sharpe Ratio

    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (default: 252 trading days)

    Returns:
        Annualized Sharpe ratio (capped at reasonable bounds)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Clip returns to realistic bounds first
    returns = np.clip(returns, -0.99, 1.0)

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return < 1e-10:
        return 0.0

    # Annualized risk-free rate per period
    rf_per_period = risk_free_rate / periods_per_year

    sharpe = (mean_return - rf_per_period) / std_return * np.sqrt(periods_per_year)

    # Cap at reasonable bounds (Sharpe > 5 or < -5 is suspicious)
    sharpe = np.clip(sharpe, -5.0, 5.0)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of total std)

    Uses standard Sortino definition: downside = returns below target (typically 0)

    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Trading periods per year
        target_return: Minimum acceptable return per period (default: 0)

    Returns:
        Annualized Sortino ratio (capped at reasonable bounds)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    rf_per_period = risk_free_rate / periods_per_year

    # Standard Sortino uses returns below target (typically 0)
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        # No downside returns: risk-adjusted ratio is undefined
        return 0.0

    # Downside deviation: std of returns below target
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std < 1e-10:
        return 0.0

    sortino = (mean_return - rf_per_period) / downside_std * np.sqrt(periods_per_year)

    # Cap at reasonable bounds (Sortino > 10 or < -10 is suspicious)
    sortino = np.clip(sortino, -10.0, 10.0)

    return float(sortino)


def calculate_max_drawdown(
    returns: Union[np.ndarray, pd.Series],
    return_series: bool = False
) -> Union[float, tuple]:
    """
    Calculate maximum drawdown

    Args:
        returns: Array of returns
        return_series: If True, return (max_dd, drawdown_series)

    Returns:
        Maximum drawdown (negative value, capped at -1.0 = -100%) or (max_dd, drawdown_series)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0 if not return_series else (0.0, np.array([]))

    # Clip individual returns to realistic bounds
    # No single-period return can exceed -100% (total loss) in reality
    returns_clipped = np.clip(returns, -0.99, 1.0)

    # Cumulative returns with equity floor
    # Equity cannot go below 0 (you can't lose more than 100%)
    cum_returns = np.cumprod(1 + returns_clipped)
    cum_returns = np.maximum(cum_returns, 1e-10)  # Equity floor

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdown series
    drawdown = (cum_returns - running_max) / running_max

    # Cap drawdown at -100% (physically impossible to exceed)
    drawdown = np.maximum(drawdown, -1.0)

    max_dd = np.min(drawdown)

    if return_series:
        return float(max_dd), drawdown
    return float(max_dd)


def calculate_calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Calmar Ratio = Annual Return / |Max Drawdown|

    Args:
        returns: Array of returns
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio (capped at reasonable bounds)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Clip returns before computing
    returns = np.clip(returns, -0.99, 1.0)

    # Annualized return
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year

    if n_years <= 0:
        return 0.0

    # Handle edge cases for annual return calculation
    if total_return <= -1:
        annual_return = -1.0  # Total loss
    else:
        annual_return = (1 + total_return) ** (1 / n_years) - 1

    # Max drawdown
    max_dd = calculate_max_drawdown(returns)

    if abs(max_dd) < 0.001:  # Less than 0.1% drawdown
        # Drawdown too small for meaningful risk-adjusted return
        return 0.0

    calmar = annual_return / abs(max_dd)

    # Cap at reasonable bounds (Calmar > 10 is exceptional)
    calmar = np.clip(calmar, -10.0, 10.0)

    return float(calmar)


def compute_all_metrics(
    returns: Union[np.ndarray, pd.Series],
    predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
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

    # Convert returns to numpy if needed
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = returns
    returns_array = returns_array[~np.isnan(returns_array)]

    # Return metrics
    metrics['total_return'] = FinancialMetrics.total_return(returns)
    metrics['annualized_return'] = FinancialMetrics.annualized_return(returns, periods_per_year)
    cum_returns = FinancialMetrics.cumulative_returns(returns)
    metrics['cumulative_return_final'] = cum_returns[-1] if len(cum_returns) > 0 else 0.0

    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown metrics
    max_dd = calculate_max_drawdown(returns)
    metrics['max_drawdown'] = max_dd
    metrics['drawdown_duration'] = FinancialMetrics.drawdown_duration(returns)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, periods_per_year)

    # Volatility
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

    # Validate all metrics and replace invalid values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isinf(value):
                logger.warning(f"Metric '{key}' is infinite, capping to bounded value")
                metrics[key] = 10.0 if value > 0 else -10.0
            elif np.isnan(value):
                logger.warning(f"Metric '{key}' is NaN, setting to 0")
                metrics[key] = 0.0

    # Deflated Sharpe Ratio (backtest overfitting adjustment)
    try:
        n_trials = max(5, len(returns_array) // 250)  # heuristic for number of tried variants
        variance_sharpe = np.var(returns_array) * periods_per_year if len(returns_array) > 1 else 0.0
        metrics['deflated_sharpe_ratio'] = FinancialMetrics.deflated_sharpe_ratio(
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            n_trials=n_trials,
            variance_sharpe=variance_sharpe,
            skewness=metrics.get('skewness', 0.0),
            kurtosis=metrics.get('kurtosis', 3.0)
        )
    except Exception as e:
        logger.warning(f"Failed to compute deflated Sharpe ratio: {e}")
        metrics['deflated_sharpe_ratio'] = 0.0

    return metrics
