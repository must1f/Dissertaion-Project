"""
Comprehensive Risk Metrics for Monte Carlo Simulation

This module implements financial risk metrics for comparing standard
vs regime-switching Monte Carlo simulations.

Risk Metrics Implemented:
=========================

1. Value at Risk (VaR)
   - Historical simulation
   - Parametric (Gaussian)
   - Cornish-Fisher expansion (accounts for skewness/kurtosis)

2. Expected Shortfall (ES) / Conditional VaR (CVaR)
   - Average loss beyond VaR
   - More coherent risk measure than VaR

3. Maximum Drawdown (MDD)
   - Largest peak-to-trough decline
   - Critical for portfolio management

4. Higher Moments
   - Skewness: Asymmetry in return distribution
   - Kurtosis: Fat tails / tail risk

5. Risk-Adjusted Returns
   - Sharpe Ratio
   - Sortino Ratio (downside risk only)
   - Calmar Ratio (return / max drawdown)

Why These Metrics Matter for Regime-Switching:
=============================================
Standard Monte Carlo underestimates:
- VaR: By ignoring volatility clustering, extreme losses are underweighted
- ES: The average tail loss is too optimistic
- MDD: Path-dependent effects of regime persistence are missed
- Kurtosis: Regime mixing naturally produces excess kurtosis

Author: Dissertation Research Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.special import ndtri  # Inverse normal CDF

from ..utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VaRResult:
    """Value at Risk computation result"""
    var_historical: float      # Historical simulation VaR
    var_parametric: float      # Gaussian VaR
    var_cornish_fisher: float  # Cornish-Fisher adjusted VaR
    confidence_level: float    # e.g., 0.95 or 0.99
    n_observations: int


@dataclass
class ESResult:
    """Expected Shortfall computation result"""
    es_historical: float       # Historical ES
    es_parametric: float       # Gaussian ES
    confidence_level: float
    n_tail_observations: int


@dataclass
class DrawdownResult:
    """Drawdown analysis result"""
    max_drawdown: float           # Maximum drawdown (negative)
    max_drawdown_duration: int    # Duration in periods
    avg_drawdown: float           # Average drawdown
    drawdown_series: np.ndarray   # Full drawdown time series
    peak_indices: np.ndarray      # Indices of peaks
    trough_indices: np.ndarray    # Indices of troughs


@dataclass
class RiskMetricsResult:
    """Comprehensive risk metrics result"""
    # Value at Risk
    var_95: VaRResult
    var_99: VaRResult

    # Expected Shortfall
    es_95: ESResult
    es_99: ESResult

    # Drawdown
    drawdown: DrawdownResult

    # Distribution metrics
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float         # Excess kurtosis

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Tail metrics
    tail_ratio: float       # |5th percentile| / 95th percentile
    gain_to_pain: float     # Sum of returns / sum of |negative returns|

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for easy comparison"""
        return {
            'var_95_hist': self.var_95.var_historical,
            'var_95_param': self.var_95.var_parametric,
            'var_95_cf': self.var_95.var_cornish_fisher,
            'var_99_hist': self.var_99.var_historical,
            'var_99_param': self.var_99.var_parametric,
            'var_99_cf': self.var_99.var_cornish_fisher,
            'es_95': self.es_95.es_historical,
            'es_99': self.es_99.es_historical,
            'max_drawdown': self.drawdown.max_drawdown,
            'avg_drawdown': self.drawdown.avg_drawdown,
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'tail_ratio': self.tail_ratio,
            'gain_to_pain': self.gain_to_pain,
        }


# =============================================================================
# Value at Risk Functions
# =============================================================================

def compute_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'all'
) -> VaRResult:
    """
    Compute Value at Risk using multiple methods

    VaR Definition:
        The α-VaR is the loss threshold such that:
        P(Loss > VaR) = 1 - α

    Methods:
    1. Historical: Direct percentile of empirical distribution
    2. Parametric: Assumes Gaussian returns
    3. Cornish-Fisher: Adjusts for skewness and kurtosis

    Args:
        returns: Array of returns (can be terminal returns or daily)
        confidence_level: Confidence level (0.95 = 95%)
        method: 'historical', 'parametric', 'cornish_fisher', or 'all'

    Returns:
        VaRResult with VaR values
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    alpha = 1 - confidence_level  # Left tail probability

    # Historical VaR: Direct percentile
    var_historical = np.percentile(returns, alpha * 100)

    # Parametric VaR: Assumes N(μ, σ²)
    mu = np.mean(returns)
    sigma = np.std(returns)
    z_alpha = ndtri(alpha)  # Quantile of standard normal
    var_parametric = mu + sigma * z_alpha

    # Cornish-Fisher VaR: Adjusts for non-normality
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis

    # Cornish-Fisher expansion for adjusted z-score
    # z_cf = z + (z² - 1)S/6 + (z³ - 3z)K/24 - (2z³ - 5z)S²/36
    z = z_alpha
    z_cf = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)

    var_cornish_fisher = mu + sigma * z_cf

    return VaRResult(
        var_historical=float(var_historical),
        var_parametric=float(var_parametric),
        var_cornish_fisher=float(var_cornish_fisher),
        confidence_level=confidence_level,
        n_observations=n
    )


def compute_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> ESResult:
    """
    Compute Expected Shortfall (Conditional VaR)

    ES Definition:
        ES_α = E[Loss | Loss > VaR_α]

    Expected Shortfall is a "coherent" risk measure (unlike VaR) because:
    1. It satisfies subadditivity: ES(A+B) ≤ ES(A) + ES(B)
    2. It considers the magnitude of tail losses, not just frequency

    Args:
        returns: Array of returns
        confidence_level: Confidence level

    Returns:
        ESResult with ES values
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    alpha = 1 - confidence_level

    # Historical ES: Average of returns below VaR
    var_threshold = np.percentile(returns, alpha * 100)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) > 0:
        es_historical = np.mean(tail_returns)
    else:
        es_historical = var_threshold

    # Parametric ES: Assumes Gaussian
    # ES = μ - σ * φ(z_α) / α  where φ is standard normal PDF
    mu = np.mean(returns)
    sigma = np.std(returns)
    z_alpha = ndtri(alpha)
    phi_z = stats.norm.pdf(z_alpha)
    es_parametric = mu - sigma * phi_z / alpha

    return ESResult(
        es_historical=float(es_historical),
        es_parametric=float(es_parametric),
        confidence_level=confidence_level,
        n_tail_observations=len(tail_returns)
    )


# =============================================================================
# Drawdown Functions
# =============================================================================

def compute_maximum_drawdown(
    prices: np.ndarray,
    return_series: bool = True
) -> DrawdownResult:
    """
    Compute maximum drawdown and related statistics

    Drawdown Definition:
        DD_t = (Peak_t - Price_t) / Peak_t
        where Peak_t = max(Price_0, ..., Price_t)

    Maximum Drawdown:
        MDD = max(DD_0, ..., DD_T)

    Why Drawdown Matters:
    - Path-dependent risk measure
    - Captures sustained losses (not just single-day)
    - Regime persistence leads to longer, deeper drawdowns

    Args:
        prices: Price series (cumulative wealth)
        return_series: If True, also return full drawdown series

    Returns:
        DrawdownResult with MDD and statistics
    """
    prices = prices.flatten()

    # Running maximum (peak)
    running_max = np.maximum.accumulate(prices)

    # Drawdown series (negative values = drawdown)
    drawdown = (prices - running_max) / running_max

    # Maximum drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)

    # Find peak before max drawdown
    peak_idx = np.argmax(prices[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Drawdown duration
    dd_duration = max_dd_idx - peak_idx

    # Average drawdown (only during drawdown periods)
    avg_dd = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0

    # Find all peaks and troughs
    peaks = []
    troughs = []

    in_drawdown = False
    peak_val = prices[0]
    peak_idx_local = 0

    for i in range(1, len(prices)):
        if prices[i] > peak_val:
            if in_drawdown:
                # End of drawdown
                troughs.append(i - 1)
                in_drawdown = False
            peak_val = prices[i]
            peak_idx_local = i
            peaks.append(i)
        elif prices[i] < peak_val * 0.99:  # 1% threshold
            in_drawdown = True

    return DrawdownResult(
        max_drawdown=float(max_dd),
        max_drawdown_duration=int(dd_duration),
        avg_drawdown=float(avg_dd),
        drawdown_series=drawdown if return_series else np.array([]),
        peak_indices=np.array(peaks),
        trough_indices=np.array(troughs)
    )


def compute_drawdown_from_returns(returns: np.ndarray) -> DrawdownResult:
    """
    Compute drawdown from return series

    Args:
        returns: Array of returns (log or simple)

    Returns:
        DrawdownResult
    """
    # Convert returns to cumulative wealth
    cumulative_returns = np.cumsum(returns)
    wealth = np.exp(cumulative_returns)  # Assuming log returns

    return compute_maximum_drawdown(wealth)


# =============================================================================
# Risk-Adjusted Metrics
# =============================================================================

def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sharpe Ratio

    Sharpe Ratio = (E[R] - R_f) / σ(R)

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_per_period
    sharpe = np.mean(excess_returns) / np.std(excess_returns)

    # Annualize
    sharpe_annual = sharpe * np.sqrt(periods_per_year)

    return float(sharpe_annual)


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Compute Sortino Ratio

    Sortino Ratio = (E[R] - R_f) / σ_downside(R)

    Only penalizes downside volatility, not upside.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        target_return: Minimum acceptable return (MAR)

    Returns:
        Annualized Sortino ratio
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Convert to per-period
    rf_per_period = risk_free_rate / periods_per_year
    target_per_period = target_return / periods_per_year

    # Downside deviation
    downside_returns = returns[returns < target_per_period] - target_per_period
    if len(downside_returns) == 0:
        return np.inf  # No downside

    downside_std = np.sqrt(np.mean(downside_returns**2))

    if downside_std == 0:
        return np.inf

    excess_return = np.mean(returns) - rf_per_period
    sortino = excess_return / downside_std

    # Annualize
    sortino_annual = sortino * np.sqrt(periods_per_year)

    return float(sortino_annual)


def compute_calmar_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar Ratio

    Calmar Ratio = Annual Return / |Max Drawdown|

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Compute max drawdown
    dd_result = compute_drawdown_from_returns(returns)
    max_dd = abs(dd_result.max_drawdown)

    if max_dd == 0:
        return np.inf

    # Annual return
    total_return = np.sum(returns)
    years = len(returns) / periods_per_year
    annual_return = total_return / years if years > 0 else 0

    calmar = annual_return / max_dd

    return float(calmar)


# =============================================================================
# Tail Risk Metrics
# =============================================================================

def compute_tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
    """
    Compute tail ratio: |left tail| / right tail

    A ratio > 1 indicates heavier left tail (more downside risk).

    Args:
        returns: Array of returns
        percentile: Percentile for tail (default 5%)

    Returns:
        Tail ratio
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    left_tail = np.percentile(returns, percentile)
    right_tail = np.percentile(returns, 100 - percentile)

    if right_tail == 0:
        return np.inf if left_tail != 0 else 1.0

    return abs(left_tail) / right_tail


def compute_gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Compute Gain-to-Pain Ratio

    GtP = Sum of returns / Sum of |negative returns|

    Higher is better. Measures efficiency of gains vs pain of losses.

    Args:
        returns: Array of returns

    Returns:
        Gain-to-pain ratio
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    total_gain = np.sum(returns)
    total_pain = np.sum(np.abs(returns[returns < 0]))

    if total_pain == 0:
        return np.inf if total_gain > 0 else 0.0

    return float(total_gain / total_pain)


# =============================================================================
# Main Risk Calculator
# =============================================================================

class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculator

    Computes all risk metrics for a given return series or
    Monte Carlo simulation result.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute_all(
        self,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> RiskMetricsResult:
        """
        Compute all risk metrics

        Args:
            returns: Array of returns (daily or terminal)
            prices: Optional price series for drawdown

        Returns:
            RiskMetricsResult with all metrics
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]

        # VaR
        var_95 = compute_var(returns, confidence_level=0.95)
        var_99 = compute_var(returns, confidence_level=0.99)

        # Expected Shortfall
        es_95 = compute_expected_shortfall(returns, confidence_level=0.95)
        es_99 = compute_expected_shortfall(returns, confidence_level=0.99)

        # Drawdown (from prices if available, else from returns)
        if prices is not None:
            drawdown = compute_maximum_drawdown(prices)
        else:
            drawdown = compute_drawdown_from_returns(returns)

        # Distribution metrics
        mean_return = float(np.mean(returns))
        volatility = float(np.std(returns))
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))

        # Risk-adjusted metrics
        sharpe = compute_sharpe_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )
        sortino = compute_sortino_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )
        calmar = compute_calmar_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )

        # Tail metrics
        tail_ratio = compute_tail_ratio(returns)
        gain_to_pain = compute_gain_to_pain_ratio(returns)

        return RiskMetricsResult(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            drawdown=drawdown,
            mean_return=mean_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            tail_ratio=tail_ratio,
            gain_to_pain=gain_to_pain
        )

    def compare_simulations(
        self,
        standard_returns: np.ndarray,
        regime_returns: np.ndarray,
        standard_prices: Optional[np.ndarray] = None,
        regime_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare risk metrics between two simulations

        Args:
            standard_returns: Returns from standard MC
            regime_returns: Returns from regime-switching MC
            standard_prices: Optional prices from standard MC
            regime_prices: Optional prices from regime-switching MC

        Returns:
            Dict with 'standard', 'regime_switching', and 'difference' metrics
        """
        standard_metrics = self.compute_all(standard_returns, standard_prices)
        regime_metrics = self.compute_all(regime_returns, regime_prices)

        std_dict = standard_metrics.to_dict()
        reg_dict = regime_metrics.to_dict()

        # Compute differences
        diff_dict = {}
        for key in std_dict:
            std_val = std_dict[key]
            reg_val = reg_dict[key]

            if std_val != 0:
                diff_dict[key] = (reg_val - std_val) / abs(std_val) * 100  # Percentage
            else:
                diff_dict[key] = reg_val - std_val

        return {
            'standard': std_dict,
            'regime_switching': reg_dict,
            'difference_pct': diff_dict
        }

    def generate_report(
        self,
        comparison: Dict[str, Dict[str, float]],
        include_interpretation: bool = True
    ) -> str:
        """
        Generate a formatted risk comparison report

        Args:
            comparison: Output from compare_simulations()
            include_interpretation: Add financial interpretation

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RISK METRICS COMPARISON: STANDARD vs REGIME-SWITCHING MONTE CARLO")
        lines.append("=" * 80)
        lines.append("")

        # VaR comparison
        lines.append("VALUE AT RISK (VaR)")
        lines.append("-" * 40)
        lines.append(f"{'Metric':<25} {'Standard':<15} {'Regime-Switch':<15} {'Diff %':<10}")
        lines.append("-" * 40)

        var_metrics = ['var_95_hist', 'var_99_hist', 'var_95_cf', 'var_99_cf']
        for m in var_metrics:
            std = comparison['standard'][m]
            reg = comparison['regime_switching'][m]
            diff = comparison['difference_pct'][m]
            lines.append(f"{m:<25} {std:>12.4f}   {reg:>12.4f}   {diff:>8.1f}%")

        lines.append("")

        # ES comparison
        lines.append("EXPECTED SHORTFALL (ES)")
        lines.append("-" * 40)
        es_metrics = ['es_95', 'es_99']
        for m in es_metrics:
            std = comparison['standard'][m]
            reg = comparison['regime_switching'][m]
            diff = comparison['difference_pct'][m]
            lines.append(f"{m:<25} {std:>12.4f}   {reg:>12.4f}   {diff:>8.1f}%")

        lines.append("")

        # Distribution
        lines.append("DISTRIBUTION CHARACTERISTICS")
        lines.append("-" * 40)
        dist_metrics = ['mean_return', 'volatility', 'skewness', 'kurtosis']
        for m in dist_metrics:
            std = comparison['standard'][m]
            reg = comparison['regime_switching'][m]
            diff = comparison['difference_pct'][m]
            lines.append(f"{m:<25} {std:>12.4f}   {reg:>12.4f}   {diff:>8.1f}%")

        lines.append("")

        # Drawdown
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * 40)
        dd_metrics = ['max_drawdown', 'avg_drawdown']
        for m in dd_metrics:
            std = comparison['standard'][m]
            reg = comparison['regime_switching'][m]
            diff = comparison['difference_pct'][m]
            lines.append(f"{m:<25} {std:>12.4f}   {reg:>12.4f}   {diff:>8.1f}%")

        lines.append("")

        # Risk-adjusted
        lines.append("RISK-ADJUSTED METRICS")
        lines.append("-" * 40)
        ra_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        for m in ra_metrics:
            std = comparison['standard'][m]
            reg = comparison['regime_switching'][m]
            diff = comparison['difference_pct'][m]
            lines.append(f"{m:<25} {std:>12.4f}   {reg:>12.4f}   {diff:>8.1f}%")

        lines.append("")
        lines.append("=" * 80)

        if include_interpretation:
            lines.append("")
            lines.append("INTERPRETATION")
            lines.append("-" * 40)

            # Key findings
            var_diff = comparison['difference_pct']['var_99_hist']
            kurt_diff = comparison['difference_pct']['kurtosis']
            mdd_diff = comparison['difference_pct']['max_drawdown']

            lines.append("")
            lines.append("Key Findings:")
            lines.append(f"1. VaR(99%) is {abs(var_diff):.1f}% more extreme under regime-switching")
            lines.append(f"2. Kurtosis is {kurt_diff:.1f}% higher (fatter tails)")
            lines.append(f"3. Max drawdown is {abs(mdd_diff):.1f}% larger")
            lines.append("")
            lines.append("Implication:")
            lines.append("Standard Monte Carlo UNDERESTIMATES tail risk because it ignores:")
            lines.append("  - Volatility clustering (high vol follows high vol)")
            lines.append("  - Regime persistence (crises are prolonged)")
            lines.append("  - Non-normality from regime mixing")
            lines.append("")
            lines.append("For risk management, regime-switching provides more conservative,")
            lines.append("realistic estimates of potential losses.")
            lines.append("=" * 80)

        return "\n".join(lines)


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    """Demonstrate risk metrics computation"""

    print("=" * 60)
    print("RISK METRICS DEMONSTRATION")
    print("=" * 60)

    # Generate sample returns (mimicking two different distributions)
    np.random.seed(42)

    # Standard: Normal distribution
    standard_returns = np.random.normal(0.0004, 0.012, 10000)

    # Regime-switching: Mixture (heavier tails)
    regime_1 = np.random.normal(0.0006, 0.008, 4000)  # Low vol
    regime_2 = np.random.normal(0.0002, 0.015, 4000)  # Normal
    regime_3 = np.random.normal(-0.001, 0.03, 2000)   # Crisis
    regime_returns = np.concatenate([regime_1, regime_2, regime_3])
    np.random.shuffle(regime_returns)

    # Calculate metrics
    calculator = RiskMetricsCalculator()
    comparison = calculator.compare_simulations(standard_returns, regime_returns)

    # Print report
    report = calculator.generate_report(comparison)
    print(report)
