"""
Regime-Based Performance Analysis

Provides comprehensive analysis of strategy performance across market regimes:
- Per-regime Sharpe, Sortino, drawdown
- Crisis period performance (2008, COVID, etc.)
- Regime transition analysis

References:
    - Ang, A. & Bekaert, G. (2002). "International Asset Allocation
      With Regime Shifts." RFS.
    - Guidolin, M. & Timmermann, A. (2007). "Asset Allocation Under
      Multivariate Regime Switching." JEF.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .regime_detector import (
    RegimeLabel,
    HMMRegimeDetector,
    VolatilityClusterDetector,
    RollingVolatilityDetector,
    get_regime_detector,
)
from .financial_metrics import FinancialMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Crisis Period Definitions
# ============================================================================

class CrisisPeriod(Enum):
    """Predefined crisis periods for analysis"""
    DOT_COM_CRASH = ("2000-03-01", "2002-10-01", "Dot-com Crash")
    GFC_2008 = ("2007-10-01", "2009-03-01", "Global Financial Crisis")
    EURO_CRISIS = ("2010-04-01", "2012-07-01", "European Debt Crisis")
    CHINA_DEVAL = ("2015-08-01", "2016-02-01", "China Devaluation")
    COVID_CRASH = ("2020-02-19", "2020-03-23", "COVID-19 Crash")
    COVID_RECOVERY = ("2020-03-23", "2020-08-01", "COVID-19 Recovery")
    INFLATION_2022 = ("2022-01-01", "2022-10-01", "2022 Inflation Crisis")

    def __init__(self, start_date: str, end_date: str, name: str):
        self.start_date = start_date
        self.end_date = end_date
        self.crisis_name = name


@dataclass
class RegimeMetrics:
    """Performance metrics for a specific regime"""
    regime: str
    count: int  # Number of observations
    proportion: float  # Proportion of total time
    mean_return: float  # Annualized mean return
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_duration: float  # Average days in regime


@dataclass
class CrisisMetrics:
    """Performance metrics for a crisis period"""
    crisis_name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    total_return: float
    max_drawdown: float
    recovery_days: Optional[int]  # Days to recover to pre-crisis level
    sharpe_ratio: float
    sortino_ratio: float
    benchmark_return: Optional[float] = None  # If benchmark provided
    alpha: Optional[float] = None  # Excess return vs benchmark


@dataclass
class TransitionMetrics:
    """Metrics for regime transitions"""
    from_regime: str
    to_regime: str
    count: int  # Number of transitions
    avg_return_after: float  # Average return after transition
    avg_vol_after: float  # Average volatility after transition
    transition_probability: float


# ============================================================================
# Per-Regime Analysis
# ============================================================================

def compute_regime_metrics(
    returns: Union[np.ndarray, pd.Series],
    regimes: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, RegimeMetrics]:
    """
    Compute comprehensive metrics for each regime.

    Args:
        returns: Return series
        regimes: Regime labels (0=LOW_VOL, 1=NORMAL, 2=HIGH_VOL)
        timestamps: Optional timestamps for duration calculation
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary mapping regime names to RegimeMetrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    results = {}
    regime_names = {0: 'LOW_VOL', 1: 'NORMAL', 2: 'HIGH_VOL'}

    total_obs = len(returns)

    for regime_id, regime_name in regime_names.items():
        mask = regimes == regime_id
        regime_returns = returns[mask]
        count = np.sum(mask)

        if count < 5:
            results[regime_name] = RegimeMetrics(
                regime=regime_name,
                count=count,
                proportion=count / total_obs if total_obs > 0 else 0,
                mean_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                sortino_ratio=np.nan,
                max_drawdown=np.nan,
                win_rate=np.nan,
                avg_duration=np.nan,
            )
            continue

        # Calculate metrics
        mean_ret = float(np.mean(regime_returns) * 252)
        vol = float(np.std(regime_returns, ddof=1) * np.sqrt(252))

        sharpe = FinancialMetrics.sharpe_ratio(
            regime_returns, risk_free_rate=risk_free_rate
        )
        sortino = FinancialMetrics.sortino_ratio(
            regime_returns, risk_free_rate=risk_free_rate
        )
        max_dd = FinancialMetrics.max_drawdown(regime_returns)
        win_rate = float(np.mean(regime_returns > 0))

        # Calculate average duration in regime
        avg_duration = _calculate_avg_regime_duration(regimes, regime_id)

        results[regime_name] = RegimeMetrics(
            regime=regime_name,
            count=count,
            proportion=count / total_obs,
            mean_return=mean_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_duration=avg_duration,
        )

    return results


def _calculate_avg_regime_duration(
    regimes: np.ndarray,
    target_regime: int
) -> float:
    """Calculate average duration of stays in a regime."""
    durations = []
    current_duration = 0

    for regime in regimes:
        if regime == target_regime:
            current_duration += 1
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return float(np.mean(durations)) if durations else 0.0


# ============================================================================
# Crisis Period Analysis
# ============================================================================

def compute_crisis_performance(
    returns: Union[np.ndarray, pd.Series],
    timestamps: pd.DatetimeIndex,
    crisis_periods: Optional[List[CrisisPeriod]] = None,
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, CrisisMetrics]:
    """
    Compute performance during predefined crisis periods.

    Args:
        returns: Return series
        timestamps: Timestamps for returns
        crisis_periods: List of crisis periods to analyze (default: all)
        benchmark_returns: Optional benchmark returns for alpha calculation

    Returns:
        Dictionary mapping crisis names to CrisisMetrics
    """
    if isinstance(returns, pd.Series):
        returns_series = returns
    else:
        returns_series = pd.Series(returns, index=timestamps)

    if benchmark_returns is not None:
        if isinstance(benchmark_returns, np.ndarray):
            benchmark_series = pd.Series(benchmark_returns, index=timestamps)
        else:
            benchmark_series = benchmark_returns
    else:
        benchmark_series = None

    if crisis_periods is None:
        crisis_periods = list(CrisisPeriod)

    results = {}

    for crisis in crisis_periods:
        try:
            start = pd.to_datetime(crisis.start_date)
            end = pd.to_datetime(crisis.end_date)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse crisis dates for {crisis}: {e}")
            continue

        # Filter to crisis period
        mask = (returns_series.index >= start) & (returns_series.index <= end)
        crisis_returns = returns_series[mask]

        if len(crisis_returns) < 5:
            continue

        # Calculate metrics
        total_return = float(np.prod(1 + crisis_returns) - 1)
        max_dd = FinancialMetrics.max_drawdown(crisis_returns.values)
        sharpe = FinancialMetrics.sharpe_ratio(crisis_returns.values)
        sortino = FinancialMetrics.sortino_ratio(crisis_returns.values)
        duration_days = (end - start).days

        # Calculate recovery time
        recovery_days = _calculate_recovery_days(
            returns_series, end, crisis_returns.iloc[0] if len(crisis_returns) > 0 else 0
        )

        # Benchmark comparison
        bench_return = None
        alpha = None
        if benchmark_series is not None:
            bench_crisis = benchmark_series[mask]
            if len(bench_crisis) > 0:
                bench_return = float(np.prod(1 + bench_crisis) - 1)
                alpha = total_return - bench_return

        results[crisis.crisis_name] = CrisisMetrics(
            crisis_name=crisis.crisis_name,
            start_date=start,
            end_date=end,
            duration_days=duration_days,
            total_return=total_return,
            max_drawdown=max_dd,
            recovery_days=recovery_days,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            benchmark_return=bench_return,
            alpha=alpha,
        )

    return results


def _calculate_recovery_days(
    returns: pd.Series,
    crisis_end: datetime,
    pre_crisis_cumret: float,
    max_days: int = 500
) -> Optional[int]:
    """Calculate days to recover to pre-crisis level after crisis ends."""
    post_crisis = returns[returns.index > crisis_end]

    if len(post_crisis) == 0:
        return None

    cumulative = np.cumprod(1 + post_crisis.values) - 1

    # Find first day where cumulative return exceeds pre-crisis level
    recovery_mask = cumulative >= pre_crisis_cumret

    if np.any(recovery_mask):
        return int(np.argmax(recovery_mask)) + 1

    return None  # Not recovered within max_days


# ============================================================================
# Regime Transition Analysis
# ============================================================================

def compute_transition_metrics(
    returns: Union[np.ndarray, pd.Series],
    regimes: np.ndarray,
    lookahead: int = 5,
) -> List[TransitionMetrics]:
    """
    Analyze what happens after regime transitions.

    Args:
        returns: Return series
        regimes: Regime labels
        lookahead: Days to analyze after transition

    Returns:
        List of TransitionMetrics for each transition type
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    regime_names = {0: 'LOW_VOL', 1: 'NORMAL', 2: 'HIGH_VOL'}
    results = []

    for from_regime in range(3):
        for to_regime in range(3):
            if from_regime == to_regime:
                continue

            # Find transition points
            transitions = []
            for i in range(1, len(regimes)):
                if regimes[i-1] == from_regime and regimes[i] == to_regime:
                    transitions.append(i)

            if len(transitions) < 3:
                continue

            # Calculate post-transition metrics
            post_returns = []
            post_vols = []

            for t in transitions:
                if t + lookahead <= len(returns):
                    post_returns.append(np.mean(returns[t:t+lookahead]))
                    post_vols.append(np.std(returns[t:t+lookahead]))

            if len(post_returns) == 0:
                continue

            # Transition probability
            from_count = np.sum(regimes == from_regime)
            trans_prob = len(transitions) / from_count if from_count > 0 else 0

            results.append(TransitionMetrics(
                from_regime=regime_names[from_regime],
                to_regime=regime_names[to_regime],
                count=len(transitions),
                avg_return_after=float(np.mean(post_returns) * 252),
                avg_vol_after=float(np.mean(post_vols) * np.sqrt(252)),
                transition_probability=trans_prob,
            ))

    return results


# ============================================================================
# Comprehensive Regime Analysis
# ============================================================================

def compute_full_regime_analysis(
    returns: Union[np.ndarray, pd.Series],
    timestamps: Optional[pd.DatetimeIndex] = None,
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    regime_method: str = 'rolling',
    risk_free_rate: float = 0.02,
) -> Dict[str, any]:
    """
    Perform comprehensive regime-based analysis.

    Args:
        returns: Return series
        timestamps: Timestamps (required for crisis analysis)
        benchmark_returns: Optional benchmark for comparison
        regime_method: 'hmm', 'kmeans', or 'rolling'
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with:
        - regime_metrics: Per-regime performance
        - crisis_metrics: Crisis period performance
        - transition_metrics: Regime transition analysis
        - detector: Fitted regime detector
        - regimes: Regime classifications
    """
    if isinstance(returns, pd.Series):
        if timestamps is None:
            timestamps = returns.index
        returns_arr = returns.values
    else:
        returns_arr = returns

    # Fit regime detector
    detector = get_regime_detector(method=regime_method)
    detector.fit(returns_arr)
    regimes = detector.predict(returns_arr)

    # Compute regime metrics
    regime_metrics = compute_regime_metrics(
        returns_arr, regimes, timestamps, risk_free_rate
    )

    # Compute crisis metrics (if timestamps provided)
    crisis_metrics = {}
    if timestamps is not None:
        crisis_metrics = compute_crisis_performance(
            returns_arr, timestamps, benchmark_returns=benchmark_returns
        )

    # Compute transition metrics
    transition_metrics = compute_transition_metrics(returns_arr, regimes)

    return {
        'regime_metrics': regime_metrics,
        'crisis_metrics': crisis_metrics,
        'transition_metrics': transition_metrics,
        'detector': detector,
        'regimes': regimes,
    }


def regime_metrics_to_dataframe(
    regime_metrics: Dict[str, RegimeMetrics]
) -> pd.DataFrame:
    """Convert regime metrics to DataFrame for display."""
    records = []
    for name, metrics in regime_metrics.items():
        records.append({
            'Regime': name,
            'Days': metrics.count,
            'Proportion': f"{metrics.proportion:.1%}",
            'Ann. Return': f"{metrics.mean_return:.1%}" if not np.isnan(metrics.mean_return) else "N/A",
            'Volatility': f"{metrics.volatility:.1%}" if not np.isnan(metrics.volatility) else "N/A",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}" if not np.isnan(metrics.sharpe_ratio) else "N/A",
            'Sortino': f"{metrics.sortino_ratio:.2f}" if not np.isnan(metrics.sortino_ratio) else "N/A",
            'Max DD': f"{metrics.max_drawdown:.1%}" if not np.isnan(metrics.max_drawdown) else "N/A",
            'Win Rate': f"{metrics.win_rate:.1%}" if not np.isnan(metrics.win_rate) else "N/A",
            'Avg Duration': f"{metrics.avg_duration:.1f}" if not np.isnan(metrics.avg_duration) else "N/A",
        })

    return pd.DataFrame(records)


def crisis_metrics_to_dataframe(
    crisis_metrics: Dict[str, CrisisMetrics]
) -> pd.DataFrame:
    """Convert crisis metrics to DataFrame for display."""
    records = []
    for name, metrics in crisis_metrics.items():
        records.append({
            'Crisis': name,
            'Start': metrics.start_date.strftime('%Y-%m-%d'),
            'End': metrics.end_date.strftime('%Y-%m-%d'),
            'Days': metrics.duration_days,
            'Return': f"{metrics.total_return:.1%}",
            'Max DD': f"{metrics.max_drawdown:.1%}",
            'Recovery': f"{metrics.recovery_days}d" if metrics.recovery_days else "N/A",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}",
            'Alpha': f"{metrics.alpha:.1%}" if metrics.alpha else "N/A",
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Regime Analysis Demo")
    print("=" * 60)

    # Generate synthetic data with regime changes
    np.random.seed(42)
    n_days = 1000

    # Create dates
    dates = pd.date_range(start='2018-01-01', periods=n_days, freq='B')

    # Simulate returns with regime structure
    returns = np.concatenate([
        np.random.randn(200) * 0.01,   # Low vol
        np.random.randn(200) * 0.02,   # Normal
        np.random.randn(100) * 0.04,   # High vol (crisis)
        np.random.randn(300) * 0.015,  # Normal
        np.random.randn(200) * 0.01,   # Low vol
    ])

    print(f"\nGenerated {n_days} days of synthetic returns")

    # Run full analysis
    analysis = compute_full_regime_analysis(
        returns=returns,
        timestamps=dates,
        regime_method='rolling',
    )

    # Display regime metrics
    print("\n" + "-" * 40)
    print("Per-Regime Performance:")
    regime_df = regime_metrics_to_dataframe(analysis['regime_metrics'])
    print(regime_df.to_string(index=False))

    # Display transition metrics
    print("\n" + "-" * 40)
    print("Regime Transitions:")
    for trans in analysis['transition_metrics'][:5]:
        print(f"  {trans.from_regime} -> {trans.to_regime}: "
              f"{trans.count} transitions, "
              f"avg return after: {trans.avg_return_after:.1%}")

    # Display regime distribution over time
    regimes = analysis['regimes']
    print("\n" + "-" * 40)
    print("Regime Distribution:")
    for regime_id, name in enumerate(['LOW_VOL', 'NORMAL', 'HIGH_VOL']):
        count = np.sum(regimes == regime_id)
        print(f"  {name}: {count} days ({count/len(regimes):.1%})")

    print("\n" + "=" * 60)
    print("Demo complete!")
