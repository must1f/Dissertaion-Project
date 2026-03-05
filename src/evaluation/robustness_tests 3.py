"""
Robustness Testing Suite

Comprehensive robustness testing for trading strategies:
- Pre-2008 train / Post-2008 test splits
- Crisis-only performance analysis
- Parameter sensitivity grid search
- Out-of-sample degradation analysis
- Monte Carlo simulation of strategy statistics

This module helps validate that strategy performance is robust and not
due to overfitting or favorable market conditions.

References:
    - Harvey, C. et al. (2016). "...and the Cross-Section of Expected
      Returns." RFS.
    - Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor
import warnings

from .financial_metrics import FinancialMetrics
from .walk_forward_validation import WalkForwardValidator, WalkForwardSummary
from .crisis_analyzer import CrisisAnalyzer, crisis_results_to_dataframe
from .regime_analysis import compute_regime_metrics, compute_full_regime_analysis
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RobustnessTestResult:
    """Result from a single robustness test"""
    test_name: str
    passed: bool
    score: float  # 0-1, higher is better
    details: Dict
    recommendations: List[str]


@dataclass
class RobustnessReport:
    """Comprehensive robustness report"""
    overall_score: float  # 0-1
    tests_passed: int
    tests_failed: int
    test_results: List[RobustnessTestResult]
    recommendations: List[str]
    is_robust: bool  # Overall assessment


# ============================================================================
# Pre/Post Split Testing
# ============================================================================

def test_pre_post_2008(
    returns: Union[np.ndarray, pd.Series],
    timestamps: pd.DatetimeIndex,
    split_date: str = "2008-09-15",  # Lehman collapse
    min_samples: int = 252,
) -> RobustnessTestResult:
    """
    Test strategy on pre-2008 vs post-2008 data.

    This tests if strategy works in both pre and post GFC environments.

    Args:
        returns: Strategy returns
        timestamps: Return timestamps
        split_date: Date to split (default: Lehman collapse)
        min_samples: Minimum samples required per period

    Returns:
        RobustnessTestResult
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns, index=timestamps)

    split = pd.to_datetime(split_date)

    pre_returns = returns[returns.index < split]
    post_returns = returns[returns.index >= split]

    recommendations = []

    if len(pre_returns) < min_samples or len(post_returns) < min_samples:
        return RobustnessTestResult(
            test_name="Pre/Post 2008 Split",
            passed=False,
            score=0.0,
            details={'error': 'Insufficient data'},
            recommendations=['Need more historical data for this test'],
        )

    # Calculate metrics for both periods
    pre_sharpe = FinancialMetrics.sharpe_ratio(pre_returns.values)
    post_sharpe = FinancialMetrics.sharpe_ratio(post_returns.values)

    pre_return = FinancialMetrics.total_return(pre_returns.values)
    post_return = FinancialMetrics.total_return(post_returns.values)

    pre_dd = FinancialMetrics.max_drawdown(pre_returns.values)
    post_dd = FinancialMetrics.max_drawdown(post_returns.values)

    # Scoring criteria:
    # - Both periods should have positive Sharpe
    # - Sharpe shouldn't degrade too much
    # - Max drawdown should be manageable in both

    both_positive = pre_sharpe > 0 and post_sharpe > 0
    sharpe_ratio = post_sharpe / pre_sharpe if pre_sharpe > 0.1 else 0
    drawdown_ok = pre_dd > -0.4 and post_dd > -0.4

    if not both_positive:
        recommendations.append("Strategy has negative Sharpe in one period")
    if sharpe_ratio < 0.5:
        recommendations.append("Significant Sharpe degradation post-2008")
    if not drawdown_ok:
        recommendations.append("Excessive drawdown in one period")

    # Calculate score
    score = 0.0
    if both_positive:
        score += 0.4
    if sharpe_ratio > 0.5:
        score += 0.3 * min(sharpe_ratio, 1.0)
    if drawdown_ok:
        score += 0.3

    passed = score >= 0.6

    return RobustnessTestResult(
        test_name="Pre/Post 2008 Split",
        passed=passed,
        score=score,
        details={
            'pre_sharpe': pre_sharpe,
            'post_sharpe': post_sharpe,
            'pre_return': pre_return,
            'post_return': post_return,
            'pre_max_dd': pre_dd,
            'post_max_dd': post_dd,
            'sharpe_ratio': sharpe_ratio,
        },
        recommendations=recommendations,
    )


# ============================================================================
# Crisis Performance Testing
# ============================================================================

def test_crisis_performance(
    returns: Union[np.ndarray, pd.Series],
    timestamps: pd.DatetimeIndex,
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    max_crisis_dd: float = -0.3,  # Max acceptable DD during crisis
    min_crises_outperformed: float = 0.5,  # % of crises to outperform
) -> RobustnessTestResult:
    """
    Test strategy performance during crisis periods.

    Args:
        returns: Strategy returns
        timestamps: Return timestamps
        benchmark_returns: Optional benchmark
        max_crisis_dd: Maximum acceptable drawdown during crisis
        min_crises_outperformed: Minimum % of crises to outperform benchmark

    Returns:
        RobustnessTestResult
    """
    analyzer = CrisisAnalyzer()
    results = analyzer.analyze(
        returns=returns,
        timestamps=timestamps,
        benchmark_returns=benchmark_returns,
    )

    valid_results = [r for r in results if r.in_sample and not np.isnan(r.strategy_return)]

    if len(valid_results) < 2:
        return RobustnessTestResult(
            test_name="Crisis Performance",
            passed=False,
            score=0.0,
            details={'error': 'Insufficient crisis periods in sample'},
            recommendations=['Strategy needs to cover more historical crises'],
        )

    recommendations = []

    # Analyze results
    crisis_dds = [r.max_drawdown for r in valid_results]
    crisis_returns = [r.strategy_return for r in valid_results]

    avg_crisis_dd = np.mean(crisis_dds)
    worst_crisis_dd = np.min(crisis_dds)
    avg_crisis_return = np.mean(crisis_returns)

    # Benchmark comparison
    if benchmark_returns is not None:
        comparison = analyzer.compare_vs_benchmark(results)
        outperform_pct = comparison.crises_outperformed / max(1, comparison.total_crises_analyzed)
    else:
        outperform_pct = 0.5  # Neutral if no benchmark

    # Scoring
    dd_score = 1.0 - min(1.0, abs(worst_crisis_dd) / abs(max_crisis_dd))
    outperform_score = min(1.0, outperform_pct / min_crises_outperformed)

    if worst_crisis_dd < max_crisis_dd:
        recommendations.append(f"Crisis drawdown ({worst_crisis_dd:.1%}) exceeds threshold")
    if outperform_pct < min_crises_outperformed:
        recommendations.append("Underperforming benchmark during too many crises")

    score = (dd_score * 0.6 + outperform_score * 0.4)
    passed = score >= 0.5 and worst_crisis_dd > max_crisis_dd

    return RobustnessTestResult(
        test_name="Crisis Performance",
        passed=passed,
        score=score,
        details={
            'n_crises_analyzed': len(valid_results),
            'avg_crisis_return': avg_crisis_return,
            'avg_crisis_dd': avg_crisis_dd,
            'worst_crisis_dd': worst_crisis_dd,
            'outperform_pct': outperform_pct,
        },
        recommendations=recommendations,
    )


# ============================================================================
# Walk-Forward Robustness
# ============================================================================

def test_walk_forward_robustness(
    returns: Union[np.ndarray, pd.Series],
    timestamps: Optional[pd.DatetimeIndex] = None,
    n_folds: int = 5,
    min_positive_pct: float = 0.7,  # % of folds with positive test Sharpe
    max_degradation: float = 0.6,   # Max train/test Sharpe ratio degradation
) -> RobustnessTestResult:
    """
    Test walk-forward validation robustness.

    Args:
        returns: Strategy returns
        timestamps: Optional timestamps
        n_folds: Number of walk-forward folds
        min_positive_pct: Minimum % of folds with positive OOS Sharpe
        max_degradation: Maximum acceptable avg overfitting ratio

    Returns:
        RobustnessTestResult
    """
    validator = WalkForwardValidator(
        method='anchored',
        n_folds=n_folds,
    )

    summary = validator.validate(returns, timestamps)

    recommendations = []

    if summary.n_folds < 3:
        return RobustnessTestResult(
            test_name="Walk-Forward Robustness",
            passed=False,
            score=0.0,
            details={'error': 'Insufficient folds generated'},
            recommendations=['Need more data for walk-forward validation'],
        )

    # Analyze results
    positive_pct = summary.positive_test_pct
    degradation = summary.degradation_pct
    consistency = 1 / (1 + summary.test_sharpe_consistency) if summary.test_sharpe_consistency < float('inf') else 0

    if positive_pct < min_positive_pct:
        recommendations.append(f"Only {positive_pct:.0%} of folds have positive OOS Sharpe")
    if degradation > max_degradation:
        recommendations.append(f"High OOS degradation ({degradation:.0%} of folds)")
    if summary.avg_test_sharpe < 0.3:
        recommendations.append("Average OOS Sharpe is weak")

    # Scoring
    positive_score = min(1.0, positive_pct / min_positive_pct)
    degradation_score = max(0, 1 - degradation / max_degradation)
    sharpe_score = min(1.0, max(0, summary.avg_test_sharpe / 1.0))

    score = (positive_score * 0.4 + degradation_score * 0.3 + sharpe_score * 0.3)
    passed = positive_pct >= min_positive_pct and summary.avg_test_sharpe > 0

    return RobustnessTestResult(
        test_name="Walk-Forward Robustness",
        passed=passed,
        score=score,
        details={
            'n_folds': summary.n_folds,
            'avg_train_sharpe': summary.avg_train_sharpe,
            'avg_test_sharpe': summary.avg_test_sharpe,
            'test_sharpe_std': summary.std_test_sharpe,
            'positive_test_pct': positive_pct,
            'degradation_pct': degradation,
        },
        recommendations=recommendations,
    )


# ============================================================================
# Regime Robustness
# ============================================================================

def test_regime_robustness(
    returns: Union[np.ndarray, pd.Series],
    timestamps: Optional[pd.DatetimeIndex] = None,
    min_regime_sharpe: float = 0.0,  # Minimum acceptable Sharpe in each regime
) -> RobustnessTestResult:
    """
    Test strategy performance across market regimes.

    Args:
        returns: Strategy returns
        timestamps: Optional timestamps
        min_regime_sharpe: Minimum Sharpe required in each regime

    Returns:
        RobustnessTestResult
    """
    analysis = compute_full_regime_analysis(
        returns=returns,
        timestamps=timestamps,
        regime_method='rolling',
    )

    regime_metrics = analysis['regime_metrics']
    recommendations = []

    # Check performance in each regime
    sharpes = {}
    for regime_name, metrics in regime_metrics.items():
        if not np.isnan(metrics.sharpe_ratio):
            sharpes[regime_name] = metrics.sharpe_ratio

    if len(sharpes) < 2:
        return RobustnessTestResult(
            test_name="Regime Robustness",
            passed=False,
            score=0.0,
            details={'error': 'Insufficient regime data'},
            recommendations=['Need data across multiple market regimes'],
        )

    # Check for negative Sharpe in any regime
    negative_regimes = [r for r, s in sharpes.items() if s < min_regime_sharpe]
    if negative_regimes:
        recommendations.append(f"Poor performance in regimes: {', '.join(negative_regimes)}")

    # Calculate consistency across regimes
    sharpe_values = list(sharpes.values())
    sharpe_std = np.std(sharpe_values)
    sharpe_mean = np.mean(sharpe_values)
    consistency = 1 - (sharpe_std / (abs(sharpe_mean) + 0.1))

    # Scoring
    positive_all = all(s >= min_regime_sharpe for s in sharpe_values)
    positive_score = 1.0 if positive_all else 0.5 * (1 - len(negative_regimes) / len(sharpes))
    consistency_score = max(0, consistency)

    score = positive_score * 0.6 + consistency_score * 0.4
    passed = positive_all and sharpe_mean > 0.3

    return RobustnessTestResult(
        test_name="Regime Robustness",
        passed=passed,
        score=score,
        details={
            'regime_sharpes': sharpes,
            'sharpe_consistency': consistency,
            'all_positive': positive_all,
        },
        recommendations=recommendations,
    )


# ============================================================================
# Monte Carlo Robustness
# ============================================================================

def test_monte_carlo_robustness(
    returns: Union[np.ndarray, pd.Series],
    n_simulations: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> RobustnessTestResult:
    """
    Monte Carlo simulation to test statistic robustness.

    Simulates bootstrap distributions of key metrics.

    Args:
        returns: Strategy returns
        n_simulations: Number of Monte Carlo simulations
        confidence: Confidence level for intervals
        seed: Random seed

    Returns:
        RobustnessTestResult
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if seed is not None:
        np.random.seed(seed)

    recommendations = []

    # Bootstrap Sharpe ratio
    sharpe_point, sharpe_lower, sharpe_upper = FinancialMetrics.bootstrapped_sharpe_ci(
        returns, confidence=confidence, n_bootstrap=n_simulations
    )

    # Bootstrap max drawdown
    dd_samples = []
    n = len(returns)
    for _ in range(n_simulations):
        sample = np.random.choice(returns, size=n, replace=True)
        dd_samples.append(FinancialMetrics.max_drawdown(sample))

    dd_mean = np.mean(dd_samples)
    dd_std = np.std(dd_samples)
    dd_lower = np.percentile(dd_samples, (1 - confidence) / 2 * 100)
    dd_upper = np.percentile(dd_samples, (1 + confidence) / 2 * 100)

    # Check if confidence interval includes zero (for Sharpe)
    sharpe_significant = sharpe_lower > 0

    if not sharpe_significant:
        recommendations.append("Sharpe ratio confidence interval includes zero")
    if sharpe_upper - sharpe_lower > 1.5:
        recommendations.append("Wide confidence interval - high uncertainty")

    # Scoring
    significance_score = 1.0 if sharpe_significant else 0.3
    precision_score = max(0, 1 - (sharpe_upper - sharpe_lower) / 3)
    point_score = min(1.0, max(0, sharpe_point / 1.5))

    score = significance_score * 0.5 + precision_score * 0.25 + point_score * 0.25
    passed = sharpe_significant and sharpe_point > 0.3

    return RobustnessTestResult(
        test_name="Monte Carlo Robustness",
        passed=passed,
        score=score,
        details={
            'sharpe_point': sharpe_point,
            'sharpe_ci': (sharpe_lower, sharpe_upper),
            'sharpe_significant': sharpe_significant,
            'max_dd_mean': dd_mean,
            'max_dd_ci': (dd_lower, dd_upper),
        },
        recommendations=recommendations,
    )


# ============================================================================
# Comprehensive Robustness Suite
# ============================================================================

def run_full_robustness_suite(
    returns: Union[np.ndarray, pd.Series],
    timestamps: pd.DatetimeIndex,
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    tests: Optional[List[str]] = None,
) -> RobustnessReport:
    """
    Run full robustness testing suite.

    Args:
        returns: Strategy returns
        timestamps: Return timestamps
        benchmark_returns: Optional benchmark
        tests: Specific tests to run (default: all)

    Returns:
        RobustnessReport with all results
    """
    all_tests = {
        'pre_post_2008': lambda: test_pre_post_2008(returns, timestamps),
        'crisis': lambda: test_crisis_performance(returns, timestamps, benchmark_returns),
        'walk_forward': lambda: test_walk_forward_robustness(returns, timestamps),
        'regime': lambda: test_regime_robustness(returns, timestamps),
        'monte_carlo': lambda: test_monte_carlo_robustness(returns),
    }

    if tests is None:
        tests = list(all_tests.keys())

    results = []
    for test_name in tests:
        if test_name in all_tests:
            try:
                result = all_tests[test_name]()
                results.append(result)
            except Exception as e:
                logger.error(f"Test '{test_name}' failed: {e}")
                results.append(RobustnessTestResult(
                    test_name=test_name,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    recommendations=[f"Test failed with error: {str(e)}"],
                ))

    # Aggregate results
    tests_passed = sum(1 for r in results if r.passed)
    tests_failed = len(results) - tests_passed
    overall_score = np.mean([r.score for r in results]) if results else 0

    # Collect all recommendations
    all_recommendations = []
    for r in results:
        all_recommendations.extend(r.recommendations)

    # Add overall recommendations
    if tests_passed < len(results) * 0.6:
        all_recommendations.append("Strategy fails multiple robustness tests - review methodology")
    if overall_score < 0.5:
        all_recommendations.append("Low overall robustness score - potential overfitting")

    is_robust = tests_passed >= len(results) * 0.7 and overall_score >= 0.5

    return RobustnessReport(
        overall_score=overall_score,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        test_results=results,
        recommendations=list(set(all_recommendations)),  # Dedupe
        is_robust=is_robust,
    )


def robustness_report_to_dataframe(report: RobustnessReport) -> pd.DataFrame:
    """Convert robustness report to DataFrame."""
    records = []

    for r in report.test_results:
        records.append({
            'Test': r.test_name,
            'Passed': 'Yes' if r.passed else 'No',
            'Score': f"{r.score:.2f}",
            'Key Finding': r.recommendations[0] if r.recommendations else 'No issues',
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Robustness Testing Suite Demo")
    print("=" * 60)

    # Generate synthetic returns
    np.random.seed(42)
    dates = pd.date_range(start='2005-01-01', end='2024-01-01', freq='B')
    n_days = len(dates)

    # Returns with some structure
    returns = np.random.randn(n_days) * 0.015 + 0.0003

    # Add crisis periods
    gfc_mask = (dates >= '2008-09-01') & (dates <= '2009-03-01')
    returns[gfc_mask] = np.random.randn(np.sum(gfc_mask)) * 0.04 - 0.002

    covid_mask = (dates >= '2020-02-20') & (dates <= '2020-03-23')
    returns[covid_mask] = np.random.randn(np.sum(covid_mask)) * 0.05 - 0.003

    returns = pd.Series(returns, index=dates)

    # Benchmark
    benchmark = np.random.randn(n_days) * 0.012 + 0.0002
    benchmark = pd.Series(benchmark, index=dates)

    print(f"\nGenerated {n_days} days of returns (2005-2024)")
    print(f"Overall Sharpe: {FinancialMetrics.sharpe_ratio(returns.values):.2f}")

    # Run suite
    print("\n" + "-" * 40)
    print("Running robustness tests...")

    report = run_full_robustness_suite(
        returns=returns,
        timestamps=dates,
        benchmark_returns=benchmark,
    )

    # Display results
    print("\n" + "-" * 40)
    print("Robustness Report:")
    print(f"  Overall Score: {report.overall_score:.2f}")
    print(f"  Tests Passed: {report.tests_passed}/{report.tests_passed + report.tests_failed}")
    print(f"  Is Robust: {'Yes' if report.is_robust else 'No'}")

    print("\nTest Results:")
    df = robustness_report_to_dataframe(report)
    print(df.to_string(index=False))

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations[:5]:
            print(f"  - {rec}")

    print("\n" + "=" * 60)
    print("Demo complete!")
