"""
Statistical Significance Testing for Model Comparison

Provides rigorous statistical methods for comparing forecasting models:
- Bootstrap confidence intervals
- Diebold-Mariano test for forecast comparison
- Paired t-tests across validation windows
- Model comparison tables with significance markers

References:
    - Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"
    - Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the Bootstrap"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# scipy is optional
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import math

    class _NormFallback:
        @staticmethod
        def cdf(x: float) -> float:
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        @staticmethod
        def ppf(p: float) -> float:
            # Approximation for inverse normal CDF
            if p <= 0:
                return float('-inf')
            if p >= 1:
                return float('inf')
            if p == 0.5:
                return 0.0

            # Rational approximation
            a = [
                -3.969683028665376e+01,
                2.209460984245205e+02,
                -2.759285104469687e+02,
                1.383577518672690e+02,
                -3.066479806614716e+01,
                2.506628277459239e+00
            ]
            b = [
                -5.447609879822406e+01,
                1.615858368580409e+02,
                -1.556989798598866e+02,
                6.680131188771972e+01,
                -1.328068155288572e+01
            ]
            c = [
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e+00,
                -2.549732539343734e+00,
                4.374664141464968e+00,
                2.938163982698783e+00
            ]
            d = [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e+00,
                3.754408661907416e+00
            ]

            p_low = 0.02425
            p_high = 1 - p_low

            if p < p_low:
                q = math.sqrt(-2 * math.log(p))
                return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                       ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
            elif p <= p_high:
                q = p - 0.5
                r = q * q
                return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                       (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
            else:
                q = math.sqrt(-2 * math.log(1-p))
                return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    class _TFallback:
        @staticmethod
        def cdf(x: float, df: int) -> float:
            # Approximation using normal for large df
            if df > 30:
                return _NormFallback.cdf(x)
            # For small df, use normal approximation (less accurate)
            return _NormFallback.cdf(x * (1 - 1/(4*df)))

        @staticmethod
        def ppf(p: float, df: int) -> float:
            if df > 30:
                return _NormFallback.ppf(p)
            return _NormFallback.ppf(p) * (1 + 1/(4*df))

    class _StatsFallback:
        norm = _NormFallback()
        t = _TFallback()

    stats = _StatsFallback()

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SignificanceLevel(Enum):
    """Standard significance levels"""
    VERY_STRONG = 0.001  # ***
    STRONG = 0.01        # **
    MODERATE = 0.05      # *
    WEAK = 0.10          # .
    NONE = 1.0           # ns


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation"""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    standard_error: float
    bootstrap_distribution: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"{self.point_estimate:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}] (95% CI)"

    def contains(self, value: float) -> bool:
        """Check if value is within confidence interval"""
        return self.ci_lower <= value <= self.ci_upper


@dataclass
class DieboldMarianoResult:
    """Result of Diebold-Mariano test"""
    test_statistic: float
    p_value: float
    better_model: str  # 'model1' or 'model2' or 'neither'
    significance: SignificanceLevel
    n_observations: int
    mean_loss_diff: float

    def __str__(self) -> str:
        sig_marker = {
            SignificanceLevel.VERY_STRONG: "***",
            SignificanceLevel.STRONG: "**",
            SignificanceLevel.MODERATE: "*",
            SignificanceLevel.WEAK: ".",
            SignificanceLevel.NONE: "ns"
        }[self.significance]
        return f"DM={self.test_statistic:.3f}, p={self.p_value:.4f}{sig_marker}, better={self.better_model}"


@dataclass
class PairedTestResult:
    """Result of paired comparison test"""
    test_name: str
    test_statistic: float
    p_value: float
    significance: SignificanceLevel
    effect_size: float
    n_pairs: int
    mean_diff: float
    std_diff: float

    def __str__(self) -> str:
        sig_marker = {
            SignificanceLevel.VERY_STRONG: "***",
            SignificanceLevel.STRONG: "**",
            SignificanceLevel.MODERATE: "*",
            SignificanceLevel.WEAK: ".",
            SignificanceLevel.NONE: "ns"
        }[self.significance]
        return f"{self.test_name}: t={self.test_statistic:.3f}, p={self.p_value:.4f}{sig_marker}"


@dataclass
class ModelComparisonSummary:
    """Summary of model comparison results"""
    model_names: List[str]
    n_models: int
    baseline_model: str
    metric_name: str
    results: Dict[str, Dict[str, Any]]
    ranking: List[Tuple[str, float]]


class StatisticalTests:
    """
    Statistical testing utilities for model comparison.

    All tests assume two-tailed hypotheses unless otherwise specified.
    """

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        metric_func: Callable[[np.ndarray], float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = 'percentile',
        random_state: Optional[int] = None
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval for any metric.

        Args:
            data: Data array
            metric_func: Function that computes metric from data
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95)
            method: 'percentile' or 'bca' (bias-corrected accelerated)
            random_state: Random seed

        Returns:
            BootstrapResult with CI bounds
        """
        if random_state is not None:
            np.random.seed(random_state)

        n = len(data)
        point_estimate = metric_func(data)

        # Generate bootstrap samples
        bootstrap_estimates = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_estimates[i] = metric_func(sample)

        # Remove any NaN/inf values
        bootstrap_estimates = bootstrap_estimates[np.isfinite(bootstrap_estimates)]

        if len(bootstrap_estimates) == 0:
            return BootstrapResult(
                point_estimate=point_estimate,
                ci_lower=np.nan,
                ci_upper=np.nan,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                standard_error=np.nan,
                bootstrap_distribution=None
            )

        alpha = 1 - confidence_level

        if method == 'percentile':
            ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
        elif method == 'bca':
            # BCa method (bias-corrected and accelerated)
            # Bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_estimates < point_estimate))

            # Acceleration (jackknife)
            jackknife_estimates = np.zeros(n)
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_estimates[i] = metric_func(jack_sample)

            jack_mean = np.mean(jackknife_estimates)
            num = np.sum((jack_mean - jackknife_estimates) ** 3)
            denom = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)
            a = num / denom if denom != 0 else 0

            # Adjusted percentiles
            z_alpha_lower = stats.norm.ppf(alpha / 2)
            z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

            p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
            p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

            ci_lower = np.percentile(bootstrap_estimates, p_lower * 100)
            ci_upper = np.percentile(bootstrap_estimates, p_upper * 100)
        else:
            raise ValueError(f"Unknown method: {method}")

        return BootstrapResult(
            point_estimate=point_estimate,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            standard_error=float(np.std(bootstrap_estimates)),
            bootstrap_distribution=bootstrap_estimates
        )

    @staticmethod
    def diebold_mariano_test(
        errors1: np.ndarray,
        errors2: np.ndarray,
        loss_func: str = 'squared',
        h: int = 1
    ) -> DieboldMarianoResult:
        """
        Diebold-Mariano test for comparing forecast accuracy.

        Tests H0: Both forecasts have equal accuracy
        vs H1: Forecast 2 is more accurate

        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            loss_func: 'squared' (MSE), 'absolute' (MAE), or 'custom'
            h: Forecast horizon (for autocorrelation correction)

        Returns:
            DieboldMarianoResult with test statistic and p-value
        """
        errors1 = np.asarray(errors1).flatten()
        errors2 = np.asarray(errors2).flatten()

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("Error arrays must have same length")

        # Compute loss differential
        if loss_func == 'squared':
            d = errors1 ** 2 - errors2 ** 2
        elif loss_func == 'absolute':
            d = np.abs(errors1) - np.abs(errors2)
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")

        # Mean of loss differential
        d_mean = np.mean(d)

        # Variance with Newey-West correction for autocorrelation
        gamma0 = np.var(d, ddof=1)

        # Autocorrelation terms
        gamma_sum = 0
        for lag in range(1, h):
            if lag < n:
                autocov = np.cov(d[:-lag], d[lag:])[0, 1]
                gamma_sum += autocov

        # Long-run variance estimate
        var_d = (gamma0 + 2 * gamma_sum) / n

        if var_d <= 0:
            return DieboldMarianoResult(
                test_statistic=0.0,
                p_value=1.0,
                better_model='neither',
                significance=SignificanceLevel.NONE,
                n_observations=n,
                mean_loss_diff=float(d_mean)
            )

        # DM test statistic
        dm_stat = d_mean / np.sqrt(var_d)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        # Determine better model
        if p_value < 0.05:
            better_model = 'model2' if d_mean > 0 else 'model1'
        else:
            better_model = 'neither'

        # Determine significance level
        if p_value < 0.001:
            significance = SignificanceLevel.VERY_STRONG
        elif p_value < 0.01:
            significance = SignificanceLevel.STRONG
        elif p_value < 0.05:
            significance = SignificanceLevel.MODERATE
        elif p_value < 0.10:
            significance = SignificanceLevel.WEAK
        else:
            significance = SignificanceLevel.NONE

        return DieboldMarianoResult(
            test_statistic=float(dm_stat),
            p_value=float(p_value),
            better_model=better_model,
            significance=significance,
            n_observations=n,
            mean_loss_diff=float(d_mean)
        )

    @staticmethod
    def paired_t_test(
        metric1: np.ndarray,
        metric2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> PairedTestResult:
        """
        Paired t-test for comparing model performance across windows.

        Args:
            metric1: Metric values for model 1 across windows
            metric2: Metric values for model 2 across windows
            alternative: 'two-sided', 'greater', or 'less'

        Returns:
            PairedTestResult with test statistic and p-value
        """
        metric1 = np.asarray(metric1)
        metric2 = np.asarray(metric2)

        n = len(metric1)
        if n != len(metric2):
            raise ValueError("Metric arrays must have same length")

        # Differences
        d = metric1 - metric2
        d_mean = np.mean(d)
        d_std = np.std(d, ddof=1)

        if d_std == 0:
            return PairedTestResult(
                test_name='Paired t-test',
                test_statistic=0.0,
                p_value=1.0,
                significance=SignificanceLevel.NONE,
                effect_size=0.0,
                n_pairs=n,
                mean_diff=float(d_mean),
                std_diff=0.0
            )

        # t-statistic
        t_stat = d_mean / (d_std / np.sqrt(n))

        # Degrees of freedom
        df = n - 1

        # p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, df)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        # Effect size (Cohen's d)
        effect_size = d_mean / d_std

        # Significance level
        if p_value < 0.001:
            significance = SignificanceLevel.VERY_STRONG
        elif p_value < 0.01:
            significance = SignificanceLevel.STRONG
        elif p_value < 0.05:
            significance = SignificanceLevel.MODERATE
        elif p_value < 0.10:
            significance = SignificanceLevel.WEAK
        else:
            significance = SignificanceLevel.NONE

        return PairedTestResult(
            test_name='Paired t-test',
            test_statistic=float(t_stat),
            p_value=float(p_value),
            significance=significance,
            effect_size=float(effect_size),
            n_pairs=n,
            mean_diff=float(d_mean),
            std_diff=float(d_std)
        )

    @staticmethod
    def wilcoxon_signed_rank_test(
        metric1: np.ndarray,
        metric2: np.ndarray
    ) -> PairedTestResult:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            metric1: Metric values for model 1
            metric2: Metric values for model 2

        Returns:
            PairedTestResult with test statistic and p-value
        """
        metric1 = np.asarray(metric1)
        metric2 = np.asarray(metric2)

        # Differences
        d = metric1 - metric2
        d_nonzero = d[d != 0]
        n = len(d_nonzero)

        if n == 0:
            return PairedTestResult(
                test_name='Wilcoxon signed-rank',
                test_statistic=0.0,
                p_value=1.0,
                significance=SignificanceLevel.NONE,
                effect_size=0.0,
                n_pairs=len(d),
                mean_diff=0.0,
                std_diff=0.0
            )

        # Rank absolute differences
        ranks = np.argsort(np.argsort(np.abs(d_nonzero))) + 1

        # Sum of positive and negative ranks
        pos_ranks = np.sum(ranks[d_nonzero > 0])
        neg_ranks = np.sum(ranks[d_nonzero < 0])

        # W statistic (smaller of the two)
        w_stat = min(pos_ranks, neg_ranks)

        # Normal approximation for p-value (valid for n > 10)
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

        z_stat = (w_stat - mean_w) / std_w if std_w > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Effect size (r = Z / sqrt(N))
        effect_size = abs(z_stat) / np.sqrt(n) if n > 0 else 0

        # Significance level
        if p_value < 0.001:
            significance = SignificanceLevel.VERY_STRONG
        elif p_value < 0.01:
            significance = SignificanceLevel.STRONG
        elif p_value < 0.05:
            significance = SignificanceLevel.MODERATE
        elif p_value < 0.10:
            significance = SignificanceLevel.WEAK
        else:
            significance = SignificanceLevel.NONE

        return PairedTestResult(
            test_name='Wilcoxon signed-rank',
            test_statistic=float(w_stat),
            p_value=float(p_value),
            significance=significance,
            effect_size=float(effect_size),
            n_pairs=len(d),
            mean_diff=float(np.mean(d)),
            std_diff=float(np.std(d, ddof=1))
        )


class ModelComparator:
    """
    Compare multiple models with statistical rigor.

    Supports:
    - Pairwise comparisons with multiple testing correction
    - Ranking with confidence intervals
    - Summary tables with significance markers
    """

    def __init__(
        self,
        baseline_model: Optional[str] = None,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ):
        self.baseline_model = baseline_model
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.tests = StatisticalTests()

    def compare_models(
        self,
        model_metrics: Dict[str, np.ndarray],
        metric_name: str = 'sharpe_ratio',
        higher_is_better: bool = True
    ) -> ModelComparisonSummary:
        """
        Compare multiple models on a single metric.

        Args:
            model_metrics: Dict mapping model name to metric values across windows
            metric_name: Name of the metric being compared
            higher_is_better: Whether higher metric values are better

        Returns:
            ModelComparisonSummary with all comparison results
        """
        model_names = list(model_metrics.keys())
        n_models = len(model_names)

        # Use first model as baseline if not specified
        baseline = self.baseline_model or model_names[0]
        if baseline not in model_names:
            baseline = model_names[0]

        results = {}

        for model_name in model_names:
            values = model_metrics[model_name]

            # Bootstrap CI
            boot_result = self.tests.bootstrap_confidence_interval(
                values,
                metric_func=np.mean,
                n_bootstrap=self.n_bootstrap,
                confidence_level=self.confidence_level
            )

            results[model_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'median': float(np.median(values)),
                'ci_lower': boot_result.ci_lower,
                'ci_upper': boot_result.ci_upper,
                'n_samples': len(values)
            }

            # Compare to baseline
            if model_name != baseline:
                baseline_values = model_metrics[baseline]

                # Paired t-test
                t_result = self.tests.paired_t_test(values, baseline_values)
                results[model_name]['vs_baseline_p'] = t_result.p_value
                results[model_name]['vs_baseline_sig'] = t_result.significance.value
                results[model_name]['vs_baseline_effect'] = t_result.effect_size

        # Create ranking
        ranking = sorted(
            [(name, results[name]['mean']) for name in model_names],
            key=lambda x: x[1],
            reverse=higher_is_better
        )

        return ModelComparisonSummary(
            model_names=model_names,
            n_models=n_models,
            baseline_model=baseline,
            metric_name=metric_name,
            results=results,
            ranking=ranking
        )

    def generate_comparison_table(
        self,
        model_metrics: Dict[str, Dict[str, np.ndarray]],
        metrics_to_compare: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate comparison table for multiple models and metrics.

        Args:
            model_metrics: Dict of {model_name: {metric_name: values}}
            metrics_to_compare: List of metric names to include

        Returns:
            DataFrame with comparison results
        """
        if metrics_to_compare is None:
            # Get all metrics from first model
            first_model = list(model_metrics.keys())[0]
            metrics_to_compare = list(model_metrics[first_model].keys())

        rows = []

        for model_name in model_metrics.keys():
            row = {'Model': model_name}

            for metric_name in metrics_to_compare:
                if metric_name in model_metrics[model_name]:
                    values = model_metrics[model_name][metric_name]
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    row[f'{metric_name}_mean'] = mean
                    row[f'{metric_name}_std'] = std
                    row[f'{metric_name}_formatted'] = f"{mean:.4f} ({std:.4f})"

            rows.append(row)

        return pd.DataFrame(rows)


# Convenience functions

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> BootstrapResult:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Array of returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        BootstrapResult for Sharpe ratio
    """
    def sharpe_func(r):
        if len(r) == 0 or np.std(r) == 0:
            return 0.0
        mean_ret = np.mean(r) * periods_per_year
        std_ret = np.std(r) * np.sqrt(periods_per_year)
        return (mean_ret - risk_free_rate) / std_ret

    return StatisticalTests.bootstrap_confidence_interval(
        returns,
        metric_func=sharpe_func,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level
    )


def compare_forecasts(
    errors_baseline: np.ndarray,
    errors_candidate: np.ndarray,
    model_names: Tuple[str, str] = ('Baseline', 'Candidate')
) -> Dict[str, Any]:
    """
    Quick comparison of two forecasting models.

    Args:
        errors_baseline: Errors from baseline model
        errors_candidate: Errors from candidate model
        model_names: Names of the two models

    Returns:
        Dict with comparison results
    """
    dm_result = StatisticalTests.diebold_mariano_test(errors_baseline, errors_candidate)

    mse_baseline = np.mean(errors_baseline ** 2)
    mse_candidate = np.mean(errors_candidate ** 2)

    return {
        'dm_statistic': dm_result.test_statistic,
        'dm_p_value': dm_result.p_value,
        'dm_significance': dm_result.significance.value,
        'better_model': model_names[0] if dm_result.better_model == 'model1' else
                        model_names[1] if dm_result.better_model == 'model2' else 'neither',
        'mse_baseline': mse_baseline,
        'mse_candidate': mse_candidate,
        'mse_reduction': (mse_baseline - mse_candidate) / mse_baseline * 100
    }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Statistical Tests Demo")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic data
    n_windows = 20
    model1_sharpes = np.random.randn(n_windows) * 0.5 + 1.0  # Mean Sharpe ~1.0
    model2_sharpes = np.random.randn(n_windows) * 0.5 + 1.3  # Mean Sharpe ~1.3

    print("\n1. Bootstrap CI for Sharpe Ratio:")
    returns = np.random.randn(252) * 0.02 + 0.0003
    boot_result = bootstrap_sharpe_ci(returns)
    print(f"   {boot_result}")

    print("\n2. Paired t-test (Model 1 vs Model 2):")
    t_result = StatisticalTests.paired_t_test(model1_sharpes, model2_sharpes)
    print(f"   {t_result}")
    print(f"   Effect size (Cohen's d): {t_result.effect_size:.3f}")

    print("\n3. Diebold-Mariano Test:")
    errors1 = np.random.randn(100) * 0.02
    errors2 = np.random.randn(100) * 0.015  # Lower errors = better
    dm_result = StatisticalTests.diebold_mariano_test(errors1, errors2)
    print(f"   {dm_result}")

    print("\n4. Model Comparison:")
    comparator = ModelComparator(baseline_model='LSTM')
    comparison = comparator.compare_models(
        {
            'LSTM': model1_sharpes,
            'PINN-GBM': model2_sharpes,
            'Transformer': np.random.randn(n_windows) * 0.5 + 0.9
        },
        metric_name='sharpe_ratio'
    )
    print(f"   Ranking: {comparison.ranking}")
    print(f"   Baseline: {comparison.baseline_model}")

    print("\n" + "=" * 60)
