"""
Volatility Forecast Evaluation Metrics

Comprehensive metrics for evaluating volatility forecasting models,
including statistical measures, economic metrics, and diagnostic tests.

Metrics:
    Statistical:
        - MSE, MAE, RMSE (standard errors)
        - QLIKE (quasi-likelihood, preferred for variance)
        - HMSE (heteroskedasticity-adjusted MSE)
        - Mincer-Zarnowitz R² (forecast efficiency)
        - Log-likelihood (Gaussian assumption)

    Economic:
        - VaR breach rate with Kupiec test
        - Expected Shortfall accuracy
        - Volatility timing value
        - Risk-adjusted performance

    Diagnostic:
        - Diebold-Mariano test
        - Model Confidence Set (MCS)
        - Autocorrelation of forecast errors

References:
    - Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect
      Volatility Proxies." Journal of Econometrics.
    - Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
      Journal of Business & Economic Statistics.
    - Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The Model Confidence Set."
      Econometrica.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.stats import norm, chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

logger = get_logger(__name__)


@dataclass
class VolatilityMetricsResult:
    """Container for volatility forecast evaluation results."""
    mse: float
    mae: float
    rmse: float
    qlike: float
    hmse: float
    r2: float
    log_likelihood: float
    directional_accuracy: float
    additional_metrics: Dict[str, float] = None


# =============================================================================
# CORE STATISTICAL METRICS
# =============================================================================

class VolatilityMetrics:
    """
    Statistical metrics for volatility forecast evaluation.
    """

    @staticmethod
    def mse(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
    ) -> float:
        """
        Mean Squared Error on variance.

        MSE = (1/T) Σ (σ̂² - σ²)²

        Args:
            predicted_var: Predicted variance
            realized_var: Realized variance

        Returns:
            MSE value
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        valid = ~(np.isnan(predicted_var) | np.isnan(realized_var))
        if valid.sum() == 0:
            return np.nan

        return float(np.mean((predicted_var[valid] - realized_var[valid]) ** 2))

    @staticmethod
    def mae(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
    ) -> float:
        """
        Mean Absolute Error on variance.

        MAE = (1/T) Σ |σ̂² - σ²|
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        valid = ~(np.isnan(predicted_var) | np.isnan(realized_var))
        if valid.sum() == 0:
            return np.nan

        return float(np.mean(np.abs(predicted_var[valid] - realized_var[valid])))

    @staticmethod
    def rmse(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
    ) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(VolatilityMetrics.mse(predicted_var, realized_var)))

    @staticmethod
    def qlike(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """
        QLIKE (Quasi-Likelihood) Loss.

        QLIKE = (1/T) Σ [σ²/σ̂² - ln(σ²/σ̂²) - 1]

        This is the preferred loss function for variance forecasts because:
        1. It's scale-independent
        2. It's robust to heteroskedasticity
        3. It's consistent even when realized variance is a noisy proxy

        Reference:
            Patton, A.J. (2011). "Volatility Forecast Comparison Using
            Imperfect Volatility Proxies."

        Args:
            predicted_var: Predicted variance
            realized_var: Realized variance
            eps: Small constant for numerical stability

        Returns:
            QLIKE loss (lower is better)
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        valid = ~(np.isnan(predicted_var) | np.isnan(realized_var))

        pred = np.maximum(predicted_var[valid], eps)
        real = np.maximum(realized_var[valid], eps)

        ratio = real / pred
        qlike = ratio - np.log(ratio) - 1

        return float(np.mean(qlike))

    @staticmethod
    def hmse(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """
        Heteroskedasticity-adjusted MSE.

        HMSE = (1/T) Σ [(σ̂² - σ²) / σ²]²

        This scales errors by realized variance, giving equal weight
        to percentage errors across different volatility regimes.

        Args:
            predicted_var: Predicted variance
            realized_var: Realized variance
            eps: Small constant for numerical stability

        Returns:
            HMSE value
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        valid = ~(np.isnan(predicted_var) | np.isnan(realized_var))

        pred = predicted_var[valid]
        real = np.maximum(realized_var[valid], eps)

        relative_error = (pred - real) / real

        return float(np.mean(relative_error ** 2))

    @staticmethod
    def mincer_zarnowitz_r2(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
    ) -> float:
        """
        Mincer-Zarnowitz Regression R².

        Regresses realized on predicted:
            RV_t = α + β·PredVar_t + ε_t

        Under efficient forecasts:
            - α = 0 (no bias)
            - β = 1 (correct scaling)
            - R² measures forecast informativeness

        Returns:
            R² from M-Z regression
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        valid = ~(np.isnan(predicted_var) | np.isnan(realized_var))
        if valid.sum() < 10:
            return 0.0

        pred = predicted_var[valid]
        real = realized_var[valid]

        # Simple linear regression
        X = np.column_stack([np.ones(len(pred)), pred])
        try:
            coeffs = np.linalg.lstsq(X, real, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0

        # Calculate R²
        fitted = X @ coeffs
        ss_res = np.sum((real - fitted) ** 2)
        ss_tot = np.sum((real - real.mean()) ** 2)

        if ss_tot < 1e-10:
            return 0.0

        r2 = 1 - ss_res / ss_tot
        return float(max(0, min(1, r2)))

    @staticmethod
    def log_likelihood(
        returns: np.ndarray,
        predicted_var: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """
        Gaussian Log-Likelihood.

        LL = -(1/2T) Σ [ln(2πσ̂²) + r²/σ̂²]

        Assumes returns are conditionally Gaussian with predicted variance.

        Args:
            returns: Actual returns
            predicted_var: Predicted variance
            eps: Small constant for numerical stability

        Returns:
            Average log-likelihood (higher is better)
        """
        returns = np.asarray(returns).flatten()
        predicted_var = np.asarray(predicted_var).flatten()

        valid = ~(np.isnan(returns) | np.isnan(predicted_var))
        if valid.sum() == 0:
            return np.nan

        r = returns[valid]
        v = np.maximum(predicted_var[valid], eps)

        ll = -0.5 * (np.log(2 * np.pi * v) + r ** 2 / v)

        return float(np.mean(ll))

    @staticmethod
    def directional_accuracy(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
    ) -> float:
        """
        Directional accuracy of volatility changes.

        Measures how often the model correctly predicts whether
        volatility will increase or decrease.

        Args:
            predicted_var: Predicted variance
            realized_var: Realized variance

        Returns:
            Proportion of correct direction predictions (0 to 1)
        """
        predicted_var = np.asarray(predicted_var).flatten()
        realized_var = np.asarray(realized_var).flatten()

        if len(predicted_var) < 2:
            return 0.5

        # Direction of changes
        pred_diff = np.diff(predicted_var)
        real_diff = np.diff(realized_var)

        valid = ~(np.isnan(pred_diff) | np.isnan(real_diff))
        if valid.sum() == 0:
            return 0.5

        correct = np.sign(pred_diff[valid]) == np.sign(real_diff[valid])

        return float(np.mean(correct))

    @staticmethod
    def compute_all(
        predicted_var: np.ndarray,
        realized_var: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> VolatilityMetricsResult:
        """
        Compute all volatility forecast metrics.

        Args:
            predicted_var: Predicted variance
            realized_var: Realized variance
            returns: Actual returns (optional, for log-likelihood)

        Returns:
            VolatilityMetricsResult with all metrics
        """
        mse = VolatilityMetrics.mse(predicted_var, realized_var)
        mae = VolatilityMetrics.mae(predicted_var, realized_var)
        rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan
        qlike = VolatilityMetrics.qlike(predicted_var, realized_var)
        hmse = VolatilityMetrics.hmse(predicted_var, realized_var)
        r2 = VolatilityMetrics.mincer_zarnowitz_r2(predicted_var, realized_var)
        dir_acc = VolatilityMetrics.directional_accuracy(predicted_var, realized_var)

        if returns is not None:
            ll = VolatilityMetrics.log_likelihood(returns, predicted_var)
        else:
            ll = np.nan

        return VolatilityMetricsResult(
            mse=mse,
            mae=mae,
            rmse=rmse,
            qlike=qlike,
            hmse=hmse,
            r2=r2,
            log_likelihood=ll,
            directional_accuracy=dir_acc,
        )


# =============================================================================
# ECONOMIC METRICS
# =============================================================================

class EconomicVolatilityMetrics:
    """
    Economic metrics for evaluating volatility forecasts in
    risk management and trading contexts.
    """

    @staticmethod
    def var_breach_rate(
        returns: np.ndarray,
        predicted_vol: np.ndarray,
        confidence: float = 0.99,
    ) -> Dict[str, float]:
        """
        Value-at-Risk breach rate analysis.

        VaR_t = -z_α · σ̂_t

        Tests whether actual breach rate matches expected rate.
        Uses Kupiec test for unconditional coverage.

        Args:
            returns: Actual returns
            predicted_vol: Predicted volatility (NOT variance)
            confidence: VaR confidence level (e.g., 0.99 for 99% VaR)

        Returns:
            Dictionary with breach rate and test statistics
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available for VaR tests")
            return {'breach_rate': np.nan, 'expected_rate': 1 - confidence}

        returns = np.asarray(returns).flatten()
        predicted_vol = np.asarray(predicted_vol).flatten()

        valid = ~(np.isnan(returns) | np.isnan(predicted_vol))
        returns = returns[valid]
        predicted_vol = predicted_vol[valid]

        # VaR threshold (negative since it's a loss)
        z_score = norm.ppf(1 - confidence)
        var_threshold = z_score * predicted_vol  # Negative

        # Breaches (returns worse than VaR)
        breaches = returns < var_threshold
        breach_rate = breaches.mean()
        expected_rate = 1 - confidence

        T = len(returns)
        n = breaches.sum()

        # Kupiec test (likelihood ratio test for correct coverage)
        if 0 < n < T:
            lr_uc = -2 * (
                n * np.log(expected_rate) + (T - n) * np.log(1 - expected_rate)
                - n * np.log(n / T) - (T - n) * np.log(1 - n / T)
            )
            p_value = 1 - chi2.cdf(lr_uc, df=1)
        else:
            lr_uc = np.nan
            p_value = np.nan

        return {
            'breach_rate': float(breach_rate),
            'expected_rate': expected_rate,
            'breach_count': int(n),
            'total_obs': T,
            'kupiec_lr': float(lr_uc) if not np.isnan(lr_uc) else None,
            'kupiec_pvalue': float(p_value) if not np.isnan(p_value) else None,
            'passes_kupiec': p_value > 0.05 if not np.isnan(p_value) else None,
            'coverage_ratio': breach_rate / expected_rate if expected_rate > 0 else np.nan,
        }

    @staticmethod
    def expected_shortfall_accuracy(
        returns: np.ndarray,
        predicted_vol: np.ndarray,
        confidence: float = 0.975,
    ) -> Dict[str, float]:
        """
        Expected Shortfall (CVaR) accuracy.

        ES_α = E[R | R < VaR_α]

        Under Gaussian assumption:
        ES_α = -σ · φ(z_α) / (1 - α)

        Args:
            returns: Actual returns
            predicted_vol: Predicted volatility
            confidence: Confidence level

        Returns:
            Dictionary with ES metrics
        """
        if not HAS_SCIPY:
            return {'es_predicted': np.nan, 'es_realized': np.nan}

        returns = np.asarray(returns).flatten()
        predicted_vol = np.asarray(predicted_vol).flatten()

        valid = ~(np.isnan(returns) | np.isnan(predicted_vol))
        returns = returns[valid]
        predicted_vol = predicted_vol[valid]

        z = norm.ppf(1 - confidence)
        phi_z = norm.pdf(z)

        # Predicted ES (under Gaussian)
        predicted_es = -predicted_vol * phi_z / (1 - confidence)

        # VaR threshold
        var_threshold = z * predicted_vol

        # Realized ES (mean of breaching returns)
        breach_mask = returns < var_threshold

        if breach_mask.sum() > 0:
            realized_es = np.mean(returns[breach_mask])
        else:
            realized_es = np.nan

        # Average predicted ES
        avg_predicted_es = np.mean(predicted_es)

        return {
            'es_predicted': float(avg_predicted_es),
            'es_realized': float(realized_es) if not np.isnan(realized_es) else None,
            'es_ratio': float(realized_es / avg_predicted_es) if not np.isnan(realized_es) and avg_predicted_es != 0 else None,
            'n_breaches': int(breach_mask.sum()),
        }

    @staticmethod
    def volatility_timing_value(
        returns: np.ndarray,
        predicted_vol: np.ndarray,
        realized_vol: np.ndarray,
        target_vol: float = 0.15,
    ) -> Dict[str, float]:
        """
        Economic value of volatility timing.

        Compares performance of volatility-targeting strategy using
        predicted volatility vs oracle (realized volatility).

        Strategy: w_t = σ_target / σ̂_t (with leverage limits)

        Args:
            returns: Asset returns
            predicted_vol: Predicted volatility
            realized_vol: Realized volatility (oracle)
            target_vol: Target annual volatility (default: 15%)

        Returns:
            Dictionary with strategy performance metrics
        """
        returns = np.asarray(returns).flatten()
        predicted_vol = np.asarray(predicted_vol).flatten()
        realized_vol = np.asarray(realized_vol).flatten()

        valid = ~(np.isnan(returns) | np.isnan(predicted_vol) | np.isnan(realized_vol))
        returns = returns[valid]
        predicted_vol = predicted_vol[valid]
        realized_vol = realized_vol[valid]

        if len(returns) < 20:
            return {'sharpe_predicted': np.nan, 'sharpe_oracle': np.nan}

        # Daily target volatility
        target_vol_daily = target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

        # Strategy weights
        weights_pred = np.clip(target_vol_daily / np.maximum(predicted_vol, 1e-6), 0.25, 2.0)
        weights_oracle = np.clip(target_vol_daily / np.maximum(realized_vol, 1e-6), 0.25, 2.0)

        # Strategy returns
        strat_returns_pred = weights_pred * returns
        strat_returns_oracle = weights_oracle * returns

        # Performance metrics
        def compute_sharpe(rets):
            if len(rets) == 0 or np.std(rets) < 1e-10:
                return 0.0
            annual_ret = np.mean(rets) * TRADING_DAYS_PER_YEAR
            annual_vol = np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR)
            return (annual_ret - RISK_FREE_RATE) / annual_vol

        sharpe_pred = compute_sharpe(strat_returns_pred)
        sharpe_oracle = compute_sharpe(strat_returns_oracle)
        sharpe_buyhold = compute_sharpe(returns)

        # Value added
        value_vs_oracle = sharpe_pred / sharpe_oracle if sharpe_oracle > 0 else np.nan
        value_vs_buyhold = sharpe_pred - sharpe_buyhold

        # Realized volatility of strategy
        realized_vol_pred = np.std(strat_returns_pred) * np.sqrt(TRADING_DAYS_PER_YEAR)
        realized_vol_oracle = np.std(strat_returns_oracle) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return {
            'sharpe_predicted': float(sharpe_pred),
            'sharpe_oracle': float(sharpe_oracle),
            'sharpe_buyhold': float(sharpe_buyhold),
            'sharpe_capture': float(value_vs_oracle) if not np.isnan(value_vs_oracle) else None,
            'value_vs_buyhold': float(value_vs_buyhold),
            'realized_vol_strategy': float(realized_vol_pred),
            'realized_vol_oracle': float(realized_vol_oracle),
            'target_vol': target_vol,
            'vol_tracking_error': float(abs(realized_vol_pred - target_vol)),
        }


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

class VolatilityDiagnostics:
    """
    Diagnostic tests for comparing volatility forecasting models.
    """

    @staticmethod
    def diebold_mariano_test(
        errors1: np.ndarray,
        errors2: np.ndarray,
        horizon: int = 1,
        loss_fn: str = 'squared',
        alternative: str = 'two-sided',
    ) -> Dict[str, float]:
        """
        Diebold-Mariano test for equal predictive ability.

        H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)

        Positive DM statistic means model 2 is better.

        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            horizon: Forecast horizon (for HAC standard errors)
            loss_fn: 'squared' or 'absolute'
            alternative: 'two-sided', 'less' (model 1 better), 'greater' (model 2 better)

        Returns:
            Dictionary with test results
        """
        if not HAS_SCIPY:
            return {'dm_statistic': np.nan, 'p_value': np.nan}

        errors1 = np.asarray(errors1).flatten()
        errors2 = np.asarray(errors2).flatten()

        n = min(len(errors1), len(errors2))
        if n < 20:
            return {'dm_statistic': np.nan, 'p_value': np.nan, 'error': 'Insufficient data'}

        errors1 = errors1[:n]
        errors2 = errors2[:n]

        # Loss differential
        if loss_fn == 'squared':
            d = errors1 ** 2 - errors2 ** 2
        elif loss_fn == 'absolute':
            d = np.abs(errors1) - np.abs(errors2)
        else:
            d = errors1 ** 2 - errors2 ** 2

        d_bar = np.mean(d)

        # Newey-West variance estimator
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0

        for k in range(1, horizon):
            if k < n:
                gamma_k = np.cov(d[:-k], d[k:])[0, 1]
                gamma_sum += 2 * gamma_k

        var_d = (gamma_0 + gamma_sum) / n

        if var_d <= 0:
            return {'dm_statistic': np.nan, 'p_value': np.nan, 'error': 'Non-positive variance'}

        dm_stat = d_bar / np.sqrt(var_d)

        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        elif alternative == 'less':
            p_value = norm.cdf(dm_stat)
        else:  # greater
            p_value = 1 - norm.cdf(dm_stat)

        return {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'mean_loss_diff': float(d_bar),
            'model_1_better': dm_stat < 0 and p_value < 0.05,
            'model_2_better': dm_stat > 0 and p_value < 0.05,
            'significant': p_value < 0.05,
        }

    @staticmethod
    def model_confidence_set(
        losses: Dict[str, np.ndarray],
        alpha: float = 0.1,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Hansen's Model Confidence Set.

        Identifies the set of models that cannot be statistically
        distinguished from the best model at significance level alpha.

        Args:
            losses: Dictionary mapping model names to loss arrays
            alpha: Significance level
            n_bootstrap: Number of bootstrap samples
            seed: Random seed

        Returns:
            Dictionary with MCS results
        """
        if seed is not None:
            np.random.seed(seed)

        model_names = list(losses.keys())
        n_models = len(model_names)
        n = len(list(losses.values())[0])

        if n_models < 2:
            return {
                'included_models': model_names,
                'excluded_models': [],
                'p_values': {m: 1.0 for m in model_names},
            }

        loss_matrix = np.array([losses[m] for m in model_names])

        included = set(model_names)
        p_values = {m: 1.0 for m in model_names}
        excluded_order = []

        while len(included) > 1:
            current_models = list(included)
            current_idx = [model_names.index(m) for m in current_models]
            current_losses = loss_matrix[current_idx]

            # Average losses
            avg_losses = current_losses.mean(axis=1)

            # Find worst model
            worst_idx = np.argmax(avg_losses)
            worst_model = current_models[worst_idx]

            # Bootstrap test
            t_stats = []
            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(n, n, replace=True)
                boot_losses = current_losses[:, boot_idx]
                boot_avg = boot_losses.mean(axis=1)
                t_stats.append(boot_avg.max() - boot_avg.mean())

            # Test statistic
            test_stat = avg_losses.max() - avg_losses.mean()
            critical_value = np.percentile(t_stats, 100 * (1 - alpha))

            # P-value
            p_val = np.mean([t >= test_stat for t in t_stats])
            p_values[worst_model] = float(p_val)

            if test_stat > critical_value:
                included.remove(worst_model)
                excluded_order.append(worst_model)
            else:
                break

        return {
            'included_models': list(included),
            'excluded_models': excluded_order,
            'p_values': p_values,
            'alpha': alpha,
        }

    @staticmethod
    def forecast_error_autocorrelation(
        errors: np.ndarray,
        max_lag: int = 10,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Test for autocorrelation in forecast errors.

        Under efficient forecasts, errors should be serially uncorrelated.

        Args:
            errors: Forecast errors
            max_lag: Maximum lag to test

        Returns:
            Dictionary with ACF values and Ljung-Box test
        """
        errors = np.asarray(errors).flatten()
        errors = errors[~np.isnan(errors)]
        n = len(errors)

        if n < max_lag + 10:
            return {'acf': np.array([]), 'ljung_box_stat': np.nan}

        # ACF
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        errors_centered = errors - errors.mean()
        var_errors = np.var(errors_centered)

        for k in range(1, max_lag + 1):
            acf[k] = np.sum(errors_centered[k:] * errors_centered[:-k]) / ((n - k) * var_errors)

        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum(acf[1:] ** 2 / np.arange(n - 1, n - max_lag - 1, -1))

        if HAS_SCIPY:
            lb_pvalue = 1 - chi2.cdf(lb_stat, df=max_lag)
        else:
            lb_pvalue = np.nan

        return {
            'acf': acf,
            'ljung_box_stat': float(lb_stat),
            'ljung_box_pvalue': float(lb_pvalue) if not np.isnan(lb_pvalue) else None,
            'significant_autocorr': lb_pvalue < 0.05 if not np.isnan(lb_pvalue) else None,
        }


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_volatility_forecast(
    predicted_var: np.ndarray,
    realized_var: np.ndarray,
    returns: Optional[np.ndarray] = None,
    model_name: str = "Model",
    compute_economic: bool = True,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a volatility forecast.

    Args:
        predicted_var: Predicted variance
        realized_var: Realized variance
        returns: Actual returns (optional, for economic metrics)
        model_name: Name of the model
        compute_economic: Whether to compute economic metrics

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {'model': model_name}

    # Statistical metrics
    stats = VolatilityMetrics.compute_all(predicted_var, realized_var, returns)
    results.update({
        'mse': stats.mse,
        'mae': stats.mae,
        'rmse': stats.rmse,
        'qlike': stats.qlike,
        'hmse': stats.hmse,
        'r2': stats.r2,
        'log_likelihood': stats.log_likelihood,
        'directional_accuracy': stats.directional_accuracy,
    })

    # Economic metrics
    if compute_economic and returns is not None:
        predicted_vol = np.sqrt(np.maximum(predicted_var, 1e-10))
        realized_vol = np.sqrt(np.maximum(realized_var, 1e-10))

        # VaR analysis
        var_results = EconomicVolatilityMetrics.var_breach_rate(
            returns, predicted_vol, confidence=0.99
        )
        results.update({
            'var_breach_rate': var_results['breach_rate'],
            'var_expected_rate': var_results['expected_rate'],
            'kupiec_pvalue': var_results.get('kupiec_pvalue'),
        })

        # Volatility timing
        timing_results = EconomicVolatilityMetrics.volatility_timing_value(
            returns, predicted_vol, realized_vol
        )
        results.update({
            'sharpe_strategy': timing_results['sharpe_predicted'],
            'sharpe_oracle': timing_results['sharpe_oracle'],
            'sharpe_capture': timing_results.get('sharpe_capture'),
        })

    return results


def compare_volatility_models(
    predictions: Dict[str, np.ndarray],
    realized_var: np.ndarray,
    returns: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Compare multiple volatility forecasting models.

    Args:
        predictions: Dictionary mapping model names to predicted variances
        realized_var: Realized variance
        returns: Actual returns (optional)
        alpha: Significance level for MCS

    Returns:
        Dictionary with comparison results
    """
    # Evaluate each model
    results = []
    losses = {}

    for model_name, pred_var in predictions.items():
        eval_results = evaluate_volatility_forecast(
            pred_var, realized_var, returns, model_name
        )
        results.append(eval_results)

        # Store QLIKE losses for MCS
        pred = np.maximum(np.asarray(pred_var).flatten(), 1e-8)
        real = np.maximum(np.asarray(realized_var).flatten(), 1e-8)
        losses[model_name] = real / pred - np.log(real / pred) - 1

    results_df = pd.DataFrame(results)

    # Model Confidence Set
    mcs_results = VolatilityDiagnostics.model_confidence_set(losses, alpha=alpha)

    # Pairwise DM tests
    model_names = list(predictions.keys())
    dm_matrix = pd.DataFrame(
        index=model_names,
        columns=model_names,
        dtype=float
    )

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i < j:
                dm_result = VolatilityDiagnostics.diebold_mariano_test(
                    predictions[m1] - realized_var,
                    predictions[m2] - realized_var,
                )
                dm_matrix.loc[m1, m2] = dm_result['dm_statistic']
                dm_matrix.loc[m2, m1] = -dm_result['dm_statistic']
            elif i == j:
                dm_matrix.loc[m1, m2] = 0.0

    return {
        'results': results_df,
        'mcs': mcs_results,
        'dm_matrix': dm_matrix,
        'best_model_qlike': results_df.loc[results_df['qlike'].idxmin(), 'model'],
        'best_model_r2': results_df.loc[results_df['r2'].idxmax(), 'model'],
    }
