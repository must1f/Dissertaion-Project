"""
Baseline Volatility Forecasting Models

Traditional econometric models for volatility forecasting to serve
as benchmarks against neural network models.

Models:
    - NaiveRollingVol: Simple rolling window volatility
    - EWMA: Exponentially Weighted Moving Average (RiskMetrics)
    - GARCHModel: GARCH(1,1) with MLE estimation
    - GJRGARCHModel: GJR-GARCH for asymmetric volatility (leverage effect)
    - EGARCHModel: Exponential GARCH for asymmetry

References:
    - Bollerslev, T. (1986). "Generalized Autoregressive Conditional
      Heteroskedasticity." Journal of Econometrics.
    - Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). "On the Relation
      between the Expected Value and the Volatility of the Nominal Excess
      Return on Stocks." Journal of Finance.
    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns:
      A New Approach." Econometrica.
"""

import warnings
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)

# Try to import arch library for GARCH models
try:
    from arch import arch_model
    from arch.univariate import ConstantMean, GARCH, EGARCH, EWMAVariance
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch library not installed. GARCH models will use fallback implementations.")


@dataclass
class VolatilityForecast:
    """Container for volatility forecast results."""
    variance: np.ndarray  # Predicted variance
    volatility: np.ndarray  # Predicted volatility (sqrt of variance)
    params: Optional[Dict] = None  # Model parameters
    model_name: str = ""
    fitted: bool = False


# =============================================================================
# NAIVE ROLLING VOLATILITY
# =============================================================================

class NaiveRollingVol:
    """
    Naive Rolling Volatility Baseline.

    Predicts future variance as the rolling mean of squared returns
    over a lookback window.

    σ̂²_{t+h} = (1/K) Σᵢ₌₀^{K-1} r²_{t-i}

    This is the simplest possible volatility forecast and serves
    as a minimum benchmark.
    """

    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: Number of periods for rolling window (default: 20 days ≈ 1 month)
        """
        self.lookback = lookback
        self.name = f"Rolling({lookback})"
        self._fitted = False
        self._last_variance = None

    def fit(self, returns: np.ndarray) -> "NaiveRollingVol":
        """
        Fit model (just stores last variance for prediction).

        Args:
            returns: Array of returns
        """
        returns = np.asarray(returns).flatten()
        squared_returns = returns ** 2

        if len(squared_returns) >= self.lookback:
            self._last_variance = np.mean(squared_returns[-self.lookback:])
        else:
            self._last_variance = np.mean(squared_returns)

        self._fitted = True
        return self

    def predict(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """
        Predict variance using rolling window.

        Args:
            returns: Array of returns
            horizon: Forecast horizon (ignored for this model - uses persistence)

        Returns:
            VolatilityForecast with predicted variance
        """
        returns = np.asarray(returns).flatten()
        squared_returns = returns ** 2

        # Rolling variance
        variance = pd.Series(squared_returns).rolling(
            window=self.lookback,
            min_periods=1
        ).mean().values

        return VolatilityForecast(
            variance=variance,
            volatility=np.sqrt(variance),
            params={'lookback': self.lookback},
            model_name=self.name,
            fitted=True,
        )

    def forecast_one_step(self, returns: np.ndarray) -> float:
        """
        Make a single one-step ahead forecast.

        Args:
            returns: Recent returns

        Returns:
            Predicted variance
        """
        returns = np.asarray(returns).flatten()
        squared_returns = returns ** 2

        if len(squared_returns) >= self.lookback:
            return np.mean(squared_returns[-self.lookback:])
        else:
            return np.mean(squared_returns)


# =============================================================================
# EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA)
# =============================================================================

class EWMA:
    """
    Exponentially Weighted Moving Average Volatility Model.

    RiskMetrics approach where recent observations have more weight.

    σ̂²_t = λσ̂²_{t-1} + (1-λ)r²_{t-1}

    Standard λ = 0.94 for daily data (RiskMetrics recommendation).
    """

    def __init__(self, decay: float = 0.94, min_periods: int = 10):
        """
        Args:
            decay: Decay factor λ (default: 0.94 for daily data)
            min_periods: Minimum periods before starting EWMA
        """
        if not 0 < decay < 1:
            raise ValueError("Decay must be between 0 and 1")

        self.decay = decay
        self.min_periods = min_periods
        self.name = f"EWMA({decay})"
        self._fitted = False
        self._last_variance = None

    def fit(self, returns: np.ndarray) -> "EWMA":
        """
        Fit EWMA model.

        Args:
            returns: Array of returns
        """
        returns = np.asarray(returns).flatten()
        variance = self._compute_ewma_variance(returns)
        self._last_variance = variance[-1] if len(variance) > 0 else returns[-1] ** 2
        self._fitted = True
        return self

    def _compute_ewma_variance(self, returns: np.ndarray) -> np.ndarray:
        """Compute EWMA variance series."""
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with first squared return
        variance[0] = returns[0] ** 2

        # EWMA recursion
        for t in range(1, n):
            variance[t] = self.decay * variance[t-1] + (1 - self.decay) * returns[t-1] ** 2

        return variance

    def predict(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """
        Predict variance using EWMA.

        Args:
            returns: Array of returns
            horizon: Forecast horizon

        Returns:
            VolatilityForecast
        """
        returns = np.asarray(returns).flatten()
        variance = self._compute_ewma_variance(returns)

        # For multi-step, EWMA assumes variance persistence
        if horizon > 1:
            # h-step forecast is same as 1-step for EWMA (no mean reversion)
            pass

        return VolatilityForecast(
            variance=variance,
            volatility=np.sqrt(variance),
            params={'decay': self.decay},
            model_name=self.name,
            fitted=True,
        )

    def forecast_one_step(self, returns: np.ndarray) -> float:
        """Make single one-step forecast."""
        variance = self._compute_ewma_variance(returns)
        return variance[-1]

    @property
    def half_life(self) -> float:
        """Half-life of the EWMA in periods."""
        return np.log(0.5) / np.log(self.decay)


# =============================================================================
# GARCH(1,1) MODEL
# =============================================================================

class GARCHModel:
    """
    GARCH(1,1) Model for volatility forecasting.

    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    Where:
        ω > 0 (intercept)
        α ≥ 0 (shock impact)
        β ≥ 0 (persistence)
        α + β < 1 (stationarity)

    Properties:
        Unconditional variance: σ̄² = ω / (1 - α - β)
        Half-life of shocks: ln(0.5) / ln(α + β)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = 'normal',
        rescale: bool = True,
    ):
        """
        Args:
            p: GARCH lag order
            q: ARCH lag order
            dist: Error distribution ('normal', 't', 'skewt')
            rescale: Whether to rescale returns (recommended for numerical stability)
        """
        self.p = p
        self.q = q
        self.dist = dist
        self.rescale = rescale
        self.name = f"GARCH({p},{q})"

        self._fitted = False
        self._model = None
        self._result = None
        self._scale = 1.0
        self._params = {}

    def fit(self, returns: np.ndarray) -> "GARCHModel":
        """
        Fit GARCH model using Maximum Likelihood Estimation.

        Args:
            returns: Array of returns
        """
        returns = np.asarray(returns).flatten()

        if not HAS_ARCH:
            # Fallback to simple GARCH estimation
            return self._fit_fallback(returns)

        # Scale returns to percentage for better optimization
        if self.rescale:
            self._scale = 100.0
            returns_scaled = returns * self._scale
        else:
            returns_scaled = returns

        try:
            self._model = arch_model(
                returns_scaled,
                vol='Garch',
                p=self.p,
                q=self.q,
                dist=self.dist,
            )
            self._result = self._model.fit(disp='off', show_warning=False)

            # Extract parameters
            params = self._result.params
            self._params = {
                'omega': params.get('omega', params.iloc[0]) / (self._scale ** 2),
                'alpha': params.get('alpha[1]', params.iloc[1] if len(params) > 1 else 0.1),
                'beta': params.get('beta[1]', params.iloc[2] if len(params) > 2 else 0.85),
            }

            # Calculate derived quantities
            persistence = self._params['alpha'] + self._params['beta']
            if persistence < 1:
                self._params['unconditional_var'] = self._params['omega'] / (1 - persistence)
            else:
                self._params['unconditional_var'] = np.var(returns)

            self._params['persistence'] = persistence
            self._params['half_life'] = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

            self._fitted = True
            logger.info(f"GARCH fitted: ω={self._params['omega']:.2e}, "
                       f"α={self._params['alpha']:.4f}, β={self._params['beta']:.4f}")

        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}. Using fallback.")
            return self._fit_fallback(returns)

        return self

    def _fit_fallback(self, returns: np.ndarray) -> "GARCHModel":
        """
        Fallback GARCH estimation without arch library.

        Uses moment-based estimation.
        """
        returns = np.asarray(returns).flatten()
        squared_returns = returns ** 2

        # Simple moment-based estimation
        var_r = np.var(returns)
        autocorr_sq = np.corrcoef(squared_returns[1:], squared_returns[:-1])[0, 1]

        # Typical GARCH parameters
        alpha = max(0.05, min(0.15, autocorr_sq * 0.5)) if not np.isnan(autocorr_sq) else 0.1
        beta = 0.85
        persistence = alpha + beta

        if persistence >= 1:
            beta = 0.94 - alpha

        omega = var_r * (1 - persistence)

        self._params = {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': alpha + beta,
            'unconditional_var': var_r,
            'half_life': np.log(0.5) / np.log(alpha + beta) if alpha + beta < 1 else np.inf,
        }

        self._fitted = True
        return self

    def predict(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """
        Predict variance using fitted GARCH model.

        Args:
            returns: Array of returns
            horizon: Forecast horizon

        Returns:
            VolatilityForecast
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        returns = np.asarray(returns).flatten()

        if HAS_ARCH and self._result is not None:
            # Use arch library for prediction
            if self.rescale:
                returns_scaled = returns * self._scale
            else:
                returns_scaled = returns

            try:
                # Refit on new data if needed
                model = arch_model(
                    returns_scaled,
                    vol='Garch',
                    p=self.p,
                    q=self.q,
                )
                result = model.fit(disp='off', show_warning=False)

                # Get conditional variance
                variance = result.conditional_volatility ** 2 / (self._scale ** 2)

                return VolatilityForecast(
                    variance=variance,
                    volatility=np.sqrt(variance),
                    params=self._params,
                    model_name=self.name,
                    fitted=True,
                )
            except Exception:
                pass

        # Fallback: manual GARCH recursion
        return self._predict_fallback(returns, horizon)

    def _predict_fallback(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """Manual GARCH variance computation."""
        n = len(returns)
        variance = np.zeros(n)

        omega = self._params['omega']
        alpha = self._params['alpha']
        beta = self._params['beta']

        # Initialize with unconditional variance
        variance[0] = self._params['unconditional_var']

        # GARCH recursion
        for t in range(1, n):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

        return VolatilityForecast(
            variance=variance,
            volatility=np.sqrt(variance),
            params=self._params,
            model_name=self.name,
            fitted=True,
        )

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Multi-step ahead variance forecast.

        Uses the formula:
        E[σ²_{t+h}|I_t] = σ̄² + (α+β)^{h-1} (σ²_{t+1} - σ̄²)

        Args:
            horizon: Number of steps ahead

        Returns:
            Array of h-step forecasts
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        if self._result is not None:
            try:
                forecasts = self._result.forecast(horizon=horizon)
                return forecasts.variance.values[-1, :] / (self._scale ** 2)
            except Exception:
                pass

        # Fallback formula
        unconditional = self._params['unconditional_var']
        persistence = self._params['persistence']

        # Assume current variance is unconditional (conservative)
        current_var = unconditional

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = unconditional + (persistence ** h) * (current_var - unconditional)

        return forecasts

    def forecast_one_step(self, returns: np.ndarray) -> float:
        """Make single one-step variance forecast."""
        result = self.predict(returns, horizon=1)
        return result.variance[-1]


# =============================================================================
# GJR-GARCH MODEL (ASYMMETRIC)
# =============================================================================

class GJRGARCHModel:
    """
    GJR-GARCH Model for asymmetric volatility (leverage effect).

    σ²_t = ω + (α + γ·I_{r<0})·r²_{t-1} + β·σ²_{t-1}

    Where I_{r<0} = 1 if r_{t-1} < 0 (indicator for negative return).

    γ > 0 captures the leverage effect: negative returns have larger
    impact on future volatility than positive returns.
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,  # Asymmetric order
        q: int = 1,
        dist: str = 'normal',
    ):
        """
        Args:
            p: GARCH lag order
            o: Asymmetric lag order
            q: ARCH lag order
            dist: Error distribution
        """
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.name = f"GJR-GARCH({p},{o},{q})"

        self._fitted = False
        self._model = None
        self._result = None
        self._scale = 100.0
        self._params = {}

    def fit(self, returns: np.ndarray) -> "GJRGARCHModel":
        """Fit GJR-GARCH model."""
        returns = np.asarray(returns).flatten()

        if not HAS_ARCH:
            return self._fit_fallback(returns)

        returns_scaled = returns * self._scale

        try:
            self._model = arch_model(
                returns_scaled,
                vol='Garch',
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
            )
            self._result = self._model.fit(disp='off', show_warning=False)

            params = self._result.params
            self._params = {
                'omega': params.get('omega', params.iloc[0]) / (self._scale ** 2),
                'alpha': params.get('alpha[1]', params.iloc[1] if len(params) > 1 else 0.05),
                'gamma': params.get('gamma[1]', params.iloc[2] if len(params) > 2 else 0.1),
                'beta': params.get('beta[1]', params.iloc[3] if len(params) > 3 else 0.85),
            }

            # News impact
            alpha = self._params['alpha']
            gamma = self._params['gamma']
            beta = self._params['beta']

            self._params['persistence'] = alpha + beta + 0.5 * gamma
            self._params['positive_shock_impact'] = alpha
            self._params['negative_shock_impact'] = alpha + gamma

            self._fitted = True
            logger.info(f"GJR-GARCH fitted: γ={self._params['gamma']:.4f} "
                       f"(leverage effect: {self._params['negative_shock_impact']/self._params['positive_shock_impact']:.2f}x)")

        except Exception as e:
            logger.warning(f"GJR-GARCH fitting failed: {e}. Using fallback.")
            return self._fit_fallback(returns)

        return self

    def _fit_fallback(self, returns: np.ndarray) -> "GJRGARCHModel":
        """Fallback GJR-GARCH estimation."""
        returns = np.asarray(returns).flatten()

        # Estimate leverage effect from data
        neg_returns = returns[returns < 0]
        pos_returns = returns[returns > 0]

        neg_var = np.var(neg_returns) if len(neg_returns) > 10 else np.var(returns)
        pos_var = np.var(pos_returns) if len(pos_returns) > 10 else np.var(returns)

        leverage_ratio = neg_var / pos_var if pos_var > 0 else 1.5

        alpha = 0.05
        gamma = max(0.05, min(0.2, (leverage_ratio - 1) * 0.1))
        beta = 0.85
        omega = np.var(returns) * (1 - alpha - beta - 0.5 * gamma)

        self._params = {
            'omega': max(1e-8, omega),
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'persistence': alpha + beta + 0.5 * gamma,
            'positive_shock_impact': alpha,
            'negative_shock_impact': alpha + gamma,
        }

        self._fitted = True
        return self

    def predict(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """Predict variance using GJR-GARCH."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        returns = np.asarray(returns).flatten()
        n = len(returns)
        variance = np.zeros(n)

        omega = self._params['omega']
        alpha = self._params['alpha']
        gamma = self._params['gamma']
        beta = self._params['beta']

        # Initialize
        variance[0] = np.var(returns[:min(20, n)])

        # GJR-GARCH recursion
        for t in range(1, n):
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            variance[t] = omega + (alpha + gamma * indicator) * returns[t-1]**2 + beta * variance[t-1]

        return VolatilityForecast(
            variance=variance,
            volatility=np.sqrt(variance),
            params=self._params,
            model_name=self.name,
            fitted=True,
        )

    def forecast_one_step(self, returns: np.ndarray) -> float:
        """Make single one-step variance forecast."""
        result = self.predict(returns)
        return result.variance[-1]


# =============================================================================
# EGARCH MODEL
# =============================================================================

class EGARCHModel:
    """
    Exponential GARCH Model.

    log(σ²_t) = ω + α·g(z_{t-1}) + β·log(σ²_{t-1})

    Where:
        z_t = r_t / σ_t (standardized residual)
        g(z) = θz + γ(|z| - E|z|)

    Advantages:
        - No positivity constraints needed (models log variance)
        - Allows negative α (leverage effect)
        - Better handles extreme observations
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = 'normal',
    ):
        """
        Args:
            p: EGARCH lag order
            o: Asymmetric lag order
            q: ARCH lag order
            dist: Error distribution
        """
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.name = f"EGARCH({p},{o},{q})"

        self._fitted = False
        self._model = None
        self._result = None
        self._scale = 100.0
        self._params = {}

    def fit(self, returns: np.ndarray) -> "EGARCHModel":
        """Fit EGARCH model."""
        returns = np.asarray(returns).flatten()

        if not HAS_ARCH:
            logger.warning("EGARCH requires arch library. Using GJR-GARCH fallback.")
            self._fallback_model = GJRGARCHModel(p=self.p, q=self.q)
            self._fallback_model.fit(returns)
            self._params = self._fallback_model._params
            self._fitted = True
            return self

        returns_scaled = returns * self._scale

        try:
            self._model = arch_model(
                returns_scaled,
                vol='EGARCH',
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
            )
            self._result = self._model.fit(disp='off', show_warning=False)

            params = self._result.params
            self._params = {
                'omega': params.get('omega', 0),
                'alpha': params.get('alpha[1]', 0),
                'gamma': params.get('gamma[1]', 0),
                'beta': params.get('beta[1]', 0),
            }

            self._fitted = True
            logger.info(f"EGARCH fitted: ω={self._params['omega']:.4f}, "
                       f"α={self._params['alpha']:.4f}, γ={self._params['gamma']:.4f}")

        except Exception as e:
            logger.warning(f"EGARCH fitting failed: {e}. Using GJR-GARCH fallback.")
            self._fallback_model = GJRGARCHModel(p=self.p, q=self.q)
            self._fallback_model.fit(returns)
            self._params = self._fallback_model._params
            self._fitted = True

        return self

    def predict(self, returns: np.ndarray, horizon: int = 1) -> VolatilityForecast:
        """Predict variance using EGARCH."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        returns = np.asarray(returns).flatten()

        if HAS_ARCH and self._result is not None:
            returns_scaled = returns * self._scale

            try:
                model = arch_model(
                    returns_scaled,
                    vol='EGARCH',
                    p=self.p,
                    o=self.o,
                    q=self.q,
                )
                result = model.fit(disp='off', show_warning=False)

                variance = result.conditional_volatility ** 2 / (self._scale ** 2)

                return VolatilityForecast(
                    variance=variance,
                    volatility=np.sqrt(variance),
                    params=self._params,
                    model_name=self.name,
                    fitted=True,
                )
            except Exception:
                pass

        # Fallback to GJR-GARCH
        if hasattr(self, '_fallback_model'):
            return self._fallback_model.predict(returns, horizon)

        # Simple fallback
        variance = pd.Series(returns**2).ewm(span=20).mean().values

        return VolatilityForecast(
            variance=variance,
            volatility=np.sqrt(variance),
            params=self._params,
            model_name=self.name,
            fitted=True,
        )

    def forecast_one_step(self, returns: np.ndarray) -> float:
        """Make single one-step variance forecast."""
        result = self.predict(returns)
        return result.variance[-1]


# =============================================================================
# REALIZED VOLATILITY MODELS (HAR-RV)
# =============================================================================

class HARRV:
    """
    Heterogeneous Autoregressive model for Realized Volatility.

    RV_t = β₀ + β_d·RV_{t-1} + β_w·RV_{t-5:t-1} + β_m·RV_{t-22:t-1} + ε_t

    Where:
        RV_{t-5:t-1} = average of last 5 days
        RV_{t-22:t-1} = average of last 22 days

    This model captures heterogeneous market participants:
        - Short-term traders (daily)
        - Medium-term (weekly)
        - Long-term (monthly)

    Reference:
        Corsi, F. (2009). "A Simple Approximate Long-Memory Model
        of Realized Volatility." Journal of Financial Econometrics.
    """

    def __init__(self):
        self.name = "HAR-RV"
        self._fitted = False
        self._params = {}
        self._coefficients = None

    def fit(self, realized_variance: np.ndarray) -> "HARRV":
        """
        Fit HAR model to realized variance.

        Args:
            realized_variance: Array of realized variance
        """
        rv = np.asarray(realized_variance).flatten()
        n = len(rv)

        if n < 25:
            raise ValueError("Need at least 25 observations for HAR model")

        # Create features
        # Daily: RV_{t-1}
        rv_daily = rv[21:-1]

        # Weekly: average of RV_{t-5:t-1}
        rv_weekly = np.array([rv[i-5:i].mean() for i in range(22, n)])

        # Monthly: average of RV_{t-22:t-1}
        rv_monthly = np.array([rv[i-22:i].mean() for i in range(22, n)])

        # Target: RV_t
        target = rv[22:]

        # Stack features
        X = np.column_stack([
            np.ones(len(target)),  # Intercept
            rv_daily,
            rv_weekly,
            rv_monthly,
        ])

        # OLS estimation
        try:
            self._coefficients = np.linalg.lstsq(X, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to simple persistence
            self._coefficients = np.array([rv.mean() * 0.1, 0.3, 0.3, 0.3])

        self._params = {
            'intercept': self._coefficients[0],
            'beta_daily': self._coefficients[1],
            'beta_weekly': self._coefficients[2],
            'beta_monthly': self._coefficients[3],
        }

        self._fitted = True
        logger.info(f"HAR-RV fitted: β_d={self._params['beta_daily']:.4f}, "
                   f"β_w={self._params['beta_weekly']:.4f}, β_m={self._params['beta_monthly']:.4f}")

        return self

    def predict(self, realized_variance: np.ndarray) -> VolatilityForecast:
        """
        Predict using HAR model.

        Args:
            realized_variance: Array of realized variance

        Returns:
            VolatilityForecast
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        rv = np.asarray(realized_variance).flatten()
        n = len(rv)

        predictions = np.zeros(n)
        predictions[:22] = rv[:22]  # Use actual for warmup

        for i in range(22, n):
            rv_daily = rv[i-1]
            rv_weekly = rv[i-5:i].mean()
            rv_monthly = rv[i-22:i].mean()

            predictions[i] = (
                self._params['intercept']
                + self._params['beta_daily'] * rv_daily
                + self._params['beta_weekly'] * rv_weekly
                + self._params['beta_monthly'] * rv_monthly
            )

        # Ensure positive
        predictions = np.maximum(predictions, 1e-10)

        return VolatilityForecast(
            variance=predictions,
            volatility=np.sqrt(predictions),
            params=self._params,
            model_name=self.name,
            fitted=True,
        )

    def forecast_one_step(self, realized_variance: np.ndarray) -> float:
        """Make single one-step variance forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        rv = np.asarray(realized_variance).flatten()

        if len(rv) < 22:
            return rv[-1]

        rv_daily = rv[-1]
        rv_weekly = rv[-5:].mean()
        rv_monthly = rv[-22:].mean()

        prediction = (
            self._params['intercept']
            + self._params['beta_daily'] * rv_daily
            + self._params['beta_weekly'] * rv_weekly
            + self._params['beta_monthly'] * rv_monthly
        )

        return max(prediction, 1e-10)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_volatility_baseline(
    model_type: str,
    **kwargs,
) -> Union[NaiveRollingVol, EWMA, GARCHModel, GJRGARCHModel, EGARCHModel, HARRV]:
    """
    Factory function to create baseline volatility models.

    Args:
        model_type: One of 'rolling', 'ewma', 'garch', 'gjr_garch', 'egarch', 'har'
        **kwargs: Model-specific arguments

    Returns:
        Instantiated baseline model
    """
    model_type = model_type.lower().replace('-', '_')

    if model_type in ['rolling', 'naive', 'rolling_vol']:
        return NaiveRollingVol(lookback=kwargs.get('lookback', 20))

    elif model_type == 'ewma':
        return EWMA(
            decay=kwargs.get('decay', 0.94),
            min_periods=kwargs.get('min_periods', 10),
        )

    elif model_type in ['garch', 'garch11']:
        return GARCHModel(
            p=kwargs.get('p', 1),
            q=kwargs.get('q', 1),
            dist=kwargs.get('dist', 'normal'),
        )

    elif model_type in ['gjr_garch', 'gjr', 'tgarch']:
        return GJRGARCHModel(
            p=kwargs.get('p', 1),
            o=kwargs.get('o', 1),
            q=kwargs.get('q', 1),
            dist=kwargs.get('dist', 'normal'),
        )

    elif model_type == 'egarch':
        return EGARCHModel(
            p=kwargs.get('p', 1),
            o=kwargs.get('o', 1),
            q=kwargs.get('q', 1),
            dist=kwargs.get('dist', 'normal'),
        )

    elif model_type in ['har', 'har_rv']:
        return HARRV()

    else:
        raise ValueError(f"Unknown baseline model: {model_type}. "
                        f"Available: rolling, ewma, garch, gjr_garch, egarch, har")
