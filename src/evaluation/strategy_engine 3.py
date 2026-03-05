"""
Strategy Engine — Model-Agnostic Signal-to-Position Mapping

Implements the 4 mandatory signal-to-position mappings from the dissertation
evaluation specification.  Every strategy enforces a 1-period lag so that the
position used to earn r_t is decided using information available at t-1 only.

Integrates with the existing web-app pipeline via:
  training_service.py  →  compute_research_metrics()
                        →  compute_strategy_returns()   ← now delegates here
  backend/api/metrics   →  FinancialMetrics.*

Usage::

    from src.evaluation.strategy_engine import StrategyEngine

    positions = StrategyEngine.sign_strategy(predicted_returns)
    net_returns, stats = StrategyEngine.compute_net_returns(
        positions, actual_returns, cost_per_unit_turnover=0.001
    )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Configuration for strategy evaluation."""
    strategy: str = "sign"            # sign | threshold | vol_scaled | probability
    transaction_cost: float = 0.001   # cost per unit turnover (10 bps)
    threshold: float = 0.0            # for threshold strategy
    w_max: float = 1.0                # max abs position for vol_scaled
    risk_free_rate: float = 0.02      # annualised
    periods_per_year: int = 252


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Model-agnostic signal-to-position mapper with mandatory 1-period lagging.

    All ``*_strategy`` class-methods return an array of **lagged** positions
    the same length as the input, ready to be multiplied by actual returns::

        r_net = positions * actual_returns - c * |Δpositions|

    The lag is applied internally so callers never need to worry about it.
    """

    # ----- strategy A: sign ------------------------------------------------
    @staticmethod
    def sign_strategy(predicted_returns: np.ndarray) -> np.ndarray:
        """w_t = sign(r̂_{t+1}), lagged by 1 period.

        Long (+1) when the model predicts a positive return, short (−1)
        otherwise.  Position at time *t* is based on the prediction made
        at *t − 1*.
        """
        predicted_returns = np.asarray(predicted_returns).flatten()
        raw = np.sign(predicted_returns)
        return StrategyEngine._lag(raw)

    # ----- strategy B: threshold -------------------------------------------
    @staticmethod
    def threshold_strategy(
        predicted_returns: np.ndarray,
        tau: float = 0.0005,
    ) -> np.ndarray:
        """w_t = sign(r̂) if |r̂| > τ  else 0, lagged by 1 period.

        Reduces trading frequency by ignoring weak signals.
        τ should be tuned on the validation set, never on the test set.
        """
        predicted_returns = np.asarray(predicted_returns).flatten()
        raw = np.where(
            np.abs(predicted_returns) > tau,
            np.sign(predicted_returns),
            0.0,
        )
        return StrategyEngine._lag(raw)

    # ----- strategy C: volatility-scaled -----------------------------------
    @staticmethod
    def volatility_scaled_strategy(
        predicted_returns: np.ndarray,
        rolling_vol: np.ndarray,
        w_max: float = 1.0,
    ) -> np.ndarray:
        """w_t = clip(r̂ / σ̂, −w_max, w_max), lagged by 1 period.

        Confidence-weighted sizing: larger positions when expected return
        is high relative to recent volatility.
        ``rolling_vol`` should be estimated from past data only (e.g.
        trailing 20-day standard deviation of returns).
        """
        predicted_returns = np.asarray(predicted_returns).flatten()
        rolling_vol = np.asarray(rolling_vol).flatten()
        # guard against div-by-zero
        safe_vol = np.clip(rolling_vol, 1e-10, None)
        raw = np.clip(predicted_returns / safe_vol, -w_max, w_max)
        return StrategyEngine._lag(raw)

    # ----- strategy D: probability -----------------------------------------
    @staticmethod
    def probability_strategy(predicted_probs: np.ndarray) -> np.ndarray:
        """w_t = 2·p − 1, lagged by 1 period.

        For classification models that output P(r > 0).
        Produces positions in [−1, +1].
        """
        predicted_probs = np.asarray(predicted_probs).flatten()
        raw = 2.0 * predicted_probs - 1.0
        return StrategyEngine._lag(raw)

    # ----- helper: apply 1-period lag --------------------------------------
    @staticmethod
    def _lag(positions: np.ndarray) -> np.ndarray:
        """Shift positions forward by 1 period (mandatory look-ahead prevention).

        Position at index *t* becomes the position that was decided at *t − 1*.
        Index 0 has no prior decision → flat (0).
        """
        lagged = np.zeros_like(positions)
        lagged[1:] = positions[:-1]
        return lagged

    # ----- returns & turnover ----------------------------------------------
    @staticmethod
    def compute_net_returns(
        positions: np.ndarray,
        actual_returns: np.ndarray,
        cost_per_unit_turnover: float = 0.001,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute net strategy returns with turnover-based transaction costs.

        .. math::

            r^{\\text{net}}_t = w_{t-1} \\cdot r_t - c \\cdot |\\Delta w_t|

        Parameters
        ----------
        positions : array
            Already-lagged position weights (output of a ``*_strategy`` method).
        actual_returns : array
            Realised simple returns.
        cost_per_unit_turnover : float
            Transaction cost per unit of turnover (default 10 bps).

        Returns
        -------
        net_returns : array
            Net strategy returns after costs.
        trading_stats : dict
            Turnover, exposure, trade count statistics.
        """
        positions = np.asarray(positions).flatten()
        actual_returns = np.asarray(actual_returns).flatten()

        n = min(len(positions), len(actual_returns))
        positions = positions[:n]
        actual_returns = actual_returns[:n]

        # Gross returns
        gross_returns = positions * actual_returns

        # Turnover = |Δw_t|
        turnover = np.abs(np.diff(np.concatenate([[0.0], positions])))

        # Net returns
        net_returns = gross_returns - cost_per_unit_turnover * turnover

        # Trading statistics
        n_trades = int(np.sum(np.abs(np.diff(np.sign(positions))) > 0))
        avg_daily_turnover = float(np.mean(turnover))
        exposure_pct = float(np.mean(np.abs(positions) > 0)) * 100
        total_turnover = float(np.sum(turnover))

        trading_stats = {
            "n_trades": n_trades,
            "avg_daily_turnover": avg_daily_turnover,
            "total_turnover": total_turnover,
            "exposure_pct": exposure_pct,
            "pct_days_trading": float(np.mean(turnover > 0)) * 100,
            "avg_position_size": float(np.mean(np.abs(positions[positions != 0])))
            if np.any(positions != 0) else 0.0,
        }

        return net_returns, trading_stats

    # ----- convenience: run a named strategy end-to-end --------------------
    @staticmethod
    def run(
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray,
        config: Optional[StrategyConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Run a complete strategy: signal → position → net returns.

        Parameters
        ----------
        predicted_returns : array
            Model predictions (returns or probabilities depending on strategy).
        actual_returns : array
            Realised simple returns.
        config : StrategyConfig, optional
            Strategy configuration.  Defaults to sign strategy with 10 bps cost.

        Returns
        -------
        net_returns : array
        positions : array
        trading_stats : dict
        """
        if config is None:
            config = StrategyConfig()

        predicted_returns = np.asarray(predicted_returns).flatten()
        actual_returns = np.asarray(actual_returns).flatten()

        # Clip to realistic bounds
        actual_returns = np.clip(actual_returns, -0.20, 0.20)
        predicted_returns = np.clip(predicted_returns, -0.20, 0.20)

        # Select strategy
        if config.strategy == "sign":
            positions = StrategyEngine.sign_strategy(predicted_returns)
        elif config.strategy == "threshold":
            positions = StrategyEngine.threshold_strategy(
                predicted_returns, tau=config.threshold
            )
        elif config.strategy == "vol_scaled":
            # Compute trailing 20-day rolling vol from actual returns
            rolling_vol = _rolling_std(actual_returns, window=20)
            positions = StrategyEngine.volatility_scaled_strategy(
                predicted_returns, rolling_vol, w_max=config.w_max
            )
        elif config.strategy == "probability":
            positions = StrategyEngine.probability_strategy(predicted_returns)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")

        # Compute net returns
        net_returns, trading_stats = StrategyEngine.compute_net_returns(
            positions, actual_returns, config.transaction_cost
        )

        return net_returns, positions, trading_stats


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rolling_std(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling standard deviation (causal — only past data)."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.std(returns[i - window:i], ddof=1)
    # Fill initial NaNs with the first valid value
    first_valid = out[window] if window < n else 0.01
    out[:window] = first_valid
    return out
