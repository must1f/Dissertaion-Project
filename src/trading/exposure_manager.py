"""
Dynamic Exposure Management for Trading

Implements dissertation-grade exposure sizing with:
- Volatility targeting: w_t = w_hat / sigma_t with leverage cap
- Turnover optimization to reduce excessive trading
- Regime-aware exposure scaling
- Kelly-based risk budgeting

References:
    - Moskowitz, T. et al. (2012). "Time Series Momentum." JFE.
    - Harvey, C. et al. (2018). "The Impact of Volatility Targeting." JFM.
    - Barroso, P. & Santa-Clara, P. (2015). "Momentum has its moments." JFE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RegimeState(Enum):
    """Market regime states for exposure scaling"""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"


@dataclass
class ExposureResult:
    """Result of exposure calculation"""
    target_exposure: float  # Target portfolio exposure (can be > 1 with leverage)
    gross_exposure: float  # Total absolute exposure
    net_exposure: float  # Long - Short exposure
    leverage_ratio: float  # Gross / Net Asset Value
    volatility_scalar: float  # Volatility adjustment factor
    regime_scalar: float  # Regime adjustment factor
    turnover_cost: float  # Estimated turnover cost
    position_weights: Dict[str, float]  # Per-asset weights
    metadata: Dict = field(default_factory=dict)


@dataclass
class TurnoverResult:
    """Result of turnover optimization"""
    optimized_weights: Dict[str, float]
    original_weights: Dict[str, float]
    turnover: float  # Total turnover (0-2)
    turnover_cost: float  # Estimated cost from turnover
    trades_required: int
    positions_unchanged: int


class VolatilityTargetSizer:
    """
    Volatility Targeting Position Sizer

    Implements the volatility targeting approach from Moskowitz et al. (2012):
    w_t = (target_vol / realized_vol_t) * base_weight

    This approach:
    - Increases exposure when volatility is low
    - Decreases exposure when volatility is high
    - Improves risk-adjusted returns by maintaining consistent volatility contribution

    References:
        Harvey, C. et al. (2018). "The Impact of Volatility Targeting"
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized target vol
        lookback_period: int = 21,  # ~1 month for vol estimation
        min_leverage: float = 0.1,  # Minimum exposure
        max_leverage: float = 2.0,  # Maximum leverage (200%)
        vol_floor: float = 0.05,  # 5% minimum vol to avoid extreme leverage
        vol_ceiling: float = 0.60,  # 60% max vol to avoid near-zero exposure
        annualization_factor: int = 252,
        ewma_halflife: Optional[int] = None,  # Use EWMA if specified
    ):
        """
        Initialize volatility target sizer.

        Args:
            target_volatility: Annualized target portfolio volatility
            lookback_period: Days for rolling volatility estimation
            min_leverage: Minimum exposure floor
            max_leverage: Maximum leverage ceiling
            vol_floor: Minimum realized vol (prevents extreme leverage)
            vol_ceiling: Maximum realized vol (prevents near-zero exposure)
            annualization_factor: Trading days per year
            ewma_halflife: Half-life for exponential weighting (optional)
        """
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
        self.annualization_factor = annualization_factor
        self.ewma_halflife = ewma_halflife

        logger.info(
            f"VolatilityTargetSizer initialized: "
            f"target_vol={target_volatility:.1%}, "
            f"max_leverage={max_leverage:.1f}x"
        )

    def estimate_volatility(
        self,
        returns: Union[np.ndarray, pd.Series],
        use_ewma: bool = True
    ) -> float:
        """
        Estimate current realized volatility.

        Args:
            returns: Historical returns array/series
            use_ewma: Use exponentially weighted MA for vol estimation

        Returns:
            Annualized volatility estimate
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) < 5:
            logger.warning("Insufficient returns for vol estimation")
            return self.target_volatility

        # Use most recent data
        recent_returns = returns[-self.lookback_period:]

        if use_ewma and self.ewma_halflife:
            # Exponentially weighted volatility
            weights = np.exp(-np.log(2) * np.arange(len(recent_returns))[::-1] / self.ewma_halflife)
            weights /= weights.sum()
            vol = np.sqrt(np.sum(weights * recent_returns**2))
        else:
            # Simple rolling volatility
            vol = np.std(recent_returns, ddof=1)

        # Annualize
        vol_annualized = vol * np.sqrt(self.annualization_factor)

        return float(vol_annualized)

    def calculate_exposure(
        self,
        returns: Union[np.ndarray, pd.Series],
        base_weight: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate target exposure using volatility targeting.

        The core formula: w_t = (target_vol / realized_vol_t) * base_weight

        Args:
            returns: Historical returns for vol estimation
            base_weight: Base position weight (before vol adjustment)

        Returns:
            Tuple of (target_exposure, volatility_scalar)
        """
        # Estimate current volatility
        realized_vol = self.estimate_volatility(returns)

        # Apply floor and ceiling
        realized_vol = np.clip(realized_vol, self.vol_floor, self.vol_ceiling)

        # Calculate volatility scalar
        vol_scalar = self.target_volatility / realized_vol

        # Calculate target exposure
        target_exposure = base_weight * vol_scalar

        # Apply leverage limits
        target_exposure = np.clip(
            target_exposure,
            self.min_leverage * base_weight,
            self.max_leverage * base_weight
        )

        logger.debug(
            f"Vol targeting: realized_vol={realized_vol:.2%}, "
            f"scalar={vol_scalar:.2f}, exposure={target_exposure:.2f}"
        )

        return float(target_exposure), float(vol_scalar)


class TurnoverOptimizer:
    """
    Turnover Optimization

    Reduces excessive trading by:
    - Implementing no-trade bands around target weights
    - Penalizing turnover in optimization
    - Batching small trades

    This approach reduces transaction costs and improves net returns.
    """

    def __init__(
        self,
        no_trade_threshold: float = 0.02,  # 2% band before rebalancing
        max_turnover_per_period: float = 0.5,  # Max 50% portfolio turnover
        transaction_cost: float = 0.001,  # 10 bps per trade
        min_trade_size: float = 0.01,  # Min 1% trade to execute
    ):
        """
        Initialize turnover optimizer.

        Args:
            no_trade_threshold: Deviation from target to trigger rebalance
            max_turnover_per_period: Maximum allowed turnover per period
            transaction_cost: Estimated cost per unit of turnover
            min_trade_size: Minimum trade size to execute
        """
        self.no_trade_threshold = no_trade_threshold
        self.max_turnover_per_period = max_turnover_per_period
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size

    def optimize_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> TurnoverResult:
        """
        Optimize trades to minimize turnover while tracking targets.

        Uses no-trade bands: only trade if deviation exceeds threshold.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            TurnoverResult with optimized weights
        """
        optimized = {}
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        total_turnover = 0.0
        trades_required = 0
        positions_unchanged = 0

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            deviation = target - current

            if abs(deviation) < self.no_trade_threshold:
                # Within no-trade band - keep current weight
                optimized[asset] = current
                positions_unchanged += 1
            elif abs(deviation) < self.min_trade_size:
                # Trade too small - keep current
                optimized[asset] = current
                positions_unchanged += 1
            else:
                # Execute trade
                optimized[asset] = target
                total_turnover += abs(deviation)
                trades_required += 1

        # Cap total turnover if needed
        if total_turnover > self.max_turnover_per_period:
            scale = self.max_turnover_per_period / total_turnover
            for asset in optimized:
                current = current_weights.get(asset, 0.0)
                deviation = optimized[asset] - current
                optimized[asset] = current + deviation * scale
            total_turnover = self.max_turnover_per_period

        turnover_cost = total_turnover * self.transaction_cost

        return TurnoverResult(
            optimized_weights=optimized,
            original_weights=dict(target_weights),
            turnover=total_turnover,
            turnover_cost=turnover_cost,
            trades_required=trades_required,
            positions_unchanged=positions_unchanged
        )


class RegimeAwareExposureScaler:
    """
    Regime-Aware Exposure Scaling

    Adjusts exposure based on detected market regime:
    - LOW_VOL: Can increase exposure (regime_scalar > 1)
    - NORMAL: Neutral (regime_scalar = 1)
    - HIGH_VOL: Reduce exposure (regime_scalar < 1)
    - CRISIS: Significantly reduce exposure

    Based on research showing conditional volatility persistence.
    """

    def __init__(
        self,
        regime_scalars: Optional[Dict[RegimeState, float]] = None,
        vol_percentiles: Tuple[float, float, float] = (0.25, 0.75, 0.95),
        lookback_period: int = 252,  # 1 year for regime classification
    ):
        """
        Initialize regime-aware scaler.

        Args:
            regime_scalars: Override default scalars per regime
            vol_percentiles: Percentiles for LOW/NORMAL/HIGH/CRISIS thresholds
            lookback_period: Days for regime classification
        """
        self.regime_scalars = regime_scalars or {
            RegimeState.LOW_VOL: 1.2,   # 20% increase
            RegimeState.NORMAL: 1.0,    # No change
            RegimeState.HIGH_VOL: 0.7,  # 30% reduction
            RegimeState.CRISIS: 0.3,    # 70% reduction
        }
        self.vol_percentiles = vol_percentiles
        self.lookback_period = lookback_period
        self._vol_thresholds: Optional[Tuple[float, float, float]] = None

    def calibrate_thresholds(
        self,
        historical_volatility: Union[np.ndarray, pd.Series]
    ):
        """
        Calibrate regime thresholds from historical volatility.

        Args:
            historical_volatility: Historical volatility series
        """
        if isinstance(historical_volatility, pd.Series):
            historical_volatility = historical_volatility.values

        historical_volatility = historical_volatility[~np.isnan(historical_volatility)]

        if len(historical_volatility) < 50:
            logger.warning("Insufficient history for regime calibration")
            self._vol_thresholds = (0.10, 0.20, 0.35)  # Defaults
            return

        self._vol_thresholds = tuple(
            np.percentile(historical_volatility, p * 100)
            for p in self.vol_percentiles
        )

        logger.info(
            f"Regime thresholds calibrated: "
            f"LOW<{self._vol_thresholds[0]:.1%}, "
            f"HIGH>{self._vol_thresholds[1]:.1%}, "
            f"CRISIS>{self._vol_thresholds[2]:.1%}"
        )

    def classify_regime(self, current_volatility: float) -> RegimeState:
        """
        Classify current market regime.

        Args:
            current_volatility: Current volatility estimate

        Returns:
            RegimeState classification
        """
        if self._vol_thresholds is None:
            # Use defaults
            thresholds = (0.10, 0.20, 0.35)
        else:
            thresholds = self._vol_thresholds

        if current_volatility < thresholds[0]:
            return RegimeState.LOW_VOL
        elif current_volatility < thresholds[1]:
            return RegimeState.NORMAL
        elif current_volatility < thresholds[2]:
            return RegimeState.HIGH_VOL
        else:
            return RegimeState.CRISIS

    def get_exposure_scalar(
        self,
        current_volatility: float,
        override_regime: Optional[RegimeState] = None
    ) -> Tuple[float, RegimeState]:
        """
        Get exposure scaling factor for current conditions.

        Args:
            current_volatility: Current volatility estimate
            override_regime: Optionally override detected regime

        Returns:
            Tuple of (scalar, regime_state)
        """
        if override_regime:
            regime = override_regime
        else:
            regime = self.classify_regime(current_volatility)

        scalar = self.regime_scalars[regime]

        logger.debug(f"Regime: {regime.value}, scalar: {scalar:.2f}")

        return scalar, regime


class ExposureManager:
    """
    Central Exposure Management

    Combines volatility targeting, turnover optimization, and regime-aware
    scaling into a unified exposure management system.

    Key features:
    - Volatility-targeted position sizing
    - Regime-aware exposure scaling
    - Turnover optimization to reduce trading costs
    - Position-level and portfolio-level constraints

    Usage:
        manager = ExposureManager(target_volatility=0.15)
        result = manager.calculate_exposure(
            returns=historical_returns,
            current_weights={'AAPL': 0.3, 'MSFT': 0.3},
            target_weights={'AAPL': 0.4, 'MSFT': 0.4}
        )
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        min_leverage: float = 0.1,
        no_trade_threshold: float = 0.02,
        transaction_cost: float = 0.001,
        regime_aware: bool = True,
        vol_lookback: int = 21,
        ewma_halflife: Optional[int] = 10,
    ):
        """
        Initialize exposure manager.

        Args:
            target_volatility: Annualized target portfolio volatility
            max_leverage: Maximum gross exposure
            min_leverage: Minimum gross exposure
            no_trade_threshold: No-trade band width
            transaction_cost: Estimated transaction cost per unit
            regime_aware: Whether to use regime-aware scaling
            vol_lookback: Lookback period for volatility estimation
            ewma_halflife: EWMA half-life for volatility (None for simple)
        """
        self.vol_sizer = VolatilityTargetSizer(
            target_volatility=target_volatility,
            lookback_period=vol_lookback,
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            ewma_halflife=ewma_halflife,
        )

        self.turnover_optimizer = TurnoverOptimizer(
            no_trade_threshold=no_trade_threshold,
            transaction_cost=transaction_cost,
        )

        self.regime_scaler = RegimeAwareExposureScaler() if regime_aware else None
        self.regime_aware = regime_aware

        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

        # Track exposure history
        self._exposure_history: List[ExposureResult] = []

        logger.info(
            f"ExposureManager initialized: "
            f"target_vol={target_volatility:.1%}, "
            f"regime_aware={regime_aware}"
        )

    def calibrate(
        self,
        historical_returns: Union[np.ndarray, pd.Series],
    ):
        """
        Calibrate the exposure manager with historical data.

        Args:
            historical_returns: Historical returns for calibration
        """
        if isinstance(historical_returns, pd.Series):
            returns = historical_returns.values
        else:
            returns = historical_returns

        # Calculate rolling volatility for regime calibration
        if self.regime_scaler:
            rolling_vol = pd.Series(returns).rolling(21).std() * np.sqrt(252)
            self.regime_scaler.calibrate_thresholds(rolling_vol.dropna())

        logger.info("ExposureManager calibrated")

    def calculate_exposure(
        self,
        returns: Union[np.ndarray, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        target_weights: Optional[Dict[str, float]] = None,
        base_exposure: float = 1.0,
    ) -> ExposureResult:
        """
        Calculate optimal exposure given current conditions.

        Args:
            returns: Recent returns for volatility estimation
            current_weights: Current portfolio weights (for turnover calc)
            target_weights: Target portfolio weights
            base_exposure: Base exposure level before adjustments

        Returns:
            ExposureResult with all exposure details
        """
        # Step 1: Calculate volatility-adjusted exposure
        vol_adjusted_exposure, vol_scalar = self.vol_sizer.calculate_exposure(
            returns, base_weight=base_exposure
        )

        # Step 2: Apply regime-aware scaling
        regime_scalar = 1.0
        regime_state = RegimeState.NORMAL

        if self.regime_scaler:
            current_vol = self.vol_sizer.estimate_volatility(returns)
            regime_scalar, regime_state = self.regime_scaler.get_exposure_scalar(current_vol)

        target_exposure = vol_adjusted_exposure * regime_scalar
        target_exposure = np.clip(target_exposure, self.min_leverage, self.max_leverage)

        # Step 3: Optimize turnover if weights provided
        turnover_cost = 0.0
        optimized_weights = target_weights or {}

        if current_weights and target_weights:
            # Scale target weights by exposure
            scaled_targets = {
                k: v * target_exposure for k, v in target_weights.items()
            }

            turnover_result = self.turnover_optimizer.optimize_trades(
                current_weights=current_weights,
                target_weights=scaled_targets,
            )

            optimized_weights = turnover_result.optimized_weights
            turnover_cost = turnover_result.turnover_cost

        # Calculate gross and net exposure
        if optimized_weights:
            gross_exposure = sum(abs(w) for w in optimized_weights.values())
            net_exposure = sum(optimized_weights.values())
        else:
            gross_exposure = target_exposure
            net_exposure = target_exposure

        result = ExposureResult(
            target_exposure=target_exposure,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage_ratio=gross_exposure,  # Assuming NAV = 1
            volatility_scalar=vol_scalar,
            regime_scalar=regime_scalar,
            turnover_cost=turnover_cost,
            position_weights=optimized_weights,
            metadata={
                'regime': regime_state.value,
                'realized_vol': self.vol_sizer.estimate_volatility(returns),
                'target_vol': self.target_volatility,
            }
        )

        # Track history
        self._exposure_history.append(result)

        # Keep only recent history
        if len(self._exposure_history) > 1000:
            self._exposure_history = self._exposure_history[-1000:]

        logger.info(
            f"Exposure calculated: target={target_exposure:.2f}, "
            f"vol_scalar={vol_scalar:.2f}, regime={regime_state.value}"
        )

        return result

    def get_exposure_history(self) -> List[ExposureResult]:
        """Get exposure calculation history."""
        return self._exposure_history.copy()

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics from exposure history."""
        if not self._exposure_history:
            return {}

        exposures = [r.target_exposure for r in self._exposure_history]
        vol_scalars = [r.volatility_scalar for r in self._exposure_history]
        regime_scalars = [r.regime_scalar for r in self._exposure_history]
        turnover_costs = [r.turnover_cost for r in self._exposure_history]

        return {
            'mean_exposure': np.mean(exposures),
            'std_exposure': np.std(exposures),
            'min_exposure': np.min(exposures),
            'max_exposure': np.max(exposures),
            'mean_vol_scalar': np.mean(vol_scalars),
            'mean_regime_scalar': np.mean(regime_scalars),
            'total_turnover_cost': np.sum(turnover_costs),
        }


def compare_exposure_methods(
    returns: np.ndarray,
    target_volatility: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different exposure sizing methods.

    Args:
        returns: Historical returns
        target_volatility: Target portfolio volatility

    Returns:
        Dictionary comparing method results
    """
    results = {}

    # Method 1: Static exposure (no volatility targeting)
    static_exposure = 1.0
    static_vol = np.std(returns) * np.sqrt(252)
    results['static'] = {
        'exposure': static_exposure,
        'expected_vol': static_vol,
        'target_vol': target_volatility,
    }

    # Method 2: Simple volatility targeting
    vol_sizer = VolatilityTargetSizer(
        target_volatility=target_volatility,
        max_leverage=2.0,
    )
    vol_exposure, vol_scalar = vol_sizer.calculate_exposure(returns)
    results['vol_targeting'] = {
        'exposure': vol_exposure,
        'expected_vol': static_vol * vol_exposure,
        'vol_scalar': vol_scalar,
    }

    # Method 3: Full exposure management (vol + regime)
    manager = ExposureManager(
        target_volatility=target_volatility,
        regime_aware=True,
    )
    manager.calibrate(returns)
    full_result = manager.calculate_exposure(returns)
    results['full_management'] = {
        'exposure': full_result.target_exposure,
        'expected_vol': full_result.metadata['realized_vol'] * full_result.target_exposure,
        'vol_scalar': full_result.volatility_scalar,
        'regime_scalar': full_result.regime_scalar,
        'regime': full_result.metadata['regime'],
    }

    return results


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Exposure Manager Demo")
    print("=" * 60)

    # Generate synthetic returns
    np.random.seed(42)
    n_days = 500

    # Simulate with varying volatility regimes
    returns = np.concatenate([
        np.random.randn(100) * 0.01,  # Low vol
        np.random.randn(100) * 0.02,  # Normal vol
        np.random.randn(100) * 0.04,  # High vol
        np.random.randn(100) * 0.02,  # Normal vol
        np.random.randn(100) * 0.01,  # Low vol
    ])

    print(f"\nGenerated {len(returns)} days of returns")
    print(f"Overall vol: {np.std(returns) * np.sqrt(252):.2%}")

    # Compare methods
    print("\n" + "-" * 40)
    print("Comparing exposure methods:")
    comparison = compare_exposure_methods(returns, target_volatility=0.15)

    for method, metrics in comparison.items():
        print(f"\n  {method}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.3f}")
            else:
                print(f"    {key}: {value}")

    # Full manager demo
    print("\n" + "-" * 40)
    print("ExposureManager Demo:")

    manager = ExposureManager(
        target_volatility=0.15,
        max_leverage=2.0,
        regime_aware=True,
    )
    manager.calibrate(returns)

    # Calculate exposure at different points
    for i in [100, 200, 300, 400]:
        result = manager.calculate_exposure(returns[:i])
        print(
            f"  Day {i}: exposure={result.target_exposure:.2f}, "
            f"regime={result.metadata['regime']}, "
            f"vol={result.metadata['realized_vol']:.2%}"
        )

    print("\n" + "=" * 60)
    print("Demo complete!")
