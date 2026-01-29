"""
Position Sizing Strategies for Trading

Implements various position sizing methods including:
- Fixed risk percentage
- Kelly Criterion (full and fractional)
- Volatility-based sizing
- Confidence-based sizing (using uncertainty estimates)

Reference:
    Kelly, J. L. (1956). "A New Interpretation of Information Rate."
    Bell System Technical Journal.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    position_size: float  # Number of shares
    dollar_amount: float  # Dollar value of position
    portfolio_fraction: float  # Fraction of portfolio (0-1)
    method: str  # Sizing method used
    confidence: float = 1.0  # Confidence in signal (0-1)
    metadata: Dict = None  # Additional info


class PositionSizer:
    """Base class for position sizing strategies"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.20,  # Max 20% per position
        min_position_pct: float = 0.01   # Min 1% per position
    ):
        """
        Initialize position sizer

        Args:
            initial_capital: Starting capital
            max_position_pct: Maximum fraction of capital per position
            min_position_pct: Minimum fraction of capital per position
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    def clip_position(self, fraction: float) -> float:
        """
        Clip position size to allowed range

        Args:
            fraction: Desired fraction of capital

        Returns:
            Clipped fraction
        """
        return np.clip(fraction, self.min_position_pct, self.max_position_pct)

    def calculate(
        self,
        current_capital: float,
        current_price: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size

        Args:
            current_capital: Current account value
            current_price: Current stock price
            **kwargs: Strategy-specific parameters

        Returns:
            PositionSizeResult
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class FixedRiskSizer(PositionSizer):
    """
    Fixed risk percentage per trade

    Simple and conservative approach: risk fixed % of capital per trade.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,  # Risk 2% per trade
        **kwargs
    ):
        """
        Initialize fixed risk sizer

        Args:
            risk_per_trade: Fraction of capital to risk per trade
            **kwargs: Passed to PositionSizer
        """
        super().__init__(**kwargs)
        self.risk_per_trade = risk_per_trade

    def calculate(
        self,
        current_capital: float,
        current_price: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size based on fixed risk

        Args:
            current_capital: Current account value
            current_price: Current stock price

        Returns:
            PositionSizeResult
        """
        # Dollar amount to risk
        dollar_amount = current_capital * self.risk_per_trade

        # Position size (number of shares)
        position_size = int(dollar_amount / current_price)

        # Actual fraction invested
        actual_fraction = (position_size * current_price) / current_capital

        return PositionSizeResult(
            position_size=position_size,
            dollar_amount=position_size * current_price,
            portfolio_fraction=actual_fraction,
            method="FixedRisk",
            metadata={'risk_pct': self.risk_per_trade}
        )


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizing

    Maximizes log-wealth growth by sizing positions based on win rate
    and profit/loss ratios.

    Kelly fraction: f* = (p*b - q) / b
    where:
        p = probability of win
        q = probability of loss (1 - p)
        b = odds (avg_win / avg_loss)
    """

    def __init__(
        self,
        fractional_kelly: float = 0.5,  # Use half-Kelly (safer)
        **kwargs
    ):
        """
        Initialize Kelly Criterion sizer

        Args:
            fractional_kelly: Fraction of Kelly to use (0.25-1.0)
                0.25 = quarter Kelly (very conservative)
                0.5 = half Kelly (recommended)
                1.0 = full Kelly (aggressive, high variance)
            **kwargs: Passed to PositionSizer
        """
        super().__init__(**kwargs)
        self.fractional_kelly = fractional_kelly

        if not 0 < fractional_kelly <= 1.0:
            raise ValueError("fractional_kelly must be in (0, 1]")

        logger.info(f"Kelly Criterion initialized: fractional={fractional_kelly}")

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction

        Args:
            win_rate: Probability of winning trade (0-1)
            avg_win: Average profit on winning trades (%)
            avg_loss: Average loss on losing trades (%) - should be positive

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate: {win_rate}")
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid avg_win/avg_loss: {avg_win}/{avg_loss}")
            return 0.0

        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # Odds ratio

        f_kelly = (p * b - q) / b

        # Clip to valid range
        f_kelly = max(0, f_kelly)

        # Apply fractional Kelly
        f_kelly = f_kelly * self.fractional_kelly

        # Apply position limits
        f_kelly = self.clip_position(f_kelly)

        return f_kelly

    def calculate(
        self,
        current_capital: float,
        current_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion

        Args:
            current_capital: Current account value
            current_price: Current stock price
            win_rate: Historical win rate (0-1)
            avg_win: Average win percentage (e.g., 0.05 for 5%)
            avg_loss: Average loss percentage (positive, e.g., 0.03 for 3%)
            confidence: Model confidence (0-1) - scales Kelly fraction
            **kwargs: Additional parameters

        Returns:
            PositionSizeResult
        """
        # Calculate base Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        # Adjust by confidence if provided
        if confidence < 1.0:
            adjusted_fraction = kelly_fraction * confidence
            logger.debug(f"Kelly adjusted by confidence: {kelly_fraction:.3f} -> {adjusted_fraction:.3f}")
        else:
            adjusted_fraction = kelly_fraction

        # Dollar amount to invest
        dollar_amount = current_capital * adjusted_fraction

        # Position size (number of shares)
        position_size = int(dollar_amount / current_price)

        # Actual fraction invested
        actual_fraction = (position_size * current_price) / current_capital

        return PositionSizeResult(
            position_size=position_size,
            dollar_amount=position_size * current_price,
            portfolio_fraction=actual_fraction,
            method=f"Kelly_{self.fractional_kelly}",
            confidence=confidence,
            metadata={
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'kelly_fraction': kelly_fraction,
                'fractional_kelly': self.fractional_kelly
            }
        )


class VolatilityBasedSizer(PositionSizer):
    """
    Size positions inversely to volatility

    Higher volatility → smaller position
    Lower volatility → larger position
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # Target 15% volatility per position
        **kwargs
    ):
        """
        Initialize volatility-based sizer

        Args:
            target_volatility: Target volatility per position (annualized)
            **kwargs: Passed to PositionSizer
        """
        super().__init__(**kwargs)
        self.target_volatility = target_volatility

    def calculate(
        self,
        current_capital: float,
        current_price: float,
        stock_volatility: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size based on volatility

        Args:
            current_capital: Current account value
            current_price: Current stock price
            stock_volatility: Stock volatility (annualized)

        Returns:
            PositionSizeResult
        """
        if stock_volatility <= 0:
            logger.warning(f"Invalid volatility: {stock_volatility}")
            stock_volatility = self.target_volatility

        # Fraction to invest: inversely proportional to volatility
        fraction = self.target_volatility / stock_volatility

        # Clip to valid range
        fraction = self.clip_position(fraction)

        # Dollar amount
        dollar_amount = current_capital * fraction

        # Position size
        position_size = int(dollar_amount / current_price)

        # Actual fraction
        actual_fraction = (position_size * current_price) / current_capital

        return PositionSizeResult(
            position_size=position_size,
            dollar_amount=position_size * current_price,
            portfolio_fraction=actual_fraction,
            method="VolatilityBased",
            metadata={
                'target_volatility': self.target_volatility,
                'stock_volatility': stock_volatility
            }
        )


class ConfidenceBasedSizer(PositionSizer):
    """
    Size positions based on model confidence

    Uses uncertainty estimates from MC Dropout or ensembles to scale position size.
    """

    def __init__(
        self,
        base_risk: float = 0.05,  # Base 5% risk when confidence = 1.0
        **kwargs
    ):
        """
        Initialize confidence-based sizer

        Args:
            base_risk: Base risk when confidence is 100%
            **kwargs: Passed to PositionSizer
        """
        super().__init__(**kwargs)
        self.base_risk = base_risk

    def calculate(
        self,
        current_capital: float,
        current_price: float,
        confidence: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size based on confidence

        Args:
            current_capital: Current account value
            current_price: Current stock price
            confidence: Model confidence (0-1)

        Returns:
            PositionSizeResult
        """
        # Scale risk by confidence
        adjusted_risk = self.base_risk * confidence

        # Clip to valid range
        adjusted_risk = self.clip_position(adjusted_risk)

        # Dollar amount
        dollar_amount = current_capital * adjusted_risk

        # Position size
        position_size = int(dollar_amount / current_price)

        # Actual fraction
        actual_fraction = (position_size * current_price) / current_capital

        return PositionSizeResult(
            position_size=position_size,
            dollar_amount=position_size * current_price,
            portfolio_fraction=actual_fraction,
            method="ConfidenceBased",
            confidence=confidence,
            metadata={'base_risk': self.base_risk}
        )


def compare_position_sizing_methods(
    current_capital: float,
    current_price: float,
    win_rate: float = 0.55,
    avg_win: float = 0.03,
    avg_loss: float = 0.02,
    confidence: float = 0.8,
    stock_volatility: float = 0.25
) -> Dict[str, PositionSizeResult]:
    """
    Compare different position sizing methods

    Args:
        current_capital: Current account value
        current_price: Current stock price
        win_rate: Historical win rate
        avg_win: Average win percentage
        avg_loss: Average loss percentage
        confidence: Model confidence
        stock_volatility: Stock volatility

    Returns:
        Dictionary mapping method name to PositionSizeResult
    """
    results = {}

    # Fixed risk
    fixed_sizer = FixedRiskSizer(risk_per_trade=0.02)
    results['Fixed 2%'] = fixed_sizer.calculate(current_capital, current_price)

    # Full Kelly
    full_kelly = KellyCriterionSizer(fractional_kelly=1.0)
    results['Full Kelly'] = full_kelly.calculate(
        current_capital, current_price, win_rate, avg_win, avg_loss
    )

    # Half Kelly
    half_kelly = KellyCriterionSizer(fractional_kelly=0.5)
    results['Half Kelly'] = half_kelly.calculate(
        current_capital, current_price, win_rate, avg_win, avg_loss
    )

    # Quarter Kelly
    quarter_kelly = KellyCriterionSizer(fractional_kelly=0.25)
    results['Quarter Kelly'] = quarter_kelly.calculate(
        current_capital, current_price, win_rate, avg_win, avg_loss
    )

    # Kelly with confidence
    kelly_conf = KellyCriterionSizer(fractional_kelly=0.5)
    results['Half Kelly + Confidence'] = kelly_conf.calculate(
        current_capital, current_price, win_rate, avg_win, avg_loss, confidence
    )

    # Volatility-based
    vol_sizer = VolatilityBasedSizer(target_volatility=0.15)
    results['Volatility Based'] = vol_sizer.calculate(
        current_capital, current_price, stock_volatility
    )

    # Confidence-based
    conf_sizer = ConfidenceBasedSizer(base_risk=0.05)
    results['Confidence Based'] = conf_sizer.calculate(
        current_capital, current_price, confidence
    )

    return results


def print_comparison_table(results: Dict[str, PositionSizeResult]):
    """
    Print comparison table of position sizing methods

    Args:
        results: Dictionary of sizing results
    """
    print("\n" + "=" * 80)
    print("Position Sizing Method Comparison")
    print("=" * 80)
    print(f"{'Method':<25} {'Shares':<8} {'$ Amount':<12} {'% Portfolio':<12} {'Confidence':<10}")
    print("-" * 80)

    for method, result in results.items():
        print(f"{method:<25} {result.position_size:<8} "
              f"${result.dollar_amount:<11,.2f} "
              f"{result.portfolio_fraction*100:<11.2f}% "
              f"{result.confidence:<9.2f}")

    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    print("Kelly Criterion Position Sizing Demo")

    # Example scenario
    capital = 100000.0
    price = 150.0
    win_rate = 0.55  # 55% win rate
    avg_win = 0.05   # 5% average win
    avg_loss = 0.03  # 3% average loss
    confidence = 0.75  # 75% confidence from model
    volatility = 0.30  # 30% annualized volatility

    # Compare methods
    results = compare_position_sizing_methods(
        current_capital=capital,
        current_price=price,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        confidence=confidence,
        stock_volatility=volatility
    )

    # Print results
    print_comparison_table(results)

    # Recommendation
    print("\nRecommendation:")
    print("  Half Kelly: Good balance of growth and risk management")
    print("  Half Kelly + Confidence: Adapts to model uncertainty (recommended)")
    print("  Quarter Kelly: More conservative, lower variance")
    print("  Full Kelly: Maximum growth but high variance (not recommended)")
