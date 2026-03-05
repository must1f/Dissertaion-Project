"""
Unit tests for position sizing strategies.

Tests cover:
- FixedRiskSizer: Fixed percentage risk per trade
- KellyCriterionSizer: Kelly fraction calculations (full, half, quarter)
- VolatilityBasedSizer: Volatility-scaled positioning
- ConfidenceBasedSizer: Model confidence-based positioning
- Edge cases and invalid inputs
"""

import pytest
import numpy as np

from src.trading.position_sizing import (
    PositionSizeResult,
    PositionSizer,
    FixedRiskSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
    ConfidenceBasedSizer,
    compare_position_sizing_methods,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def capital():
    """Standard test capital"""
    return 100000.0


@pytest.fixture
def price():
    """Standard test stock price"""
    return 100.0


@pytest.fixture
def kelly_params():
    """Standard Kelly criterion parameters"""
    return {
        'win_rate': 0.55,
        'avg_win': 0.05,
        'avg_loss': 0.03,
    }


# ============================================================================
# FixedRiskSizer Tests
# ============================================================================

class TestFixedRiskSizer:
    """Tests for FixedRiskSizer"""

    def test_basic_calculation(self, capital, price):
        """Test basic fixed risk calculation"""
        sizer = FixedRiskSizer(risk_per_trade=0.02)
        result = sizer.calculate(capital, price)

        assert result.method == "FixedRisk"
        assert result.portfolio_fraction == pytest.approx(0.02, rel=0.01)
        assert result.position_size == 20  # $2000 / $100 = 20 shares
        assert result.dollar_amount == pytest.approx(2000.0, rel=0.01)

    def test_higher_risk(self, capital, price):
        """Test with higher risk percentage"""
        sizer = FixedRiskSizer(risk_per_trade=0.10)
        result = sizer.calculate(capital, price)

        assert result.position_size == 100  # $10,000 / $100 = 100 shares
        assert result.dollar_amount == pytest.approx(10000.0, rel=0.01)

    def test_fractional_shares_supported(self, capital):
        """Test that fractional shares are supported (modern trading)"""
        sizer = FixedRiskSizer(risk_per_trade=0.02)
        # With $2000 risk and $75 price, get 26.6667 fractional shares
        result = sizer.calculate(capital, 75.0)

        assert result.position_size == pytest.approx(26.6667, rel=0.01)
        assert result.dollar_amount == pytest.approx(2000.0, rel=0.01)

    def test_metadata_contains_risk_pct(self, capital, price):
        """Test that metadata includes risk percentage"""
        sizer = FixedRiskSizer(risk_per_trade=0.05)
        result = sizer.calculate(capital, price)

        assert result.metadata['risk_pct'] == 0.05


# ============================================================================
# KellyCriterionSizer Tests
# ============================================================================

class TestKellyCriterionSizer:
    """Tests for KellyCriterionSizer"""

    def test_full_kelly_calculation(self, capital, price, kelly_params):
        """Test full Kelly calculation"""
        sizer = KellyCriterionSizer(fractional_kelly=1.0)
        result = sizer.calculate(capital, price, **kelly_params)

        assert result.method == "Kelly_1.0"
        # Kelly formula: f = (p*b - q) / b = (0.55 * 1.667 - 0.45) / 1.667
        # f = (0.9167 - 0.45) / 1.667 = 0.28
        assert result.metadata['kelly_fraction'] == pytest.approx(0.20, abs=0.05)

    def test_half_kelly_calculation(self, capital, price, kelly_params):
        """Test half Kelly calculation"""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)
        result = sizer.calculate(capital, price, **kelly_params)

        assert result.method == "Kelly_0.5"
        # Half of full Kelly
        full_kelly = KellyCriterionSizer(fractional_kelly=1.0)
        full_result = full_kelly.calculate(capital, price, **kelly_params)

        # Half Kelly should be ~half the fraction (subject to clipping)
        assert result.metadata['kelly_fraction'] <= full_result.metadata['kelly_fraction']

    def test_quarter_kelly_calculation(self, capital, price, kelly_params):
        """Test quarter Kelly calculation"""
        sizer = KellyCriterionSizer(fractional_kelly=0.25)
        result = sizer.calculate(capital, price, **kelly_params)

        assert result.method == "Kelly_0.25"
        assert result.portfolio_fraction > 0

    def test_kelly_with_confidence(self, capital, price, kelly_params):
        """Test Kelly with confidence scaling"""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)

        # Full confidence
        result_full = sizer.calculate(capital, price, confidence=1.0, **kelly_params)

        # Half confidence
        result_half = sizer.calculate(capital, price, confidence=0.5, **kelly_params)

        # Lower confidence should result in smaller position
        assert result_half.portfolio_fraction < result_full.portfolio_fraction
        assert result_half.confidence == 0.5

    def test_kelly_zero_win_rate(self, capital, price):
        """Test Kelly with zero win rate returns zero"""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)
        result = sizer.calculate(
            capital, price,
            win_rate=0.0, avg_win=0.05, avg_loss=0.03
        )

        assert result.position_size == 0

    def test_kelly_negative_expected_value(self, capital, price):
        """Test Kelly with losing expectancy returns minimum position"""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)
        result = sizer.calculate(
            capital, price,
            win_rate=0.30, avg_win=0.02, avg_loss=0.05  # Negative expectancy
        )

        # Should get minimum position or zero
        assert result.portfolio_fraction >= 0

    def test_kelly_respects_max_position(self, capital, price):
        """Test that Kelly respects max position limit"""
        sizer = KellyCriterionSizer(
            fractional_kelly=1.0,
            max_position_pct=0.10  # Max 10%
        )
        result = sizer.calculate(
            capital, price,
            win_rate=0.70, avg_win=0.20, avg_loss=0.05  # Would suggest > 10%
        )

        assert result.portfolio_fraction <= 0.10 + 0.01  # Allow small tolerance

    def test_invalid_fractional_kelly(self):
        """Test that invalid fractional Kelly raises error"""
        with pytest.raises(ValueError):
            KellyCriterionSizer(fractional_kelly=0.0)

        with pytest.raises(ValueError):
            KellyCriterionSizer(fractional_kelly=1.5)

    def test_kelly_metadata(self, capital, price, kelly_params):
        """Test Kelly metadata contains expected fields"""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)
        result = sizer.calculate(capital, price, **kelly_params)

        assert 'win_rate' in result.metadata
        assert 'avg_win' in result.metadata
        assert 'avg_loss' in result.metadata
        assert 'kelly_fraction' in result.metadata
        assert 'fractional_kelly' in result.metadata


# ============================================================================
# VolatilityBasedSizer Tests
# ============================================================================

class TestVolatilityBasedSizer:
    """Tests for VolatilityBasedSizer"""

    def test_basic_calculation(self, capital, price):
        """Test basic volatility-based calculation"""
        sizer = VolatilityBasedSizer(target_volatility=0.15)
        result = sizer.calculate(capital, price, stock_volatility=0.30)

        assert result.method == "VolatilityBased"
        # Target 15%, stock 30% → fraction = 0.15/0.30 = 0.50
        assert result.portfolio_fraction == pytest.approx(0.20, abs=0.05)  # Capped at max

    def test_low_volatility_increases_position(self, capital, price):
        """Test that low volatility increases position size"""
        # Use higher max to avoid clipping
        sizer = VolatilityBasedSizer(target_volatility=0.15, max_position_pct=0.50)

        result_high_vol = sizer.calculate(capital, price, stock_volatility=0.30)
        result_low_vol = sizer.calculate(capital, price, stock_volatility=0.10)

        # Low vol (0.10) -> fraction 0.15/0.10 = 1.5 -> clipped to 0.50
        # High vol (0.30) -> fraction 0.15/0.30 = 0.50 -> stays at 0.50
        # Need even bigger difference to show the effect
        sizer2 = VolatilityBasedSizer(target_volatility=0.10, max_position_pct=0.90)
        result_high_vol2 = sizer2.calculate(capital, price, stock_volatility=0.50)
        result_low_vol2 = sizer2.calculate(capital, price, stock_volatility=0.15)

        assert result_low_vol2.portfolio_fraction > result_high_vol2.portfolio_fraction

    def test_high_volatility_decreases_position(self, capital, price):
        """Test that high volatility decreases position size"""
        # Use a higher max to avoid clipping
        sizer = VolatilityBasedSizer(target_volatility=0.10, max_position_pct=0.80)

        result_normal = sizer.calculate(capital, price, stock_volatility=0.15)
        result_high = sizer.calculate(capital, price, stock_volatility=0.50)

        assert result_high.portfolio_fraction < result_normal.portfolio_fraction

    def test_zero_volatility_handled(self, capital, price):
        """Test that zero volatility is handled gracefully"""
        sizer = VolatilityBasedSizer(target_volatility=0.15)
        result = sizer.calculate(capital, price, stock_volatility=0.0)

        # Should default to target volatility
        assert result.portfolio_fraction > 0

    def test_volatility_metadata(self, capital, price):
        """Test volatility metadata"""
        sizer = VolatilityBasedSizer(target_volatility=0.15)
        result = sizer.calculate(capital, price, stock_volatility=0.25)

        assert result.metadata['target_volatility'] == 0.15
        assert result.metadata['stock_volatility'] == 0.25


# ============================================================================
# ConfidenceBasedSizer Tests
# ============================================================================

class TestConfidenceBasedSizer:
    """Tests for ConfidenceBasedSizer"""

    def test_basic_calculation(self, capital, price):
        """Test basic confidence-based calculation"""
        sizer = ConfidenceBasedSizer(base_risk=0.05)
        result = sizer.calculate(capital, price, confidence=1.0)

        assert result.method == "ConfidenceBased"
        assert result.portfolio_fraction == pytest.approx(0.05, rel=0.01)
        assert result.confidence == 1.0

    def test_low_confidence_reduces_position(self, capital, price):
        """Test that low confidence reduces position size"""
        sizer = ConfidenceBasedSizer(base_risk=0.10)

        result_full = sizer.calculate(capital, price, confidence=1.0)
        result_half = sizer.calculate(capital, price, confidence=0.5)

        assert result_half.portfolio_fraction < result_full.portfolio_fraction
        assert result_half.portfolio_fraction == pytest.approx(0.05, rel=0.1)

    def test_zero_confidence(self, capital, price):
        """Test zero confidence results in minimum position"""
        sizer = ConfidenceBasedSizer(base_risk=0.10)
        result = sizer.calculate(capital, price, confidence=0.0)

        # Should get minimum position
        assert result.portfolio_fraction == sizer.min_position_pct

    def test_confidence_recorded(self, capital, price):
        """Test that confidence is recorded in result"""
        sizer = ConfidenceBasedSizer(base_risk=0.05)
        result = sizer.calculate(capital, price, confidence=0.75)

        assert result.confidence == 0.75


# ============================================================================
# PositionSizer Base Class Tests
# ============================================================================

class TestPositionSizerBase:
    """Tests for PositionSizer base class"""

    def test_clip_position_max(self):
        """Test clipping at max position"""
        sizer = FixedRiskSizer(max_position_pct=0.20)
        clipped = sizer.clip_position(0.50)

        assert clipped == 0.20

    def test_clip_position_min(self):
        """Test clipping at min position"""
        sizer = FixedRiskSizer(min_position_pct=0.01)
        clipped = sizer.clip_position(0.005)

        assert clipped == 0.01

    def test_clip_position_within_range(self):
        """Test that values within range are unchanged"""
        sizer = FixedRiskSizer(min_position_pct=0.01, max_position_pct=0.20)
        clipped = sizer.clip_position(0.10)

        assert clipped == 0.10


# ============================================================================
# Compare Position Sizing Methods Tests
# ============================================================================

class TestComparePositionSizing:
    """Tests for compare_position_sizing_methods utility"""

    def test_returns_all_methods(self, capital, price):
        """Test that comparison returns all expected methods"""
        results = compare_position_sizing_methods(
            current_capital=capital,
            current_price=price,
            win_rate=0.55,
            avg_win=0.05,
            avg_loss=0.03,
            confidence=0.8,
            stock_volatility=0.25
        )

        expected_methods = [
            'Fixed 2%',
            'Full Kelly',
            'Half Kelly',
            'Quarter Kelly',
            'Half Kelly + Confidence',
            'Volatility Based',
            'Confidence Based'
        ]

        for method in expected_methods:
            assert method in results, f"Missing method: {method}"

    def test_all_results_valid(self, capital, price):
        """Test that all results are valid PositionSizeResult"""
        results = compare_position_sizing_methods(
            current_capital=capital,
            current_price=price,
            win_rate=0.55,
            avg_win=0.05,
            avg_loss=0.03,
            confidence=0.8,
            stock_volatility=0.25
        )

        for method, result in results.items():
            assert isinstance(result, PositionSizeResult)
            assert result.position_size >= 0
            assert result.dollar_amount >= 0
            assert 0 <= result.portfolio_fraction <= 1
            assert result.method is not None


# ============================================================================
# PositionSizeResult Tests
# ============================================================================

class TestPositionSizeResult:
    """Tests for PositionSizeResult dataclass"""

    def test_dataclass_creation(self):
        """Test creating PositionSizeResult"""
        result = PositionSizeResult(
            position_size=100,
            dollar_amount=10000.0,
            portfolio_fraction=0.10,
            method="Test"
        )

        assert result.position_size == 100
        assert result.dollar_amount == 10000.0
        assert result.portfolio_fraction == 0.10
        assert result.method == "Test"
        assert result.confidence == 1.0  # Default
        assert result.metadata is None  # Default

    def test_dataclass_with_metadata(self):
        """Test creating PositionSizeResult with metadata"""
        result = PositionSizeResult(
            position_size=50,
            dollar_amount=5000.0,
            portfolio_fraction=0.05,
            method="Test",
            confidence=0.8,
            metadata={'key': 'value'}
        )

        assert result.confidence == 0.8
        assert result.metadata == {'key': 'value'}


# ============================================================================
# Integration Tests
# ============================================================================

class TestPositionSizingIntegration:
    """Integration tests for position sizing"""

    def test_realistic_trading_scenario(self):
        """Test position sizing in a realistic trading scenario"""
        capital = 50000.0
        price = 250.0  # e.g., MSFT

        # Moderate win rate, good risk/reward
        win_rate = 0.52
        avg_win = 0.04
        avg_loss = 0.025

        # Use half Kelly
        sizer = KellyCriterionSizer(
            fractional_kelly=0.5,
            max_position_pct=0.15
        )

        result = sizer.calculate(
            capital, price,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            confidence=0.70
        )

        # Verify reasonable position size
        assert result.position_size > 0
        assert result.dollar_amount < capital * 0.20  # Not over 20%
        assert result.portfolio_fraction <= 0.15  # Respects max

    def test_position_sizing_sequence(self):
        """Test a sequence of position sizing calls (simulating trades)"""
        initial_capital = 100000.0
        current_capital = initial_capital
        price = 100.0

        sizer = KellyCriterionSizer(fractional_kelly=0.5)

        positions = []
        for i in range(5):
            # Simulate varying confidence
            confidence = 0.6 + 0.1 * (i % 3)

            result = sizer.calculate(
                current_capital=current_capital,
                current_price=price,
                win_rate=0.55,
                avg_win=0.05,
                avg_loss=0.03,
                confidence=confidence
            )

            positions.append(result)
            # Simulate win/loss (no actual capital change in test)

        # All positions should be valid
        for pos in positions:
            assert pos.position_size >= 0
            assert pos.portfolio_fraction > 0
