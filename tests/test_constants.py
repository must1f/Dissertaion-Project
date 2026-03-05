"""
Tests for Financial Constants Module

Validates that all constants are correctly defined and helper functions work.
"""

import pytest
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import (
    # Time constants
    TRADING_DAYS_PER_YEAR,
    SQRT_TRADING_DAYS,
    DAILY_TIME_STEP,
    # Market assumptions
    RISK_FREE_RATE,
    DAILY_RISK_FREE_RATE,
    TRANSACTION_COST,
    # Default parameters
    DEFAULT_ANNUAL_RETURN,
    DEFAULT_ANNUAL_VOLATILITY,
    DEFAULT_DAILY_VOLATILITY,
    # Risk management
    MAX_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    # PINN defaults
    DEFAULT_LAMBDA_GBM,
    DEFAULT_LAMBDA_OU,
    DEFAULT_LAMBDA_BS,
    # Helpers
    annualize_return,
    annualize_volatility,
    daily_return,
    daily_volatility,
    annualize_sharpe,
)


class TestTimeConstants:
    """Tests for time-related constants."""

    def test_trading_days_per_year(self):
        """Standard US market has 252 trading days."""
        assert TRADING_DAYS_PER_YEAR == 252

    def test_sqrt_trading_days(self):
        """Pre-computed square root should be correct."""
        assert SQRT_TRADING_DAYS == pytest.approx(math.sqrt(252), rel=1e-10)

    def test_daily_time_step(self):
        """Daily time step should be 1/252."""
        assert DAILY_TIME_STEP == pytest.approx(1.0 / 252.0, rel=1e-10)


class TestMarketAssumptions:
    """Tests for market assumption constants."""

    def test_risk_free_rate_reasonable(self):
        """Risk-free rate should be between 0% and 10%."""
        assert 0.0 <= RISK_FREE_RATE <= 0.10

    def test_daily_risk_free_rate_derived(self):
        """Daily rate should be annual / 252."""
        assert DAILY_RISK_FREE_RATE == pytest.approx(
            RISK_FREE_RATE / TRADING_DAYS_PER_YEAR, rel=1e-10
        )

    def test_transaction_cost_reasonable(self):
        """Transaction cost should be between 0% and 1%."""
        assert 0.0 < TRANSACTION_COST < 0.01


class TestDefaultParameters:
    """Tests for default market parameters."""

    def test_default_return_reasonable(self):
        """Default annual return should be realistic (0-20%)."""
        assert 0.0 < DEFAULT_ANNUAL_RETURN <= 0.20

    def test_default_volatility_reasonable(self):
        """Default volatility should be realistic (5-50%)."""
        assert 0.05 <= DEFAULT_ANNUAL_VOLATILITY <= 0.50

    def test_daily_volatility_derived(self):
        """Daily volatility should be annual / sqrt(252)."""
        assert DEFAULT_DAILY_VOLATILITY == pytest.approx(
            DEFAULT_ANNUAL_VOLATILITY / SQRT_TRADING_DAYS, rel=1e-10
        )


class TestRiskManagement:
    """Tests for risk management constants."""

    def test_max_position_size_reasonable(self):
        """Max position should be between 1% and 100%."""
        assert 0.01 <= MAX_POSITION_SIZE <= 1.0

    def test_stop_loss_reasonable(self):
        """Stop loss should be between 0.5% and 20%."""
        assert 0.005 <= DEFAULT_STOP_LOSS <= 0.20


class TestPINNDefaults:
    """Tests for PINN-specific constants."""

    def test_lambda_values_non_negative(self):
        """Physics constraint weights should be non-negative."""
        assert DEFAULT_LAMBDA_GBM >= 0
        assert DEFAULT_LAMBDA_OU >= 0
        assert DEFAULT_LAMBDA_BS >= 0

    def test_lambda_values_reasonable(self):
        """Weights should be small relative to data loss."""
        assert DEFAULT_LAMBDA_GBM <= 1.0
        assert DEFAULT_LAMBDA_OU <= 1.0
        assert DEFAULT_LAMBDA_BS <= 1.0


class TestHelperFunctions:
    """Tests for conversion helper functions."""

    def test_annualize_return(self):
        """Annualize daily return."""
        daily = 0.0004  # ~10% annual
        annual = annualize_return(daily)
        assert annual == pytest.approx(daily * TRADING_DAYS_PER_YEAR)

    def test_annualize_volatility(self):
        """Annualize daily volatility."""
        daily_vol = 0.01  # ~16% annual
        annual_vol = annualize_volatility(daily_vol)
        assert annual_vol == pytest.approx(daily_vol * SQRT_TRADING_DAYS)

    def test_daily_return(self):
        """Convert annual to daily return."""
        annual = 0.10  # 10% annual
        daily = daily_return(annual)
        assert daily == pytest.approx(annual / TRADING_DAYS_PER_YEAR)

    def test_daily_volatility(self):
        """Convert annual to daily volatility."""
        annual_vol = 0.20  # 20% annual
        daily_vol = daily_volatility(annual_vol)
        assert daily_vol == pytest.approx(annual_vol / SQRT_TRADING_DAYS)

    def test_annualize_sharpe(self):
        """Annualize Sharpe ratio."""
        daily_sharpe = 0.05
        annual_sharpe = annualize_sharpe(daily_sharpe)
        assert annual_sharpe == pytest.approx(daily_sharpe * SQRT_TRADING_DAYS)

    def test_roundtrip_return(self):
        """Daily and annual conversion should roundtrip."""
        original = 0.10
        roundtrip = annualize_return(daily_return(original))
        assert roundtrip == pytest.approx(original, rel=1e-10)

    def test_roundtrip_volatility(self):
        """Volatility conversion should roundtrip."""
        original = 0.20
        roundtrip = annualize_volatility(daily_volatility(original))
        assert roundtrip == pytest.approx(original, rel=1e-10)


def test_constants_importable_from_src():
    """Verify constants can be imported from src package."""
    from src import (
        TRADING_DAYS_PER_YEAR,
        RISK_FREE_RATE,
        TRANSACTION_COST,
        annualize_volatility,
    )

    assert TRADING_DAYS_PER_YEAR == 252
    assert callable(annualize_volatility)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
