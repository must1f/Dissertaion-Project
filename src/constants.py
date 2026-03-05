"""
Financial Constants Module

Centralized definitions for all financial constants used throughout the codebase.
This eliminates magic numbers and ensures consistency across all calculations.

Usage:
    from src.constants import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess_return = returns - RISK_FREE_RATE
"""

import math

# =============================================================================
# TIME CONSTANTS
# =============================================================================

# Number of trading days in a year (US market standard)
TRADING_DAYS_PER_YEAR: int = 252

# Square root of trading days (precomputed for efficiency)
SQRT_TRADING_DAYS: float = math.sqrt(TRADING_DAYS_PER_YEAR)

# Time step in years (1 trading day)
DAILY_TIME_STEP: float = 1.0 / TRADING_DAYS_PER_YEAR

# Trading weeks per year
TRADING_WEEKS_PER_YEAR: int = 52

# Trading months per year
TRADING_MONTHS_PER_YEAR: int = 12


# =============================================================================
# MARKET ASSUMPTIONS
# =============================================================================

# Risk-free rate (annual, as decimal)
# Based on typical T-bill rates
RISK_FREE_RATE: float = 0.02  # 2% annual

# Daily risk-free rate (derived)
DAILY_RISK_FREE_RATE: float = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR


# =============================================================================
# TRANSACTION COSTS
# =============================================================================

# Default transaction cost (as decimal)
# 0.3% is realistic for retail equity trading including spread + commission
TRANSACTION_COST: float = 0.003  # 0.3%

# Institutional transaction cost (lower due to volume discounts)
INSTITUTIONAL_TRANSACTION_COST: float = 0.001  # 0.1%


# =============================================================================
# DEFAULT MARKET PARAMETERS
# =============================================================================

# Default expected annual return (for simulations when no data available)
DEFAULT_ANNUAL_RETURN: float = 0.08  # 8% (historical S&P 500 average)

# Default expected daily return (derived)
DEFAULT_DAILY_RETURN: float = DEFAULT_ANNUAL_RETURN / TRADING_DAYS_PER_YEAR

# Default annual volatility (for simulations when no data available)
DEFAULT_ANNUAL_VOLATILITY: float = 0.20  # 20% (typical equity volatility)

# Default daily volatility (derived)
DEFAULT_DAILY_VOLATILITY: float = DEFAULT_ANNUAL_VOLATILITY / SQRT_TRADING_DAYS


# =============================================================================
# RISK MANAGEMENT DEFAULTS
# =============================================================================

# Maximum position size as fraction of portfolio
MAX_POSITION_SIZE: float = 0.20  # 20%

# Default stop-loss level
DEFAULT_STOP_LOSS: float = 0.02  # 2%

# Default risk per trade (Kelly fraction approximation)
RISK_PER_TRADE: float = 0.02  # 2%

# No-trade threshold (minimum expected return to justify trading)
NO_TRADE_THRESHOLD: float = 0.02  # 2%


# =============================================================================
# PHYSICS-INFORMED DEFAULTS (PINN)
# =============================================================================

# Default physics constraint weights
DEFAULT_LAMBDA_GBM: float = 0.1  # Geometric Brownian Motion weight
DEFAULT_LAMBDA_OU: float = 0.1   # Ornstein-Uhlenbeck weight
DEFAULT_LAMBDA_BS: float = 0.1   # Black-Scholes weight
DEFAULT_LAMBDA_LANGEVIN: float = 0.1  # Langevin dynamics weight


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Small epsilon for numerical stability (avoid division by zero)
EPSILON: float = 1e-8

# Maximum loss value considered healthy
MAX_HEALTHY_LOSS: float = 1e6

# Clipping range for normalized values
NORM_CLIP_MIN: float = -10.0
NORM_CLIP_MAX: float = 10.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def annualize_return(daily_return: float) -> float:
    """Convert daily return to annualized return."""
    return daily_return * TRADING_DAYS_PER_YEAR


def annualize_volatility(daily_vol: float) -> float:
    """Convert daily volatility to annualized volatility."""
    return daily_vol * SQRT_TRADING_DAYS


def daily_return(annual_return: float) -> float:
    """Convert annualized return to daily return."""
    return annual_return / TRADING_DAYS_PER_YEAR


def daily_volatility(annual_vol: float) -> float:
    """Convert annualized volatility to daily volatility."""
    return annual_vol / SQRT_TRADING_DAYS


def annualize_sharpe(daily_sharpe: float) -> float:
    """Convert daily Sharpe ratio to annualized."""
    return daily_sharpe * SQRT_TRADING_DAYS
