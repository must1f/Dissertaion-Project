"""
Physics-Informed Neural Network (PINN) Framework for Financial Forecasting
"""

__version__ = "0.1.0"
__author__ = "PINN Financial Research"

# Export commonly used constants
from .constants import (
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
    RISK_PER_TRADE,
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
