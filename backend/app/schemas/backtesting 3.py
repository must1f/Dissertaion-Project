"""Pydantic schemas for backtesting endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""

    FIXED = "fixed"
    KELLY_FULL = "kelly_full"
    KELLY_HALF = "kelly_half"
    KELLY_QUARTER = "kelly_quarter"
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"


class TradeAction(str, Enum):
    """Trade actions."""

    BUY = "BUY"
    SELL = "SELL"


class BacktestRequest(BaseModel):
    """Request to run a backtest. Only S&P 500 data is supported."""

    model_key: str = Field(..., description="Model to use for signals")
    ticker: str = Field(
        default=DEFAULT_TICKER,
        description=f"Stock ticker to backtest (only {', '.join(ALLOWED_TICKERS)} supported)"
    )

    # Date range
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")

    # Capital settings
    initial_capital: float = Field(default=100000.0, ge=1000)

    # Cost settings
    commission_rate: float = Field(default=0.001, ge=0, le=0.1)
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.1)

    # Risk management
    max_position_size: float = Field(default=0.2, ge=0.01, le=1.0)
    stop_loss: float = Field(default=0.02, ge=0, le=0.5)
    take_profit: float = Field(default=0.05, ge=0, le=1.0)

    # Position sizing
    position_sizing_method: PositionSizingMethod = Field(
        default=PositionSizingMethod.FIXED
    )

    # Signal settings
    signal_threshold: float = Field(default=0.01, ge=0, le=0.1)

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        """Validate that only allowed tickers are used for backtesting."""
        return validate_ticker(v)


class Trade(BaseModel):
    """Individual trade record."""

    id: str
    timestamp: datetime
    ticker: str
    action: TradeAction
    price: float
    quantity: float
    value: float
    commission: float
    slippage: float

    # Position info
    position_before: float
    position_after: float

    # P&L (for closing trades)
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    # Signal info
    signal_confidence: Optional[float] = None
    predicted_return: Optional[float] = None


class PortfolioSnapshot(BaseModel):
    """Portfolio state at a point in time."""

    timestamp: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: Optional[float] = None
    cumulative_return: float


class DrawdownInfo(BaseModel):
    """Drawdown information."""

    start_date: datetime
    end_date: Optional[datetime] = None
    recovery_date: Optional[datetime] = None
    drawdown_percent: float
    duration_days: int
    recovery_days: Optional[int] = None


class BacktestResults(BaseModel):
    """Complete backtest results."""

    # Configuration
    model_key: str
    ticker: str
    start_date: datetime
    end_date: datetime
    initial_capital: float

    # Summary metrics
    final_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: Optional[float] = None
    total_trades: int

    # Detailed metrics
    metrics: Dict[str, Any]

    # Time series
    portfolio_history: List[PortfolioSnapshot]
    equity_curve: List[float]
    returns: List[float]

    # Trades
    trades: List[Trade]
    winning_trades: int
    losing_trades: int

    # Drawdowns
    drawdowns: List[DrawdownInfo]


class BacktestResponse(BaseModel):
    """Response for backtest request."""

    success: bool
    result_id: str
    results: BacktestResults
    processing_time_ms: float


class BacktestSummary(BaseModel):
    """Summary of a backtest result."""

    result_id: str
    model_key: str
    ticker: str
    run_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int


class BacktestListResponse(BaseModel):
    """Response listing backtest results."""

    results: List[BacktestSummary]
    total: int
    page: int
    page_size: int


class TradeHistoryResponse(BaseModel):
    """Response with trade history for a backtest."""

    result_id: str
    trades: List[Trade]
    total: int
    summary: Dict[str, Any]
