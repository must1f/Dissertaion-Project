"""Pydantic schemas for trading agent endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker


class TradingMode(str, Enum):
    """Trading mode options."""
    PAPER = "paper"
    SIMULATION = "simulation"
    BACKTEST = "backtest"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalAction(str, Enum):
    """Trading signal actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Alert types."""
    SIGNAL = "signal"
    RISK = "risk"
    PRICE = "price"
    EXECUTION = "execution"
    ERROR = "error"


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"


# ============== Request Schemas ==============

class AgentConfigRequest(BaseModel):
    """Request to start or configure trading agent."""

    model_key: str = Field(..., description="Model to use for predictions")
    ticker: str = Field(
        default=DEFAULT_TICKER,
        description=f"Stock ticker (only {', '.join(ALLOWED_TICKERS)} supported)"
    )
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode"
    )
    initial_capital: float = Field(
        default=100000.0,
        ge=1000.0,
        le=10000000.0,
        description="Initial capital"
    )
    signal_threshold: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Minimum expected return to generate signal"
    )
    max_position_size: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Maximum position size as fraction of portfolio"
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.40,
        le=0.95,
        description="Minimum confidence threshold for trading"
    )
    stop_loss_pct: float = Field(
        default=0.02,
        ge=0.01,
        le=0.10,
        description="Stop loss percentage"
    )
    take_profit_pct: float = Field(
        default=0.05,
        ge=0.02,
        le=0.20,
        description="Take profit percentage"
    )
    position_sizing: PositionSizingMethod = Field(
        default=PositionSizingMethod.CONFIDENCE,
        description="Position sizing method"
    )

    # Simulation specific
    simulation_start_date: Optional[str] = Field(
        default=None,
        description="Start date for simulation (YYYY-MM-DD)"
    )
    simulation_end_date: Optional[str] = Field(
        default=None,
        description="End date for simulation (YYYY-MM-DD)"
    )
    simulation_speed: float = Field(
        default=1.0,
        ge=0.1,
        le=100.0,
        description="Simulation speed multiplier"
    )

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        return validate_ticker(v)


class ManualOrderRequest(BaseModel):
    """Request for manual order placement."""

    ticker: str = Field(default=DEFAULT_TICKER)
    side: SignalAction = Field(..., description="BUY or SELL")
    quantity: Optional[float] = Field(
        default=None,
        ge=0.001,
        description="Number of shares (auto-calculated if not provided)"
    )
    order_type: OrderType = Field(default=OrderType.MARKET)
    limit_price: Optional[float] = Field(default=None)
    stop_price: Optional[float] = Field(default=None)
    reason: str = Field(default="Manual order")

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        return validate_ticker(v)


class ClosePositionRequest(BaseModel):
    """Request to close a position."""

    ticker: str = Field(default=DEFAULT_TICKER)
    quantity: Optional[float] = Field(
        default=None,
        description="Shares to close (all if not specified)"
    )

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        return validate_ticker(v)


# ============== Response Schemas ==============

class Signal(BaseModel):
    """Trading signal."""

    time: str
    ticker: str
    action: SignalAction
    confidence: float
    price: float
    expected_return: float
    model_key: str
    uncertainty_std: Optional[float] = None


class Position(BaseModel):
    """Portfolio position."""

    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class Trade(BaseModel):
    """Executed trade record."""

    trade_id: str
    timestamp: str
    ticker: str
    side: str
    quantity: float
    price: float
    value: float
    commission: float
    pnl: float
    pnl_pct: float
    model_used: str
    signal_confidence: float


class Order(BaseModel):
    """Trading order."""

    order_id: str
    timestamp: str
    ticker: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str
    filled_quantity: float = 0
    filled_price: float = 0.0
    commission: float = 0.0
    reason: str = ""


class Alert(BaseModel):
    """Trading alert."""

    alert_id: str
    timestamp: str
    alert_type: str
    severity: str
    ticker: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None


class PortfolioAllocation(BaseModel):
    """Portfolio allocation entry."""

    name: str
    value: float
    color: str


class ConfidenceHistoryPoint(BaseModel):
    """Confidence history data point."""

    time: str
    confidence: float
    signal: int  # 1 for BUY, -1 for SELL, 0 for HOLD


class PerformanceMetrics(BaseModel):
    """Trading performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: Optional[str] = None


class MarketData(BaseModel):
    """Current market data."""

    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    open: float
    high: float
    low: float
    timestamp: str
    is_market_open: bool


class AgentStatus(BaseModel):
    """Trading agent status response."""

    is_running: bool
    trading_mode: str
    model_key: Optional[str] = None
    ticker: str = DEFAULT_TICKER

    # Portfolio summary
    cash: float = 100000.0
    positions_value: float = 0.0
    total_value: float = 100000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    # Statistics
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    avg_confidence: float = 0.0
    pnl_today: float = 0.0

    # Configuration
    config: Optional[Dict[str, Any]] = None

    # Data for charts
    signals: List[Signal] = []
    positions: List[Position] = []
    portfolio_allocation: List[PortfolioAllocation] = []
    confidence_history: List[ConfidenceHistoryPoint] = []

    # Performance
    performance: Optional[PerformanceMetrics] = None

    # Alerts
    recent_alerts: List[Alert] = []

    # Market data
    market_data: Optional[MarketData] = None

    # Timestamps
    started_at: Optional[str] = None
    last_update: Optional[str] = None


class AgentStartResponse(BaseModel):
    """Response when starting agent."""

    success: bool
    message: str
    agent_id: str
    config: Dict[str, Any]


class AgentStopResponse(BaseModel):
    """Response when stopping agent."""

    success: bool
    message: str
    final_portfolio_value: float
    total_pnl: float
    total_trades: int


class TradeHistoryResponse(BaseModel):
    """Response for trade history."""

    trades: List[Trade]
    total: int
    page: int
    page_size: int
    summary: PerformanceMetrics


class OrderHistoryResponse(BaseModel):
    """Response for order history."""

    orders: List[Order]
    total: int
    page: int
    page_size: int


class AlertHistoryResponse(BaseModel):
    """Response for alert history."""

    alerts: List[Alert]
    total: int
    unread_count: int


class PortfolioHistoryPoint(BaseModel):
    """Portfolio history data point."""

    timestamp: str
    total_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float


class PortfolioHistoryResponse(BaseModel):
    """Response for portfolio history."""

    history: List[PortfolioHistoryPoint]
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float


class RiskMetrics(BaseModel):
    """Risk metrics for the portfolio."""

    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float  # Conditional VaR
    beta: float
    volatility: float
    correlation_spy: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


class RiskMetricsResponse(BaseModel):
    """Response for risk metrics."""

    metrics: RiskMetrics
    calculated_at: str
    lookback_days: int
