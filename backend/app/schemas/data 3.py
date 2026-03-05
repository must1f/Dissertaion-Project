"""Pydantic schemas for data endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker, validate_tickers


class StockInfo(BaseModel):
    """Basic stock information."""

    ticker: str
    name: Optional[str] = None
    first_date: Optional[datetime] = None
    last_date: Optional[datetime] = None
    record_count: int = 0


class StockListResponse(BaseModel):
    """Response for listing available stocks."""

    stocks: List[StockInfo]
    total: int


class OHLCVData(BaseModel):
    """OHLCV data point."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str


class StockDataResponse(BaseModel):
    """Response containing stock price data."""

    ticker: str
    data: List[OHLCVData]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    count: int


class FeatureData(BaseModel):
    """Engineered feature data point."""

    timestamp: datetime
    ticker: str

    # Price features
    log_return: Optional[float] = None
    simple_return: Optional[float] = None

    # Volatility features
    rolling_volatility_5: Optional[float] = None
    rolling_volatility_20: Optional[float] = None
    rolling_volatility_60: Optional[float] = None

    # Momentum features
    momentum_5: Optional[float] = None
    momentum_10: Optional[float] = None
    momentum_20: Optional[float] = None
    momentum_60: Optional[float] = None

    # Technical indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None


class FeaturesResponse(BaseModel):
    """Response containing engineered features."""

    ticker: str
    features: List[FeatureData]
    feature_names: List[str]
    count: int


class FetchDataRequest(BaseModel):
    """Request to fetch new stock data. Only S&P 500 data is supported."""

    tickers: List[str] = Field(
        default=[DEFAULT_TICKER],
        min_length=1,
        max_length=1,
        description=f"Stock tickers (only {', '.join(ALLOWED_TICKERS)} supported)"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format"
    )
    interval: str = Field(
        default="1d",
        description="Data interval (1d, 1h, etc.)"
    )
    force_refresh: bool = Field(
        default=False,
        description="Force refresh even if data exists"
    )

    @field_validator('tickers')
    @classmethod
    def validate_allowed_tickers(cls, v: List[str]) -> List[str]:
        """Validate that only allowed tickers are requested."""
        return validate_tickers(v)


class FetchDataResponse(BaseModel):
    """Response after fetching data."""

    success: bool
    tickers_fetched: List[str]
    records_added: int
    message: str
    errors: Optional[Dict[str, str]] = None
