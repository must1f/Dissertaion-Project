"""Pydantic schemas for prediction endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker, validate_tickers


class UncertaintyMethod(str, Enum):
    """Methods for uncertainty estimation."""

    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    BOTH = "both"


class SignalAction(str, Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PredictionRequest(BaseModel):
    """Request for running predictions. Only S&P 500 data is supported."""

    ticker: str = Field(
        default=DEFAULT_TICKER,
        description=f"Stock ticker symbol (only {', '.join(ALLOWED_TICKERS)} supported)"
    )
    model_key: str = Field(..., description="Model to use for prediction")
    sequence_length: int = Field(
        default=60,
        ge=10,
        le=200,
        description="Number of historical days to use"
    )
    horizon: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Prediction horizon in days"
    )
    estimate_uncertainty: bool = Field(
        default=True,
        description="Whether to estimate prediction uncertainty"
    )
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.MC_DROPOUT,
        description="Method for uncertainty estimation"
    )
    n_mc_samples: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of MC dropout samples"
    )
    generate_signal: bool = Field(
        default=True,
        description="Whether to generate trading signal"
    )
    signal_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Threshold for signal generation"
    )

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        """Validate that only allowed tickers are used for predictions."""
        return validate_ticker(v)


class PredictionInterval(BaseModel):
    """Prediction confidence interval."""

    lower: float
    upper: float
    confidence: float = 0.95


class PredictionResult(BaseModel):
    """Single prediction result."""

    timestamp: datetime
    ticker: str
    model_key: str

    # Predictions
    predicted_price: float
    predicted_return: float
    current_price: float

    # Uncertainty (optional)
    uncertainty_std: Optional[float] = None
    prediction_interval: Optional[PredictionInterval] = None
    confidence_score: Optional[float] = None

    # Signal (optional)
    signal_action: Optional[SignalAction] = None
    expected_return: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response for prediction request."""

    success: bool
    prediction: PredictionResult
    model_info: Dict[str, Any]
    physics_parameters: Optional[Dict[str, float]] = None
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions. Only S&P 500 data is supported."""

    tickers: List[str] = Field(
        default=[DEFAULT_TICKER],
        min_length=1,
        max_length=1,
        description=f"Stock tickers (only {', '.join(ALLOWED_TICKERS)} supported)"
    )
    model_key: str
    sequence_length: int = 60
    estimate_uncertainty: bool = True

    @field_validator('tickers')
    @classmethod
    def validate_allowed_tickers(cls, v: List[str]) -> List[str]:
        """Validate that only allowed tickers are requested."""
        return validate_tickers(v)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    success: bool
    predictions: List[PredictionResult]
    failed_tickers: List[str] = []
    total_processing_time_ms: float


class PredictionHistoryItem(BaseModel):
    """Historical prediction record."""

    id: str
    timestamp: datetime
    ticker: str
    model_key: str
    predicted_price: float
    actual_price: Optional[float] = None
    error: Optional[float] = None
    signal_action: Optional[SignalAction] = None


class PredictionHistoryResponse(BaseModel):
    """Response for prediction history."""

    predictions: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int


class LatestPredictionResponse(BaseModel):
    """Response for latest predictions for a ticker."""

    ticker: str
    predictions: Dict[str, PredictionResult]  # model_key -> prediction
    consensus_signal: Optional[SignalAction] = None
    last_updated: datetime
