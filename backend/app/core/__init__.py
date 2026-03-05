"""Core package - database, exception handling, and configuration."""

from .database import DatabaseManager, get_db
from .exceptions import (
    APIException,
    ModelNotFoundError,
    ModelNotTrainedError,
    DataNotFoundError,
    DataFetchError,
    PredictionError,
    TrainingError,
    BacktestError,
    ValidationError,
    DatabaseError,
)
from .config import (
    ALLOWED_TICKERS,
    DEFAULT_TICKER,
    TICKER_NAMES,
    is_ticker_allowed,
    validate_ticker,
    validate_tickers,
)

__all__ = [
    "DatabaseManager",
    "get_db",
    "APIException",
    "ModelNotFoundError",
    "ModelNotTrainedError",
    "DataNotFoundError",
    "DataFetchError",
    "PredictionError",
    "TrainingError",
    "BacktestError",
    "ValidationError",
    "DatabaseError",
    "ALLOWED_TICKERS",
    "DEFAULT_TICKER",
    "TICKER_NAMES",
    "is_ticker_allowed",
    "validate_ticker",
    "validate_tickers",
]
