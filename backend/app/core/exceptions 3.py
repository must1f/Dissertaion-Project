"""Custom exceptions for the API."""

from typing import Any, Optional


class APIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ModelNotFoundError(APIException):
    """Raised when a model is not found."""

    def __init__(self, model_key: str):
        super().__init__(
            message=f"Model '{model_key}' not found",
            status_code=404,
            details={"model_key": model_key},
        )


class ModelNotTrainedError(APIException):
    """Raised when trying to use an untrained model."""

    def __init__(self, model_key: str):
        super().__init__(
            message=f"Model '{model_key}' has not been trained yet",
            status_code=400,
            details={"model_key": model_key},
        )


class DataNotFoundError(APIException):
    """Raised when requested data is not found."""

    def __init__(self, ticker: str, message: Optional[str] = None):
        super().__init__(
            message=message or f"No data found for ticker '{ticker}'",
            status_code=404,
            details={"ticker": ticker},
        )


class DataFetchError(APIException):
    """Raised when data fetching fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class PredictionError(APIException):
    """Raised when prediction fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class TrainingError(APIException):
    """Raised when training fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class BacktestError(APIException):
    """Raised when backtesting fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class ValidationError(APIException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=422,
            details=details,
        )


class DatabaseError(APIException):
    """Raised when database operations fail."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )
