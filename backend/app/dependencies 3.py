"""Dependency injection for FastAPI."""

from typing import Generator

from backend.app.core.database import DatabaseManager, get_db
from backend.app.services.model_service import ModelService
from backend.app.services.data_service import DataService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.training_service import TrainingService
from backend.app.services.backtest_service import BacktestService
from backend.app.services.metrics_service import MetricsService


# Service singletons
_model_service: ModelService = None
_data_service: DataService = None
_prediction_service: PredictionService = None
_training_service: TrainingService = None
_backtest_service: BacktestService = None
_metrics_service: MetricsService = None


def get_model_service() -> ModelService:
    """Get model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_data_service() -> DataService:
    """Get data service instance."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def get_training_service() -> TrainingService:
    """Get training service instance."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service


def get_backtest_service() -> BacktestService:
    """Get backtest service instance."""
    global _backtest_service
    if _backtest_service is None:
        _backtest_service = BacktestService()
    return _backtest_service


def get_metrics_service() -> MetricsService:
    """Get metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
