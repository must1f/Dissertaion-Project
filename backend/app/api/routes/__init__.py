"""API routes package."""

from .data import router as data_router
from .models import router as models_router
from .predictions import router as predictions_router
from .training import router as training_router
from .backtesting import router as backtesting_router
from .metrics import router as metrics_router
from .monte_carlo import router as monte_carlo_router
from .websocket import router as websocket_router
from .trading import router as trading_router
from .dissertation import router as dissertation_router
from .spectral import router as spectral_router

__all__ = [
    "data_router",
    "models_router",
    "predictions_router",
    "training_router",
    "backtesting_router",
    "metrics_router",
    "monte_carlo_router",
    "websocket_router",
    "trading_router",
    "dissertation_router",
    "spectral_router",
]
