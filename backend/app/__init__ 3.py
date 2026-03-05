"""PINN Financial Forecasting Backend API."""

__version__ = "1.0.0"

from .config import settings
from .main import app

__all__ = ["app", "settings", "__version__"]
