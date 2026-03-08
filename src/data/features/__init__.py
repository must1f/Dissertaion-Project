"""Feature engineering aggregator."""

from .returns import add_adjusted_returns
from .volatility import add_rolling_volatility
from .trend import add_momentum_trend
from .cross_asset import add_cross_asset_spreads
from .regime import add_regime_markers

__all__ = [
    "add_adjusted_returns",
    "add_rolling_volatility",
    "add_momentum_trend",
    "add_cross_asset_spreads",
    "add_regime_markers",
]
