"""
Universe configuration helpers for multi-asset experiments.

This module centralizes the definition of asset universes and provides
deterministic hashing so caching, dataset versions, and experiment configs
can reference the exact same universe definition.
"""

from dataclasses import dataclass, field, asdict
from hashlib import sha256
from typing import List, Dict, Any


# Stable base universe for the benchmark dataset
BASE_UNIVERSE = [
    "SPY",  # S&P 500 ETF (target)
    "QQQ",  # NASDAQ 100
    "IWM",  # Russell 2000
    "XLK",  # Tech sector
    "XLF",  # Financials
    "XLE",  # Energy
    "^VIX",  # Implied volatility
    "^TNX",  # 10Y yield
]

# Optional macro/commodity overlays
OPTIONAL_UNIVERSE = [
    "GC=F",  # Gold futures
    "CL=F",  # WTI crude futures
]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _normalize_symbol(symbol: str) -> str:
    alias_map = {
        "VIX": "^VIX",
        "TNX": "^TNX",
    }
    return alias_map.get(symbol, symbol)


@dataclass
class UniverseDefinition:
    """Defines the assets and calendar for an experiment."""

    name: str = "core_multi_asset"
    symbols: List[str] = field(default_factory=lambda: list(BASE_UNIVERSE + OPTIONAL_UNIVERSE))
    interval: str = "1d"
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"
    calendar: str = "NYSE"
    holidays: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        payload = f"{self.name}|{self.interval}|{self.start_date}|{self.end_date}|{','.join(sorted(self.symbols))}"
        return sha256(payload.encode()).hexdigest()[:12]


def default_universe() -> UniverseDefinition:
    """Convenience accessor for the core benchmark universe."""

    return UniverseDefinition()


def universe_from_config(data_cfg) -> UniverseDefinition:
    """
    Build a UniverseDefinition from a DataConfig-like object.

    Accepts objects that have attributes: universe_name, tickers, start_date,
    end_date, interval, and calendar.
    """

    base = list(getattr(data_cfg, "base_universe", BASE_UNIVERSE))
    optional = list(getattr(data_cfg, "optional_universe", OPTIONAL_UNIVERSE))
    include_optional = bool(getattr(data_cfg, "include_optional_assets", True))
    provided = list(getattr(data_cfg, "tickers", []) or [])

    symbols_seed = base + (optional if include_optional else []) + provided
    symbols = _dedupe_preserve_order([_normalize_symbol(s) for s in symbols_seed])

    return UniverseDefinition(
        name=getattr(data_cfg, "universe_name", "core_multi_asset"),
        symbols=symbols,
        interval=getattr(data_cfg, "interval", "1d"),
        start_date=getattr(data_cfg, "start_date", "2005-01-01"),
        end_date=getattr(data_cfg, "end_date", "2025-12-31"),
        calendar=getattr(data_cfg, "calendar", "NYSE"),
        holidays=list(getattr(data_cfg, "master_calendar_holidays", []) or []),
    )
