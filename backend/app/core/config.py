"""Application configuration and constants."""

from typing import List

# Allowed tickers - restricted to S&P 500 only
# All models are trained exclusively on S&P 500 data
ALLOWED_TICKERS: List[str] = ["^GSPC"]

# Default ticker for all operations
DEFAULT_TICKER: str = "^GSPC"

# Ticker display names
TICKER_NAMES = {
    "^GSPC": "S&P 500 Index",
}


def is_ticker_allowed(ticker: str) -> bool:
    """Check if a ticker is in the allowed list."""
    return ticker.upper() in [t.upper() for t in ALLOWED_TICKERS]


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize a ticker symbol.

    Raises ValueError if ticker is not allowed.
    Returns the normalized ticker symbol.
    """
    normalized = ticker.upper()
    if normalized not in [t.upper() for t in ALLOWED_TICKERS]:
        allowed_str = ", ".join(ALLOWED_TICKERS)
        raise ValueError(
            f"Ticker '{ticker}' is not allowed. "
            f"Only S&P 500 data is supported. Allowed tickers: {allowed_str}"
        )
    return normalized


def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate and normalize a list of ticker symbols.

    Raises ValueError if any ticker is not allowed.
    Returns list of normalized ticker symbols.
    """
    return [validate_ticker(t) for t in tickers]
