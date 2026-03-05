"""
Real-time Market Data Service

Provides live and historical stock market data with:
- Real-time price streaming (via yfinance)
- Intraday data (1m, 5m, 15m, 1h intervals)
- Technical indicators calculation
- WebSocket-style callbacks for price updates
- Data caching for performance

Note: This is for educational/research purposes only.
For production trading, use proper market data providers (IEX, Polygon, etc.)
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue
import pandas as pd
import numpy as np
import yfinance as yf

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketQuote:
    """Real-time market quote"""
    ticker: str
    timestamp: datetime
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    change: float = 0.0
    change_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'change': self.change,
            'change_pct': self.change_pct
        }


@dataclass
class MarketData:
    """Historical market data with technical indicators"""
    ticker: str
    data: pd.DataFrame
    last_update: datetime
    indicators: Dict[str, pd.Series] = field(default_factory=dict)


class RealTimeDataService:
    """
    Real-time market data service

    Features:
    - Live price fetching with configurable intervals
    - Historical data retrieval
    - Technical indicator calculation
    - Price change callbacks
    - Data caching
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        update_interval: float = 5.0,  # seconds
        cache_duration: int = 60  # seconds
    ):
        """
        Initialize real-time data service

        Args:
            tickers: List of tickers to track
            update_interval: How often to fetch new data (seconds)
            cache_duration: How long to cache data (seconds)
        """
        self.tickers = tickers or []
        self.update_interval = update_interval
        self.cache_duration = cache_duration

        # Data storage
        self._quotes: Dict[str, MarketQuote] = {}
        self._market_data: Dict[str, MarketData] = {}
        self._price_history: Dict[str, List[float]] = {}

        # Callbacks
        self._callbacks: List[Callable[[MarketQuote], None]] = []

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._update_queue = Queue()

        logger.info(f"RealTimeDataService initialized for {len(self.tickers)} tickers")

    def add_ticker(self, ticker: str):
        """Add a ticker to track"""
        ticker = ticker.upper()
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"Added ticker: {ticker}")

    def remove_ticker(self, ticker: str):
        """Remove a ticker from tracking"""
        ticker = ticker.upper()
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            if ticker in self._quotes:
                del self._quotes[ticker]
            logger.info(f"Removed ticker: {ticker}")

    def add_callback(self, callback: Callable[[MarketQuote], None]):
        """Add a callback for price updates"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MarketQuote], None]):
        """Remove a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, quote: MarketQuote):
        """Notify all callbacks of a price update"""
        for callback in self._callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_quote(self, ticker: str) -> Optional[MarketQuote]:
        """
        Get the latest quote for a ticker

        Args:
            ticker: Ticker symbol

        Returns:
            MarketQuote or None if not available
        """
        ticker = ticker.upper()

        with self._lock:
            if ticker in self._quotes:
                return self._quotes[ticker]

        # Fetch if not cached
        return self.fetch_quote(ticker)

    def fetch_quote(self, ticker: str) -> Optional[MarketQuote]:
        """
        Fetch a fresh quote from the market

        Args:
            ticker: Ticker symbol

        Returns:
            MarketQuote or None on error
        """
        ticker = ticker.upper()

        try:
            stock = yf.Ticker(ticker)

            # Get current price info
            info = stock.info

            # Get intraday data for more accurate pricing
            hist = stock.history(period='1d', interval='1m')

            if hist.empty:
                # Fallback to daily data
                hist = stock.history(period='5d')
                if hist.empty:
                    logger.warning(f"No data available for {ticker}")
                    return None

            latest = hist.iloc[-1]
            prev_close = info.get('previousClose', latest['Close'])

            quote = MarketQuote(
                ticker=ticker,
                timestamp=datetime.now(),
                price=float(latest['Close']),
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                volume=int(latest['Volume']),
                bid=info.get('bid'),
                ask=info.get('ask'),
                bid_size=info.get('bidSize'),
                ask_size=info.get('askSize'),
                change=float(latest['Close'] - prev_close) if prev_close else 0.0,
                change_pct=float((latest['Close'] - prev_close) / prev_close * 100) if prev_close else 0.0
            )

            with self._lock:
                self._quotes[ticker] = quote

                # Track price history
                if ticker not in self._price_history:
                    self._price_history[ticker] = []
                self._price_history[ticker].append(quote.price)

                # Limit history size
                if len(self._price_history[ticker]) > 1000:
                    self._price_history[ticker] = self._price_history[ticker][-1000:]

            return quote

        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {e}")
            return None

    def fetch_historical_data(
        self,
        ticker: str,
        period: str = '60d',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a ticker

        Args:
            ticker: Ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker.upper()

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return None

            # Rename columns to lowercase
            hist.columns = [c.lower() for c in hist.columns]
            hist['ticker'] = ticker
            hist['time'] = hist.index

            # Cache the data
            with self._lock:
                self._market_data[ticker] = MarketData(
                    ticker=ticker,
                    data=hist,
                    last_update=datetime.now()
                )

            logger.info(f"Fetched {len(hist)} records for {ticker}")
            return hist

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def fetch_intraday_data(
        self,
        ticker: str,
        days: int = 1,
        interval: str = '5m'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for a ticker

        Args:
            ticker: Ticker symbol
            days: Number of days of intraday data
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)

        Returns:
            DataFrame with intraday OHLCV data
        """
        period = f'{days}d'
        return self.fetch_historical_data(ticker, period=period, interval=interval)

    def calculate_indicators(
        self,
        ticker: str,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for a ticker

        Args:
            ticker: Ticker symbol
            data: Optional DataFrame (fetches if not provided)

        Returns:
            Dictionary of indicator Series
        """
        ticker = ticker.upper()

        if data is None:
            if ticker in self._market_data:
                data = self._market_data[ticker].data
            else:
                data = self.fetch_historical_data(ticker)

        if data is None or data.empty:
            return {}

        indicators = {}

        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            # Moving Averages
            indicators['sma_20'] = close.rolling(window=20).mean()
            indicators['sma_50'] = close.rolling(window=50).mean()
            indicators['ema_12'] = close.ewm(span=12, adjust=False).mean()
            indicators['ema_26'] = close.ewm(span=26, adjust=False).mean()

            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            sma_20 = indicators['sma_20']
            std_20 = close.rolling(window=20).std()
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20

            # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(window=14).mean()

            # Volume indicators
            indicators['volume_sma'] = volume.rolling(window=20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']

            # Volatility
            returns = close.pct_change()
            indicators['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)

            # Store in cache
            with self._lock:
                if ticker in self._market_data:
                    self._market_data[ticker].indicators = indicators

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            return {}

    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """Get cached market data for a ticker"""
        ticker = ticker.upper()
        with self._lock:
            return self._market_data.get(ticker)

    def get_all_quotes(self) -> Dict[str, MarketQuote]:
        """Get all current quotes"""
        with self._lock:
            return dict(self._quotes)

    def get_price_history(self, ticker: str) -> List[float]:
        """Get recent price history for a ticker"""
        ticker = ticker.upper()
        with self._lock:
            return list(self._price_history.get(ticker, []))

    def start_streaming(self):
        """Start the background price streaming"""
        if self._running:
            logger.warning("Streaming already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._thread.start()
        logger.info("Price streaming started")

    def stop_streaming(self):
        """Stop the background price streaming"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Price streaming stopped")

    def _streaming_loop(self):
        """Background loop for fetching prices"""
        while self._running:
            for ticker in self.tickers:
                if not self._running:
                    break

                quote = self.fetch_quote(ticker)
                if quote:
                    self._notify_callbacks(quote)

                # Small delay between tickers to avoid rate limiting
                time.sleep(0.1)

            # Wait for next update interval
            time.sleep(self.update_interval)

    def is_market_open(self) -> bool:
        """
        Check if US stock market is open

        Returns:
            True if market is open
        """
        now = datetime.now()

        # Check if weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        # This is a simplified check - doesn't account for holidays
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status

        Returns:
            Dictionary with market status information
        """
        now = datetime.now()
        is_open = self.is_market_open()

        status = {
            'is_open': is_open,
            'current_time': now.isoformat(),
            'day_of_week': now.strftime('%A'),
            'tickers_tracked': len(self.tickers),
            'quotes_cached': len(self._quotes),
            'streaming': self._running
        }

        if not is_open:
            # Calculate time until market opens
            if now.weekday() >= 5:
                # Weekend - calculate to Monday
                days_until_monday = 7 - now.weekday()
                next_open = now + timedelta(days=days_until_monday)
            else:
                next_open = now + timedelta(days=1 if now.hour >= 16 else 0)

            next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
            status['next_open'] = next_open.isoformat()

        return status


class SimulatedDataService(RealTimeDataService):
    """
    Simulated data service for backtesting and paper trading

    Uses historical data to simulate real-time price feeds
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        speed_multiplier: float = 1.0
    ):
        """
        Initialize simulated data service

        Args:
            tickers: List of tickers
            start_date: Simulation start date (YYYY-MM-DD)
            end_date: Simulation end date (YYYY-MM-DD)
            speed_multiplier: How fast to run simulation (1.0 = real-time)
        """
        super().__init__(tickers=tickers)

        self.start_date = start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.speed_multiplier = speed_multiplier

        self._simulation_data: Dict[str, pd.DataFrame] = {}
        self._current_index: Dict[str, int] = {}
        self._simulation_running = False

        logger.info(f"SimulatedDataService initialized: {self.start_date} to {self.end_date}")

    def load_simulation_data(self):
        """Load historical data for simulation"""
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date, interval='1h')

                if not hist.empty:
                    hist.columns = [c.lower() for c in hist.columns]
                    hist['ticker'] = ticker
                    self._simulation_data[ticker] = hist
                    self._current_index[ticker] = 0
                    logger.info(f"Loaded {len(hist)} simulation records for {ticker}")

            except Exception as e:
                logger.error(f"Error loading simulation data for {ticker}: {e}")

    def get_next_quote(self, ticker: str) -> Optional[MarketQuote]:
        """Get the next simulated quote"""
        ticker = ticker.upper()

        if ticker not in self._simulation_data:
            return None

        data = self._simulation_data[ticker]
        idx = self._current_index.get(ticker, 0)

        if idx >= len(data):
            return None

        row = data.iloc[idx]
        timestamp = data.index[idx]

        # Get previous close for change calculation
        prev_close = data.iloc[idx - 1]['close'] if idx > 0 else row['close']

        quote = MarketQuote(
            ticker=ticker,
            timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
            price=float(row['close']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
            change=float(row['close'] - prev_close),
            change_pct=float((row['close'] - prev_close) / prev_close * 100) if prev_close else 0.0
        )

        # Advance index
        self._current_index[ticker] = idx + 1

        # Store quote
        with self._lock:
            self._quotes[ticker] = quote

        return quote

    def reset_simulation(self):
        """Reset simulation to beginning"""
        for ticker in self._current_index:
            self._current_index[ticker] = 0
        self._quotes.clear()
        logger.info("Simulation reset to beginning")

    def run_simulation(self, callback: Optional[Callable[[MarketQuote], None]] = None):
        """
        Run the simulation

        Args:
            callback: Optional callback for each quote
        """
        if not self._simulation_data:
            self.load_simulation_data()

        self._simulation_running = True

        while self._simulation_running:
            all_done = True

            for ticker in self.tickers:
                quote = self.get_next_quote(ticker)

                if quote:
                    all_done = False
                    self._notify_callbacks(quote)
                    if callback:
                        callback(quote)

            if all_done:
                break

            # Simulated delay
            time.sleep(0.1 / self.speed_multiplier)

        logger.info("Simulation completed")

    def stop_simulation(self):
        """Stop running simulation"""
        self._simulation_running = False


# Convenience functions

def get_live_quote(ticker: str) -> Optional[MarketQuote]:
    """Quick function to get a live quote"""
    service = RealTimeDataService()
    return service.fetch_quote(ticker)


def get_multiple_quotes(tickers: List[str]) -> Dict[str, MarketQuote]:
    """Get quotes for multiple tickers"""
    service = RealTimeDataService(tickers=tickers)
    quotes = {}
    for ticker in tickers:
        quote = service.fetch_quote(ticker)
        if quote:
            quotes[ticker] = quote
    return quotes


def get_intraday_with_indicators(
    ticker: str,
    interval: str = '5m',
    days: int = 1
) -> Optional[pd.DataFrame]:
    """Get intraday data with technical indicators"""
    service = RealTimeDataService(tickers=[ticker])
    data = service.fetch_intraday_data(ticker, days=days, interval=interval)

    if data is not None:
        indicators = service.calculate_indicators(ticker, data)
        for name, series in indicators.items():
            data[name] = series

    return data
