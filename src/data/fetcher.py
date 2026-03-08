"""
Data fetcher for financial time series data from multiple sources
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, cast
from datetime import datetime
import pandas as pd
import yfinance as yf

# tqdm is optional; provide a no-op fallback so training doesn't fail
try:  # pragma: no cover
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_db
from .cache import CacheManager
from .universe import universe_from_config, UniverseDefinition
from .quality import run_qa

logger = get_logger(__name__)


class DataFetcher:
    """
    Fetches financial data from multiple sources:
    - yfinance (primary, free, no API key required)
    - Alpha Vantage (backup, requires API key, rate limited)
    """

    def __init__(self, config=None):
        """
        Initialize data fetcher

        Args:
            config: Optional Config object
        """
        self.config = config or get_config()
        self.db = get_db()
        cache_root = Path(self.config.data_dir) / "raw_cache"
        self.cache = CacheManager(cache_root)

        # Check database connection status
        if not self.db.is_connected():
            logger.warning("Database not available, will use Parquet files for data storage")


    def fetch_yfinance(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch data using yfinance (primary source)

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with columns: time, ticker, open, high, low, close, volume, adjusted_close
        """
        logger.info(f"Fetching {len(tickers)} tickers from yfinance: {start_date} to {end_date}")

        all_data = []

        for ticker in tqdm(tickers, desc="Downloading tickers"):
            try:
                # Download data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,  # Keep unadjusted prices for trading
                )

                if not isinstance(data, pd.DataFrame) or data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                # Prepare DataFrame
                df = cast(pd.DataFrame, pd.DataFrame(data.copy()))

                # Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df['ticker'] = ticker
                df['time'] = df.index
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adjusted_close'
                })

                # Some instruments may not provide adjusted_close; fall back to close.
                if 'adjusted_close' not in df.columns and 'close' in df.columns:
                    df['adjusted_close'] = df['close']

                # Select and reorder columns
                df = df[['time', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']].copy()

                # Remove timezone info for database compatibility
                df['time'] = pd.DatetimeIndex(pd.to_datetime(df['time'], errors='coerce')).tz_localize(None)

                all_data.append(df)

                # Rate limiting (be nice to Yahoo servers)
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to fetch {ticker} from yfinance: {e}")
                continue

        if not all_data:
            logger.error("No data fetched from yfinance")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully fetched {len(combined_df)} records for {len(all_data)} tickers")

        return combined_df

    def fetch_and_store(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data and store in database with Parquet backup

        Args:
            tickers: List of tickers (None = use config)
            start_date: Start date (None = use config)
            end_date: End date (None = use config)
            force_refresh: If True, refetch even if data exists

        Returns:
            DataFrame with fetched data
        """
        tickers = tickers or self.config.data.tickers
        start_date = start_date or self.config.data.start_date
        end_date = end_date or self.config.data.end_date

        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")

        # Check if data already exists
        if not force_refresh:
            existing_data = self.db.get_stock_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )

            if not existing_data.empty:
                logger.info(f"Found {len(existing_data)} existing records in database")
                # Check completeness
                expected_tickers = set(tickers)
                actual_tickers = set(existing_data['ticker'].unique())

                if expected_tickers == actual_tickers:
                    logger.info("All tickers present in database, skipping fetch")
                    return existing_data

        # Fetch data
        df = self.fetch_yfinance(tickers, start_date, end_date)

        if df.empty:
            logger.error("Failed to fetch data")
            return df

        # Store in database
        try:
            logger.info("Storing data in TimescaleDB...")
            self.db.bulk_insert_stock_prices(df)
            logger.info("Data stored successfully in database")
        except Exception as e:
            logger.error(f"Failed to store in database: {e}")
            # Continue to save Parquet backup

        # Save Parquet backup
        try:
            parquet_path = self.config.data_dir / "parquet" / f"stock_prices_{start_date}_{end_date}.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)
            logger.info(f"Parquet backup saved to {parquet_path}")
        except Exception as e:
            logger.error(f"Failed to save Parquet backup: {e}")

        return df

    def fetch_multi_asset_cached(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        force_refresh: Optional[bool] = None,
        dataset_tag: str = "raw_cache",
    ) -> pd.DataFrame:
        """
        Universe-aware fetch with Parquet cache and QA report.
        """

        data_cfg = getattr(self.config, "data", None)
        universe: UniverseDefinition = universe_from_config(data_cfg)
        start_date = start_date or universe.start_date
        end_date = end_date or universe.end_date
        interval = interval or universe.interval
        force_refresh = force_refresh if force_refresh is not None else getattr(data_cfg, "force_refresh", False)

        cache_paths = self.cache.paths(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            dataset_tag=dataset_tag,
        )

        ttl_days = getattr(data_cfg, "cache_ttl_days", None)

        if not force_refresh:
            cached, meta, _ = self.cache.load_with_meta(cache_paths, ttl_days=ttl_days)
            if cached is not None and not cached.empty:
                logger.info("Loaded multi-asset data from cache")
                return cached

        df = self.fetch_yfinance(universe.symbols, start_date, end_date, interval)
        if df.empty:
            logger.error("No data returned for universe fetch")
            return df

        qa_report = run_qa(df)
        metadata = {
            "source": "yfinance",
            "force_refresh": force_refresh,
            "tickers": universe.symbols,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "cache_key": cache_paths.cache_key(),
            "universe_hash": universe.hash(),
        }
        self.cache.save(
            cache_paths,
            df,
            metadata=metadata,
            qa_report=qa_report,
        )

        logger.info("Saved multi-asset data to cache")
        return df

    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load data from Parquet file

        Args:
            filename: Parquet filename

        Returns:
            DataFrame with stock prices
        """
        parquet_path = self.config.data_dir / "parquet" / filename
        logger.info(f"Loading data from {parquet_path}")

        try:
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} records from Parquet")
            return df
        except Exception as e:
            logger.error(f"Failed to load from Parquet: {e}")
            return pd.DataFrame()

    def get_sp500_tickers(self) -> List[str]:
        """
        Fetch current S&P 500 constituent tickers from Wikipedia

        Returns:
            List of ticker symbols
        """
        try:
            logger.info("Fetching S&P 500 tickers from Wikipedia")
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = table['Symbol'].tolist()

            # Clean tickers (some have dots that need conversion)
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            logger.info(f"Found {len(tickers)} S&P 500 tickers")
            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            return []


if __name__ == "__main__":
    """Run data fetcher as standalone script"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.utils.logger import setup_logger
    
    # Setup logging
    setup_logger(level="INFO")
    
    logger.info("=" * 80)
    logger.info("DATA FETCHING")
    logger.info("=" * 80)
    
    # Fetch and store data
    config = get_config()
    fetcher = DataFetcher(config)
    
    # Fetch top 10 tickers (for quick demo)
    tickers = config.data.tickers[:10]
    logger.info(f"Fetching data for {len(tickers)} tickers: {tickers}")
    
    df = fetcher.fetch_and_store(
        tickers=tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=False
    )
    
    if df.empty:
        logger.error("Failed to fetch data!")
        sys.exit(1)
    
    logger.info(f"✓ Successfully fetched {len(df)} records for {len(df['ticker'].unique())} tickers")
    logger.info("=" * 80)
