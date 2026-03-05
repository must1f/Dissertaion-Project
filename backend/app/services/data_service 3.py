"""Service for data management with direct yfinance integration."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import DataNotFoundError, DataFetchError
from backend.app.schemas.data import (
    StockInfo,
    OHLCVData,
    FeatureData,
    StockDataResponse,
    FeaturesResponse,
)

# Import from existing src/ (optional, for preprocessing)
try:
    from src.data.preprocessor import DataPreprocessor
    HAS_PREPROCESSOR = True
except ImportError:
    HAS_PREPROCESSOR = False
    DataPreprocessor = None


class DataService:
    """Service for managing stock data with incremental updates."""

    # Master parquet file for all stock data
    MASTER_PARQUET = "stock_prices_master.parquet"

    def __init__(self):
        """Initialize data service."""
        self._preprocessor: Optional[DataPreprocessor] = None
        self._data_cache: Dict[str, pd.DataFrame] = {}

    @property
    def preprocessor(self) -> "DataPreprocessor":
        """Lazy load preprocessor."""
        if self._preprocessor is None:
            if not HAS_PREPROCESSOR:
                raise RuntimeError("Preprocessor not available")
            self._preprocessor = DataPreprocessor()
        return self._preprocessor

    def _get_master_parquet_path(self) -> Path:
        """Get path to master parquet file."""
        parquet_dir = settings.data_path / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        return parquet_dir / self.MASTER_PARQUET

    def _load_master_data(self) -> pd.DataFrame:
        """Load all data from master parquet file."""
        master_path = self._get_master_parquet_path()
        if master_path.exists():
            try:
                df = pd.read_parquet(master_path)
                # Normalize time column
                if "time" in df.columns and "timestamp" not in df.columns:
                    df["timestamp"] = df["time"]
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            except Exception as e:
                print(f"Error loading master parquet: {e}")
        return pd.DataFrame()

    def _save_master_data(self, df: pd.DataFrame) -> None:
        """Save data to master parquet file (merge with existing)."""
        master_path = self._get_master_parquet_path()

        # Ensure we have the time column
        if "timestamp" in df.columns and "time" not in df.columns:
            df["time"] = df["timestamp"]

        try:
            if master_path.exists():
                existing = pd.read_parquet(master_path)
                # Combine and deduplicate
                combined = pd.concat([existing, df], ignore_index=True)
                time_col = "time" if "time" in combined.columns else "timestamp"
                combined = combined.drop_duplicates(subset=["ticker", time_col], keep="last")
                combined = combined.sort_values([time_col])
                combined.to_parquet(master_path, compression="snappy", index=False)
                print(f"Updated master parquet: {len(combined)} total records")
            else:
                df.to_parquet(master_path, compression="snappy", index=False)
                print(f"Created master parquet: {len(df)} records")
        except Exception as e:
            print(f"Error saving master parquet: {e}")

    def get_ticker_date_range(self, ticker: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the earliest and latest dates we have for a ticker."""
        df = self._load_master_data()
        if df.empty or "ticker" not in df.columns:
            return None, None

        ticker_data = df[df["ticker"] == ticker]
        if ticker_data.empty:
            return None, None

        time_col = "timestamp" if "timestamp" in ticker_data.columns else "time"
        if time_col not in ticker_data.columns:
            return None, None

        ticker_data[time_col] = pd.to_datetime(ticker_data[time_col])
        return ticker_data[time_col].min(), ticker_data[time_col].max()

    def get_available_stocks(self) -> List[StockInfo]:
        """Get list of available stock tickers."""
        stocks = []

        # Check for parquet files in data directory (search recursively)
        data_dir = settings.data_path
        if data_dir.exists():
            for parquet_file in data_dir.glob("**/*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    if "ticker" in df.columns:
                        # Normalize time column name
                        time_col = "timestamp" if "timestamp" in df.columns else "time" if "time" in df.columns else None
                        for ticker in df["ticker"].unique():
                            ticker_data = df[df["ticker"] == ticker]
                            stocks.append(
                                StockInfo(
                                    ticker=ticker,
                                    first_date=ticker_data[time_col].min() if time_col else None,
                                    last_date=ticker_data[time_col].max() if time_col else None,
                                    record_count=len(ticker_data),
                                )
                            )
                except Exception as e:
                    print(f"Error reading {parquet_file}: {e}")
                    continue

        # Add common tickers if none found
        if not stocks:
            for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY"]:
                stocks.append(StockInfo(ticker=ticker, record_count=0))

        return stocks

    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> StockDataResponse:
        """Get stock OHLCV data with optional interval resampling."""
        # Check cache (include interval in cache key)
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        if cache_key in self._data_cache:
            df = self._data_cache[cache_key]
        else:
            df = self._load_stock_data(ticker, start_date, end_date)

            # If nothing was found for the requested window, try fetching fresh data
            if df.empty and HAS_YFINANCE:
                try:
                    df = self.fetch_data([ticker], start_date, end_date)
                except Exception:
                    # Swallow and fall through to not mask the original cause
                    pass

            # Resample to monthly if requested
            if not df.empty and interval == "1mo":
                df = self._resample_to_monthly(df)

            self._data_cache[cache_key] = df

        if df.empty:
            raise DataNotFoundError(ticker)

        # Convert to response format
        data = []
        for _, row in df.iterrows():
            data.append(
                OHLCVData(
                    timestamp=row.get("timestamp") or row.name,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0),
                    ticker=ticker,
                )
            )

        return StockDataResponse(
            ticker=ticker,
            data=data,
            start_date=df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
            end_date=df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
            count=len(data),
        )

    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV data to monthly."""
        if df.empty:
            return df

        # Ensure we have a timestamp column
        time_col = "timestamp" if "timestamp" in df.columns else "time"
        if time_col not in df.columns:
            return df

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

        # Resample to monthly using OHLCV aggregation rules
        monthly = df.resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # Reset index and add timestamp column back
        monthly = monthly.reset_index()
        monthly = monthly.rename(columns={time_col: "timestamp"})

        # Add ticker column back if it was present
        if "ticker" in df.columns:
            monthly["ticker"] = df["ticker"].iloc[0] if len(df) > 0 else ""
        elif "ticker" in self._data_cache:
            monthly["ticker"] = self._data_cache.get("ticker", "")

        return monthly

    def _load_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load stock data from files or fetch if needed."""
        # Collect data from ALL parquet files for this ticker
        all_ticker_data = []
        data_dir = settings.data_path

        for parquet_file in data_dir.glob("**/*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                if "ticker" in df.columns and ticker in df["ticker"].values:
                    ticker_data = df[df["ticker"] == ticker].copy()

                    # Normalize timestamp column name (parquet uses 'time', API expects 'timestamp')
                    if "time" in ticker_data.columns and "timestamp" not in ticker_data.columns:
                        ticker_data["timestamp"] = ticker_data["time"]

                    if not ticker_data.empty and "timestamp" in ticker_data.columns:
                        all_ticker_data.append(ticker_data)
            except Exception as e:
                print(f"Error reading {parquet_file}: {e}")
                continue

        # Combine all data from different parquet files
        if all_ticker_data:
            combined = pd.concat(all_ticker_data, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])

            # Remove duplicates (same ticker + timestamp)
            combined = combined.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
            combined = combined.sort_values("timestamp")

            # Filter by date if specified
            if start_date:
                combined = combined[combined["timestamp"] >= start_date]
            if end_date:
                combined = combined[combined["timestamp"] <= end_date]

            if not combined.empty:
                return combined

        # If no data found in files for the date range, try fetching
        if HAS_YFINANCE:
            try:
                print(f"No local data for {ticker} from {start_date} to {end_date}, fetching from yfinance...")
                return self.fetch_data([ticker], start_date, end_date)
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")

        return pd.DataFrame()

    def get_stock_features(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> FeaturesResponse:
        """Get engineered features for a stock."""
        # Get raw data first
        df = self._load_stock_data(ticker, start_date, end_date)
        if df.empty:
            raise DataNotFoundError(ticker)

        # Resample to monthly if requested
        if interval == "1mo":
            df = self._resample_to_monthly(df)

        # Calculate features
        if HAS_PREPROCESSOR:
            try:
                df = self.preprocessor.calculate_returns(df)
                df = self.preprocessor.calculate_volatility(df)
                df = self.preprocessor.calculate_momentum(df)
                df = self.preprocessor.calculate_technical_indicators(df)
            except Exception as e:
                print(f"Error calculating features: {e}")

        # Convert to response format
        features = []
        feature_cols = [
            "log_return", "simple_return",
            "rolling_volatility_5", "rolling_volatility_20", "rolling_volatility_60",
            "momentum_5", "momentum_10", "momentum_20", "momentum_60",
            "rsi_14", "macd", "macd_signal", "bollinger_upper", "bollinger_lower", "atr_14",
        ]

        available_features = [col for col in feature_cols if col in df.columns]

        for _, row in df.iterrows():
            feature_data = FeatureData(
                timestamp=row.get("timestamp") or row.name,
                ticker=ticker,
            )
            for col in available_features:
                if not pd.isna(row.get(col)):
                    setattr(feature_data, col, float(row[col]))
            features.append(feature_data)

        return FeaturesResponse(
            ticker=ticker,
            features=features,
            feature_names=available_features,
            count=len(features),
        )

    def _fetch_from_yfinance(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch data directly from yfinance."""
        if not HAS_YFINANCE:
            raise DataFetchError("yfinance is not installed")

        all_data = []
        for ticker in tickers:
            try:
                print(f"Downloading {ticker} from yfinance: {start_date} to {end_date}")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                )

                if data.empty:
                    print(f"No data returned for {ticker}")
                    continue

                df = data.copy()

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["ticker"] = ticker
                df["time"] = df.index
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adjusted_close",
                })

                # Select columns
                cols = ["time", "ticker", "open", "high", "low", "close", "volume"]
                if "adjusted_close" in df.columns:
                    cols.append("adjusted_close")
                df = df[[c for c in cols if c in df.columns]]

                # Remove timezone info
                df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                all_data.append(df)

                print(f"Got {len(df)} records for {ticker}")

            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def fetch_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch new data from yfinance and persist to parquet."""
        if not HAS_YFINANCE:
            raise DataFetchError("yfinance is not installed. Run: pip install yfinance")

        # Set default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

        # Clear in-memory cache for affected tickers when force_refresh
        if force_refresh:
            for ticker in tickers:
                self.clear_cache(ticker)

        try:
            print(f"Fetching {tickers} from yfinance: {start_date} to {end_date}")

            df = self._fetch_from_yfinance(tickers, start_date, end_date, interval)

            print(f"yfinance returned {len(df) if df is not None else 0} records")

            # If yfinance returns empty, try fetching "latest available" data
            if df is None or df.empty:
                print(f"No data for {start_date} to {end_date}, trying to fetch latest available...")
                fallback_start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
                fallback_end = datetime.now().strftime("%Y-%m-%d")

                df = self._fetch_from_yfinance(tickers, fallback_start, fallback_end, interval)
                print(f"Fallback fetch returned {len(df) if df is not None else 0} records")

            # Persist fetched data to parquet for future requests
            if df is not None and not df.empty:
                self._persist_to_parquet(df, tickers, start_date, end_date)

            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Fetch error: {e}")
            raise DataFetchError(f"Failed to fetch data: {str(e)}")

    def _persist_to_parquet(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Persist fetched data to master parquet file."""
        if df is None or df.empty:
            return

        # Save to master parquet (merges automatically)
        self._save_master_data(df)

    def fetch_latest(
        self,
        ticker: str,
        years: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch latest data for a ticker with incremental updates.

        - Gets 10 years of historical data
        - Only fetches data we don't already have
        - Merges with existing data (never deletes old data)
        - Returns summary of what was fetched
        """
        if not HAS_YFINANCE:
            raise DataFetchError("yfinance is not installed")

        today = datetime.now()
        target_start = today - timedelta(days=365 * years)

        # Check what data we already have
        existing_start, existing_end = self.get_ticker_date_range(ticker)

        print(f"Existing data for {ticker}: {existing_start} to {existing_end}")

        fetch_ranges = []
        records_fetched = 0

        if existing_start is None or existing_end is None:
            # No existing data - fetch full 10 years
            fetch_ranges.append((target_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")))
            print(f"No existing data, will fetch full {years} years")
        else:
            # Check if we need older data (before existing_start)
            if target_start.date() < existing_start.date():
                fetch_ranges.append((
                    target_start.strftime("%Y-%m-%d"),
                    (existing_start - timedelta(days=1)).strftime("%Y-%m-%d")
                ))
                print(f"Will fetch older data: {target_start.date()} to {existing_start.date()}")

            # Check if we need newer data (after existing_end)
            if existing_end.date() < today.date():
                fetch_ranges.append((
                    (existing_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d")
                ))
                print(f"Will fetch newer data: {existing_end.date()} to {today.date()}")

        if not fetch_ranges:
            print(f"Data for {ticker} is already up to date")
            return {
                "ticker": ticker,
                "records_fetched": 0,
                "message": "Data is already up to date",
                "date_range": {
                    "start": existing_start.strftime("%Y-%m-%d") if existing_start else None,
                    "end": existing_end.strftime("%Y-%m-%d") if existing_end else None,
                }
            }

        # Fetch each missing range
        all_new_data = []
        for start_date, end_date in fetch_ranges:
            print(f"Fetching {ticker}: {start_date} to {end_date}")
            df = self._fetch_from_yfinance([ticker], start_date, end_date, interval="1d")
            if df is not None and not df.empty:
                all_new_data.append(df)
                records_fetched += len(df)
                print(f"Got {len(df)} records")

        # Merge and save all new data
        if all_new_data:
            combined_new = pd.concat(all_new_data, ignore_index=True)
            self._save_master_data(combined_new)
            print(f"Saved {len(combined_new)} new records to master parquet")

        # Clear cache for this ticker
        self.clear_cache(ticker)

        # Get updated date range
        new_start, new_end = self.get_ticker_date_range(ticker)

        return {
            "ticker": ticker,
            "records_fetched": records_fetched,
            "message": f"Fetched {records_fetched} new records" if records_fetched > 0 else "No new data available",
            "date_range": {
                "start": new_start.strftime("%Y-%m-%d") if new_start else None,
                "end": new_end.strftime("%Y-%m-%d") if new_end else None,
            }
        }

    # Default feature columns matching the training pipeline (train_pinn_variants.py)
    TRAINING_FEATURE_COLS = [
        'close', 'volume',
        'log_return', 'simple_return',
        'rolling_volatility_5', 'rolling_volatility_20',
        'momentum_5', 'momentum_20',
        'rsi_14', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower', 'atr_14',
    ]

    def prepare_sequences(
        self,
        ticker: str,
        sequence_length: int = 60,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare sequences for model input.

        Uses the same 14 features that models were trained with by default.
        Loads raw OHLCV data, computes engineered features, and normalises
        them with StandardScaler (matching the training pipeline).

        Returns:
            Tuple of (sequences, targets, df_with_scaler_info)
            The returned DataFrame has two extra attributes attached:
              - ``_scaler_mean``: per-column mean used for normalisation
              - ``_scaler_std``:  per-column std  used for normalisation
        """
        # Load raw OHLCV data (contains close, volume, etc.)
        raw_df = self._load_stock_data(ticker)
        if raw_df.empty:
            try:
                self.fetch_data([ticker], force_refresh=True)
                raw_df = self._load_stock_data(ticker)
            except Exception:
                pass
        if raw_df.empty:
            raise DataNotFoundError(ticker, "No raw data available")

        # Calculate engineered features on the raw dataframe
        if HAS_PREPROCESSOR:
            try:
                raw_df = self.preprocessor.calculate_returns(raw_df)
                raw_df = self.preprocessor.calculate_volatility(raw_df)
                raw_df = self.preprocessor.calculate_momentum(raw_df)
                raw_df = self.preprocessor.calculate_technical_indicators(raw_df)
            except Exception as e:
                print(f"Error calculating features for sequences: {e}")

        # Set timestamp as index
        if "timestamp" in raw_df.columns:
            raw_df = raw_df.set_index("timestamp")
        elif "time" in raw_df.columns:
            raw_df = raw_df.set_index("time")

        # Select features — default to the 14 columns used during training
        if feature_cols is None:
            feature_cols = self.TRAINING_FEATURE_COLS

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in raw_df.columns]
        if not available_cols:
            raise DataNotFoundError(ticker, "No valid features available")

        # Drop NaN rows
        df = raw_df[available_cols].dropna()

        # ── Normalise with StandardScaler (matches training pipeline) ──
        scaler_mean = df[available_cols].mean()
        scaler_std  = df[available_cols].std().replace(0, 1)  # avoid div-by-zero
        df_norm = (df[available_cols] - scaler_mean) / scaler_std

        # Attach scaler stats so the caller can inverse-transform model output
        df_norm._scaler_mean = scaler_mean  # type: ignore[attr-defined]
        df_norm._scaler_std  = scaler_std   # type: ignore[attr-defined]

        # If we still don't have enough rows for the desired sequence length, relax gracefully
        if len(df_norm) <= sequence_length:
            if len(df_norm) < 2:
                return np.array([]), np.array([]), df_norm
            sequence_length = max(2, len(df_norm) - 1)

        # Create sequences from normalised data
        sequences = []
        targets = []

        for i in range(sequence_length, len(df_norm)):
            sequences.append(df_norm.iloc[i - sequence_length:i][available_cols].values)
            targets.append(df_norm.iloc[i]["log_return"] if "log_return" in available_cols else 0)

        return np.array(sequences), np.array(targets), df_norm

    def clear_cache(self, ticker: Optional[str] = None):
        """Clear data cache."""
        if ticker:
            keys_to_remove = [k for k in self._data_cache if k.startswith(ticker)]
            for key in keys_to_remove:
                del self._data_cache[key]
        else:
            self._data_cache.clear()
