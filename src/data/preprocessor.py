"""
Data preprocessing: feature engineering, normalization, stationarity testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import pandas_ta as ta

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_db

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocess financial time series data:
    - Feature engineering (technical indicators)
    - Normalization
    - Stationarity testing
    - Train/val/test splitting
    """

    def __init__(self, config=None):
        """
        Initialize preprocessor

        Args:
            config: Optional Config object
        """
        self.config = config or get_config()
        self.db = get_db()
        self.scalers: Dict[str, StandardScaler] = {}  # Per-ticker scalers

    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculate log returns

        Args:
            df: DataFrame with price data
            price_col: Column name for price

        Returns:
            DataFrame with log_return column
        """
        df = df.copy()
        df = df.sort_values(['ticker', 'time'])

        # Log returns
        df['log_return'] = df.groupby('ticker')[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # Simple returns
        df['simple_return'] = df.groupby('ticker')[price_col].pct_change()

        return df

    def calculate_volatility(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns)

        Args:
            df: DataFrame with return data
            windows: List of window sizes

        Returns:
            DataFrame with volatility columns
        """
        df = df.copy()

        for window in windows:
            df[f'rolling_volatility_{window}'] = df.groupby('ticker')['log_return'].transform(
                lambda x: x.rolling(window=window).std()
            )

        return df

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 60],
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            df: DataFrame with price data
            windows: List of window sizes
            price_col: Price column

        Returns:
            DataFrame with momentum columns
        """
        df = df.copy()

        for window in windows:
            # Rate of change
            df[f'momentum_{window}'] = df.groupby('ticker')[price_col].transform(
                lambda x: (x / x.shift(window) - 1)
            )

            # Moving average
            df[f'sma_{window}'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=window).mean()
            )

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas_ta

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        results = []

        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('time')

            # RSI (Relative Strength Index)
            ticker_df['rsi_14'] = ta.rsi(ticker_df['close'], length=14)

            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(ticker_df['close'])
            if macd is not None:
                ticker_df['macd'] = macd['MACD_12_26_9']
                ticker_df['macd_signal'] = macd['MACDs_12_26_9']
                ticker_df['macd_hist'] = macd['MACDh_12_26_9']

            # Bollinger Bands
            bbands = ta.bbands(ticker_df['close'], length=20)
            if bbands is not None:
                ticker_df['bollinger_upper'] = bbands['BBU_20_2.0']
                ticker_df['bollinger_middle'] = bbands['BBM_20_2.0']
                ticker_df['bollinger_lower'] = bbands['BBL_20_2.0']

            # ATR (Average True Range) - volatility measure
            ticker_df['atr_14'] = ta.atr(
                ticker_df['high'],
                ticker_df['low'],
                ticker_df['close'],
                length=14
            )

            # OBV (On-Balance Volume)
            ticker_df['obv'] = ta.obv(ticker_df['close'], ticker_df['volume'])

            # Stochastic Oscillator
            stoch = ta.stoch(ticker_df['high'], ticker_df['low'], ticker_df['close'])
            if stoch is not None:
                ticker_df['stoch_k'] = stoch['STOCHk_14_3_3']
                ticker_df['stoch_d'] = stoch['STOCHd_14_3_3']

            results.append(ticker_df)

        combined_df = pd.concat(results, ignore_index=True)
        logger.info(f"Calculated technical indicators for {len(df['ticker'].unique())} tickers")

        return combined_df

    def test_stationarity(
        self,
        series: pd.Series,
        significance_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity

        Args:
            series: Time series to test
            significance_level: Significance level for test

        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        series = series.dropna()

        if len(series) < 10:
            return {
                'is_stationary': None,
                'adf_statistic': None,
                'p_value': None,
                'error': 'Insufficient data'
            }

        try:
            result = adfuller(series, autolag='AIC')

            return {
                'is_stationary': result[1] < significance_level,
                'adf_statistic': result[0],
                'p_value': result[1],
                'used_lag': result[2],
                'n_obs': result[3],
                'critical_values': result[4]
            }

        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return {
                'is_stationary': None,
                'adf_statistic': None,
                'p_value': None,
                'error': str(e)
            }

    def test_all_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Test stationarity for all tickers

        Args:
            df: DataFrame with time series data

        Returns:
            DataFrame with stationarity test results
        """
        results = []

        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker]

            # Test price stationarity
            price_result = self.test_stationarity(ticker_df['close'])
            price_result['ticker'] = ticker
            price_result['series'] = 'price'
            results.append(price_result)

            # Test returns stationarity
            if 'log_return' in ticker_df.columns:
                return_result = self.test_stationarity(ticker_df['log_return'])
                return_result['ticker'] = ticker
                return_result['series'] = 'log_return'
                results.append(return_result)

        results_df = pd.DataFrame(results)
        logger.info(f"Stationarity test completed for {len(df['ticker'].unique())} tickers")

        # Log summary
        if 'is_stationary' in results_df.columns:
            stationary_count = results_df['is_stationary'].sum()
            total_count = len(results_df)
            logger.info(f"{stationary_count}/{total_count} series are stationary")

        return results_df

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Normalize features per ticker to handle scale differences

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to normalize
            method: 'standard' (z-score) or 'minmax' (0-1 range)

        Returns:
            Tuple of (normalized DataFrame, dictionary of scalers)
        """
        df = df.copy()
        scalers = {}

        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df.loc[ticker_mask, feature_cols]

            # Choose scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Fit and transform
            normalized_data = scaler.fit_transform(ticker_data)
            df.loc[ticker_mask, feature_cols] = normalized_data

            scalers[ticker] = scaler

        logger.info(f"Normalized {len(feature_cols)} features for {len(scalers)} tickers")

        return df, scalers

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'close',
        sequence_length: int = 60,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for time series prediction

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
            target_col: Target column to predict
            sequence_length: Number of time steps in each sequence
            forecast_horizon: Steps ahead to predict

        Returns:
            Tuple of (X sequences, y targets, ticker labels)
        """
        X_list = []
        y_list = []
        ticker_list = []

        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('time')

            # Get feature and target arrays
            features = ticker_df[feature_cols].values
            targets = ticker_df[target_col].values

            # Create sequences
            for i in range(len(ticker_df) - sequence_length - forecast_horizon + 1):
                X_list.append(features[i:i + sequence_length])
                y_list.append(targets[i + sequence_length + forecast_horizon - 1])
                ticker_list.append(ticker)

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Created {len(X)} sequences with shape {X.shape}")

        return X, y, ticker_list

    def split_temporal(
        self,
        df: pd.DataFrame,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/val/test split (no data leakage)

        Args:
            df: DataFrame to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = train_ratio or self.config.data.train_ratio
        val_ratio = val_ratio or self.config.data.val_ratio
        test_ratio = test_ratio or self.config.data.test_ratio

        # Verify ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Sort by time
        df = df.sort_values('time')

        # Calculate split points
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def process_and_store(
        self,
        df: pd.DataFrame,
        calculate_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline and store to database

        Args:
            df: Raw stock price data
            calculate_indicators: Whether to calculate technical indicators

        Returns:
            Processed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")

        # Calculate returns
        df = self.calculate_returns(df)

        # Calculate volatility
        df = self.calculate_volatility(df)

        # Calculate momentum
        df = self.calculate_momentum(df)

        # Calculate technical indicators
        if calculate_indicators:
            df = self.calculate_technical_indicators(df)

        # Drop rows with NaN (from rolling calculations)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")

        # Store features in database
        try:
            feature_cols = [
                'log_return', 'simple_return',
                'rolling_volatility_5', 'rolling_volatility_20', 'rolling_volatility_60',
                'momentum_5', 'momentum_20',
                'rsi_14', 'macd', 'macd_signal',
                'bollinger_upper', 'bollinger_lower', 'atr_14'
            ]

            features_df = df[['time', 'ticker'] + [col for col in feature_cols if col in df.columns]]

            self.db.write_dataframe(
                features_df,
                table_name='features',
                schema='finance',
                if_exists='append'
            )
            logger.info("Features stored in database")

        except Exception as e:
            logger.error(f"Failed to store features in database: {e}")

        logger.info("Preprocessing pipeline completed")

        return df
