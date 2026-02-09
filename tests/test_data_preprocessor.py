"""
Comprehensive tests for data preprocessing functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test data preprocessing functionality"""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'time': dates.tolist() * 3,
            'ticker': ['AAPL'] * 100 + ['GOOGL'] * 100 + ['MSFT'] * 100,
            'open': np.random.uniform(100, 200, 300),
            'high': np.random.uniform(100, 200, 300),
            'low': np.random.uniform(100, 200, 300),
            'close': np.random.uniform(100, 200, 300),
            'volume': np.random.uniform(1000000, 10000000, 300),
        })
        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        return df.sort_values(['ticker', 'time']).reset_index(drop=True)

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor()

    def test_calculate_returns(self, preprocessor, sample_df):
        """Test log returns calculation"""
        result = preprocessor.calculate_returns(sample_df, price_col='close')
        
        # Check log_return column exists
        assert 'log_return' in result.columns
        assert 'simple_return' in result.columns
        
        # Check first value is NaN (no previous value)
        for ticker in result['ticker'].unique():
            ticker_data = result[result['ticker'] == ticker]
            assert pd.isna(ticker_data['log_return'].iloc[0])
            assert pd.isna(ticker_data['simple_return'].iloc[0])
        
        # Check returns are calculated correctly
        aapl_data = result[result['ticker'] == 'AAPL']
        prices = aapl_data['close'].values
        expected_log_returns = np.log(prices[1:] / prices[:-1])
        calculated_log_returns = aapl_data['log_return'].iloc[1:].values
        np.testing.assert_allclose(calculated_log_returns, expected_log_returns, rtol=1e-5)

    def test_calculate_returns_multiindex(self, preprocessor):
        """Test returns calculation with MultiIndex columns"""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        # Create MultiIndex columns
        df = pd.DataFrame({
            ('close', 'AAPL'): np.random.uniform(100, 200, 50),
            ('ticker', ''): ['AAPL'] * 50,
            ('time', ''): dates,
        })
        
        result = preprocessor.calculate_returns(df, price_col='close')
        
        # Check that MultiIndex was flattened
        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'log_return' in result.columns

    def test_calculate_volatility(self, preprocessor, sample_df):
        """Test rolling volatility calculation"""
        # First calculate returns
        df_with_returns = preprocessor.calculate_returns(sample_df)
        
        # Calculate volatility with multiple windows
        windows = [5, 20, 60]
        result = preprocessor.calculate_volatility(df_with_returns, windows=windows)
        
        # Check volatility columns exist
        for window in windows:
            col_name = f'rolling_volatility_{window}'
            assert col_name in result.columns
            
            # Check values are non-negative (where not NaN)
            non_nan_values = result[col_name].dropna()
            assert (non_nan_values >= 0).all()
        
        # Check that first N-1 values are NaN (window size)
        for ticker in result['ticker'].unique():
            ticker_data = result[result['ticker'] == ticker]
            assert ticker_data['rolling_volatility_5'].iloc[:4].isna().all()
            assert ticker_data['rolling_volatility_20'].iloc[:19].isna().all()

    def test_calculate_momentum(self, preprocessor, sample_df):
        """Test momentum indicators calculation"""
        windows = [5, 10, 20]
        result = preprocessor.calculate_momentum(sample_df, windows=windows, price_col='close')
        
        # Check momentum columns exist
        for window in windows:
            assert f'momentum_{window}' in result.columns
            assert f'sma_{window}' in result.columns
        
        # Verify momentum calculation for one ticker
        aapl_data = result[result['ticker'] == 'AAPL']
        prices = aapl_data['close'].values
        
        # Check 5-period momentum
        expected_momentum_5 = (prices[5:] / prices[:-5] - 1)
        calculated_momentum_5 = aapl_data['momentum_5'].iloc[5:].values
        np.testing.assert_allclose(calculated_momentum_5, expected_momentum_5, rtol=1e-5)

    def test_calculate_technical_indicators(self, preprocessor, sample_df):
        """Test technical indicators calculation"""
        result = preprocessor.calculate_technical_indicators(sample_df)
        
        # Check that key technical indicators exist
        expected_indicators = ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 
                              'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                              'atr_14', 'obv', 'stoch_k', 'stoch_d']
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
        
        # Check RSI is between 0 and 100 (where not NaN)
        rsi_values = result['rsi_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()
        
        # Check Bollinger Bands ordering: lower <= middle <= upper
        bb_data = result[['bollinger_lower', 'bollinger_middle', 'bollinger_upper']].dropna()
        assert (bb_data['bollinger_lower'] <= bb_data['bollinger_middle']).all()
        assert (bb_data['bollinger_middle'] <= bb_data['bollinger_upper']).all()
        
        # Check ATR is non-negative
        atr_values = result['atr_14'].dropna()
        assert (atr_values >= 0).all()

    def test_test_stationarity(self, preprocessor):
        """Test stationarity testing (ADF test)"""
        # Test with stationary series (white noise)
        stationary_series = pd.Series(np.random.randn(100))
        result = preprocessor.test_stationarity(stationary_series)
        
        assert 'is_stationary' in result
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert isinstance(result['is_stationary'], bool)
        
        # Test with non-stationary series (random walk)
        non_stationary_series = pd.Series(np.cumsum(np.random.randn(100)))
        result_ns = preprocessor.test_stationarity(non_stationary_series)
        
        # Random walk typically has higher p-value (less stationary)
        assert result_ns['p_value'] is not None

    def test_test_stationarity_insufficient_data(self, preprocessor):
        """Test stationarity with insufficient data"""
        short_series = pd.Series([1, 2, 3])
        result = preprocessor.test_stationarity(short_series)
        
        assert result['is_stationary'] is None
        assert 'error' in result
        assert result['error'] == 'Insufficient data'

    def test_test_stationarity_with_nans(self, preprocessor):
        """Test stationarity with NaN values"""
        series_with_nans = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 10)
        result = preprocessor.test_stationarity(series_with_nans)
        
        # Should handle NaNs gracefully
        assert result['is_stationary'] is not None or 'error' in result

    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame(columns=['time', 'ticker', 'close'])
        
        # Should not raise error, just return empty with expected columns
        result = preprocessor.calculate_returns(empty_df)
        assert len(result) == 0
        assert 'log_return' in result.columns

    def test_single_ticker_processing(self, preprocessor, sample_df):
        """Test processing with single ticker"""
        single_ticker_df = sample_df[sample_df['ticker'] == 'AAPL']
        
        result = preprocessor.calculate_returns(single_ticker_df)
        assert len(result) == len(single_ticker_df)
        assert result['ticker'].unique()[0] == 'AAPL'

    def test_scaler_persistence(self, preprocessor):
        """Test that scalers are stored per ticker"""
        # Initially should have no scalers
        assert len(preprocessor.scalers) == 0
        
        # Add a scaler for a ticker
        preprocessor.scalers['AAPL'] = StandardScaler()
        
        # Check it persists
        assert 'AAPL' in preprocessor.scalers
        assert isinstance(preprocessor.scalers['AAPL'], StandardScaler)

    def test_chain_operations(self, preprocessor, sample_df):
        """Test chaining multiple preprocessing operations"""
        # Chain: returns -> volatility -> momentum -> technical indicators
        result = preprocessor.calculate_returns(sample_df)
        result = preprocessor.calculate_volatility(result, windows=[5, 20])
        result = preprocessor.calculate_momentum(result, windows=[5, 10])
        result = preprocessor.calculate_technical_indicators(result)
        
        # Verify all expected columns exist
        expected_cols = ['log_return', 'simple_return', 'rolling_volatility_5', 
                        'rolling_volatility_20', 'momentum_5', 'momentum_10',
                        'sma_5', 'sma_10', 'rsi_14', 'macd']
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column after chaining: {col}"

    def test_price_data_consistency(self, preprocessor, sample_df):
        """Test that OHLC relationships are maintained"""
        # Verify high >= low, high >= open, high >= close
        assert (sample_df['high'] >= sample_df['low']).all()
        assert (sample_df['high'] >= sample_df['open']).all()
        assert (sample_df['high'] >= sample_df['close']).all()
        
        # Verify low <= open, low <= close
        assert (sample_df['low'] <= sample_df['open']).all()
        assert (sample_df['low'] <= sample_df['close']).all()


class TestDataPreprocessorEdgeCases:
    """Test edge cases and error handling"""

    def test_extreme_values(self):
        """Test handling of extreme price values"""
        preprocessor = DataPreprocessor()
        
        # Create DataFrame with extreme values
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=10),
            'ticker': ['TEST'] * 10,
            'close': [1e-10, 1e-5, 1e5, 1e10, 1e-10, 1e-5, 1e5, 1e10, 1e-10, 1e-5]
        })
        
        result = preprocessor.calculate_returns(df)
        
        # Should not raise error, but returns might be extreme
        assert 'log_return' in result.columns
        assert not np.any(np.isinf(result['log_return'].dropna()))

    def test_zero_prices(self):
        """Test handling of zero prices"""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=10),
            'ticker': ['TEST'] * 10,
            'close': [100, 0, 100, 100, 100, 100, 100, 100, 100, 100]
        })
        
        result = preprocessor.calculate_returns(df)
        
        # Log return with zero denominator should be inf or nan
        assert pd.isna(result['log_return'].iloc[1]) or np.isinf(result['log_return'].iloc[1])

    def test_constant_prices(self):
        """Test with constant prices (no volatility)"""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=100),
            'ticker': ['TEST'] * 100,
            'close': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'open': [100.0] * 100,
            'volume': [1000000] * 100
        })
        
        result = preprocessor.calculate_returns(df)
        result = preprocessor.calculate_volatility(result, windows=[5, 20])
        
        # Returns should be 0 (or NaN for first value)
        returns = result['log_return'].dropna()
        np.testing.assert_allclose(returns, 0.0, atol=1e-10)
        
        # Volatility should be 0
        volatility = result['rolling_volatility_5'].dropna()
        np.testing.assert_allclose(volatility, 0.0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
