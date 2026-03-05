"""
Unit tests for data cleaning utilities.

Tests cover:
- Missing value handling
- Outlier detection and treatment
- Duplicate removal
- Suspicious value detection
- Data quality assessment
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_cleaner import (
    DataCleaner,
    ImputationMethod,
    OutlierMethod,
    OutlierTreatment,
    DataQualityReport,
    CleaningResult,
    clean_financial_data,
    validate_cleaned_data
)


@pytest.fixture
def sample_financial_data():
    """Create sample financial data with known issues"""
    np.random.seed(42)
    n = 100

    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'time': dates,
        'ticker': 'AAPL',
        'open': 100 + np.random.randn(n) * 2,
        'high': 102 + np.random.randn(n) * 2,
        'low': 98 + np.random.randn(n) * 2,
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000000, 10000000, n)
    })

    return df


@pytest.fixture
def data_with_missing():
    """Create data with missing values"""
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=n, freq='D'),
        'ticker': 'AAPL',
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000000, 10000000, n).astype(float)
    })

    # Introduce missing values
    df.loc[5:7, 'close'] = np.nan
    df.loc[10:12, 'volume'] = np.nan
    df.loc[20, 'close'] = np.nan

    return df


@pytest.fixture
def data_with_outliers():
    """Create data with outliers"""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=n, freq='D'),
        'ticker': 'AAPL',
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000000, 10000000, n).astype(float)
    })

    # Add outliers
    df.loc[10, 'close'] = 200  # Extreme high
    df.loc[20, 'close'] = 50   # Extreme low
    df.loc[30, 'volume'] = 1e12  # Extreme volume

    return df


@pytest.fixture
def data_with_suspicious():
    """Create data with suspicious values"""
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=n, freq='D'),
        'ticker': 'AAPL',
        'open': 100 + np.random.randn(n) * 2,
        'high': 102 + np.random.randn(n) * 2,
        'low': 98 + np.random.randn(n) * 2,
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000000, 10000000, n)
    })

    # Add suspicious values
    df.loc[5, 'close'] = -10  # Negative price
    df.loc[10, 'high'] = 95   # High < low
    df.loc[10, 'low'] = 100
    df.loc[15, 'volume'] = 0  # Zero volume

    return df


class TestMissingValueHandling:
    """Tests for missing value handling"""

    def test_forward_fill(self, data_with_missing):
        """Test forward fill imputation"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.FORWARD_FILL)
        result = cleaner.handle_missing(data_with_missing)

        assert result['close'].isnull().sum() == 0
        assert result['volume'].isnull().sum() == 0

    def test_backward_fill(self, data_with_missing):
        """Test backward fill imputation"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.BACKWARD_FILL)
        result = cleaner.handle_missing(data_with_missing)

        assert result['close'].isnull().sum() == 0
        assert result['volume'].isnull().sum() == 0

    def test_linear_interpolation(self, data_with_missing):
        """Test linear interpolation"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.LINEAR)
        result = cleaner.handle_missing(data_with_missing)

        assert result['close'].isnull().sum() == 0

    def test_mean_imputation(self, data_with_missing):
        """Test mean imputation"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.MEAN)
        original_mean = data_with_missing['close'].mean()
        result = cleaner.handle_missing(data_with_missing)

        assert result['close'].isnull().sum() == 0
        # After imputation, mean should be similar
        assert abs(result['close'].mean() - original_mean) < 1

    def test_drop_missing(self, data_with_missing):
        """Test dropping rows with missing values"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.DROP)
        original_len = len(data_with_missing)
        result = cleaner.handle_missing(data_with_missing)

        assert result['close'].isnull().sum() == 0
        assert len(result) < original_len

    def test_selective_column_imputation(self, data_with_missing):
        """Test imputation on specific columns only"""
        cleaner = DataCleaner(imputation_method=ImputationMethod.FORWARD_FILL)
        result = cleaner.handle_missing(data_with_missing, columns=['close'])

        assert result['close'].isnull().sum() == 0
        # Volume should still have missing values
        assert result['volume'].isnull().sum() > 0


class TestOutlierDetection:
    """Tests for outlier detection and treatment"""

    def test_iqr_detection(self, data_with_outliers):
        """Test IQR-based outlier detection"""
        cleaner = DataCleaner(
            outlier_method=OutlierMethod.IQR,
            iqr_multiplier=1.5
        )
        outliers = cleaner._detect_outliers(data_with_outliers['close'])

        # Should detect at least our injected outliers
        assert outliers.sum() >= 2

    def test_zscore_detection(self, data_with_outliers):
        """Test Z-score-based outlier detection"""
        cleaner = DataCleaner(
            outlier_method=OutlierMethod.ZSCORE,
            zscore_threshold=3.0
        )
        outliers = cleaner._detect_outliers(data_with_outliers['close'])

        # Should detect our extreme outliers
        assert outliers.sum() >= 2

    def test_mad_detection(self, data_with_outliers):
        """Test MAD-based outlier detection"""
        cleaner = DataCleaner(outlier_method=OutlierMethod.MAD)
        outliers = cleaner._detect_outliers(data_with_outliers['close'])

        # Should detect outliers
        assert outliers.sum() >= 2

    def test_outlier_clipping(self, data_with_outliers):
        """Test outlier clipping treatment"""
        cleaner = DataCleaner(
            outlier_treatment=OutlierTreatment.CLIP,
            outlier_method=OutlierMethod.IQR
        )
        original_max = data_with_outliers['close'].max()
        result = cleaner.handle_outliers(data_with_outliers)

        # Max should be reduced after clipping
        assert result['close'].max() < original_max

    def test_outlier_winsorization(self, data_with_outliers):
        """Test outlier winsorization"""
        cleaner = DataCleaner(outlier_treatment=OutlierTreatment.WINSORIZE)
        result = cleaner.handle_outliers(data_with_outliers)

        # Extreme values should be replaced
        assert result['close'].max() < data_with_outliers['close'].max()
        assert result['close'].min() > data_with_outliers['close'].min()

    def test_outlier_removal(self, data_with_outliers):
        """Test outlier removal"""
        cleaner = DataCleaner(outlier_treatment=OutlierTreatment.REMOVE)
        original_len = len(data_with_outliers)
        result = cleaner.handle_outliers(data_with_outliers)

        assert len(result) < original_len

    def test_outlier_flagging(self, data_with_outliers):
        """Test outlier flagging"""
        cleaner = DataCleaner(outlier_treatment=OutlierTreatment.FLAG)
        result = cleaner.handle_outliers(data_with_outliers)

        # Should have flag columns
        assert 'close_outlier' in result.columns
        assert result['close_outlier'].sum() > 0


class TestSuspiciousValueDetection:
    """Tests for suspicious value detection and fixing"""

    def test_detect_negative_prices(self, data_with_suspicious):
        """Test detection of negative prices"""
        cleaner = DataCleaner()
        suspicious = cleaner._find_suspicious_values(data_with_suspicious)

        neg_price_issues = [s for s in suspicious if s['type'] == 'negative_price']
        assert len(neg_price_issues) > 0

    def test_detect_high_low_violation(self, data_with_suspicious):
        """Test detection of high < low violations"""
        cleaner = DataCleaner()
        suspicious = cleaner._find_suspicious_values(data_with_suspicious)

        hl_issues = [s for s in suspicious if s['type'] == 'high_low_violation']
        assert len(hl_issues) > 0

    def test_detect_zero_volume(self, data_with_suspicious):
        """Test detection of zero volume"""
        cleaner = DataCleaner()
        suspicious = cleaner._find_suspicious_values(data_with_suspicious)

        zero_vol_issues = [s for s in suspicious if s['type'] == 'zero_volume']
        assert len(zero_vol_issues) > 0

    def test_fix_negative_prices(self, data_with_suspicious):
        """Test fixing negative prices"""
        cleaner = DataCleaner()
        result = cleaner.fix_suspicious_values(data_with_suspicious)

        assert (result['close'] >= 0).all()

    def test_fix_high_low_violation(self, data_with_suspicious):
        """Test fixing high < low violations"""
        cleaner = DataCleaner()
        result = cleaner.fix_suspicious_values(data_with_suspicious)

        assert (result['high'] >= result['low']).all()


class TestDuplicateRemoval:
    """Tests for duplicate row handling"""

    def test_remove_duplicates(self, sample_financial_data):
        """Test duplicate removal"""
        # Create duplicates
        df = pd.concat([sample_financial_data, sample_financial_data.iloc[:5]])
        original_len = len(df)

        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df)

        assert len(result) < original_len
        assert len(result) == len(sample_financial_data)

    def test_keep_first_duplicate(self, sample_financial_data):
        """Test keeping first duplicate"""
        df = pd.concat([sample_financial_data, sample_financial_data.iloc[:5]])

        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df, keep='first')

        assert len(result) == len(sample_financial_data)

    def test_keep_last_duplicate(self, sample_financial_data):
        """Test keeping last duplicate"""
        df = pd.concat([sample_financial_data, sample_financial_data.iloc[:5]])

        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df, keep='last')

        assert len(result) == len(sample_financial_data)


class TestDataQualityAssessment:
    """Tests for data quality assessment"""

    def test_quality_score_perfect_data(self, sample_financial_data):
        """Test quality score for clean data"""
        cleaner = DataCleaner()
        report = cleaner.assess_quality(sample_financial_data)

        assert report.quality_score > 0.8
        assert report.total_rows == len(sample_financial_data)

    def test_quality_score_with_issues(self, data_with_missing):
        """Test quality score decreases with data issues"""
        cleaner = DataCleaner()
        report = cleaner.assess_quality(data_with_missing)

        # Score should be lower due to missing values
        assert report.quality_score < 1.0
        assert sum(report.missing_count.values()) > 0

    def test_recommendations_generated(self, data_with_missing):
        """Test that recommendations are generated"""
        cleaner = DataCleaner()
        report = cleaner.assess_quality(data_with_missing)

        assert len(report.recommendations) > 0

    def test_quality_report_structure(self, sample_financial_data):
        """Test quality report has all required fields"""
        cleaner = DataCleaner()
        report = cleaner.assess_quality(sample_financial_data)

        assert isinstance(report, DataQualityReport)
        assert hasattr(report, 'total_rows')
        assert hasattr(report, 'missing_count')
        assert hasattr(report, 'outlier_count')
        assert hasattr(report, 'quality_score')
        assert hasattr(report, 'recommendations')


class TestFullCleaningPipeline:
    """Tests for full cleaning pipeline"""

    def test_full_pipeline(self, data_with_suspicious):
        """Test full cleaning pipeline"""
        cleaner = DataCleaner()
        result = cleaner.clean(data_with_suspicious)

        assert isinstance(result, CleaningResult)
        assert result.quality_after.quality_score >= result.quality_before.quality_score
        assert len(result.records) > 0
        assert len(result.transformations_applied) > 0

    def test_pipeline_improves_quality(self, data_with_suspicious):
        """Test that pipeline improves data quality"""
        cleaner = DataCleaner()
        result = cleaner.clean(data_with_suspicious)

        # Quality should improve
        assert result.quality_after.quality_score >= result.quality_before.quality_score

        # Suspicious values should be fixed
        assert (result.cleaned_data['close'] >= 0).all()
        assert (result.cleaned_data['high'] >= result.cleaned_data['low']).all()

    def test_pipeline_records_transformations(self, data_with_suspicious):
        """Test that transformations are recorded"""
        cleaner = DataCleaner(log_all_changes=True)
        result = cleaner.clean(data_with_suspicious)

        assert len(result.records) > 0
        for record in result.records:
            assert hasattr(record, 'operation')
            assert hasattr(record, 'affected_count')


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_clean_financial_data(self, data_with_suspicious):
        """Test convenience function"""
        result = clean_financial_data(
            data_with_suspicious,
            imputation_method='ffill',
            outlier_method='iqr',
            outlier_treatment='clip'
        )

        assert isinstance(result, CleaningResult)
        assert len(result.cleaned_data) > 0

    def test_validate_cleaned_data_valid(self, sample_financial_data):
        """Test validation of valid data"""
        is_valid, issues = validate_cleaned_data(sample_financial_data)

        assert is_valid
        assert len(issues) == 0

    def test_validate_cleaned_data_with_issues(self, data_with_missing):
        """Test validation detects issues"""
        is_valid, issues = validate_cleaned_data(data_with_missing)

        assert not is_valid
        assert len(issues) > 0


class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        cleaner = DataCleaner()

        # Should not crash
        result = cleaner.handle_missing(df)
        assert len(result) == 0

    def test_single_row(self):
        """Test handling of single row DataFrame"""
        df = pd.DataFrame({
            'close': [100.0],
            'volume': [1000000]
        })
        cleaner = DataCleaner()

        result = cleaner.clean(df)
        assert len(result.cleaned_data) == 1

    def test_all_nan_column(self):
        """Test handling of all-NaN column"""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'empty': [np.nan, np.nan, np.nan]
        })
        cleaner = DataCleaner(imputation_method=ImputationMethod.MEAN)

        result = cleaner.handle_missing(df)
        # Mean of all NaN is NaN, so column stays NaN
        assert result['empty'].isnull().all()

    def test_no_numeric_columns(self):
        """Test handling of DataFrame with no numeric columns"""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'name': ['Apple', 'Google', 'Microsoft']
        })
        cleaner = DataCleaner()

        # Should not crash on outlier detection
        result = cleaner.handle_outliers(df)
        assert len(result) == 3

    def test_preserves_non_numeric_columns(self, sample_financial_data):
        """Test that non-numeric columns are preserved"""
        cleaner = DataCleaner()
        result = cleaner.clean(sample_financial_data)

        assert 'ticker' in result.cleaned_data.columns
        assert 'time' in result.cleaned_data.columns


def test_datacleaner_from_config(tmp_path):
    """Ensure DataCleaner.from_config reads YAML and maps enums correctly."""
    cfg = tmp_path / "dc.yaml"
    cfg.write_text(
        """
missing:
  method: "linear"
outliers:
  method: "zscore"
  treatment: "clip"
  zscore_threshold: 2.5
quality:
  min_required_data_pct: 0.9
  log_all_changes: false
"""
    )

    cleaner = DataCleaner.from_config(cfg)

    assert cleaner.imputation_method == ImputationMethod.LINEAR
    assert cleaner.outlier_method == OutlierMethod.ZSCORE
    assert cleaner.outlier_treatment == OutlierTreatment.CLIP
    assert cleaner.zscore_threshold == 2.5
    assert cleaner.min_required_data_pct == 0.9
    assert cleaner.log_all_changes is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
