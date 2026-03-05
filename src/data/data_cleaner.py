"""
Data Cleaning Utilities

Provides comprehensive data cleaning for financial time series:
- Missing value detection and imputation
- Outlier detection and treatment
- Time series gap handling
- Data quality validation
- Corporate action adjustment detection

Designed for research reproducibility with detailed logging of all transformations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from pathlib import Path
import yaml  # type: ignore

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImputationMethod(Enum):
    """Methods for handling missing values"""
    DROP = "drop"  # Remove rows with missing values
    FORWARD_FILL = "ffill"  # Use last known value
    BACKWARD_FILL = "bfill"  # Use next known value
    LINEAR = "linear"  # Linear interpolation
    MEAN = "mean"  # Column mean
    MEDIAN = "median"  # Column median
    ZERO = "zero"  # Replace with zero


class OutlierMethod(Enum):
    """Methods for outlier detection"""
    IQR = "iqr"  # Interquartile range
    ZSCORE = "zscore"  # Z-score
    MAD = "mad"  # Median absolute deviation
    PERCENTILE = "percentile"  # Percentile-based


class OutlierTreatment(Enum):
    """Methods for handling detected outliers"""
    REMOVE = "remove"  # Remove outlier rows
    CLIP = "clip"  # Clip to bounds
    WINSORIZE = "winsorize"  # Replace with percentile values
    IMPUTE = "impute"  # Treat as missing and impute
    FLAG = "flag"  # Keep but flag


@dataclass
class CleaningRecord:
    """Record of a cleaning operation"""
    operation: str
    column: Optional[str]
    original_count: int
    affected_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Summary of data quality metrics"""
    total_rows: int
    total_columns: int
    missing_count: Dict[str, int]
    missing_pct: Dict[str, float]
    outlier_count: Dict[str, int]
    duplicate_rows: int
    date_gaps: List[Tuple[datetime, datetime]]
    suspicious_values: List[Dict[str, Any]]
    quality_score: float  # 0-1 score
    recommendations: List[str]


@dataclass
class CleaningResult:
    """Result of data cleaning operations"""
    cleaned_data: pd.DataFrame
    records: List[CleaningRecord]
    quality_before: DataQualityReport
    quality_after: DataQualityReport
    transformations_applied: List[str]


class DataCleaner:
    """
    Comprehensive data cleaner for financial time series.

    Provides consistent, reproducible data cleaning with full audit trail.
    """

    def __init__(
        self,
        imputation_method: ImputationMethod = ImputationMethod.FORWARD_FILL,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        outlier_treatment: OutlierTreatment = OutlierTreatment.CLIP,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        min_required_data_pct: float = 0.8,
        log_all_changes: bool = True
    ):
        """
        Initialize data cleaner.

        Args:
            imputation_method: Method for handling missing values
            outlier_method: Method for detecting outliers
            outlier_treatment: How to handle detected outliers
            iqr_multiplier: IQR multiplier for outlier detection
            zscore_threshold: Z-score threshold for outlier detection
            min_required_data_pct: Minimum data required after cleaning
            log_all_changes: Whether to log all individual changes
        """
        self.imputation_method = imputation_method
        self.outlier_method = outlier_method
        self.outlier_treatment = outlier_treatment
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.min_required_data_pct = min_required_data_pct
        self.log_all_changes = log_all_changes

        self.records: List[CleaningRecord] = []

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "DataCleaner":
        """
        Construct a DataCleaner from a YAML path or dict matching configs/data_cleaning.yaml.
        Missing keys fall back to defaults.
        """
        if isinstance(config, (str, Path)):
            cfg_path = Path(config)
            data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        else:
            data = config or {}

        missing_cfg = (data or {}).get("missing", {})
        outlier_cfg = (data or {}).get("outliers", {})
        quality_cfg = (data or {}).get("quality", {})

        return cls(
            imputation_method=ImputationMethod(missing_cfg.get("method", "ffill")),
            outlier_method=OutlierMethod(outlier_cfg.get("method", "iqr")),
            outlier_treatment=OutlierTreatment(outlier_cfg.get("treatment", "clip")),
            iqr_multiplier=float(outlier_cfg.get("iqr_multiplier", 1.5)),
            zscore_threshold=float(outlier_cfg.get("zscore_threshold", 3.0)),
            min_required_data_pct=float(quality_cfg.get("min_required_data_pct", 0.8)),
            log_all_changes=bool(quality_cfg.get("log_all_changes", True)),
        )

    def _record(
        self,
        operation: str,
        column: Optional[str],
        original_count: int,
        affected_count: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a cleaning operation"""
        record = CleaningRecord(
            operation=operation,
            column=column,
            original_count=original_count,
            affected_count=affected_count,
            details=details or {}
        )
        self.records.append(record)

        if self.log_all_changes:
            logger.debug(
                f"Cleaning: {operation} on {column or 'all'} - "
                f"affected {affected_count}/{original_count} rows"
            )

    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Assess data quality and generate report.

        Args:
            df: DataFrame to assess

        Returns:
            DataQualityReport with quality metrics
        """
        total_rows = len(df)
        total_columns = len(df.columns)

        # Missing values
        missing_count = df.isnull().sum().to_dict()
        missing_pct = {col: count / total_rows for col, count in missing_count.items()}

        # Outlier detection per numeric column
        outlier_count = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col].dropna())
            outlier_count[col] = outliers.sum()

        # Duplicate rows
        duplicate_rows = df.duplicated().sum()

        # Date gaps (if time column exists)
        date_gaps = []
        if 'time' in df.columns:
            date_gaps = self._find_date_gaps(df)

        # Suspicious values (negative prices, zero volume, etc.)
        suspicious_values = self._find_suspicious_values(df)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            missing_pct, outlier_count, duplicate_rows, len(date_gaps), len(suspicious_values), total_rows
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_pct, outlier_count, date_gaps, suspicious_values
        )

        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_count=missing_count,
            missing_pct=missing_pct,
            outlier_count=outlier_count,
            duplicate_rows=duplicate_rows,
            date_gaps=date_gaps,
            suspicious_values=suspicious_values,
            quality_score=quality_score,
            recommendations=recommendations
        )

    def _detect_outliers(
        self,
        series: pd.Series,
        method: Optional[OutlierMethod] = None
    ) -> pd.Series:
        """
        Detect outliers in a series.

        Args:
            series: Data series
            method: Detection method (defaults to instance setting)

        Returns:
            Boolean series indicating outliers
        """
        method = method or self.outlier_method

        if len(series) < 3:
            return pd.Series([False] * len(series), index=series.index)

        if method == OutlierMethod.IQR:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            return (series < lower) | (series > upper)

        elif method == OutlierMethod.ZSCORE:
            z_scores = np.abs((series - series.mean()) / (series.std() + 1e-8))
            return z_scores > self.zscore_threshold

        elif method == OutlierMethod.MAD:
            median = series.median()
            mad = np.abs(series - median).median()
            modified_z = 0.6745 * (series - median) / (mad + 1e-8)
            return np.abs(modified_z) > self.zscore_threshold

        elif method == OutlierMethod.PERCENTILE:
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            return (series < lower) | (series > upper)

        return pd.Series([False] * len(series), index=series.index)

    def _find_date_gaps(
        self,
        df: pd.DataFrame,
        time_col: str = 'time',
        max_gap_days: int = 5  # Allow weekends + holidays
    ) -> List[Tuple[datetime, datetime]]:
        """Find gaps in time series data"""
        if time_col not in df.columns:
            return []

        df_sorted = df.sort_values(time_col)
        dates = pd.to_datetime(df_sorted[time_col])

        gaps = []
        for i in range(1, len(dates)):
            gap = (dates.iloc[i] - dates.iloc[i-1]).days
            if gap > max_gap_days:
                gaps.append((dates.iloc[i-1], dates.iloc[i]))

        return gaps

    def _find_suspicious_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find potentially suspicious values in financial data"""
        suspicious = []

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    suspicious.append({
                        'type': 'negative_price',
                        'column': col,
                        'count': neg_mask.sum(),
                        'indices': df[neg_mask].index.tolist()[:10]  # First 10
                    })

        # Check for zero volume
        if 'volume' in df.columns:
            zero_vol = df['volume'] == 0
            if zero_vol.any():
                suspicious.append({
                    'type': 'zero_volume',
                    'column': 'volume',
                    'count': zero_vol.sum(),
                    'indices': df[zero_vol].index.tolist()[:10]
                })

        # Check for high > low violations
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                suspicious.append({
                    'type': 'high_low_violation',
                    'column': 'high/low',
                    'count': invalid_hl.sum(),
                    'indices': df[invalid_hl].index.tolist()[:10]
                })

        # Check for close outside high-low range
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            outside_range = (df['close'] > df['high']) | (df['close'] < df['low'])
            if outside_range.any():
                suspicious.append({
                    'type': 'close_outside_range',
                    'column': 'close',
                    'count': outside_range.sum(),
                    'indices': df[outside_range].index.tolist()[:10]
                })

        # Check for extreme returns (potential data errors)
        if 'log_return' in df.columns:
            extreme_returns = df['log_return'].abs() > 0.5  # >50% daily return
            if extreme_returns.any():
                suspicious.append({
                    'type': 'extreme_return',
                    'column': 'log_return',
                    'count': extreme_returns.sum(),
                    'indices': df[extreme_returns].index.tolist()[:10]
                })

        return suspicious

    def _calculate_quality_score(
        self,
        missing_pct: Dict[str, float],
        outlier_count: Dict[str, int],
        duplicate_rows: int,
        n_gaps: int,
        n_suspicious: int,
        total_rows: int
    ) -> float:
        """Calculate overall data quality score (0-1)"""
        # Penalize for missing values (max 30% penalty)
        avg_missing = np.mean(list(missing_pct.values())) if missing_pct else 0
        missing_penalty = min(avg_missing * 2, 0.3)

        # Penalize for outliers (max 20% penalty)
        if outlier_count:
            avg_outlier_pct = np.mean([c / total_rows for c in outlier_count.values()])
            outlier_penalty = min(avg_outlier_pct * 4, 0.2)
        else:
            outlier_penalty = 0

        # Penalize for duplicates (max 10% penalty)
        duplicate_penalty = min((duplicate_rows / total_rows) * 2, 0.1)

        # Penalize for gaps (max 20% penalty)
        gap_penalty = min(n_gaps * 0.02, 0.2)

        # Penalize for suspicious values (max 20% penalty)
        suspicious_penalty = min(n_suspicious * 0.04, 0.2)

        score = 1.0 - missing_penalty - outlier_penalty - duplicate_penalty - gap_penalty - suspicious_penalty
        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        missing_pct: Dict[str, float],
        outlier_count: Dict[str, int],
        date_gaps: List[Tuple[datetime, datetime]],
        suspicious_values: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate cleaning recommendations based on quality assessment"""
        recommendations = []

        # Missing value recommendations
        high_missing = [col for col, pct in missing_pct.items() if pct > 0.1]
        if high_missing:
            recommendations.append(
                f"Columns with >10% missing: {', '.join(high_missing)}. "
                f"Consider forward-fill for time series or dropping these columns."
            )

        # Outlier recommendations
        high_outlier = [col for col, count in outlier_count.items() if count > 10]
        if high_outlier:
            recommendations.append(
                f"Columns with many outliers: {', '.join(high_outlier)}. "
                f"Consider winsorization or checking for data errors."
            )

        # Gap recommendations
        if date_gaps:
            recommendations.append(
                f"Found {len(date_gaps)} time gaps in data. "
                f"Verify these are expected (holidays) or fill missing dates."
            )

        # Suspicious value recommendations
        for sv in suspicious_values:
            if sv['type'] == 'negative_price':
                recommendations.append(
                    f"Found {sv['count']} negative prices in {sv['column']}. "
                    f"These are likely data errors and should be corrected."
                )
            elif sv['type'] == 'high_low_violation':
                recommendations.append(
                    f"Found {sv['count']} rows where high < low. "
                    f"These are data errors and should be corrected or removed."
                )

        if not recommendations:
            recommendations.append("Data quality looks good. No major issues detected.")

        return recommendations

    def handle_missing(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[ImputationMethod] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in specified columns.

        Args:
            df: DataFrame to clean
            columns: Columns to process (None = all)
            method: Imputation method (defaults to instance setting)

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        method = method or self.imputation_method
        columns = columns or df.columns.tolist()

        original_missing = df[columns].isnull().sum().sum()

        if method == ImputationMethod.DROP:
            df = df.dropna(subset=columns)
            affected = original_missing

        elif method == ImputationMethod.FORWARD_FILL:
            df[columns] = df[columns].ffill()
            affected = original_missing - df[columns].isnull().sum().sum()

        elif method == ImputationMethod.BACKWARD_FILL:
            df[columns] = df[columns].bfill()
            affected = original_missing - df[columns].isnull().sum().sum()

        elif method == ImputationMethod.LINEAR:
            for col in columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    df[col] = df[col].interpolate(method='linear')
            affected = original_missing - df[columns].isnull().sum().sum()

        elif method == ImputationMethod.MEAN:
            for col in columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    df[col] = df[col].fillna(df[col].mean())
            affected = original_missing - df[columns].isnull().sum().sum()

        elif method == ImputationMethod.MEDIAN:
            for col in columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    df[col] = df[col].fillna(df[col].median())
            affected = original_missing - df[columns].isnull().sum().sum()

        elif method == ImputationMethod.ZERO:
            df[columns] = df[columns].fillna(0)
            affected = original_missing

        self._record(
            operation=f"handle_missing_{method.value}",
            column=None,
            original_count=len(df),
            affected_count=affected,
            details={'columns': columns, 'method': method.value}
        )

        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[OutlierMethod] = None,
        treatment: Optional[OutlierTreatment] = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in specified columns.

        Args:
            df: DataFrame to clean
            columns: Columns to process (None = all numeric)
            method: Outlier detection method
            treatment: How to handle outliers

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        method = method or self.outlier_method
        treatment = treatment or self.outlier_treatment

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        total_outliers = 0

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]
            outliers = self._detect_outliers(series.dropna(), method)

            # Map back to original index
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask[outliers.index] = outliers

            n_outliers = outlier_mask.sum()
            if n_outliers == 0:
                continue

            total_outliers += n_outliers

            if treatment == OutlierTreatment.REMOVE:
                df = df[~outlier_mask]

            elif treatment == OutlierTreatment.CLIP:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(lower, upper)

            elif treatment == OutlierTreatment.WINSORIZE:
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
                df.loc[df[col] < lower, col] = lower
                df.loc[df[col] > upper, col] = upper

            elif treatment == OutlierTreatment.IMPUTE:
                df.loc[outlier_mask, col] = np.nan
                df[col] = df[col].ffill().bfill()

            elif treatment == OutlierTreatment.FLAG:
                df[f'{col}_outlier'] = outlier_mask

            self._record(
                operation=f"handle_outliers_{treatment.value}",
                column=col,
                original_count=len(outlier_mask),
                affected_count=n_outliers,
                details={'method': method.value, 'treatment': treatment.value}
            )

        logger.info(f"Handled {total_outliers} outliers across {len(columns)} columns")
        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: DataFrame to clean
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', False)

        Returns:
            DataFrame without duplicates
        """
        original_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = original_count - len(df)

        self._record(
            operation='remove_duplicates',
            column=None,
            original_count=original_count,
            affected_count=removed_count,
            details={'subset': subset, 'keep': keep}
        )

        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")

        return df

    def fix_suspicious_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix common suspicious values in financial data.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Fix negative prices (set to NaN and forward fill)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    df.loc[neg_mask, col] = np.nan
                    df[col] = df[col].ffill()
                    self._record(
                        operation='fix_negative_prices',
                        column=col,
                        original_count=len(df),
                        affected_count=neg_mask.sum()
                    )

        # Fix high < low violations (swap values)
        if 'high' in df.columns and 'low' in df.columns:
            violation_mask = df['high'] < df['low']
            if violation_mask.any():
                df.loc[violation_mask, ['high', 'low']] = df.loc[
                    violation_mask, ['low', 'high']
                ].values
                self._record(
                    operation='fix_high_low_violation',
                    column='high/low',
                    original_count=len(df),
                    affected_count=violation_mask.sum()
                )

        return df

    def fill_time_gaps(
        self,
        df: pd.DataFrame,
        time_col: str = 'time',
        freq: str = 'D',  # Daily
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Fill gaps in time series data.

        Args:
            df: DataFrame with time column
            time_col: Name of time column
            freq: Frequency for resampling
            method: Method for filling gaps

        Returns:
            DataFrame with gaps filled
        """
        if time_col not in df.columns:
            logger.warning(f"Time column '{time_col}' not found, skipping gap fill")
            return df

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        original_count = len(df)

        # Handle multi-ticker data
        if 'ticker' in df.columns:
            filled_dfs = []
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker].copy()
                ticker_df = ticker_df.set_index(time_col)
                ticker_df = ticker_df.resample(freq).first()

                if method == 'ffill':
                    ticker_df = ticker_df.ffill()
                elif method == 'bfill':
                    ticker_df = ticker_df.bfill()
                elif method == 'linear':
                    ticker_df = ticker_df.interpolate(method='linear')

                ticker_df['ticker'] = ticker
                ticker_df = ticker_df.reset_index()
                filled_dfs.append(ticker_df)

            df = pd.concat(filled_dfs, ignore_index=True)
        else:
            df = df.set_index(time_col)
            df = df.resample(freq).first()

            if method == 'ffill':
                df = df.ffill()
            elif method == 'bfill':
                df = df.bfill()
            elif method == 'linear':
                df = df.interpolate(method='linear')

            df = df.reset_index()

        new_count = len(df)
        self._record(
            operation='fill_time_gaps',
            column=time_col,
            original_count=original_count,
            affected_count=new_count - original_count,
            details={'freq': freq, 'method': method}
        )

        if new_count > original_count:
            logger.info(f"Added {new_count - original_count} rows to fill time gaps")

        return df

    def clean(
        self,
        df: pd.DataFrame,
        fix_suspicious: bool = True,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        remove_duplicates: bool = True,
        fill_gaps: bool = False
    ) -> CleaningResult:
        """
        Run full cleaning pipeline.

        Args:
            df: DataFrame to clean
            fix_suspicious: Fix suspicious values
            handle_missing: Handle missing values
            handle_outliers: Handle outliers
            remove_duplicates: Remove duplicates
            fill_gaps: Fill time gaps

        Returns:
            CleaningResult with cleaned data and audit trail
        """
        self.records = []  # Reset records

        # Assess quality before cleaning
        quality_before = self.assess_quality(df)
        logger.info(f"Data quality before cleaning: {quality_before.quality_score:.2%}")

        transformations = []

        # Apply cleaning steps
        if fix_suspicious:
            df = self.fix_suspicious_values(df)
            transformations.append('fix_suspicious_values')

        if remove_duplicates:
            df = self.remove_duplicates(df)
            transformations.append('remove_duplicates')

        if handle_missing:
            df = self.handle_missing(df)
            transformations.append('handle_missing')

        if handle_outliers:
            df = self.handle_outliers(df)
            transformations.append('handle_outliers')

        if fill_gaps and 'time' in df.columns:
            df = self.fill_time_gaps(df)
            transformations.append('fill_time_gaps')

        # Assess quality after cleaning
        quality_after = self.assess_quality(df)
        logger.info(f"Data quality after cleaning: {quality_after.quality_score:.2%}")

        # Verify we haven't lost too much data
        if len(df) < self.min_required_data_pct * quality_before.total_rows:
            logger.warning(
                f"Cleaning removed {1 - len(df)/quality_before.total_rows:.1%} of data, "
                f"which exceeds the threshold of {1 - self.min_required_data_pct:.1%}"
            )

        return CleaningResult(
            cleaned_data=df,
            records=self.records.copy(),
            quality_before=quality_before,
            quality_after=quality_after,
            transformations_applied=transformations
        )


def clean_financial_data(
    df: pd.DataFrame,
    imputation_method: str = 'ffill',
    outlier_method: str = 'iqr',
    outlier_treatment: str = 'clip',
    **kwargs
) -> CleaningResult:
    """
    Convenience function for cleaning financial data.

    Args:
        df: DataFrame to clean
        imputation_method: Method for missing values
        outlier_method: Method for outlier detection
        outlier_treatment: How to handle outliers
        **kwargs: Additional arguments for DataCleaner

    Returns:
        CleaningResult with cleaned data
    """
    cleaner = DataCleaner(
        imputation_method=ImputationMethod(imputation_method),
        outlier_method=OutlierMethod(outlier_method),
        outlier_treatment=OutlierTreatment(outlier_treatment),
        **kwargs
    )
    return cleaner.clean(df)


def validate_cleaned_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that cleaned data meets quality requirements.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for remaining NaN
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"Columns still have NaN values: {nan_cols}")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            issues.append(f"Column '{col}' contains infinite values")

    # Check minimum row count
    if len(df) < 100:
        issues.append(f"Very few rows remaining after cleaning: {len(df)}")

    # Check for required columns
    required = ['close', 'time']
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")

    return len(issues) == 0, issues
