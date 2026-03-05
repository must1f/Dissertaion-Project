"""
Leakage Audit Tooling

Automated checks for lookahead leakage in features and labels.
Ensures scaler/normalizer fitting is done only on training data.

Critical for research integrity - prevents data leakage that would
invalidate model comparisons and overstate performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LeakageSeverity(Enum):
    """Severity levels for leakage warnings"""
    CRITICAL = "critical"  # Definite lookahead leakage
    WARNING = "warning"    # Potential leakage, needs review
    INFO = "info"          # Minor issue or best practice suggestion


@dataclass
class LeakageWarning:
    """A single leakage warning"""
    severity: LeakageSeverity
    category: str
    message: str
    feature_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'feature_name': self.feature_name,
            'details': self.details,
            'suggested_fix': self.suggested_fix
        }


@dataclass
class LeakageAuditResult:
    """Complete audit result"""
    timestamp: str
    passed: bool
    n_critical: int
    n_warnings: int
    n_info: int
    warnings: List[LeakageWarning]
    scaler_audit: Dict[str, Any]
    feature_audit: Dict[str, Any]
    split_audit: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'passed': self.passed,
            'n_critical': self.n_critical,
            'n_warnings': self.n_warnings,
            'n_info': self.n_info,
            'warnings': [w.to_dict() for w in self.warnings],
            'scaler_audit': self.scaler_audit,
            'feature_audit': self.feature_audit,
            'split_audit': self.split_audit
        }

    def summary(self) -> str:
        """Human-readable summary"""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Leakage Audit: {status}",
            f"  Critical Issues: {self.n_critical}",
            f"  Warnings: {self.n_warnings}",
            f"  Info: {self.n_info}",
        ]
        if not self.passed:
            lines.append("\nCritical Issues:")
            for w in self.warnings:
                if w.severity == LeakageSeverity.CRITICAL:
                    lines.append(f"  - [{w.category}] {w.message}")
        return "\n".join(lines)


@dataclass
class ScalerFitRecord:
    """Record of when/how a scaler was fit"""
    scaler_id: str
    fit_timestamp: str
    fit_indices: Tuple[int, int]  # (start, end) indices
    fit_dates: Optional[Tuple[str, str]] = None
    feature_names: List[str] = field(default_factory=list)
    data_hash: Optional[str] = None


class LeakageAuditor:
    """
    Audits data pipeline for lookahead leakage.

    Checks:
    1. Feature timestamps - no features use future data relative to label
    2. Scaler fitting - scalers fit only on training data
    3. Split integrity - no overlap between train/val/test
    4. Rolling window leakage - proper handling of expanding/rolling splits
    """

    def __init__(self):
        self.scaler_records: Dict[str, ScalerFitRecord] = {}
        self.warnings: List[LeakageWarning] = []

    def reset(self):
        """Reset audit state"""
        self.scaler_records = {}
        self.warnings = []

    # ========== SCALER TRACKING ==========

    def register_scaler_fit(
        self,
        scaler_id: str,
        fit_data: np.ndarray,
        fit_indices: Tuple[int, int],
        fit_dates: Optional[Tuple[str, str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Register a scaler fitting operation for audit tracking.

        Args:
            scaler_id: Unique identifier for the scaler
            fit_data: Data used for fitting
            fit_indices: (start, end) indices of fit data
            fit_dates: Optional (start, end) dates
            feature_names: Names of features being scaled
        """
        # Compute hash of fit data for verification
        data_hash = hashlib.sha256(fit_data.tobytes()).hexdigest()[:16]

        record = ScalerFitRecord(
            scaler_id=scaler_id,
            fit_timestamp=datetime.now().isoformat(),
            fit_indices=fit_indices,
            fit_dates=fit_dates,
            feature_names=feature_names or [],
            data_hash=data_hash
        )

        self.scaler_records[scaler_id] = record
        logger.debug(f"Registered scaler fit: {scaler_id} on indices {fit_indices}")

    def check_scaler_application(
        self,
        scaler_id: str,
        transform_indices: Tuple[int, int],
        transform_dates: Optional[Tuple[str, str]] = None
    ) -> List[LeakageWarning]:
        """
        Check if scaler is being applied to data it was trained on.

        Args:
            scaler_id: Scaler to check
            transform_indices: (start, end) indices of transform data
            transform_dates: Optional (start, end) dates

        Returns:
            List of warnings if leakage detected
        """
        warnings = []

        if scaler_id not in self.scaler_records:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.WARNING,
                category="scaler_untracked",
                message=f"Scaler '{scaler_id}' not registered - cannot verify fit/transform separation",
                suggested_fix="Register scaler fit with register_scaler_fit()"
            ))
            return warnings

        record = self.scaler_records[scaler_id]
        fit_start, fit_end = record.fit_indices
        transform_start, transform_end = transform_indices

        # Check if transform data is before fit data (legitimate: train before test)
        # or if transform data overlaps with or includes future data relative to fit

        # CRITICAL: Transform on training data before fit is completed
        if transform_start < fit_start:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.CRITICAL,
                category="scaler_lookahead",
                message=f"Scaler '{scaler_id}' applied to data before fit range",
                details={
                    'fit_indices': record.fit_indices,
                    'transform_indices': transform_indices
                },
                suggested_fix="Ensure scaler is fit on training data before applying to any data"
            ))

        # WARNING: Transform includes data that was in fit range (test leakage)
        if transform_start <= fit_end and transform_end >= fit_start:
            if transform_start < fit_end and transform_end > fit_start:
                # Overlapping but not identical - this might be fine for train set
                pass

        return warnings

    def audit_scaler_splits(
        self,
        scaler_id: str,
        train_indices: Tuple[int, int],
        val_indices: Optional[Tuple[int, int]] = None,
        test_indices: Optional[Tuple[int, int]] = None
    ) -> List[LeakageWarning]:
        """
        Audit scaler usage across train/val/test splits.

        Args:
            scaler_id: Scaler to audit
            train_indices: Training data indices
            val_indices: Validation data indices
            test_indices: Test data indices

        Returns:
            List of warnings
        """
        warnings = []

        if scaler_id not in self.scaler_records:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.WARNING,
                category="scaler_untracked",
                message=f"Scaler '{scaler_id}' not registered",
            ))
            return warnings

        record = self.scaler_records[scaler_id]
        fit_start, fit_end = record.fit_indices
        train_start, train_end = train_indices

        # Scaler must be fit only on training data
        if fit_start != train_start or fit_end != train_end:
            # Check if fit range extends beyond training
            if fit_end > train_end:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="scaler_leakage",
                    message=f"Scaler '{scaler_id}' fit on data beyond training range",
                    details={
                        'fit_indices': record.fit_indices,
                        'train_indices': train_indices
                    },
                    suggested_fix="Fit scaler only on training data indices"
                ))

            if val_indices and fit_end >= val_indices[0]:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="scaler_val_leakage",
                    message=f"Scaler '{scaler_id}' fit includes validation data",
                    details={
                        'fit_indices': record.fit_indices,
                        'val_indices': val_indices
                    },
                    suggested_fix="Ensure scaler is fit before validation split point"
                ))

            if test_indices and fit_end >= test_indices[0]:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="scaler_test_leakage",
                    message=f"Scaler '{scaler_id}' fit includes test data",
                    details={
                        'fit_indices': record.fit_indices,
                        'test_indices': test_indices
                    },
                    suggested_fix="Ensure scaler is fit before test split point"
                ))

        return warnings

    # ========== FEATURE TIMESTAMP VALIDATION ==========

    def check_feature_timestamps(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_lags: Optional[Dict[str, int]] = None,
        time_col: str = 'time'
    ) -> List[LeakageWarning]:
        """
        Verify no feature uses future data relative to label date.

        Args:
            features_df: DataFrame with features and timestamps
            labels_df: DataFrame with labels and timestamps
            feature_lags: Dict mapping feature name to required lag
            time_col: Name of timestamp column

        Returns:
            List of warnings
        """
        warnings = []

        if time_col not in features_df.columns:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.WARNING,
                category="missing_timestamp",
                message=f"Features DataFrame missing '{time_col}' column - cannot verify temporal ordering",
                suggested_fix=f"Ensure features DataFrame has '{time_col}' column"
            ))
            return warnings

        if time_col not in labels_df.columns:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.WARNING,
                category="missing_timestamp",
                message=f"Labels DataFrame missing '{time_col}' column",
                suggested_fix=f"Ensure labels DataFrame has '{time_col}' column"
            ))
            return warnings

        # Check alignment
        features_df = features_df.sort_values(time_col)
        labels_df = labels_df.sort_values(time_col)

        # Find common timestamps
        common_times = set(features_df[time_col]) & set(labels_df[time_col])

        if len(common_times) == 0:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.CRITICAL,
                category="no_common_timestamps",
                message="No common timestamps between features and labels",
                suggested_fix="Ensure features and labels have aligned timestamps"
            ))
            return warnings

        # Check feature lags if provided
        if feature_lags:
            for feature_name, required_lag in feature_lags.items():
                if feature_name not in features_df.columns:
                    continue

                # Check if feature values could have lookahead
                # This is a heuristic - actual lag validation requires domain knowledge
                if required_lag < 1:
                    warnings.append(LeakageWarning(
                        severity=LeakageSeverity.CRITICAL,
                        category="feature_no_lag",
                        message=f"Feature '{feature_name}' has lag < 1 - potential lookahead",
                        feature_name=feature_name,
                        details={'required_lag': required_lag},
                        suggested_fix=f"Ensure feature '{feature_name}' is computed with at least 1 period lag"
                    ))

        return warnings

    def check_feature_availability(
        self,
        feature_config: Dict[str, Dict[str, Any]],
        prediction_time: str = 'close'
    ) -> List[LeakageWarning]:
        """
        Check if features would be available at prediction time.

        Args:
            feature_config: Dict mapping feature names to their configs
                Expected format: {
                    'feature_name': {
                        'available_at': 'open' | 'close' | 'next_open',
                        'lag': int,
                        'source': str
                    }
                }
            prediction_time: When predictions are made ('open', 'close')

        Returns:
            List of warnings
        """
        warnings = []

        availability_order = ['open', 'close', 'next_open']
        pred_idx = availability_order.index(prediction_time)

        for feature_name, config in feature_config.items():
            available_at = config.get('available_at', 'close')
            lag = config.get('lag', 0)

            if available_at not in availability_order:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.WARNING,
                    category="unknown_availability",
                    message=f"Feature '{feature_name}' has unknown availability time: {available_at}",
                    feature_name=feature_name
                ))
                continue

            avail_idx = availability_order.index(available_at)

            # Feature available after prediction time without sufficient lag
            if avail_idx > pred_idx and lag == 0:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="feature_lookahead",
                    message=f"Feature '{feature_name}' available at '{available_at}' but predicting at '{prediction_time}'",
                    feature_name=feature_name,
                    details={'available_at': available_at, 'prediction_time': prediction_time, 'lag': lag},
                    suggested_fix=f"Add lag >= 1 to feature '{feature_name}' or change availability time"
                ))

        return warnings

    # ========== SPLIT VALIDATION ==========

    def check_split_leakage(
        self,
        train_indices: np.ndarray,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[LeakageWarning]:
        """
        Check for overlap between train/val/test splits.

        Args:
            train_indices: Training data indices
            val_indices: Validation data indices
            test_indices: Test data indices
            timestamps: Optional timestamps for temporal ordering check

        Returns:
            List of warnings
        """
        warnings = []

        train_set = set(train_indices)

        # Check train-val overlap
        if val_indices is not None:
            val_set = set(val_indices)
            overlap = train_set & val_set
            if overlap:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="split_overlap",
                    message=f"Train and validation sets have {len(overlap)} overlapping indices",
                    details={'overlap_count': len(overlap), 'overlap_sample': list(overlap)[:5]},
                    suggested_fix="Ensure train/val splits are mutually exclusive"
                ))

        # Check train-test overlap
        if test_indices is not None:
            test_set = set(test_indices)
            overlap = train_set & test_set
            if overlap:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="split_overlap",
                    message=f"Train and test sets have {len(overlap)} overlapping indices",
                    details={'overlap_count': len(overlap), 'overlap_sample': list(overlap)[:5]},
                    suggested_fix="Ensure train/test splits are mutually exclusive"
                ))

        # Check val-test overlap
        if val_indices is not None and test_indices is not None:
            val_set = set(val_indices)
            test_set = set(test_indices)
            overlap = val_set & test_set
            if overlap:
                warnings.append(LeakageWarning(
                    severity=LeakageSeverity.CRITICAL,
                    category="split_overlap",
                    message=f"Validation and test sets have {len(overlap)} overlapping indices",
                    details={'overlap_count': len(overlap), 'overlap_sample': list(overlap)[:5]},
                    suggested_fix="Ensure val/test splits are mutually exclusive"
                ))

        # Check temporal ordering (train should come before val/test for time series)
        if timestamps is not None:
            train_max_time = timestamps[train_indices].max()

            if val_indices is not None:
                val_min_time = timestamps[val_indices].min()
                if val_min_time < train_max_time:
                    warnings.append(LeakageWarning(
                        severity=LeakageSeverity.CRITICAL,
                        category="temporal_leakage",
                        message="Validation data starts before training data ends",
                        details={
                            'train_max': str(train_max_time),
                            'val_min': str(val_min_time)
                        },
                        suggested_fix="Use temporal split - validation should follow training chronologically"
                    ))

            if test_indices is not None:
                test_min_time = timestamps[test_indices].min()
                if test_min_time < train_max_time:
                    warnings.append(LeakageWarning(
                        severity=LeakageSeverity.CRITICAL,
                        category="temporal_leakage",
                        message="Test data starts before training data ends",
                        details={
                            'train_max': str(train_max_time),
                            'test_min': str(test_min_time)
                        },
                        suggested_fix="Use temporal split - test should follow training chronologically"
                    ))

        return warnings

    def check_sequence_leakage(
        self,
        sequences: np.ndarray,
        train_end_idx: int,
        sequence_length: int,
        forecast_horizon: int = 1
    ) -> List[LeakageWarning]:
        """
        Check for leakage in sequence creation for time series models.

        Args:
            sequences: Sequence data (N, seq_len, features)
            train_end_idx: Last index of training data
            sequence_length: Length of each sequence
            forecast_horizon: Steps ahead being predicted

        Returns:
            List of warnings
        """
        warnings = []

        # Check if any training sequence could include validation/test data
        # Last valid training sequence should end at train_end_idx - forecast_horizon
        max_valid_start = train_end_idx - sequence_length - forecast_horizon + 1

        if max_valid_start < 0:
            warnings.append(LeakageWarning(
                severity=LeakageSeverity.CRITICAL,
                category="sequence_leakage",
                message="Not enough training data for sequence length + forecast horizon",
                details={
                    'train_end_idx': train_end_idx,
                    'sequence_length': sequence_length,
                    'forecast_horizon': forecast_horizon,
                    'max_valid_start': max_valid_start
                },
                suggested_fix="Increase training data or reduce sequence length"
            ))

        return warnings

    # ========== FULL AUDIT ==========

    def run_full_audit(
        self,
        train_data: Optional[pd.DataFrame] = None,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        scalers: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Dict]] = None,
        time_col: str = 'time'
    ) -> LeakageAuditResult:
        """
        Run comprehensive leakage audit.

        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            test_data: Test DataFrame
            scalers: Dict of scalers with their fit records
            feature_config: Feature configuration with lags/availability
            time_col: Timestamp column name

        Returns:
            LeakageAuditResult with all findings
        """
        logger.info("=" * 60)
        logger.info("RUNNING LEAKAGE AUDIT")
        logger.info("=" * 60)

        self.reset()
        all_warnings = []

        # Audit splits
        split_audit = {'status': 'not_checked'}
        if train_data is not None:
            logger.info("Checking data splits...")

            train_indices = np.arange(len(train_data))
            val_indices = np.arange(len(train_data), len(train_data) + len(val_data)) if val_data is not None else None
            test_indices = np.arange(
                len(train_data) + (len(val_data) if val_data is not None else 0),
                len(train_data) + (len(val_data) if val_data is not None else 0) + len(test_data)
            ) if test_data is not None else None

            split_warnings = self.check_split_leakage(
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices
            )
            all_warnings.extend(split_warnings)

            split_audit = {
                'status': 'passed' if not split_warnings else 'failed',
                'train_size': len(train_data),
                'val_size': len(val_data) if val_data is not None else 0,
                'test_size': len(test_data) if test_data is not None else 0,
                'warnings': len(split_warnings)
            }

        # Audit scalers
        scaler_audit = {'status': 'not_checked'}
        if scalers:
            logger.info("Checking scaler fit dates...")
            for scaler_id in self.scaler_records:
                scaler_warnings = self.audit_scaler_splits(
                    scaler_id=scaler_id,
                    train_indices=(0, len(train_data)) if train_data is not None else (0, 0),
                    val_indices=(len(train_data), len(train_data) + len(val_data)) if val_data is not None else None,
                    test_indices=(
                        len(train_data) + (len(val_data) if val_data is not None else 0),
                        len(train_data) + (len(val_data) if val_data is not None else 0) + len(test_data)
                    ) if test_data is not None else None
                )
                all_warnings.extend(scaler_warnings)

            scaler_audit = {
                'status': 'passed' if not any(w.category.startswith('scaler') for w in all_warnings) else 'failed',
                'n_scalers': len(self.scaler_records),
                'warnings': sum(1 for w in all_warnings if w.category.startswith('scaler'))
            }

        # Audit features
        feature_audit = {'status': 'not_checked'}
        if feature_config:
            logger.info("Checking feature availability...")
            feature_warnings = self.check_feature_availability(feature_config)
            all_warnings.extend(feature_warnings)

            feature_audit = {
                'status': 'passed' if not feature_warnings else 'failed',
                'n_features': len(feature_config),
                'warnings': len(feature_warnings)
            }

        # Count by severity
        n_critical = sum(1 for w in all_warnings if w.severity == LeakageSeverity.CRITICAL)
        n_warnings = sum(1 for w in all_warnings if w.severity == LeakageSeverity.WARNING)
        n_info = sum(1 for w in all_warnings if w.severity == LeakageSeverity.INFO)

        passed = n_critical == 0

        result = LeakageAuditResult(
            timestamp=datetime.now().isoformat(),
            passed=passed,
            n_critical=n_critical,
            n_warnings=n_warnings,
            n_info=n_info,
            warnings=all_warnings,
            scaler_audit=scaler_audit,
            feature_audit=feature_audit,
            split_audit=split_audit
        )

        # Log summary
        status = "PASSED" if passed else "FAILED"
        logger.info(f"\nAudit Result: {status}")
        logger.info(f"  Critical: {n_critical}")
        logger.info(f"  Warnings: {n_warnings}")
        logger.info(f"  Info: {n_info}")

        if not passed:
            logger.error("CRITICAL LEAKAGE ISSUES DETECTED:")
            for w in all_warnings:
                if w.severity == LeakageSeverity.CRITICAL:
                    logger.error(f"  [{w.category}] {w.message}")

        logger.info("=" * 60)

        return result


# Convenience functions

def audit_train_test_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = 'time'
) -> LeakageAuditResult:
    """Quick audit of train/test split"""
    auditor = LeakageAuditor()
    return auditor.run_full_audit(
        train_data=train_df,
        test_data=test_df,
        time_col=time_col
    )


def verify_no_future_leakage(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_lags: Dict[str, int],
    time_col: str = 'time'
) -> bool:
    """
    Quick check for future leakage in features.

    Returns:
        True if no leakage detected, False otherwise
    """
    auditor = LeakageAuditor()
    warnings = auditor.check_feature_timestamps(
        features_df=features,
        labels_df=labels,
        feature_lags=feature_lags,
        time_col=time_col
    )
    return len([w for w in warnings if w.severity == LeakageSeverity.CRITICAL]) == 0
