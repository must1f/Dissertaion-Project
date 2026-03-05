"""
Unit tests for leakage auditing utilities.
"""

import numpy as np
import pandas as pd

from src.data.leakage_auditor import LeakageAuditor, LeakageSeverity


def test_scaler_leakage_detection():
    auditor = LeakageAuditor()

    # Register scaler fit on train indices 0-79
    fit_data = np.random.randn(80, 3)
    auditor.register_scaler_fit(
        scaler_id="std",
        fit_data=fit_data,
        fit_indices=(0, 79),
        fit_dates=("2024-01-01", "2024-03-31"),
        feature_names=["a", "b", "c"],
    )

    # Apply scaler on earlier indices (lookahead)
    warnings = auditor.check_scaler_application(
        scaler_id="std",
        transform_indices=(-10, -1),  # earlier than fit range -> critical
        transform_dates=("2023-12-01", "2023-12-10"),
    )

    assert any(w.severity == LeakageSeverity.CRITICAL for w in warnings)

    # Fit range bleeding into val/test
    split_warnings = auditor.audit_scaler_splits(
        scaler_id="std",
        train_indices=(0, 79),
        val_indices=(80, 89),
        test_indices=(90, 99),
    )
    assert not split_warnings  # fit exactly on train should be clean

    # Now pretend scaler was fit too far
    auditor.register_scaler_fit(
        scaler_id="std2",
        fit_data=fit_data,
        fit_indices=(0, 90),  # extends into val/test
    )
    split_warnings = auditor.audit_scaler_splits(
        scaler_id="std2",
        train_indices=(0, 79),
        val_indices=(80, 89),
        test_indices=(90, 99),
    )
    assert any(w.severity == LeakageSeverity.CRITICAL for w in split_warnings)


def test_feature_timestamp_leakage():
    auditor = LeakageAuditor()

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    # Create features shifted forward by 1 day relative to labels -> leakage
    features = pd.DataFrame({"time": dates + pd.Timedelta(days=1), "feat": np.arange(5)})
    labels = pd.DataFrame({"time": dates, "label": np.arange(5)})

    warnings = auditor.check_feature_timestamps(features, labels, feature_lags={"feat": 0})

    assert any(w.severity == LeakageSeverity.CRITICAL for w in warnings)
