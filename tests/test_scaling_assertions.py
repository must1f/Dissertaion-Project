import numpy as np
import pytest

from src.evaluation.financial_metrics import FinancialMetrics, assert_price_scale


def test_assert_price_scale_raises_for_z_scores():
    z = np.array([0.1, -0.2, 0.3, 0.0])
    with pytest.raises(ValueError):
        assert_price_scale(z, context="unit-test", min_std_threshold=1.0, raise_error=True)


def test_compute_all_metrics_rejects_zscore_prices_when_not_returns():
    z_pred = np.array([0.1, -0.1, 0.2, -0.05])
    z_true = np.array([0.0, 0.1, -0.2, 0.05])
    returns = np.array([0.0, 0.01, -0.02, 0.005])

    with pytest.raises(ValueError):
        FinancialMetrics.compute_all_metrics(
            returns=returns,
            predictions=z_pred,
            targets=z_true,
            predictions_are_returns=False,
            validate_price_scale=True,
            price_scale_context="unit-test",
        )
