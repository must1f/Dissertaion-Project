"""
Reporting helpers that group metrics into forecasting, financial, and regime diagnostics.
"""

from typing import Dict, Any, Optional
import numpy as np

from .metrics import calculate_metrics, calculate_financial_metrics


def strategy_returns_from_sign(pred_returns: np.ndarray, true_returns: np.ndarray) -> np.ndarray:
    """Simple long/short strategy: sign(pred) * true return."""
    pred_sign = np.sign(pred_returns)
    return pred_sign * true_returns


def summarize_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
    prefix: str = "test_",
) -> Dict[str, Any]:
    """Return structured metrics grouped by category."""
    forecast = calculate_metrics(y_true, y_pred, prefix=prefix)

    strategy_rets = strategy_returns_from_sign(y_pred, y_true)
    financial = calculate_financial_metrics(strategy_rets, prefix=prefix)

    regime: Dict[str, Any] = {}
    if regime_labels is not None and len(regime_labels) == len(y_true):
        for regime_val in np.unique(regime_labels):
            mask = regime_labels == regime_val
            if mask.sum() == 0:
                continue
            reg_prefix = f"{prefix}regime_{regime_val}_"
            regime.update(calculate_metrics(y_true[mask], y_pred[mask], prefix=reg_prefix))

    return {
        "forecasting": forecast,
        "financial": financial,
        "regime": regime,
    }
