"""Centralised registry for ML and trading metrics.

Provides a stable interface so evaluation code and the frontend can ask for
metrics by name without duplicating implementations. It wraps the existing
`MetricsCalculator` utilities and allows easy extension with custom callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import numpy as np

from .metrics import MetricsCalculator, calculate_financial_metrics


# Type alias for metric callables
MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class MetricDefinition:
    name: str
    fn: MetricFn
    description: str
    higher_is_better: bool = True


class MetricsRegistry:
    """Registry of supported metrics."""

    def __init__(self):
        self._registry: Dict[str, MetricDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("rmse", MetricsCalculator.rmse, "Root Mean Squared Error", higher_is_better=False)
        self.register("mae", MetricsCalculator.mae, "Mean Absolute Error", higher_is_better=False)
        self.register("mape", MetricsCalculator.mape, "Mean Absolute Percentage Error", higher_is_better=False)
        self.register("r2", MetricsCalculator.r2, "R-squared")
        self.register("directional_accuracy", MetricsCalculator.directional_accuracy, "Directional accuracy")

    # Public API -----------------------------------------------------------
    def register(self, name: str, fn: MetricFn, description: str, *, higher_is_better: bool = True) -> None:
        self._registry[name] = MetricDefinition(
            name=name,
            fn=fn,
            description=description,
            higher_is_better=higher_is_better,
        )

    def available(self) -> Iterable[str]:
        return self._registry.keys()

    def definitions(self) -> Mapping[str, MetricDefinition]:
        return self._registry

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_names: Iterable[str] | None = None,
    ) -> Dict[str, float]:
        """Compute a subset of metrics."""
        names = list(metric_names) if metric_names is not None else list(self._registry.keys())
        results: Dict[str, float] = {}
        for name in names:
            if name not in self._registry:
                raise KeyError(f"Metric '{name}' is not registered")
            fn = self._registry[name].fn
            results[name] = float(fn(y_true, y_pred))
        return results

    def compute_financial(self, returns: np.ndarray, benchmark_returns: np.ndarray | None = None) -> Dict[str, float]:
        """Delegate to existing financial metrics helper."""
        return calculate_financial_metrics(returns, benchmark_returns)


def get_default_registry() -> MetricsRegistry:
    """Convenience accessor for callers that don't need custom metrics."""
    return MetricsRegistry()
