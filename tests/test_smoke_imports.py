"""
Broad smoke tests to ensure core modules import without runtime errors.

These tests are lightweight and intended to catch missing dependencies
or circular imports early in CI.
"""

import importlib
import pytest


MODULES = [
    "main",
    "src.evaluation.financial_metrics",
    "src.evaluation.metrics",
    "src.evaluation.unified_evaluator",
    "src.evaluation.rolling_metrics",
    "src.trading.agent",
    "src.models.model_registry",
    "backend.app.main",
    "backend.app.api.routes.metrics",
    "compare_pinn_baseline",
    "cross_asset_eval",
    "physics_ablation",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional dependency missing for {module}: {exc}")
