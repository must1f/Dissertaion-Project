"""Metrics API endpoints."""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
import numpy as np

from backend.app.dependencies import get_metrics_service, get_model_service
from backend.app.services.metrics_service import MetricsService
from backend.app.services.model_service import ModelService
from backend.app.schemas.metrics import (
    MLMetrics,
    FinancialMetrics,
    PhysicsMetrics,
    ModelMetricsResponse,
    MetricsComparisonResponse,
    FinancialMetricsRequest,
    FinancialMetricsResponse,
    LeaderboardResponse,
)
from backend.app.core.exceptions import ModelNotFoundError

router = APIRouter()


@router.get("/financial", response_model=FinancialMetrics)
async def calculate_financial_metrics(
    returns: str = Query(..., description="Comma-separated returns (e.g., '0.01,-0.02,0.015')"),
    risk_free_rate: float = Query(0.02, description="Annual risk-free rate"),
    periods_per_year: int = Query(252, description="Trading periods per year"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Calculate financial metrics from returns."""
    try:
        returns_arr = np.array([float(r.strip()) for r in returns.split(",")])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid returns format")

    return metrics_service.calculate_financial_metrics(
        returns=returns_arr,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


@router.post("/financial", response_model=FinancialMetricsResponse)
async def calculate_financial_metrics_post(
    request: FinancialMetricsRequest,
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Calculate financial metrics from returns (POST)."""
    returns_arr = np.array(request.returns)
    benchmark = np.array(request.benchmark_returns) if request.benchmark_returns else None

    metrics = metrics_service.calculate_financial_metrics(
        returns=returns_arr,
        risk_free_rate=request.risk_free_rate,
        periods_per_year=request.periods_per_year,
        benchmark_returns=benchmark,
    )

    return FinancialMetricsResponse(
        metrics=metrics,
        input_summary={
            "n_returns": len(request.returns),
            "risk_free_rate": request.risk_free_rate,
            "periods_per_year": request.periods_per_year,
            "has_benchmark": benchmark is not None,
        },
    )


@router.get("/ml", response_model=MLMetrics)
async def calculate_ml_metrics(
    y_true: str = Query(..., description="Comma-separated true values"),
    y_pred: str = Query(..., description="Comma-separated predicted values"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Calculate ML prediction metrics."""
    try:
        true_arr = np.array([float(v.strip()) for v in y_true.split(",")])
        pred_arr = np.array([float(v.strip()) for v in y_pred.split(",")])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid values format")

    if len(true_arr) != len(pred_arr):
        raise HTTPException(status_code=400, detail="Arrays must be same length")

    return metrics_service.calculate_ml_metrics(true_arr, pred_arr)


@router.get("/physics/{model_key}", response_model=PhysicsMetrics)
async def get_physics_metrics(
    model_key: str,
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Get physics constraint metrics for a PINN model."""
    metrics = metrics_service.get_physics_metrics(model_key)
    if metrics is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_key} is not a PINN or has no physics metrics",
        )
    return metrics


@router.get("/model/{model_key}", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_key: str,
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Get all metrics for a model."""
    try:
        return metrics_service.get_model_metrics(model_key)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comparison", response_model=MetricsComparisonResponse)
async def compare_model_metrics(
    model_keys: str = Query(..., description="Comma-separated model keys"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Compare metrics across multiple models."""
    keys = [k.strip() for k in model_keys.split(",")]
    return metrics_service.compare_models(keys)


@router.get("/saved/{model_key}")
async def get_saved_metrics(
    model_key: str,
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Get saved metrics from results directory."""
    metrics = metrics_service.load_saved_metrics(model_key)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No saved metrics found for model {model_key}",
        )
    return metrics


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    metric: str = Query("sharpe_ratio", description="Metric to rank by"),
    top_n: int = Query(10, ge=1, le=50, description="Number of entries"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """Return leaderboard built from the results database."""
    try:
        return metrics_service.get_leaderboard(metric=metric, top_n=top_n)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
