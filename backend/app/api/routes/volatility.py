"""
Volatility Forecasting API Endpoints

API routes for volatility model training, prediction, backtesting,
and model comparison.
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.volatility_service import (
    VolatilityService,
    get_volatility_service,
    VOLATILITY_MODELS,
)

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================

class VolatilityDataRequest(BaseModel):
    """Request for preparing volatility data."""
    ticker: str = Field(default="SPY", description="Stock ticker")
    start_date: str = Field(default="2015-01-01", description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date")
    horizon: int = Field(default=5, ge=1, le=60, description="Forecast horizon in days")
    seq_length: int = Field(default=40, ge=10, le=100, description="Input sequence length")


class VolatilityTrainingRequest(BaseModel):
    """Request for training a volatility model."""
    model_type: str = Field(..., description="Model type key")
    ticker: str = Field(default="SPY", description="Stock ticker")
    epochs: int = Field(default=100, ge=1, le=500, description="Number of training epochs")
    batch_size: int = Field(default=64, ge=16, le=256, description="Batch size")
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1, description="Learning rate")
    hidden_dim: int = Field(default=128, ge=32, le=512, description="Hidden dimension")
    num_layers: int = Field(default=2, ge=1, le=4, description="Number of layers")
    enable_physics: bool = Field(default=True, description="Enable physics losses for PINN models")

    # Baseline model specific
    lookback: int = Field(default=20, ge=5, le=100, description="Lookback for rolling models")
    decay: float = Field(default=0.94, gt=0, lt=1, description="Decay for EWMA")


class VolatilityPredictionRequest(BaseModel):
    """Request for volatility predictions."""
    model_type: str = Field(..., description="Model type key")
    n_steps: int = Field(default=1, ge=1, le=60, description="Number of steps to predict")


class StrategyBacktestRequest(BaseModel):
    """Request for strategy backtesting."""
    model_type: str = Field(..., description="Model type for volatility predictions")
    target_vol: float = Field(default=0.15, gt=0, le=0.5, description="Target annual volatility")
    min_weight: float = Field(default=0.25, ge=0, le=1, description="Minimum position weight")
    max_weight: float = Field(default=2.0, ge=1, le=5, description="Maximum leverage")
    transaction_cost: float = Field(default=0.001, ge=0, le=0.01, description="Transaction cost")


class ModelComparisonRequest(BaseModel):
    """Request for comparing models."""
    model_types: List[str] = Field(..., min_items=1, description="Models to compare")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/models")
async def get_available_models(
    service: VolatilityService = Depends(get_volatility_service),
):
    """Get all available volatility forecasting models."""
    return service.get_available_models()


@router.get("/models/{model_type}")
async def get_model_info(
    model_type: str,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Get information about a specific model."""
    if model_type not in VOLATILITY_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    return {
        "model_type": model_type,
        **VOLATILITY_MODELS[model_type],
    }


@router.post("/data/prepare")
async def prepare_data(
    request: VolatilityDataRequest,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Prepare data for volatility forecasting."""
    result = service.prepare_data(
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        seq_length=request.seq_length,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/train")
async def train_model(
    request: VolatilityTrainingRequest,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Train a volatility forecasting model."""
    if request.model_type not in VOLATILITY_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")

    result = service.train_model(
        model_type=request.model_type,
        ticker=request.ticker,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        hidden_dim=request.hidden_dim,
        num_layers=request.num_layers,
        enable_physics=request.enable_physics,
        lookback=request.lookback,
        decay=request.decay,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/predict")
async def predict(
    request: VolatilityPredictionRequest,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Make volatility predictions."""
    result = service.predict(
        model_type=request.model_type,
        n_steps=request.n_steps,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/backtest")
async def backtest_strategy(
    request: StrategyBacktestRequest,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Backtest volatility targeting strategy."""
    result = service.backtest_strategy(
        model_type=request.model_type,
        target_vol=request.target_vol,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        transaction_cost=request.transaction_cost,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/compare")
async def compare_models(
    request: ModelComparisonRequest,
    service: VolatilityService = Depends(get_volatility_service),
):
    """Compare multiple volatility forecasting models."""
    # Validate model types
    invalid = [m for m in request.model_types if m not in VOLATILITY_MODELS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown model types: {invalid}")

    result = service.compare_models(model_types=request.model_types)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/metrics")
async def get_metrics_info():
    """Get information about available volatility metrics."""
    return {
        "statistical_metrics": [
            {"name": "MSE", "description": "Mean Squared Error on variance"},
            {"name": "MAE", "description": "Mean Absolute Error on variance"},
            {"name": "RMSE", "description": "Root Mean Squared Error"},
            {"name": "QLIKE", "description": "Quasi-Likelihood loss (preferred for variance)"},
            {"name": "HMSE", "description": "Heteroskedasticity-adjusted MSE"},
            {"name": "R2", "description": "Mincer-Zarnowitz R² (forecast efficiency)"},
            {"name": "Log-Likelihood", "description": "Gaussian log-likelihood"},
        ],
        "economic_metrics": [
            {"name": "VaR Breach Rate", "description": "Value-at-Risk breach rate"},
            {"name": "Expected Shortfall", "description": "Conditional VaR accuracy"},
            {"name": "Sharpe Ratio", "description": "Strategy risk-adjusted return"},
            {"name": "Volatility Tracking Error", "description": "Deviation from target volatility"},
        ],
        "diagnostic_tests": [
            {"name": "Diebold-Mariano", "description": "Test for equal predictive accuracy"},
            {"name": "MCS", "description": "Model Confidence Set"},
            {"name": "Ljung-Box", "description": "Test for forecast error autocorrelation"},
        ],
    }


@router.get("/baselines")
async def get_baseline_models():
    """Get information about baseline volatility models."""
    return {
        "baselines": {
            "rolling": {
                "name": "Naive Rolling Volatility",
                "description": "Simple rolling window mean of squared returns",
                "formula": "σ̂² = (1/K) Σ r²_{t-i}",
            },
            "ewma": {
                "name": "EWMA (RiskMetrics)",
                "description": "Exponentially weighted moving average",
                "formula": "σ̂² = λσ̂²_{t-1} + (1-λ)r²_{t-1}",
                "default_lambda": 0.94,
            },
            "garch": {
                "name": "GARCH(1,1)",
                "description": "Generalized Autoregressive Conditional Heteroskedasticity",
                "formula": "σ² = ω + αr²_{t-1} + βσ²_{t-1}",
            },
            "gjr_garch": {
                "name": "GJR-GARCH",
                "description": "Asymmetric GARCH for leverage effect",
                "formula": "σ² = ω + (α + γI_{r<0})r²_{t-1} + βσ²_{t-1}",
            },
        },
        "reference": "Bollerslev (1986), Glosten et al. (1993)",
    }


@router.get("/physics-constraints")
async def get_physics_constraints():
    """Get information about physics constraints for volatility PINNs."""
    return {
        "constraints": {
            "variance_ou": {
                "name": "Variance Mean-Reversion (OU)",
                "equation": "dσ² = θ(σ̄² - σ²)dt",
                "description": "Variance reverts to long-run mean",
                "learnable_params": ["θ (mean reversion speed)"],
            },
            "garch_consistency": {
                "name": "GARCH Consistency",
                "equation": "σ²_t = ω + αr²_{t-1} + βσ²_{t-1}",
                "description": "Variance follows GARCH dynamics",
                "learnable_params": ["ω", "α", "β"],
            },
            "feller_condition": {
                "name": "Feller Condition",
                "equation": "2κθ > ξ²",
                "description": "Ensures variance stays positive (CIR/Heston)",
            },
            "leverage_effect": {
                "name": "Leverage Effect",
                "equation": "Corr(r_t, σ²_{t+1}) < 0",
                "description": "Negative returns increase future volatility",
            },
            "heston_drift": {
                "name": "Heston Variance Drift",
                "equation": "dV = κ(θ - V)dt + ξ√V dW",
                "description": "Heston stochastic volatility model",
                "learnable_params": ["κ", "θ", "ξ", "ρ"],
            },
        },
        "reference": "Heston (1993), Bollerslev (1986)",
    }
