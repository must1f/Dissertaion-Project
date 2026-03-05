"""Pydantic schemas for metrics endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class MLMetrics(BaseModel):
    """Machine learning prediction metrics."""

    rmse: float = Field(..., description="Root Mean Square Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")
    directional_accuracy: float = Field(
        ...,
        description="Percentage of correct direction predictions"
    )

    # Additional metrics
    mse: Optional[float] = None
    explained_variance: Optional[float] = None


class FinancialMetrics(BaseModel):
    """Financial performance metrics."""

    # Return metrics
    total_return: float = Field(..., description="Total cumulative return")
    annual_return: float = Field(..., description="Annualized return")
    daily_return_mean: float = Field(..., description="Mean daily return")
    daily_return_std: float = Field(..., description="Daily return volatility")

    # Risk-adjusted metrics
    sharpe_ratio: float = Field(..., description="Sharpe ratio (risk-free rate = 2%)")
    sortino_ratio: float = Field(..., description="Sortino ratio (downside risk only)")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    information_ratio: Optional[float] = Field(None, description="Information ratio")

    # Drawdown metrics
    max_drawdown: float = Field(..., description="Maximum drawdown")
    max_drawdown_duration: Optional[int] = Field(
        None,
        description="Max drawdown duration in days"
    )

    # Win/loss metrics
    win_rate: float = Field(..., description="Percentage of winning trades")
    profit_factor: Optional[float] = Field(None, description="Gross profit / Gross loss")
    avg_win: Optional[float] = Field(None, description="Average winning trade return")
    avg_loss: Optional[float] = Field(None, description="Average losing trade return")

    # Trade metrics
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None


class PhysicsMetrics(BaseModel):
    """Physics constraint metrics for PINN models."""

    total_physics_loss: float
    gbm_loss: Optional[float] = None
    ou_loss: Optional[float] = None
    black_scholes_loss: Optional[float] = None
    langevin_loss: Optional[float] = None

    # Learned parameters
    theta: Optional[float] = Field(None, description="OU mean reversion speed")
    gamma: Optional[float] = Field(None, description="Langevin friction coefficient")
    temperature: Optional[float] = Field(None, description="Langevin temperature")
    mu: Optional[float] = Field(None, description="GBM drift")
    sigma: Optional[float] = Field(None, description="Volatility parameter")


class ModelMetricsResponse(BaseModel):
    """Complete metrics for a single model."""

    model_key: str
    model_name: str
    is_pinn: bool

    ml_metrics: MLMetrics
    financial_metrics: Optional[FinancialMetrics] = None
    physics_metrics: Optional[PhysicsMetrics] = None

    # Evaluation info
    evaluation_date: Optional[datetime] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None
    sample_size: Optional[int] = None


class MetricsComparisonResponse(BaseModel):
    """Comparison of metrics across models."""

    models: List[ModelMetricsResponse]
    metric_summary: Dict[str, Dict[str, float]]  # metric_name -> {model_key: value}
    best_by_metric: Dict[str, str]  # metric_name -> best model_key
    rankings: Dict[str, List[str]]  # metric_name -> [ranked model keys]


class FinancialMetricsRequest(BaseModel):
    """Request for calculating financial metrics."""

    returns: List[float] = Field(..., min_length=10)
    risk_free_rate: float = Field(default=0.02)
    periods_per_year: int = Field(default=252)
    benchmark_returns: Optional[List[float]] = None


class FinancialMetricsResponse(BaseModel):
    """Response with calculated financial metrics."""

    metrics: FinancialMetrics
    input_summary: Dict[str, Any]


# Leaderboard -----------------------------------------------------------------

class LeaderboardEntry(BaseModel):
    rank: int
    experiment_id: str
    model_name: str
    metric_value: float
    metric_name: str
    other_metrics: Dict[str, float]


class LeaderboardResponse(BaseModel):
    metric: str
    generated_at: datetime
    n_experiments: int
    entries: List[LeaderboardEntry]
