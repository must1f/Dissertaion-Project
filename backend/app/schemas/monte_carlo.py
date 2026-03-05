"""Pydantic schemas for Monte Carlo simulation endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo simulation. Only S&P 500 data is supported."""

    model_key: str = Field(..., description="Model to use for simulation")
    ticker: str = Field(
        default=DEFAULT_TICKER,
        description=f"Stock ticker (only {', '.join(ALLOWED_TICKERS)} supported)"
    )

    # Simulation parameters
    n_simulations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of simulation paths"
    )
    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Simulation horizon in days"
    )

    # Initial conditions
    initial_price: Optional[float] = Field(
        None,
        description="Starting price (uses latest if not provided)"
    )

    # Confidence levels for intervals
    confidence_levels: List[float] = Field(
        default=[0.50, 0.75, 0.90, 0.95, 0.99],
        description="Confidence levels for prediction intervals"
    )

    # Random seed for reproducibility
    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        """Validate that only allowed tickers are used for Monte Carlo simulation."""
        return validate_ticker(v)


class ConfidenceInterval(BaseModel):
    """Confidence interval at a specific level."""

    level: float
    lower: float
    upper: float


class SimulationPath(BaseModel):
    """Single simulation path (subset for visualization)."""

    path_id: int
    prices: List[float]
    final_price: float
    total_return: float
    max_drawdown: float


class DistributionStats(BaseModel):
    """Distribution statistics for a specific time point."""

    day: int
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    percentiles: Dict[str, float]  # "5", "25", "50", "75", "95"


class MonteCarloResults(BaseModel):
    """Complete Monte Carlo simulation results."""

    # Configuration
    model_key: str
    ticker: str
    n_simulations: int
    horizon_days: int
    initial_price: float
    run_date: datetime

    # Final distribution
    final_price_mean: float
    final_price_median: float
    final_price_std: float
    final_return_mean: float
    final_return_std: float

    # Confidence intervals at horizon
    confidence_intervals: List[ConfidenceInterval]

    # Distribution over time
    daily_distributions: List[DistributionStats]

    # Risk metrics
    probability_of_loss: float
    probability_of_gain: float
    value_at_risk_95: float  # 5th percentile loss
    expected_shortfall_95: float  # Average of worst 5%

    # Sample paths for visualization
    sample_paths: List[SimulationPath]

    # Histogram data for final distribution
    histogram_bins: List[float]
    histogram_counts: List[int]


class MonteCarloResponse(BaseModel):
    """Response for Monte Carlo simulation."""

    success: bool
    result_id: str
    results: MonteCarloResults
    processing_time_ms: float


class MonteCarloSummary(BaseModel):
    """Summary of Monte Carlo simulation."""

    result_id: str
    model_key: str
    ticker: str
    run_date: datetime
    n_simulations: int
    horizon_days: int
    expected_return: float
    value_at_risk_95: float


class MonteCarloListResponse(BaseModel):
    """Response listing Monte Carlo simulations."""

    results: List[MonteCarloSummary]
    total: int
