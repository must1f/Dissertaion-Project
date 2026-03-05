"""Pydantic schemas for advanced analysis endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================================
# Regime Analysis Schemas
# ============================================================================

class RegimeLabel(str, Enum):
    """Market regime labels."""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"


class RegimeMetrics(BaseModel):
    """Performance metrics for a specific regime."""
    regime: str
    count: int
    proportion: float
    mean_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    avg_duration: Optional[float] = None


class CurrentRegimeState(BaseModel):
    """Current regime state."""
    regime: RegimeLabel
    probability: float
    regime_probabilities: Dict[str, float]
    volatility: float
    transition_probability: float
    timestamp: datetime = Field(default_factory=datetime.now)


class RegimeHistoryPoint(BaseModel):
    """Single point in regime history."""
    timestamp: datetime
    regime: RegimeLabel
    probability: float
    volatility: float
    stress_window: Optional[str] = None


class RegimeHistoryResponse(BaseModel):
    """Response with regime history."""
    ticker: str
    start_date: datetime
    end_date: datetime
    total_points: int
    history: List[RegimeHistoryPoint]
    regime_summary: Dict[str, RegimeMetrics]


class RegimeAnalysisRequest(BaseModel):
    """Request for regime analysis."""
    ticker: str = Field(default="^GSPC", description="Stock ticker")
    method: str = Field(default="rolling", description="Detection method: hmm, kmeans, rolling")
    lookback_days: int = Field(default=252, description="Days of history to analyze")


class RegimeAnalysisResponse(BaseModel):
    """Response from regime analysis."""
    ticker: str
    method: str
    current_state: CurrentRegimeState
    regime_metrics: Dict[str, RegimeMetrics]
    transition_matrix: Optional[List[List[float]]] = None


# ============================================================================
# Exposure Analysis Schemas
# ============================================================================

class ExposureSnapshot(BaseModel):
    """Current exposure state."""
    target_exposure: float
    gross_exposure: float
    net_exposure: float
    leverage_ratio: float
    volatility_scalar: float
    regime_scalar: float
    regime: str
    realized_volatility: float
    target_volatility: float
    turnover_cost: float
    position_weights: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)


class ExposureHistoryPoint(BaseModel):
    """Single point in exposure history."""
    timestamp: datetime
    target_exposure: float
    gross_exposure: float
    volatility_scalar: float
    regime: str


class ExposureHistoryResponse(BaseModel):
    """Response with exposure history."""
    total_points: int
    history: List[ExposureHistoryPoint]
    summary: Dict[str, float]


class ExposureConfigRequest(BaseModel):
    """Request to configure exposure management."""
    target_volatility: float = Field(default=0.15, ge=0.01, le=0.5)
    max_leverage: float = Field(default=2.0, ge=1.0, le=4.0)
    min_leverage: float = Field(default=0.1, ge=0.0, le=1.0)
    regime_aware: bool = Field(default=True)


# ============================================================================
# Benchmark Comparison Schemas
# ============================================================================

class BenchmarkPoint(BaseModel):
    """Single point in benchmark comparison."""
    timestamp: datetime
    strategy_value: float
    benchmark_value: float
    strategy_return: float
    benchmark_return: float
    alpha: float
    regime: Optional[str] = None


class BenchmarkComparisonRequest(BaseModel):
    """Request for benchmark comparison."""
    strategy_returns: List[float] = Field(description="Strategy daily returns")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    benchmark_ticker: str = Field(default="^GSPC", description="Benchmark ticker")
    initial_capital: float = Field(default=100000.0)
    include_regimes: bool = Field(default=True)


class BenchmarkComparisonResponse(BaseModel):
    """Response from benchmark comparison."""
    strategy_final_value: float
    benchmark_final_value: float
    strategy_total_return: float
    benchmark_total_return: float
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_dd: float
    benchmark_max_dd: float
    comparison_data: List[BenchmarkPoint]


# ============================================================================
# Rolling Metrics Schemas
# ============================================================================

class RollingMetricsPoint(BaseModel):
    """Single point in rolling metrics."""
    timestamp: datetime
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    volatility: Optional[float] = None
    return_value: Optional[float] = None
    max_drawdown: Optional[float] = None


class RollingMetricsRequest(BaseModel):
    """Request for rolling metrics."""
    ticker: str = Field(default="^GSPC")
    window: int = Field(default=126, description="Rolling window in days")
    metrics: List[str] = Field(
        default=["sharpe", "volatility", "sortino"],
        description="Metrics to calculate"
    )


class RollingMetricsResponse(BaseModel):
    """Response with rolling metrics."""
    ticker: str
    window: int
    total_points: int
    data: List[RollingMetricsPoint]


# ============================================================================
# Robustness Testing Schemas
# ============================================================================

class RobustnessTestResult(BaseModel):
    """Result from a single robustness test."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]


class RobustnessRequest(BaseModel):
    """Request for robustness testing."""
    returns: List[float] = Field(description="Strategy daily returns")
    timestamps: List[str] = Field(description="Dates as YYYY-MM-DD strings")
    benchmark_returns: Optional[List[float]] = None
    tests: Optional[List[str]] = Field(
        default=None,
        description="Specific tests to run (default: all)"
    )


class RobustnessResponse(BaseModel):
    """Response from robustness testing."""
    overall_score: float
    tests_passed: int
    tests_failed: int
    is_robust: bool
    test_results: List[RobustnessTestResult]
    recommendations: List[str]


# ============================================================================
# Crisis Analysis Schemas
# ============================================================================

class CrisisPerformance(BaseModel):
    """Performance during a crisis period."""
    crisis_name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    strategy_return: float
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    max_drawdown: float
    sharpe_ratio: float
    days_to_recovery: Optional[int] = None


class CrisisAnalysisRequest(BaseModel):
    """Request for crisis analysis."""
    returns: List[float]
    timestamps: List[str]
    benchmark_returns: Optional[List[float]] = None


class CrisisAnalysisResponse(BaseModel):
    """Response from crisis analysis."""
    crises_analyzed: int
    crises_outperformed: int
    avg_alpha: Optional[float] = None
    avg_crisis_return: float
    avg_max_drawdown: float
    worst_crisis: str
    best_crisis: str
    crisis_results: List[CrisisPerformance]


class ReturnsSeries(BaseModel):
    """Time series of returns for a ticker."""
    ticker: str
    start_date: datetime
    end_date: datetime
    timestamps: List[datetime]
    returns: List[float]
