"""Schemas for spectral analysis and regime detection endpoints."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


# =============================================================================
# Spectral Analysis Schemas
# =============================================================================

class SpectralAnalysisRequest(BaseModel):
    """Request for spectral analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    window_size: int = Field(64, ge=16, le=256, description="FFT window size")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SpectralFeature(BaseModel):
    """Single spectral feature value with metadata."""
    name: str
    value: float
    description: Optional[str] = None


class SpectralSnapshot(BaseModel):
    """Spectral features for a single point in time."""
    date: str
    spectral_entropy: float = Field(..., description="Normalized spectral entropy [0-1]")
    dominant_frequency: float = Field(..., description="Dominant frequency (cycles/day)")
    dominant_period: float = Field(..., description="Dominant period (trading days)")
    power_low: float = Field(..., description="Low-frequency power (trends)")
    power_mid: float = Field(..., description="Mid-frequency power (cycles)")
    power_high: float = Field(..., description="High-frequency power (noise)")
    power_ratio: float = Field(..., description="Signal-to-noise ratio")
    autocorrelation_lag1: float = Field(..., description="Lag-1 autocorrelation")
    spectral_slope: float = Field(..., description="Spectral slope (log-log)")


class PowerSpectrumData(BaseModel):
    """Power spectrum data for visualization."""
    frequencies: List[float] = Field(..., description="Frequency bins (cycles/day)")
    power: List[float] = Field(..., description="Power at each frequency")
    frequency_bands: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Frequency band boundaries and total power"
    )


class SpectralAnalysisResponse(BaseModel):
    """Response for spectral analysis."""
    success: bool
    ticker: str
    analysis_date: datetime
    window_size: int
    current_features: SpectralSnapshot
    power_spectrum: PowerSpectrumData
    historical_features: Optional[List[SpectralSnapshot]] = None
    processing_time_ms: float


# =============================================================================
# Regime Detection Schemas
# =============================================================================

class RegimeDetectionRequest(BaseModel):
    """Request for regime detection."""
    ticker: str = Field(..., description="Stock ticker symbol")
    method: str = Field("spectral_hmm", description="Detection method: hmm, spectral_hmm, kmeans, rolling")
    n_regimes: int = Field(3, ge=2, le=5, description="Number of regimes")
    lookback_days: int = Field(504, ge=100, le=2520, description="Historical lookback period")


class RegimeCharacteristics(BaseModel):
    """Characteristics of a single regime."""
    regime_id: int
    regime_name: str
    mean_return_annual: float
    volatility_annual: float
    stationary_probability: float
    expected_duration_days: float
    spectral_entropy: Optional[float] = None
    dominant_frequency: Optional[float] = None
    power_ratio: Optional[float] = None
    sample_count: int


class TransitionMatrix(BaseModel):
    """Regime transition probability matrix."""
    matrix: List[List[float]]
    labels: List[str]


class RegimeHistoryPoint(BaseModel):
    """Regime classification at a point in time."""
    date: str
    regime: int
    regime_name: str
    probability: float
    all_probabilities: Dict[str, float]


class RegimeDetectionResponse(BaseModel):
    """Response for regime detection."""
    success: bool
    ticker: str
    method: str
    n_regimes: int
    current_regime: int
    current_regime_name: str
    current_probability: float
    regime_characteristics: List[RegimeCharacteristics]
    transition_matrix: TransitionMatrix
    recent_history: List[RegimeHistoryPoint]
    log_likelihood: Optional[float] = None
    bic: Optional[float] = None
    processing_time_ms: float


# =============================================================================
# Fan Chart Schemas
# =============================================================================

class FanChartRequest(BaseModel):
    """Request for Monte Carlo fan chart."""
    ticker: str = Field(..., description="Stock ticker symbol")
    initial_price: Optional[float] = Field(None, description="Starting price (uses current if None)")
    horizon_days: int = Field(252, ge=5, le=504, description="Forecast horizon in days")
    n_simulations: int = Field(1000, ge=100, le=10000, description="Number of Monte Carlo paths")
    percentiles: List[int] = Field(
        [5, 25, 50, 75, 95],
        description="Percentiles to compute"
    )
    use_regime_switching: bool = Field(True, description="Use regime-switching model")


class PercentileBand(BaseModel):
    """Percentile band data."""
    percentile: int
    values: List[float]


class RegimePeriod(BaseModel):
    """Contiguous regime period."""
    start: int
    end: int
    regime: int
    regime_name: str


class FanChartResponse(BaseModel):
    """Response for fan chart generation."""
    success: bool
    ticker: str
    initial_price: float
    horizon_days: int
    n_simulations: int
    dates: List[int]  # Day indices
    percentile_bands: List[PercentileBand]
    median_path: List[float]
    regime_periods: List[RegimePeriod]
    regime_probabilities: List[Dict[str, float]]
    expected_return: float
    value_at_risk_95: float
    probability_of_loss: float
    processing_time_ms: float


# =============================================================================
# Combined Analysis Schemas
# =============================================================================

class SpectralRegimeAnalysisRequest(BaseModel):
    """Request for combined spectral and regime analysis."""
    ticker: str
    include_spectral: bool = True
    include_regimes: bool = True
    include_fan_chart: bool = False
    spectral_window: int = 64
    n_regimes: int = 3
    lookback_days: int = 504


class SpectralRegimeAnalysisResponse(BaseModel):
    """Combined spectral and regime analysis response."""
    success: bool
    ticker: str
    analysis_date: datetime
    spectral_analysis: Optional[SpectralAnalysisResponse] = None
    regime_analysis: Optional[RegimeDetectionResponse] = None
    fan_chart: Optional[FanChartResponse] = None
    processing_time_ms: float
