"""Advanced analysis API endpoints."""

import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, HTTPException

# Add project root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.analysis_service import AnalysisService
from backend.app.schemas.analysis import (
    # Regime
    RegimeAnalysisRequest,
    RegimeAnalysisResponse,
    RegimeHistoryResponse,
    CurrentRegimeState,
    RegimeMetrics,
    ReturnsSeries,
    CrisisAnalysisRequest,
    CrisisAnalysisResponse,
    # Exposure
    ExposureSnapshot,
    ExposureHistoryResponse,
    ExposureConfigRequest,
    # Benchmark
    BenchmarkComparisonRequest,
    BenchmarkComparisonResponse,
    # Rolling
    RollingMetricsRequest,
    RollingMetricsResponse,
    # Robustness
    RobustnessRequest,
    RobustnessResponse,
)

router = APIRouter()

# Service singleton
_analysis_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Get or create analysis service."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service


# =============================================================================
# Regime Analysis Endpoints
# =============================================================================

@router.get("/regime/current", response_model=CurrentRegimeState)
async def get_current_regime(
    ticker: str = Query(default="^GSPC", description="Stock ticker"),
    method: str = Query(default="rolling", description="Detection method"),
    lookback_days: int = Query(default=252, ge=50, le=1000),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get current market regime state."""
    try:
        return service.get_current_regime(
            ticker=ticker,
            method=method,
            lookback_days=lookback_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Stress Testing (returns-based)
# =============================================================================

@router.post("/stress/run", response_model=CrisisAnalysisResponse)
async def run_stress_tests(
    request: CrisisAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Run stress testing on a return series."""
    try:
        returns = np.array(request.returns)
        timestamps = pd.to_datetime(request.timestamps) if request.timestamps else None
        if timestamps is None or len(timestamps) != len(returns):
            raise HTTPException(status_code=400, detail="timestamps length must match returns")

        return service.run_stress_tests(returns, pd.DatetimeIndex(timestamps))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/history", response_model=RegimeHistoryResponse)
async def get_regime_history(
    ticker: str = Query(default="^GSPC"),
    method: str = Query(default="rolling"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get regime history with metrics."""
    try:
        history, regime_metrics = service.get_regime_history(
            ticker=ticker,
            method=method,
            start_date=start_date,
            end_date=end_date,
        )

        return RegimeHistoryResponse(
            ticker=ticker,
            start_date=history[0].timestamp if history else datetime.now(),
            end_date=history[-1].timestamp if history else datetime.now(),
            total_points=len(history),
            history=history,
            regime_summary=regime_metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/returns", response_model=ReturnsSeries)
async def get_returns_series(
    ticker: str = Query(default="^GSPC"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    max_points: int = Query(default=2000, ge=10, le=5000),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Fetch return series for a ticker for stress testing."""
    try:
        returns, timestamps = service.get_returns_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_points=max_points,
        )
        if len(returns) == 0:
            raise HTTPException(status_code=404, detail=f"No returns available for {ticker}")
        return ReturnsSeries(
            ticker=ticker,
            start_date=timestamps[0],
            end_date=timestamps[-1],
            timestamps=list(timestamps),
            returns=returns.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regime/analyze", response_model=RegimeAnalysisResponse)
async def analyze_regimes(
    request: RegimeAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Run full regime analysis."""
    try:
        current_state = service.get_current_regime(
            ticker=request.ticker,
            method=request.method,
            lookback_days=request.lookback_days,
        )

        history, regime_metrics = service.get_regime_history(
            ticker=request.ticker,
            method=request.method,
        )

        return RegimeAnalysisResponse(
            ticker=request.ticker,
            method=request.method,
            current_state=current_state,
            regime_metrics=regime_metrics,
            transition_matrix=None,  # Could be added from HMM
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Exposure Analysis Endpoints
# =============================================================================

@router.post("/exposure/calculate", response_model=ExposureSnapshot)
async def calculate_exposure(
    returns: List[float],
    config: ExposureConfigRequest = None,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Calculate current exposure recommendation."""
    if config is None:
        config = ExposureConfigRequest()

    try:
        returns_arr = np.array(returns)
        return service.get_current_exposure(
            returns=returns_arr,
            target_volatility=config.target_volatility,
            regime_aware=config.regime_aware,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exposure/history", response_model=ExposureHistoryResponse)
async def get_exposure_history(
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get exposure calculation history."""
    history = service.get_exposure_history()

    summary = {}
    if history:
        exposures = [h.target_exposure for h in history]
        summary = {
            'mean_exposure': float(np.mean(exposures)),
            'min_exposure': float(np.min(exposures)),
            'max_exposure': float(np.max(exposures)),
            'current_exposure': exposures[-1] if exposures else 1.0,
        }

    return ExposureHistoryResponse(
        total_points=len(history),
        history=history,
        summary=summary,
    )


# =============================================================================
# Benchmark Comparison Endpoints
# =============================================================================

@router.post("/benchmark/compare", response_model=BenchmarkComparisonResponse)
async def compare_with_benchmark(
    request: BenchmarkComparisonRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Compare strategy performance with benchmark."""
    try:
        # Parse dates
        timestamps = pd.DatetimeIndex([
            datetime.strptime(request.start_date, '%Y-%m-%d') + pd.Timedelta(days=i)
            for i in range(len(request.strategy_returns))
        ])

        result = service.compare_with_benchmark(
            strategy_returns=np.array(request.strategy_returns),
            timestamps=timestamps,
            benchmark_ticker=request.benchmark_ticker,
            initial_capital=request.initial_capital,
            include_regimes=request.include_regimes,
        )

        return BenchmarkComparisonResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/{backtest_id}/comparison")
async def get_backtest_benchmark_comparison(
    backtest_id: str,
    benchmark_ticker: str = Query(default="^GSPC"),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get benchmark comparison for a specific backtest."""
    # This would integrate with the backtest service to get stored results
    raise HTTPException(
        status_code=501,
        detail="Endpoint requires backtest integration - use POST /benchmark/compare"
    )


# =============================================================================
# Rolling Metrics Endpoints
# =============================================================================

@router.post("/rolling-metrics", response_model=RollingMetricsResponse)
async def calculate_rolling_metrics(
    request: RollingMetricsRequest,
    returns: List[float],
    timestamps: List[str],
    service: AnalysisService = Depends(get_analysis_service),
):
    """Calculate rolling metrics for a return series."""
    try:
        ts_index = pd.DatetimeIndex([
            datetime.strptime(t, '%Y-%m-%d') for t in timestamps
        ])

        data = service.calculate_rolling_metrics(
            returns=np.array(returns),
            timestamps=ts_index,
            window=request.window,
            metrics=request.metrics,
        )

        return RollingMetricsResponse(
            ticker=request.ticker,
            window=request.window,
            total_points=len(data),
            data=data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}/rolling-metrics")
async def get_backtest_rolling_metrics(
    backtest_id: str,
    window: int = Query(default=126, ge=20, le=252),
    metrics: str = Query(default="sharpe,volatility,sortino"),
):
    """Get rolling metrics for a specific backtest."""
    raise HTTPException(
        status_code=501,
        detail="Endpoint requires backtest integration - use POST /rolling-metrics"
    )


# =============================================================================
# Robustness Testing Endpoints
# =============================================================================

@router.post("/robustness/test", response_model=RobustnessResponse)
async def run_robustness_tests(
    request: RobustnessRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Run robustness testing suite."""
    try:
        timestamps = pd.DatetimeIndex([
            datetime.strptime(t, '%Y-%m-%d') for t in request.timestamps
        ])

        benchmark = np.array(request.benchmark_returns) if request.benchmark_returns else None

        result = service.run_robustness_tests(
            returns=np.array(request.returns),
            timestamps=timestamps,
            benchmark_returns=benchmark,
            tests=request.tests,
        )

        return RobustnessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}/robustness")
async def get_backtest_robustness(
    backtest_id: str,
    tests: Optional[str] = Query(None, description="Comma-separated test names"),
):
    """Get robustness analysis for a specific backtest."""
    raise HTTPException(
        status_code=501,
        detail="Endpoint requires backtest integration - use POST /robustness/test"
    )


# =============================================================================
# Crisis Analysis Endpoints
# =============================================================================

@router.post("/crisis/analyze", response_model=CrisisAnalysisResponse)
async def analyze_crisis_performance(
    request: CrisisAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Analyze strategy performance during crisis periods."""
    try:
        timestamps = pd.DatetimeIndex([
            datetime.strptime(t, '%Y-%m-%d') for t in request.timestamps
        ])

        benchmark = np.array(request.benchmark_returns) if request.benchmark_returns else None

        result = service.analyze_crisis_performance(
            returns=np.array(request.returns),
            timestamps=timestamps,
            benchmark_returns=benchmark,
        )

        return CrisisAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}/crisis-analysis")
async def get_backtest_crisis_analysis(
    backtest_id: str,
):
    """Get crisis analysis for a specific backtest."""
    raise HTTPException(
        status_code=501,
        detail="Endpoint requires backtest integration - use POST /crisis/analyze"
    )
