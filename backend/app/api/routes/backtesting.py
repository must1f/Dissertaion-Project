"""Backtesting API endpoints."""

from typing import Optional
import time

from fastapi import APIRouter, Depends, Query, HTTPException

from backend.app.dependencies import get_backtest_service
from backend.app.services.backtest_service import BacktestService
from backend.app.schemas.backtesting import (
    BacktestRequest,
    BacktestResponse,
    BacktestListResponse,
    TradeHistoryResponse,
)
from backend.app.core.exceptions import BacktestError

router = APIRouter()


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    backtest_service: BacktestService = Depends(get_backtest_service),
):
    """Run a backtest."""
    start_time = time.time()

    try:
        results = backtest_service.run_backtest(request)
        processing_time = (time.time() - start_time) * 1000

        # Generate result ID
        import uuid
        result_id = str(uuid.uuid4())[:8]

        return BacktestResponse(
            success=True,
            result_id=result_id,
            results=results,
            processing_time_ms=processing_time,
        )
    except BacktestError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=BacktestListResponse)
async def list_backtest_results(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    model_key: Optional[str] = Query(None, description="Filter by model"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    backtest_service: BacktestService = Depends(get_backtest_service),
):
    """List backtest results."""
    result = backtest_service.list_results(
        ticker=ticker,
        model_key=model_key,
        page=page,
        page_size=page_size,
    )
    return BacktestListResponse(**result)


@router.get("/results/{result_id}")
async def get_backtest_result(
    result_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service),
):
    """Get a specific backtest result."""
    result = backtest_service.get_results(result_id)

    if result is None:
        # Try loading from disk
        result = backtest_service.load_results(result_id)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    return result


@router.get("/trades/{result_id}", response_model=TradeHistoryResponse)
async def get_trade_history(
    result_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service),
):
    """Get trade history for a backtest."""
    trades = backtest_service.get_trades(result_id)

    if trades is None:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    return TradeHistoryResponse(**trades)


@router.post("/results/{result_id}/save")
async def save_backtest_result(
    result_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service),
):
    """Save backtest result to disk."""
    try:
        backtest_service.save_results(result_id)
        return {"message": f"Result {result_id} saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
