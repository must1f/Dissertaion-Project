"""Prediction API endpoints."""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException

from backend.app.dependencies import get_prediction_service
from backend.app.services.prediction_service import PredictionService
from backend.app.schemas.predictions import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionHistoryResponse,
    LatestPredictionResponse,
    SignalAction,
)
from backend.app.core.exceptions import PredictionError, ModelNotTrainedError

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def run_prediction(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Run a prediction for a single ticker."""
    try:
        return prediction_service.predict(request)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def run_batch_prediction(
    request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Run predictions for multiple tickers."""
    try:
        result = prediction_service.batch_predict(
            tickers=request.tickers,
            model_key=request.model_key,
            sequence_length=request.sequence_length,
            estimate_uncertainty=request.estimate_uncertainty,
        )
        return BatchPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    model_key: Optional[str] = Query(None, description="Filter by model"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Get prediction history."""
    result = prediction_service.get_prediction_history(
        ticker=ticker,
        model_key=model_key,
        page=page,
        page_size=page_size,
    )
    return PredictionHistoryResponse(**result)


@router.get("/{ticker}/latest", response_model=LatestPredictionResponse)
async def get_latest_predictions(
    ticker: str,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Get latest predictions for a ticker from all models."""
    predictions = prediction_service.get_latest_predictions(ticker.upper())

    # Determine consensus signal
    signals = [p.signal_action for p in predictions.values() if p.signal_action]
    consensus = None
    if signals:
        buy_count = sum(1 for s in signals if s == SignalAction.BUY)
        sell_count = sum(1 for s in signals if s == SignalAction.SELL)

        if buy_count > sell_count:
            consensus = SignalAction.BUY
        elif sell_count > buy_count:
            consensus = SignalAction.SELL
        else:
            consensus = SignalAction.HOLD

    return LatestPredictionResponse(
        ticker=ticker.upper(),
        predictions=predictions,
        consensus_signal=consensus,
        last_updated=datetime.now(),
    )


@router.delete("/cache")
async def clear_prediction_cache(
    model_key: Optional[str] = Query(None, description="Clear cache for specific model"),
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Clear prediction service cache."""
    prediction_service.clear_cache(model_key)
    return {"message": f"Cache cleared{f' for {model_key}' if model_key else ''}"}
