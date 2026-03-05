"""Data API endpoints."""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException

from backend.app.dependencies import get_data_service
from backend.app.services.data_service import DataService
from backend.app.schemas.data import (
    StockListResponse,
    StockDataResponse,
    FeaturesResponse,
    FetchDataRequest,
    FetchDataResponse,
)
from backend.app.core.exceptions import DataNotFoundError, DataFetchError
from backend.app.core.config import ALLOWED_TICKERS, validate_ticker

router = APIRouter()


@router.get("/stocks", response_model=StockListResponse)
async def list_stocks(
    data_service: DataService = Depends(get_data_service),
):
    """List all available stocks. Only S&P 500 data is supported."""
    # Filter to only return allowed tickers
    all_stocks = data_service.get_available_stocks()
    allowed_stocks = [s for s in all_stocks if s.ticker in ALLOWED_TICKERS]
    return StockListResponse(
        stocks=allowed_stocks,
        total=len(allowed_stocks),
    )


@router.get("/stocks/{ticker}", response_model=StockDataResponse)
async def get_stock_data(
    ticker: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval: '1d' (daily) or '1mo' (monthly)"),
    data_service: DataService = Depends(get_data_service),
):
    """Get stock OHLCV data for a ticker. Only S&P 500 data is supported."""
    try:
        validated_ticker = validate_ticker(ticker)
        return data_service.get_stock_data(
            ticker=validated_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stocks/{ticker}/features", response_model=FeaturesResponse)
async def get_stock_features(
    ticker: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval: '1d' (daily) or '1mo' (monthly)"),
    data_service: DataService = Depends(get_data_service),
):
    """Get engineered features for a stock. Only S&P 500 data is supported."""
    try:
        validated_ticker = validate_ticker(ticker)
        return data_service.get_stock_features(
            ticker=validated_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/fetch", response_model=FetchDataResponse)
async def fetch_data(
    request: FetchDataRequest,
    data_service: DataService = Depends(get_data_service),
):
    """Fetch new stock data from external sources."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Fetch request: tickers={request.tickers}, start={request.start_date}, end={request.end_date}")

    try:
        df = data_service.fetch_data(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            force_refresh=request.force_refresh,
        )

        record_count = len(df) if df is not None else 0
        logger.info(f"Fetch result: {record_count} records for {request.tickers}")

        if record_count == 0:
            return FetchDataResponse(
                success=False,
                tickers_fetched=request.tickers,
                records_added=0,
                message=f"No data available from yfinance for {request.tickers} in the specified date range",
                errors={"no_data": "yfinance returned no data for this date range"},
            )

        return FetchDataResponse(
            success=True,
            tickers_fetched=request.tickers,
            records_added=record_count,
            message=f"Successfully fetched {record_count} records for {len(request.tickers)} tickers",
        )
    except DataFetchError as e:
        logger.error(f"DataFetchError: {e}")
        return FetchDataResponse(
            success=False,
            tickers_fetched=[],
            records_added=0,
            message=str(e),
            errors={"fetch_error": str(e)},
        )
    except Exception as e:
        logger.error(f"Unexpected error in fetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fetch-latest/{ticker}")
async def fetch_latest_data(
    ticker: str,
    years: int = 10,
    data_service: DataService = Depends(get_data_service),
):
    """
    Fetch latest data for a ticker with incremental updates.

    - Fetches up to 10 years of historical daily data
    - Only downloads data we don't already have
    - Never deletes existing data
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        validated_ticker = validate_ticker(ticker)
        logger.info(f"Fetch latest request: ticker={validated_ticker}, years={years}")

        result = data_service.fetch_latest(validated_ticker, years=years)

        return {
            "success": result["records_fetched"] > 0 or "up to date" in result["message"].lower(),
            "ticker": result["ticker"],
            "records_fetched": result["records_fetched"],
            "message": result["message"],
            "date_range": result["date_range"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataFetchError as e:
        logger.error(f"DataFetchError: {e}")
        return {
            "success": False,
            "ticker": ticker,
            "records_fetched": 0,
            "message": str(e),
            "date_range": None,
        }
    except Exception as e:
        logger.error(f"Unexpected error in fetch_latest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache(
    ticker: Optional[str] = Query(None, description="Clear cache for specific ticker"),
    data_service: DataService = Depends(get_data_service),
):
    """Clear data cache."""
    data_service.clear_cache(ticker)
    return {"message": f"Cache cleared{f' for {ticker}' if ticker else ''}"}
