"""API tests for data endpoints."""

from datetime import datetime

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_data_service
from backend.app.schemas.data import (
    StockInfo,
    StockDataResponse,
    OHLCVData,
    FeatureData,
    FeaturesResponse,
)
from backend.app.core.exceptions import DataFetchError
from backend.app.core.config import ALLOWED_TICKERS


client = TestClient(app)


class _DummyDataService:
    """Minimal data service stub for API tests."""

    def __init__(self):
        self.cache_cleared = None

    def get_available_stocks(self):
        return [
            StockInfo(ticker=ALLOWED_TICKERS[0], record_count=10),
            StockInfo(ticker="AAPL", record_count=5),
        ]

    def get_stock_data(self, ticker, start_date=None, end_date=None, interval="1d"):
        ts = datetime(2024, 1, 1)
        return StockDataResponse(
            ticker=ticker,
            data=[
                OHLCVData(
                    timestamp=ts,
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    volume=100,
                    ticker=ticker,
                )
            ],
            start_date=ts,
            end_date=ts,
            count=1,
        )

    def get_stock_features(self, ticker, start_date=None, end_date=None, interval="1d"):
        ts = datetime(2024, 1, 1)
        feature = FeatureData(timestamp=ts, ticker=ticker, log_return=0.1, macd=0.2)
        return FeaturesResponse(
            ticker=ticker,
            features=[feature],
            feature_names=["log_return", "macd"],
            count=1,
        )

    def fetch_data(self, *args, **kwargs):
        raise DataFetchError("boom")

    def clear_cache(self, ticker=None):
        self.cache_cleared = ticker


def override_service():
    return _DummyDataService()


# Ensure overrides cleaned up between tests

def _apply_override(service_factory=override_service):
    app.dependency_overrides[get_data_service] = service_factory


def _clear_override():
    app.dependency_overrides.pop(get_data_service, None)



def test_list_stocks_filters_to_allowed_only():
    _apply_override()
    resp = client.get("/api/data/stocks")
    _clear_override()

    assert resp.status_code == 200
    tickers = [stock["ticker"] for stock in resp.json()["stocks"]]
    assert tickers == ALLOWED_TICKERS


def test_get_stock_data_rejects_disallowed_ticker():
    resp = client.get("/api/data/stocks/AAPL")
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"].lower()


def test_get_stock_data_returns_payload_with_cacheable_fields():
    _apply_override()
    resp = client.get("/api/data/stocks/%5EGSPC")
    _clear_override()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ticker"] == ALLOWED_TICKERS[0]
    assert payload["count"] == 1
    assert payload["data"][0]["close"] == 1.5


def test_get_stock_features_serializes_feature_names():
    _apply_override()
    resp = client.get("/api/data/stocks/%5EGSPC/features")
    _clear_override()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["feature_names"] == ["log_return", "macd"]
    assert payload["features"][0]["log_return"] == 0.1


def test_fetch_data_returns_graceful_error_when_fetch_fails():
    _apply_override()
    resp = client.post(
        "/api/data/fetch",
        json={"tickers": [ALLOWED_TICKERS[0]]},
    )
    _clear_override()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is False
    assert payload["records_added"] == 0
    assert "boom" in payload["message"]


def test_clear_cache_for_specific_ticker():
    service = _DummyDataService()
    _apply_override(lambda: service)
    resp = client.delete("/api/data/cache", params={"ticker": "^GSPC"})
    _clear_override()

    assert resp.status_code == 200
    assert service.cache_cleared == "^GSPC"
    assert "^GSPC" in resp.json()["message"]
