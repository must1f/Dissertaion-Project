import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import analysis


class _StubAnalysisService:
    def get_returns_series(self, ticker: str, start_date=None, end_date=None, max_points=2000):
        returns = np.linspace(-0.01, 0.01, 20)
        timestamps = [f"2024-01-{i+1:02d}" for i in range(len(returns))]
        return returns, timestamps


def test_returns_endpoint_stub():
    client = TestClient(app)

    # override dependency to avoid external fetches
    app.dependency_overrides[analysis.get_analysis_service] = lambda: _StubAnalysisService()

    try:
        resp = client.get("/api/analysis/returns?ticker=TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "TEST"
        assert len(data["returns"]) == 20
        assert len(data["timestamps"]) == 20
    finally:
        app.dependency_overrides.clear()
