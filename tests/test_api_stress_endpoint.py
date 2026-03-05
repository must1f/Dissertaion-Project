import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import analysis
from backend.app.schemas.analysis import CrisisPerformance, CrisisAnalysisResponse


class _StubAnalysisService:
    def run_stress_tests(self, returns, timestamps):
        # Return a minimal but valid response object
        crises = [
            CrisisPerformance(
                crisis_name="Stub Crisis",
                start_date=timestamps[0],
                end_date=timestamps[-1],
                duration_days=len(timestamps),
                strategy_return=float(np.sum(returns)),
                benchmark_return=0.0,
                alpha=float(np.sum(returns)),
                max_drawdown=-0.05,
                sharpe_ratio=0.5,
                days_to_recovery=10,
            )
        ]
        return CrisisAnalysisResponse(
            crises_analyzed=1,
            crises_outperformed=1,
            avg_alpha=float(np.sum(returns)),
            avg_crisis_return=float(np.mean(returns)),
            avg_max_drawdown=-0.05,
            worst_crisis="Stub Crisis",
            best_crisis="Stub Crisis",
            crisis_results=crises,
        )

    def get_returns_series(self, *args, **kwargs):
        returns = np.linspace(-0.01, 0.01, 20)
        timestamps = [f"2024-01-{i+1:02d}" for i in range(len(returns))]
        return returns, timestamps


def test_stress_run_endpoint_stub():
    client = TestClient(app)
    app.dependency_overrides[analysis.get_analysis_service] = lambda: _StubAnalysisService()

    try:
        payload = {
            "returns": [-0.01, 0.0, 0.01],
            "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
        resp = client.post("/api/analysis/stress/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["crises_analyzed"] == 1
        assert len(data["crisis_results"]) == 1
    finally:
        app.dependency_overrides.clear()
