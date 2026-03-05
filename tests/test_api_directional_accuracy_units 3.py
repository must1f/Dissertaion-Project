import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_metrics_ml_directional_accuracy_returns_percentage():
    """
    Regression test: API should return directional_accuracy in percent (0-100),
    even though MetricsCalculator works in 0-1 units.
    """
    # Construct simple series with 50% directional match
    y_true = "0,1,0"          # diffs: [+1, -1]
    y_pred = "0,1,1"          # diffs: [+1,  0] -> one of two matches

    resp = client.get("/api/metrics/ml", params={"y_true": y_true, "y_pred": y_pred})
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "directional_accuracy" in data

    da = data["directional_accuracy"]
    # Expect percentage (50.0), not fraction (0.5)
    assert 49.9 <= da <= 50.1, f"Expected ~50%, got {da}"
