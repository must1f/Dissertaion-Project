import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.evaluation_harness import EvaluationHarness
from src.evaluation.split_manager import SplitConfig, SplitStrategy


def test_evaluation_harness_persists_window_results(tmp_path: Path):
    """Ensure window-level metrics are written to the dedicated SQLite DB."""
    split_cfg = SplitConfig(
        strategy=SplitStrategy.ROLLING,
        min_train_size=10,
        min_test_size=5,
        n_folds=3,
        embargo_size=0,
    )
    harness = EvaluationHarness(
        output_dir=tmp_path,
        save_predictions=False,
        split_config=split_cfg,
    )

    n = 60
    rng = np.random.default_rng(123)
    preds = rng.normal(0, 1, size=n)
    targs = preds + rng.normal(0, 0.05, size=n)
    returns = rng.normal(0.001, 0.01, size=n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")

    harness.evaluate(
        predictions=preds,
        targets=targs,
        model_key="dummy",
        persist_windows=True,
        returns=returns,
        timestamps=timestamps,
    )

    wf_db = tmp_path / "walk_forward_results.db"
    assert wf_db.exists()

    conn = sqlite3.connect(wf_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM window_results")
    count = cur.fetchone()[0]
    conn.close()

    assert count > 0
