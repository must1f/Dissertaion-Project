import json
from pathlib import Path

import pandas as pd

from src.evaluation.dissertation_pipeline import (
    DissertationPipelineRunner,
    load_dissertation_config,
    validate_dissertation_config,
)


def test_dissertation_config_contains_25_unique_models():
    cfg = load_dissertation_config(Path("configs/experiments/dissertation.yaml"))
    issues = validate_dissertation_config(cfg)

    assert issues == []
    assert len(cfg.unique_models) == 25
    assert "adaptive_dual_phase_pinn" in cfg.unique_models


def test_dissertation_config_rejects_duplicate_models():
    cfg = load_dissertation_config(Path("configs/experiments/dissertation.yaml"))
    cfg.groups["baseline_forecasting_models"].append("lstm")

    issues = validate_dissertation_config(cfg)
    assert any("Duplicate models" in issue for issue in issues)


def test_runner_audits_contract_flags_and_aliases(tmp_path: Path):
    cfg = load_dissertation_config(Path("configs/experiments/dissertation.yaml"))
    results_dir = tmp_path / "results"
    out_root = tmp_path / "out"
    db_path = tmp_path / "experiments.db"
    results_dir.mkdir(parents=True, exist_ok=True)

    # baseline_pinn should resolve via alias `pinn_baseline_results.json`
    (results_dir / "pinn_baseline_results.json").write_text(
        json.dumps(
            {
                "model": "pinn_baseline",
                "test_metrics": {
                    "rmse": 1.0,
                    "mae": 0.8,
                    "mape": 5.0,
                    "mse": 1.0,
                    "r2": 0.1,
                    "directional_accuracy": 0.51,
                    "sharpe_ratio": 0.3,
                    "sortino_ratio": 0.4,
                    "max_drawdown": -0.2,
                    "calmar_ratio": 0.2,
                    "total_return": 0.05,
                    "volatility": 0.2,
                    "win_rate": 0.53,
                    "annualized_return": 0.06,
                },
                "history": {
                    "train_loss": [1.0, 0.9, 0.8],
                    "val_loss": [1.1, 1.0, 0.95],
                },
            }
        )
    )

    cfg.artifacts["results_dir"] = str(results_dir)
    cfg.artifacts["output_root"] = str(out_root)
    cfg.artifacts["db_path"] = str(db_path)

    runner = DissertationPipelineRunner(cfg, config_path=Path("configs/experiments/dissertation.yaml"))
    summary = runner.run(persist_db=False)

    assert summary["models_with_results"] == 1
    assert summary["missing_models"] == 24

    audit_path = Path(summary["output_dir"]) / "model_audit.csv"
    assert audit_path.exists()
    audit_df = pd.read_csv(audit_path)
    baseline_row = audit_df.loc[audit_df["model_key"] == "baseline_pinn"].iloc[0]

    assert baseline_row["has_results"] == True  # noqa: E712
    # Missing scaler/lag/raw metrics should fail strict contract
    assert baseline_row["strict_contract_pass"] == False  # noqa: E712
