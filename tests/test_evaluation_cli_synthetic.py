"""
Smoke test for evaluation CLI: synthetic CSV + tiny checkpoint.

We avoid heavyweight training by creating a toy linear model checkpoint and a matching CSV.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.__main__ import main as eval_main


class TinyLinear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


def _create_tiny_checkpoint(tmpdir: Path, input_dim: int = 3):
    model = TinyLinear(input_dim)
    # deterministic weights
    with torch.no_grad():
        model.fc.weight.fill_(0.1)
        model.fc.bias.fill_(0.0)

    path = tmpdir / "tiny.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def _create_csv(tmpdir: Path, n: int = 20, input_dim: int = 3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, input_dim))
    y = X.sum(axis=1) * 0.1  # matches the weight above
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(input_dim)])
    df["target"] = y
    csv_path = tmpdir / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df.columns[:-1]


def test_cli_synthetic(tmp_path, monkeypatch, capsys):
    ckpt = _create_tiny_checkpoint(tmp_path)
    csv_path, feature_cols = _create_csv(tmp_path)
    # create a dummy config file to satisfy CLI
    dummy_cfg = tmp_path / "dummy.yaml"
    dummy_cfg.write_text("eval: {}\n")

    args = [
        "--config",
        str(dummy_cfg),  # minimal
        "--checkpoint",
        str(ckpt),
        "--model-type",
        "lstm",  # model registry will try to build, but we bypass by monkeypatching
        "--input-csv",
        str(csv_path),
        "--target-col",
        "target",
        "--feature-cols",
        ",".join(feature_cols),
        "--no-persist-windows",
        "--output-dir",
        str(tmp_path / "out"),
    ]

    # Monkeypatch ModelRegistry.create_model to return our tiny model
    from src.evaluation import __main__ as eval_main_module
    from src.models.model_registry import ModelRegistry

    def _fake_create(self, model_type, input_dim, **kwargs):
        return TinyLinear(input_dim)

    monkeypatch.setattr(ModelRegistry, "create_model", _fake_create, raising=True)

    # Run the CLI main with patched argv
    monkeypatch.setattr(eval_main_module.sys, "argv", ["prog"] + args)
    eval_main()

    # Ensure metrics printed
    captured = capsys.readouterr().out
    assert "Evaluation complete" in captured
