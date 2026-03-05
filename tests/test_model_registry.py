"""
Tests for model checkpointer and registry.
"""

from pathlib import Path
import torch

from src.training.model_checkpointer import ModelCheckpointer, CheckpointMetadata
from src.training.model_registry import ModelRegistry, RegistryEntry


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)


def test_model_registry_roundtrip(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    entry = RegistryEntry(
        model_name="demo",
        path="demo.pt",
        metrics={"sharpe_ratio": 1.23},
        regime="high_vol",
        git_commit="abc123",
    )
    registry.register("demo_key", entry)

    loaded = registry.get("demo_key")
    assert loaded is not None
    assert loaded.metrics["sharpe_ratio"] == 1.23

    best = registry.best_overall("sharpe_ratio")
    assert best is not None
    assert best.model_name == "demo"


def test_model_checkpointer(tmp_path: Path):
    model = DummyNet()
    ckpt = ModelCheckpointer(tmp_path)
    metadata = CheckpointMetadata(
        model_name="demo",
        experiment_id="exp1",
        seed=42,
        metrics={"sharpe_ratio": 0.9},
    )
    path = ckpt.save(model, metadata)
    payload = ckpt.load(path)
    assert "state_dict" in payload
    assert payload["metadata"]["model_name"] == "demo"

