import json
from pathlib import Path

import pytest

from src.models.model_registry import ModelRegistry
from src.models.baseline import AttentionLSTM
from scripts import train_models
from scripts import run_ablations


class DummyDataset:
    def __len__(self):
        return 4


def _core_contract():
    return {
        "dataset": {
            "fingerprint": "core-smoke",
            "target_symbol": "SPY",
            "target_type": "next_day_log_return",
            "target_column": "target",
            "price_column": "adjusted_close",
            "lookback": 5,
            "horizon": 1,
        },
        "features": {"required_core": [], "effective": []},
    }


def _vol_contract():
    return {
        "dataset": {
            "fingerprint": "vol-smoke",
            "target_symbol": "SPY",
            "target_type": "realized_vol",
            "target_column": "realized_vol",
            "price_column": "adjusted_close",
            "lookback": 5,
            "horizon": 1,
        },
        "features": {"required_core": [], "effective": []},
    }


def test_attention_registry_returns_attention_lstm(tmp_path):
    registry = ModelRegistry(tmp_path)
    model = registry.create_model("attention_lstm", input_dim=4, hidden_dim=8, num_layers=1, dropout=0.1)
    assert isinstance(model, AttentionLSTM)


def test_train_single_model_smoke_core(tmp_path, monkeypatch):
    train_models._CORE_CONTRACT_SIGNATURE = None
    ds = DummyDataset()
    result = train_models.train_single_model(
        model_type="lstm",
        train_dataset=ds,
        val_dataset=ds,
        test_dataset=ds,
        input_dim=4,
        research_mode=False,
        epochs=1,
        fairness_contract=_core_contract(),
        dataset_meta={"target_type": "next_day_log_return", "fingerprint": "core-smoke"},
        run_dir=tmp_path,
        smoke_test=True,
    )
    assert result["track"] == "core_benchmark"


def test_train_single_model_smoke_volatility(tmp_path, monkeypatch):
    train_models._CORE_CONTRACT_SIGNATURE = None
    ds = DummyDataset()
    result = train_models.train_single_model(
        model_type="vol_lstm",
        train_dataset=ds,
        val_dataset=ds,
        test_dataset=ds,
        input_dim=4,
        research_mode=False,
        epochs=1,
        fairness_contract=_vol_contract(),
        dataset_meta={"target_type": "realized_vol", "fingerprint": "vol-smoke"},
        run_dir=tmp_path,
        smoke_test=True,
        allow_target_mismatch=True,
    )
    assert result["track"] == "volatility_extension"


def test_target_mismatch_guard(monkeypatch):
    train_models._CORE_CONTRACT_SIGNATURE = None
    ds = DummyDataset()
    with pytest.raises(ValueError):
        train_models.train_single_model(
            model_type="vol_lstm",
            train_dataset=ds,
            val_dataset=ds,
            test_dataset=ds,
            input_dim=4,
            research_mode=False,
            epochs=1,
            fairness_contract=_core_contract(),
            dataset_meta={"target_type": "next_day_log_return", "fingerprint": "core-smoke"},
            run_dir=Path("."),
            smoke_test=True,
            allow_target_mismatch=False,
        )


def test_notebook_has_volatility_bundle():
    nb_path = Path("Jupyter/Colab_All_Models.ipynb")
    nb = json.loads(nb_path.read_text())
    code_cells = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
    joined = "\n".join(code_cells)
    assert "fairness_contract_volatility" in joined
    assert "target_override='realized_vol'" in joined


def test_run_ablations_smoke(tmp_path):
    comp_path = run_ablations.run_ablation_smoke(
        ablation_name="smoke_abl",
        output_dir=tmp_path,
        model="lstm",
        target_type="next_day_log_return",
    )
    assert comp_path.exists()
    content = comp_path.read_text().strip().splitlines()
    assert len(content) == 3  # header + two rows
