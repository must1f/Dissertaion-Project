import json
from pathlib import Path


def test_colab_all_models_uses_benchmark_pipeline():
    nb_path = Path("Jupyter/Colab_All_Models.ipynb")
    nb = json.loads(nb_path.read_text())

    code_cells = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
    joined = "\n".join(code_cells)

    # Legacy single-index workflow should be absent.
    assert "download_sp500_10y_data" not in joined
    assert "^GSPC" not in joined

    # New benchmark pipeline hooks should be present.
    assert "build_benchmark_dataset" in joined
    assert "fairness_contract" in joined
    assert "target_type=dataset_meta.get('target_type'" in joined
