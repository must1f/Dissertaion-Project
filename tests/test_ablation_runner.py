"""
Unit tests for AblationRunner config merging.
"""

import yaml
from pathlib import Path

from src.evaluation.ablation_runner import AblationRunner


def test_ablation_runner_merges_baseline_and_treatment(tmp_path: Path):
    ablations_yaml = tmp_path / "ablations.yaml"
    ablations_yaml.write_text(
        """
ablations:
  simple_toggle:
    baseline:
      training:
        adaptive_weighting: none
    treatment:
      training:
        adaptive_weighting: gradnorm
defaults:
  seeds: [1]
"""
    )

    base_cfg = {"training": {"epochs": 10, "adaptive_weighting": "none"}}
    runner = AblationRunner(ablations_yaml)
    variants = list(runner.generate("simple_toggle", base_cfg))

    assert len(variants) == 2
    baseline = [v for v in variants if v.label == "baseline"][0]
    treatment = [v for v in variants if v.label == "treatment"][0]

    assert baseline.config["training"]["adaptive_weighting"] == "none"
    assert treatment.config["training"]["adaptive_weighting"] == "gradnorm"
    # Base config values should persist
    assert baseline.config["training"]["epochs"] == 10
    assert treatment.config["training"]["epochs"] == 10
