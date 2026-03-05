"""
AblationRunner

Lightweight orchestration for structured ablation experiments. It reads a
YAML file that defines baseline and treatment variants, merges each with a
base experiment config, and yields concrete configs you can feed into your
training/evaluation pipeline.

Usage pattern (code):
    runner = AblationRunner("configs/ablations.yaml")
    for variant in runner.generate("curriculum", base_cfg):
        run_experiment(variant.config, tag=variant.name)

Usage pattern (CLI): see scripts/run_ablations.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

import yaml  # type: ignore


@dataclass
class AblationVariant:
    name: str        # e.g., curriculum_baseline, curriculum_treatment
    config: Dict[str, Any]
    label: str       # e.g., baseline, treatment


class AblationRunner:
    def __init__(self, ablations_path: Path | str = "configs/ablations.yaml"):
        self.ablations_path = Path(ablations_path)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.ablations_path.exists():
            raise FileNotFoundError(f"Ablations config not found: {self.ablations_path}")
        return yaml.safe_load(self.ablations_path.read_text()) or {}

    def generate(
        self,
        ablation_name: str,
        base_config: Dict[str, Any],
    ) -> Iterator[AblationVariant]:
        """Yield baseline and treatment configs for the selected ablation."""
        abl = (self.data.get("ablations") or {}).get(ablation_name)
        if abl is None:
            raise ValueError(f"Ablation '{ablation_name}' not found in {self.ablations_path}")

        for label in ["baseline", "treatment"]:
            if label not in abl:
                continue
            cfg = copy.deepcopy(base_config)
            self._deep_merge(cfg, abl[label])
            name = f"{ablation_name}_{label}"
            yield AblationVariant(name=name, config=cfg, label=label)

    @staticmethod
    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        """In-place deep merge of src into dst."""
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                AblationRunner._deep_merge(dst[k], v)
            else:
                dst[k] = v

