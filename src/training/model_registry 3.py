"""
Model registry to track best checkpoints and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any


@dataclass
class RegistryEntry:
    model_name: str
    path: str
    metrics: Dict[str, float]
    regime: Optional[str] = None
    git_commit: Optional[str] = None


class ModelRegistry:
    def __init__(self, registry_path: Path | str = "Models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        self.registry_path.write_text(json.dumps(self.data, indent=2))

    def register(self, key: str, entry: RegistryEntry) -> None:
        self.data[key] = asdict(entry)
        self.save()

    def get(self, key: str) -> Optional[RegistryEntry]:
        if key not in self.data:
            return None
        return RegistryEntry(**self.data[key])

    def best_overall(self, metric: str = "sharpe_ratio") -> Optional[RegistryEntry]:
        best_key = None
        best_val = None
        for k, v in self.data.items():
            if metric in v["metrics"]:
                val = v["metrics"][metric]
                if best_val is None or val > best_val:
                    best_val = val
                    best_key = k
        return self.get(best_key) if best_key else None
