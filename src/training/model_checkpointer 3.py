"""
Model checkpointing with metadata (commit hash, metrics, regime tags).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from ..utils.reproducibility import get_environment_info
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    model_name: str
    experiment_id: str
    seed: int
    metrics: Dict[str, float]
    regime: Optional[str] = None
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None


class ModelCheckpointer:
    def __init__(self, base_dir: Path | str = "Models/checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, metadata: CheckpointMetadata, filename: Optional[str] = None) -> Path:
        """Save model state dict plus metadata."""
        fname = filename or f"{metadata.model_name}_{metadata.experiment_id}.pt"
        path = self.base_dir / fname

        payload = {
            "state_dict": model.state_dict(),
            "metadata": asdict(metadata),
            "environment": asdict(get_environment_info()),
        }
        torch.save(payload, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    @staticmethod
    def load(path: Path | str) -> Dict[str, Any]:
        # allow list torch version object for safe globals in torch >=2.6
        from torch.serialization import add_safe_globals
        import torch.torch_version

        add_safe_globals([torch.torch_version.TorchVersion])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload
