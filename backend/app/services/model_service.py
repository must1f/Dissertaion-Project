"""Service for model management wrapping src/models/."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import ModelNotFoundError, ModelNotTrainedError
from backend.app.schemas.models import (
    ModelInfo,
    ModelStatus,
    PhysicsParameters,
    ModelWeightsInfo,
)

# Import from existing src/
try:
    from src.models.model_registry import ModelRegistry
    from src.models.pinn import PINNModel
    HAS_SRC = True
except ImportError:
    HAS_SRC = False
    ModelRegistry = None
    PINNModel = None


def _get_available_device(preferred: str) -> torch.device:
    """Get an available device, falling back to CPU if preferred is unavailable."""
    preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cpu":
        return torch.device("cpu")
    # Auto-detect best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelService:
    """Service for managing ML models."""

    def __init__(self):
        """Initialize model service."""
        self._registry: Optional[ModelRegistry] = None
        self._loaded_models: Dict[str, nn.Module] = {}
        self._device = _get_available_device(settings.default_device)

    @property
    def registry(self) -> "ModelRegistry":
        """Lazy load model registry."""
        if self._registry is None:
            if not HAS_SRC:
                raise RuntimeError("src/ modules not available")
            self._registry = ModelRegistry(settings.project_root)
        return self._registry

    def _get_model_status(self, model_key: str) -> ModelStatus:
        """Get status for a model."""
        try:
            info = self.registry.get_model_info(model_key)
            if info and info.trained:
                return ModelStatus.TRAINED
            return ModelStatus.NOT_TRAINED
        except Exception:
            return ModelStatus.NOT_TRAINED

    def _get_physics_params(self, model: nn.Module) -> Optional[PhysicsParameters]:
        """Extract physics parameters from a PINN model."""
        if not hasattr(model, "get_learned_physics_params"):
            return None

        try:
            params = model.get_learned_physics_params()
            return PhysicsParameters(
                theta=params.get("theta"),
                gamma=params.get("gamma"),
                temperature=params.get("temperature") or params.get("T"),
                mu=params.get("mu"),
                sigma=params.get("sigma"),
            )
        except Exception:
            return None

    def get_all_models(self) -> List[ModelInfo]:
        """Get information about all available models."""
        if not HAS_SRC:
            if settings.demo_mode:
                return self._get_mock_models()
            raise RuntimeError("src/ modules not available; cannot list models.")

        models = []
        try:
            all_models = self.registry.get_all_models()
            for key, info in all_models.items():
                # Skip aliases so they don't duplicate on the frontend
                if key != info.model_key:
                    continue
                    
                status = ModelStatus.TRAINED if info.trained else ModelStatus.NOT_TRAINED
                models.append(
                    ModelInfo(
                        model_key=key,
                        model_type=info.model_type,
                        display_name=info.model_name,
                        description=info.description,
                        status=status,
                        is_pinn="pinn" in key.lower() or "pinn" in info.model_type.lower(),
                        checkpoint_path=str(info.checkpoint_path) if info.checkpoint_path else None,
                    )
                )
        except Exception as e:
            print(f"Error getting models: {e}")
            if settings.demo_mode:
                return self._get_mock_models()
            raise

        return models

    def _get_mock_models(self) -> List[ModelInfo]:
        """Return mock models when src/ is not available."""
        if not settings.demo_mode:
            raise RuntimeError("Mock models are disabled when DEMO_MODE is false.")
        mock_models = [
            ("lstm", "LSTM", "LSTM", False),
            ("gru", "GRU", "GRU", False),
            ("bilstm", "BiLSTM", "Bidirectional LSTM", False),
            ("attention_lstm", "AttentionLSTM", "LSTM with Attention", False),
            ("transformer", "Transformer", "Transformer Model", False),
            ("pinn_baseline", "PINN", "PINN Baseline", True),
            ("pinn_gbm", "PINN", "PINN with GBM", True),
            ("pinn_ou", "PINN", "PINN with Ornstein-Uhlenbeck", True),
            ("pinn_black_scholes", "PINN", "PINN with Black-Scholes", True),
            ("pinn_gbm_ou", "PINN", "PINN with GBM + OU", True),
            ("stacked_pinn", "StackedPINN", "Stacked PINN", True),
        ]
        return [
            ModelInfo(
                model_key=key,
                model_type=mtype,
                display_name=name,
                status=ModelStatus.NOT_TRAINED,
                is_pinn=is_pinn,
            )
            for key, mtype, name, is_pinn in mock_models
        ]

    def get_trained_models(self) -> List[ModelInfo]:
        """Get only trained models."""
        all_models = self.get_all_models()
        return [m for m in all_models if m.status == ModelStatus.TRAINED]

    def get_model_info(self, model_key: str) -> ModelInfo:
        """Get detailed information about a specific model."""
        all_models = self.get_all_models()
        for model in all_models:
            if model.model_key == model_key:
                return model
        raise ModelNotFoundError(model_key)

    def load_model(
        self,
        model_key: str,
        device: Optional[str] = None,
        input_dim: int = 5,
    ) -> nn.Module:
        """Load a trained model."""
        # Check cache
        cache_key = f"{model_key}_{device or self._device}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        if not HAS_SRC:
            raise RuntimeError("src/ modules not available for model loading")

        # Load from registry
        target_device = torch.device(device) if device else self._device

        try:
            model = self.registry.load_model(
                model_key=model_key,
                device=target_device,
                input_dim=input_dim,
            )
        except FileNotFoundError:
            raise ModelNotTrainedError(model_key)
        except Exception as e:
            raise ModelNotFoundError(model_key)

        # Cache and return
        self._loaded_models[cache_key] = model
        return model

    def get_model_weights_info(self, model_key: str) -> ModelWeightsInfo:
        """Get information about model weights."""
        model = self.load_model(model_key)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get layer info
        layer_info = []
        for name, param in model.named_parameters():
            layer_info.append({
                "name": name,
                "shape": list(param.shape),
                "parameters": param.numel(),
                "trainable": param.requires_grad,
            })

        return ModelWeightsInfo(
            model_key=model_key,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            layer_info=layer_info,
        )

    def compare_models(
        self,
        model_keys: List[str],
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison = {
            "models": [],
            "metric_names": [],
            "best_by_metric": {},
        }

        for key in model_keys:
            try:
                model_info = self.get_model_info(key)
                model_data = {
                    "model_key": key,
                    "display_name": model_info.display_name,
                    "is_pinn": model_info.is_pinn,
                    "status": model_info.status.value,
                }

                # Add metrics if provided
                if metrics and key in metrics:
                    model_data["metrics"] = metrics[key]
                    if not comparison["metric_names"]:
                        comparison["metric_names"] = list(metrics[key].keys())

                # Get physics params if PINN
                if model_info.is_pinn and model_info.status == ModelStatus.TRAINED:
                    try:
                        model = self.load_model(key)
                        physics_params = self._get_physics_params(model)
                        if physics_params:
                            model_data["physics_parameters"] = physics_params.model_dump()
                    except Exception:
                        pass

                comparison["models"].append(model_data)
            except Exception as e:
                print(f"Error processing model {key}: {e}")
                continue

        # Determine best by metric
        if metrics and comparison["metric_names"]:
            for metric_name in comparison["metric_names"]:
                best_key = None
                best_value = None
                for key in model_keys:
                    if key in metrics and metric_name in metrics[key]:
                        value = metrics[key][metric_name]
                        # Lower is better for loss metrics, higher for accuracy metrics
                        is_lower_better = any(
                            x in metric_name.lower()
                            for x in ["loss", "error", "rmse", "mae", "mape", "drawdown"]
                        )
                        if best_value is None:
                            best_value = value
                            best_key = key
                        elif is_lower_better and value < best_value:
                            best_value = value
                            best_key = key
                        elif not is_lower_better and value > best_value:
                            best_value = value
                            best_key = key
                if best_key:
                    comparison["best_by_metric"][metric_name] = best_key

        return comparison

    def unload_model(self, model_key: str):
        """Unload a model from cache to free memory."""
        keys_to_remove = [k for k in self._loaded_models if k.startswith(model_key)]
        for key in keys_to_remove:
            del self._loaded_models[key]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_available_model_types(self) -> List[str]:
        """Get list of available model types for training."""
        return [
            "lstm",
            "gru",
            "bilstm",
            "attention_lstm",
            "transformer",
            "pinn_baseline",
            "pinn_gbm",
            "pinn_ou",
            "pinn_black_scholes",
            "pinn_gbm_ou",
            "pinn_global",
            "stacked_pinn",
            "residual_pinn",
        ]

    def list_saved_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved model checkpoint files."""
        checkpoints = []
        models_path = settings.models_path

        if not models_path.exists():
            return checkpoints

        for pt_file in models_path.glob("*.pt"):
            stat = pt_file.stat()
            # Parse model info from filename
            name = pt_file.stem  # Remove .pt extension
            parts = name.rsplit("_", 1)
            is_best = parts[-1] == "best" if len(parts) > 1 else False

            checkpoints.append({
                "filename": pt_file.name,
                "name": name,
                "path": str(pt_file),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "is_best": is_best,
            })

        # Sort by modified time, newest first
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def rename_checkpoint(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename a saved checkpoint file."""
        models_path = settings.models_path

        # Handle with or without .pt extension
        old_filename = old_name if old_name.endswith(".pt") else f"{old_name}.pt"
        new_filename = new_name if new_name.endswith(".pt") else f"{new_name}.pt"

        old_path = models_path / old_filename
        new_path = models_path / new_filename

        if not old_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {old_filename}")

        if new_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {new_filename}")

        old_path.rename(new_path)

        # Also rename associated history file if exists
        old_history = models_path / f"{old_name.replace('.pt', '')}_history.json"
        new_history = models_path / f"{new_name.replace('.pt', '')}_history.json"
        if old_history.exists():
            old_history.rename(new_history)

        return {
            "old_name": old_filename,
            "new_name": new_filename,
            "message": f"Renamed {old_filename} to {new_filename}",
        }

    def delete_checkpoint(self, name: str) -> Dict[str, str]:
        """Delete a saved checkpoint file."""
        models_path = settings.models_path

        filename = name if name.endswith(".pt") else f"{name}.pt"
        filepath = models_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename}")

        filepath.unlink()

        # Also delete associated history file if exists
        history_file = models_path / f"{name.replace('.pt', '')}_history.json"
        if history_file.exists():
            history_file.unlink()

        return {
            "deleted": filename,
            "message": f"Deleted checkpoint: {filename}",
        }
