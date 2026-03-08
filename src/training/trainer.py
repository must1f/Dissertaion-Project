"""
Trainer class for model training with early stopping, checkpointing, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

# tqdm is only used for progress bars; if it's missing we still want real training to run.
try:  # pragma: no cover - defensive import guard
    from tqdm import tqdm
except ImportError:  # lightweight fallback to keep training functional
    class tqdm:
        """Fallback tqdm that supports context manager protocol."""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_postfix(self, *args, **kwargs):
            pass  # No-op for fallback

from ..utils.config import get_config, get_research_config
from ..utils.logger import get_logger
from ..evaluation.metrics import calculate_metrics
from .model_checkpointer import ModelCheckpointer, CheckpointMetadata
from .model_registry import ModelRegistry, RegistryEntry
from ..utils.reproducibility import get_environment_info

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current score (e.g., validation loss)

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class Trainer:
    """
    Trainer for PyTorch models with comprehensive logging and checkpointing.

    Supports research mode for fair model comparisons with locked parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[any] = None,
        device: Optional[torch.device] = None,
        research_mode: bool = False,
        batch_callback: Optional[callable] = None,
        run_metadata: Optional[dict] = None,
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration object
            device: Device to train on
            research_mode: If True, use locked research parameters for fair comparison
            batch_callback: Optional callback called every N batches with progress info.
                           Signature: callback(batch_idx, total_batches, batch_loss)
        """
        self.config = config or get_config()
        self.research_mode = research_mode
        self.research_config = get_research_config() if research_mode else None

        self.device = device or torch.device(self.config.training.device)
        if not torch.cuda.is_available() and self.device.type == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpointer = ModelCheckpointer(self.config.checkpoint_dir)
        self.model_registry = ModelRegistry(self.config.project_root / "Models" / "registry.json")
        self.run_metadata = run_metadata or {}

        # Determine learning rate and regularization based on mode
        if research_mode and self.research_config:
            learning_rate = self.research_config.learning_rate
            weight_decay = self.research_config.weight_decay
            scheduler_patience = self.research_config.scheduler_patience
            scheduler_factor = self.research_config.scheduler_factor
            self.gradient_clip_norm = self.research_config.gradient_clip_norm
            logger.info("Research mode enabled - using locked training parameters")
            logger.info(f"  LR={learning_rate}, weight_decay={weight_decay}, scheduler_patience={scheduler_patience}")
        else:
            learning_rate = self.config.training.learning_rate
            weight_decay = 1e-5  # Light regularization for non-research mode
            scheduler_patience = 5
            scheduler_factor = 0.5
            self.gradient_clip_norm = 1.0

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience
        )

        # Loss criterion (default MSE, can be overridden)
        self.criterion = nn.MSELoss()

        # Early stopping - disabled in research mode for fair comparison
        if research_mode and self.research_config and not self.research_config.use_early_stopping:
            self.early_stopping = None
            logger.info("Early stopping DISABLED for research mode (all models train for full epochs)")
        else:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                mode='min'
            )

        # Batch callback for real-time progress updates
        # Interval optimized for larger batch sizes (128+) - reduces overhead
        self.batch_callback = batch_callback
        self.batch_callback_interval = 50  # Call callback every N batches (increased for performance)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'learning_rates': [],
            'epochs': [],
            'research_mode': research_mode
        }

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        if research_mode:
            logger.info(f"Research config: epochs={self.research_config.epochs}, lr={learning_rate}, batch_size={self.research_config.batch_size}")

    def train_epoch(self, enable_physics: bool = True) -> Tuple[float, Dict]:
        """
        Train for one epoch

        Args:
            enable_physics: Whether to enable physics losses (for PINN models)

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        component_sums: Dict[str, float] = defaultdict(float)

        all_predictions = []
        all_targets = []

        import time as _time
        epoch_start_time = _time.time()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        with pbar:
            for batch_idx, batch_data in enumerate(pbar):
                sequences, targets, metadata = batch_data

                # Move to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # Check if model is PINN (has compute_loss method)
                is_pinn = hasattr(self.model, 'compute_loss')

                if is_pinn:
                    # PINN forward pass
                    output = self.model(sequences)

                    # Handle different PINN model outputs:
                    # - PINNModel returns: predictions (tensor)
                    # - StackedPINN returns: (return_pred, direction_logits, attention_weights)
                    # - ResidualPINN returns: (final_pred, direction_logits, components)
                    if isinstance(output, tuple):
                        predictions = output[0]  # First element is always the main prediction
                    else:
                        predictions = output

                    # Move metadata to device
                    if 'prices' in metadata:
                        metadata['prices'] = metadata['prices'].to(self.device)
                    if 'returns' in metadata:
                        metadata['returns'] = metadata['returns'].to(self.device)
                    if 'volatilities' in metadata:
                        metadata['volatilities'] = metadata['volatilities'].to(self.device)

                    # CRITICAL: Add inputs to metadata for Black-Scholes autograd
                    # This enables computing exact derivatives via torch.autograd.grad
                    metadata['inputs'] = sequences

                    # Compute loss with physics
                    compute_out = self.model.compute_loss(
                        predictions, targets, metadata, enable_physics=enable_physics
                    )

                    if isinstance(compute_out, tuple):
                        loss, loss_dict = compute_out
                    elif isinstance(compute_out, dict):
                        loss_dict = compute_out
                        loss = loss_dict.get('total_loss', None)
                        if loss is None:
                            raise ValueError("compute_loss dict must contain 'total_loss'")
                    else:
                        raise ValueError("compute_loss must return (loss, dict) or dict")

                    total_data_loss += loss_dict.get('data_loss', 0.0)
                    total_physics_loss += loss_dict.get('physics_loss', 0.0)
                    for k, v in loss_dict.items():
                        if isinstance(v, (int, float)):
                            component_sums[k] += float(v)

                else:
                    # Standard forward pass
                    # Check if model returns tuple (LSTM/GRU models return (output, hidden))
                    model_class_name = self.model.__class__.__name__.lower()

                    if 'lstm' in model_class_name or 'gru' in model_class_name:
                        predictions, _ = self.model(sequences)
                    else:
                        predictions = self.model(sequences)

                    loss = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping (uses research config norm if in research mode)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)

                # Optimizer step
                self.optimizer.step()

                # Accumulate loss
                total_loss += loss.item()
                n_batches += 1

                # Collect predictions for metrics
                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Log timing every 200 batches (reduced frequency for performance)
                if batch_idx > 0 and batch_idx % 200 == 0:
                    elapsed = _time.time() - epoch_start_time
                    eta = (elapsed / batch_idx) * (len(self.train_loader) - batch_idx)
                    avg_so_far = total_loss / n_batches
                    logger.info(f"  Batch {batch_idx}/{len(self.train_loader)}: avg_loss={avg_so_far:.6f}, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

                # Call batch callback for real-time progress updates
                if self.batch_callback and batch_idx % self.batch_callback_interval == 0:
                    try:
                        self.batch_callback(
                            batch_idx=batch_idx,
                            total_batches=len(self.train_loader),
                            batch_loss=loss.item(),
                        )
                    except Exception as cb_err:
                        logger.warning(f"Batch callback error: {cb_err}")

        # Calculate average loss
        avg_loss = total_loss / n_batches

        # Calculate metrics
        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        metrics = calculate_metrics(all_targets, all_predictions, prefix="train_")
        metrics['train_loss'] = avg_loss

        if total_physics_loss > 0:
            metrics['train_data_loss'] = total_data_loss / n_batches
            metrics['train_physics_loss'] = total_physics_loss / n_batches

        # Log physics components if present
        for k, v in component_sums.items():
            metrics[f"train_{k}"] = v / n_batches

        return avg_loss, metrics

    @torch.no_grad()
    def validate_epoch(self, enable_physics: bool = True) -> Tuple[float, Dict]:
        """
        Validate for one epoch

        Args:
            enable_physics: Whether to enable physics losses

        Returns:
            Tuple of (average_loss, metrics_dict)
        """

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        component_sums: Dict[str, float] = defaultdict(float)

        all_predictions = []
        all_targets = []

        with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
            for sequences, targets, metadata in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                is_pinn = hasattr(self.model, 'compute_loss')

                if is_pinn:
                    output = self.model(sequences)

                    # Handle tuple outputs from StackedPINN/ResidualPINN
                    if isinstance(output, tuple):
                        predictions = output[0]
                    else:
                        predictions = output

                    if 'prices' in metadata:
                        metadata['prices'] = metadata['prices'].to(self.device)
                    if 'returns' in metadata:
                        metadata['returns'] = metadata['returns'].to(self.device)
                    if 'volatilities' in metadata:
                        metadata['volatilities'] = metadata['volatilities'].to(self.device)

                    # Add inputs for Black-Scholes autograd
                    metadata['inputs'] = sequences

                    loss, loss_dict = self.model.compute_loss(
                        predictions, targets, metadata, enable_physics=enable_physics
                    )
                    for k, v in loss_dict.items():
                        if isinstance(v, (int, float)):
                            component_sums[k] += float(v)
                else:
                    # Check if model returns tuple (LSTM/GRU models)
                    model_class_name = self.model.__class__.__name__.lower()
                    if 'lstm' in model_class_name or 'gru' in model_class_name:
                        predictions, _ = self.model(sequences)
                    else:
                        predictions = self.model(sequences)

                    loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                n_batches += 1

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches

        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        metrics = calculate_metrics(all_targets, all_predictions, prefix="val_")
        metrics['val_loss'] = avg_loss
        for k, v in component_sums.items():
            metrics[f"val_{k}"] = v / n_batches

        logger.debug(f"validate_epoch() completed: avg_loss={avg_loss:.6f}, n_batches={n_batches}")
        logger.debug(f"  metrics: {metrics}")

        return avg_loss, metrics

    def train(
        self,
        epochs: Optional[int] = None,
        enable_physics: bool = True,
        save_best: bool = True,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Full training loop

        Args:
            epochs: Number of epochs (uses config if None, research config in research mode)
            enable_physics: Whether to enable physics losses
            save_best: Whether to save best model checkpoint
            model_name: Optional model name for checkpoint files (e.g., "pinn_gbm")

        Returns:
            Training history dictionary
        """
        # In research mode, use locked epochs from research config
        if self.research_mode and self.research_config:
            epochs = self.research_config.epochs
            logger.info(f"Research mode: Using locked epoch count = {epochs}")
        else:
            epochs = epochs or self.config.training.epochs

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Physics losses enabled: {enable_physics}")
        if self.research_mode:
            logger.info("Research mode: Early stopping DISABLED for fair comparison")

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch(enable_physics=enable_physics)

            # Validate
            val_loss, val_metrics = self.validate_epoch(enable_physics=enable_physics)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            logger.info(f"Train RMSE: {train_metrics['train_rmse']:.4f} | Val RMSE: {val_metrics['val_rmse']:.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch)

            # Track separate data and physics losses if available
            if 'train_data_loss' in train_metrics:
                self.history['train_data_loss'].append(train_metrics['train_data_loss'])
            if 'train_physics_loss' in train_metrics:
                self.history['train_physics_loss'].append(train_metrics['train_physics_loss'])

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                if save_best:
                    self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                        model_name=model_name
                    )
                    logger.info(f"Best model saved (epoch {epoch})")

            # Early stopping - only if not in research mode
            if self.early_stopping is not None and self.early_stopping(val_loss):
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                logger.info(f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                break

        # Store final training info
        self.history['total_epochs_trained'] = epoch
        self.history['best_epoch'] = best_epoch
        self.history['best_val_loss'] = best_val_loss

        logger.info("\nTraining completed!")
        logger.info(f"Total epochs trained: {epoch}/{epochs}")
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        model_name: Optional[str] = None
    ):
        """Save model checkpoint"""
        # Use models directory for variant-specific checkpoints
        if model_name:
            checkpoint_dir = self.config.project_root / 'models'
        else:
            checkpoint_dir = self.config.checkpoint_dir

        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config.dict() if hasattr(self.config, 'dict') else {},
            'research_mode': self.research_mode,
            'research_config': self.research_config.dict() if self.research_config and hasattr(self.research_config, 'dict') else None
        }

        # Save latest
        if model_name:
            latest_path = checkpoint_dir / f'{model_name}_latest.pt'
        else:
            latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            if model_name:
                best_path = checkpoint_dir / f'{model_name}_best.pt'
                history_path = checkpoint_dir / f'{model_name}_history.json'
            else:
                best_path = checkpoint_dir / 'best.pth'
                history_path = checkpoint_dir / 'history.json'

            torch.save(checkpoint, best_path)

            # Save history
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

            # Persist rich checkpoint with metadata + register
            try:
                env = get_environment_info()
                metadata = CheckpointMetadata(
                    model_name=model_name or "model",
                    experiment_id=self.config.training.random_seed if hasattr(self.config, "training") else "unknown",
                    seed=self.config.training.random_seed if hasattr(self.config, "training") else -1,
                    metrics={"val_loss": val_loss},
                    regime=None,
                    git_commit=env.git_commit,
                    timestamp=env.timestamp,
                )
                saved_path = self.checkpointer.save(self.model, metadata, filename=best_path.name)

                entry_key = f"{metadata.model_name}_seed{metadata.seed}"
                registry_entry = RegistryEntry(
                    model_name=metadata.model_name,
                    path=str(saved_path),
                    metrics=metadata.metrics,
                    regime=metadata.regime,
                    git_commit=metadata.git_commit,
                )
                self.model_registry.register(entry_key, registry_entry)
                logger.info(f"Registered best checkpoint as {entry_key}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not persist registry/checkpoint metadata: {exc}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)

        logger.info(f"Checkpoint loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    @torch.no_grad()
    def evaluate(self, enable_physics: bool = True) -> Dict:
        """
        Evaluate on test set

        Args:
            enable_physics: Whether to enable physics losses (for PINN models)

        Returns:
            Dictionary of test metrics
        """
        logger.info("Evaluating on test set...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        for sequences, targets, metadata in tqdm(self.test_loader, desc="Testing"):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            model_class_name = self.model.__class__.__name__.lower()
            if 'pinn' in model_class_name:
                # PINN model
                predictions = self.model(sequences)
            elif 'lstm' in model_class_name or 'gru' in model_class_name:
                # LSTM/GRU models return (output, hidden)
                predictions, _ = self.model(sequences)
            else:
                # Other models (Transformer, etc.)
                predictions = self.model(sequences)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        from ..evaluation.reporting import summarize_metrics
        from ..utils.tracking import save_run

        report = summarize_metrics(all_targets, all_predictions, prefix="test_")

        logger.info("\nTest Results (forecasting):")
        for key, value in report["forecasting"].items():
            logger.info(f"{key}: {value:.4f}")

        logger.info("\nTest Results (financial):")
        for key, value in report["financial"].items():
            logger.info(f"{key}: {value:.4f}")

        if report.get("regime"):
            logger.info("\nTest Results (regime slices):")
            for key, value in report["regime"].items():
                logger.info(f"{key}: {value:.4f}")

        # Persist run summary with metadata if available
        run_meta = self.run_metadata.copy()
        run_meta.setdefault("research_mode", self.research_mode)
        run_meta.setdefault("device", str(self.device))
        run_meta.setdefault("model_class", self.model.__class__.__name__)
        out_dir = Path(self.config.output_dir) if hasattr(self.config, "output_dir") else Path("outputs")
        try:
            save_run(out_dir, run_meta, report)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to save run summary: {exc}")

        return report

    @torch.no_grad()
    def get_predictions(self, data_loader: Optional[DataLoader] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and targets from a data loader.

        Args:
            data_loader: Data loader to get predictions from (default: test_loader)

        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        data_loader = data_loader or self.test_loader

        self.model.eval()
        all_predictions = []
        all_targets = []

        for sequences, targets, metadata in data_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass - handle different model types
            model_class_name = self.model.__class__.__name__.lower()
            if 'pinn' in model_class_name:
                predictions = self.model(sequences)
            elif 'lstm' in model_class_name or 'gru' in model_class_name:
                predictions, _ = self.model(sequences)
            else:
                predictions = self.model(sequences)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()

        return predictions, targets
