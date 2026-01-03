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
from tqdm import tqdm
import json

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..evaluation.metrics import calculate_metrics

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
    Trainer for PyTorch models with comprehensive logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[any] = None,
        device: Optional[torch.device] = None
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
        """
        self.config = config or get_config()
        self.device = device or torch.device(self.config.training.device)
        if not torch.cuda.is_available() and self.device.type == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss criterion (default MSE, can be overridden)
        self.criterion = nn.MSELoss()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            mode='min'
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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

        all_predictions = []
        all_targets = []

        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (sequences, targets, metadata) in enumerate(pbar):
                # Move to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # Check if model is PINN
                is_pinn = hasattr(self.model, 'compute_loss')

                if is_pinn:
                    # PINN forward pass
                    predictions = self.model(sequences)

                    # Move metadata to device
                    if 'prices' in metadata:
                        metadata['prices'] = metadata['prices'].to(self.device)
                    if 'returns' in metadata:
                        metadata['returns'] = metadata['returns'].to(self.device)
                    if 'volatilities' in metadata:
                        metadata['volatilities'] = metadata['volatilities'].to(self.device)

                    # Compute loss with physics
                    loss, loss_dict = self.model.compute_loss(
                        predictions, targets, metadata, enable_physics=enable_physics
                    )

                    total_data_loss += loss_dict.get('data_loss', 0.0)
                    total_physics_loss += loss_dict.get('physics_loss', 0.0)

                else:
                    # Standard forward pass
                    if hasattr(self.model, 'base_model_type') and self.model.base_model_type in ['lstm', 'gru']:
                        predictions, _ = self.model(sequences)
                    else:
                        predictions = self.model(sequences)

                    loss = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

        all_predictions = []
        all_targets = []

        with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
            for sequences, targets, metadata in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                is_pinn = hasattr(self.model, 'compute_loss')

                if is_pinn:
                    predictions = self.model(sequences)

                    if 'prices' in metadata:
                        metadata['prices'] = metadata['prices'].to(self.device)
                    if 'returns' in metadata:
                        metadata['returns'] = metadata['returns'].to(self.device)
                    if 'volatilities' in metadata:
                        metadata['volatilities'] = metadata['volatilities'].to(self.device)

                    loss, _ = self.model.compute_loss(
                        predictions, targets, metadata, enable_physics=enable_physics
                    )
                else:
                    if hasattr(self.model, 'base_model_type') and self.model.base_model_type in ['lstm', 'gru']:
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

        return avg_loss, metrics

    def train(
        self,
        epochs: Optional[int] = None,
        enable_physics: bool = True,
        save_best: bool = True
    ) -> Dict:
        """
        Full training loop

        Args:
            epochs: Number of epochs (uses config if None)
            enable_physics: Whether to enable physics losses
            save_best: Whether to save best model checkpoint

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.training.epochs

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Physics losses enabled: {enable_physics}")

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

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                if save_best:
                    self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True
                    )
                    logger.info(f"✓ Best model saved (epoch {epoch})")

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                logger.info(f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                break

        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config.dict() if hasattr(self.config, 'dict') else {}
        }

        # Save latest
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)

            # Save history
            history_path = checkpoint_dir / 'history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

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
    def evaluate(self) -> Dict:
        """
        Evaluate on test set

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
            if hasattr(self.model, 'base_model_type') and self.model.base_model_type in ['lstm', 'gru']:
                predictions, _ = self.model(sequences)
            else:
                predictions = self.model(sequences)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        metrics = calculate_metrics(all_targets, all_predictions, prefix="test_")

        logger.info("\nTest Results:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        return metrics
