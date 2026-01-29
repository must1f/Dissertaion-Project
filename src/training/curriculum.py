"""
Curriculum Training for Physics-Informed Neural Networks

Gradually increases physics loss weights during training for better convergence
"""

import torch
import numpy as np
from typing import Dict, Literal

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CurriculumScheduler:
    """
    Curriculum scheduler for physics loss weights

    Gradually increases physics weights from initial to final values
    using various scheduling strategies
    """

    def __init__(
        self,
        initial_lambda_gbm: float = 0.0,
        final_lambda_gbm: float = 0.1,
        initial_lambda_ou: float = 0.0,
        final_lambda_ou: float = 0.1,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        strategy: Literal['linear', 'exponential', 'cosine', 'step'] = 'cosine'
    ):
        """
        Args:
            initial_lambda_gbm: Starting GBM weight
            final_lambda_gbm: Final GBM weight
            initial_lambda_ou: Starting OU weight
            final_lambda_ou: Final OU weight
            warmup_epochs: Number of epochs for warmup (pure data loss)
            total_epochs: Total number of training epochs
            strategy: Scheduling strategy
        """
        self.initial_lambda_gbm = initial_lambda_gbm
        self.final_lambda_gbm = final_lambda_gbm
        self.initial_lambda_ou = initial_lambda_ou
        self.final_lambda_ou = final_lambda_ou
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.strategy = strategy

        self.current_epoch = 0

        logger.info(f"CurriculumScheduler initialized: strategy={strategy}, "
                   f"warmup={warmup_epochs}, total={total_epochs}")
        logger.info(f"  GBM: {initial_lambda_gbm:.4f} → {final_lambda_gbm:.4f}")
        logger.info(f"  OU:  {initial_lambda_ou:.4f} → {final_lambda_ou:.4f}")

    def step(self, epoch: int) -> Dict[str, float]:
        """
        Update physics weights for current epoch

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Dict with current lambda_gbm and lambda_ou
        """
        self.current_epoch = epoch

        # During warmup: use initial weights (typically 0)
        if epoch < self.warmup_epochs:
            progress = 0.0
        else:
            # After warmup: gradually increase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)

        # Compute scaling factor based on strategy
        if self.strategy == 'linear':
            scale = progress
        elif self.strategy == 'exponential':
            scale = progress ** 2
        elif self.strategy == 'cosine':
            scale = 0.5 * (1 - np.cos(np.pi * progress))
        elif self.strategy == 'step':
            # Step increases at 25%, 50%, 75%
            if progress < 0.25:
                scale = 0.0
            elif progress < 0.5:
                scale = 0.33
            elif progress < 0.75:
                scale = 0.66
            else:
                scale = 1.0
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Interpolate weights
        lambda_gbm = self.initial_lambda_gbm + (self.final_lambda_gbm - self.initial_lambda_gbm) * scale
        lambda_ou = self.initial_lambda_ou + (self.final_lambda_ou - self.initial_lambda_ou) * scale

        return {
            'lambda_gbm': lambda_gbm,
            'lambda_ou': lambda_ou,
            'progress': progress,
            'scale': scale
        }

    def get_current_weights(self) -> Dict[str, float]:
        """Get current physics weights"""
        return self.step(self.current_epoch)

    def update_model_weights(self, model, epoch: int):
        """
        Update physics weights in model

        Args:
            model: StackedPINN or ResidualPINN model
            epoch: Current epoch
        """
        weights = self.step(epoch)

        if hasattr(model, 'lambda_gbm'):
            model.lambda_gbm = weights['lambda_gbm']
        if hasattr(model, 'lambda_ou'):
            model.lambda_ou = weights['lambda_ou']

        logger.debug(f"Epoch {epoch}: λ_gbm={weights['lambda_gbm']:.4f}, "
                    f"λ_ou={weights['lambda_ou']:.4f}")


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum that adjusts based on training performance

    Increases physics weights faster if model is learning well,
    slower if struggling
    """

    def __init__(
        self,
        initial_lambda_gbm: float = 0.0,
        final_lambda_gbm: float = 0.1,
        initial_lambda_ou: float = 0.0,
        final_lambda_ou: float = 0.1,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        strategy: str = 'cosine',
        patience: int = 5,
        threshold: float = 0.01
    ):
        """
        Args:
            patience: Epochs to wait before adjusting
            threshold: Validation loss improvement threshold
        """
        super().__init__(
            initial_lambda_gbm, final_lambda_gbm,
            initial_lambda_ou, final_lambda_ou,
            warmup_epochs, total_epochs, strategy
        )

        self.patience = patience
        self.threshold = threshold
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0
        self.adjustment_factor = 1.0

    def step_adaptive(
        self,
        epoch: int,
        val_loss: float
    ) -> Dict[str, float]:
        """
        Adaptive step based on validation performance

        Args:
            epoch: Current epoch
            val_loss: Current validation loss

        Returns:
            Dict with current weights and adjustment info
        """
        # Get base weights
        weights = super().step(epoch)

        # Check if validation improved
        if val_loss < self.best_val_loss - self.threshold:
            self.best_val_loss = val_loss
            self.epochs_since_improvement = 0
            # Model is learning well, can increase physics faster
            self.adjustment_factor = min(1.2, self.adjustment_factor * 1.05)
        else:
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.patience:
                # Model struggling, slow down physics increase
                self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
                self.epochs_since_improvement = 0

        # Apply adjustment
        weights['lambda_gbm'] *= self.adjustment_factor
        weights['lambda_ou'] *= self.adjustment_factor
        weights['adjustment_factor'] = self.adjustment_factor

        return weights
