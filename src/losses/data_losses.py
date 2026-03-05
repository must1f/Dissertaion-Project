"""
Data Loss Functions

Standard loss functions for regression tasks in financial forecasting.
All losses are implemented as nn.Module for composability and tracking.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class DataLoss(nn.Module, ABC):
    """Base class for data loss functions."""

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        pass

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MSELoss(DataLoss):
    """Mean Squared Error loss."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE: mean((pred - target)^2)"""
        loss = (predictions - targets) ** 2
        return self._reduce(loss)


class MAELoss(DataLoss):
    """Mean Absolute Error loss."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAE: mean(|pred - target|)"""
        loss = torch.abs(predictions - targets)
        return self._reduce(loss)


class HuberLoss(DataLoss):
    """
    Huber loss (smooth L1 loss).

    Less sensitive to outliers than MSE.
    Quadratic for small errors, linear for large errors.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """
        Args:
            delta: Threshold at which to switch from quadratic to linear
            reduction: Reduction method
        """
        super().__init__(reduction)
        self.delta = delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber loss."""
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return self._reduce(loss)


class LogCoshLoss(DataLoss):
    """
    Log-Cosh loss.

    Smoother than Huber, approximately equal to (x^2)/2 for small x
    and abs(x) - log(2) for large x.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute log(cosh(pred - target))"""
        diff = predictions - targets
        loss = torch.log(torch.cosh(diff + 1e-8))
        return self._reduce(loss)


class QuantileLoss(DataLoss):
    """
    Quantile loss (pinball loss).

    Used for quantile regression - predicting specific percentiles.
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean"):
        """
        Args:
            quantile: Target quantile (0.5 = median)
            reduction: Reduction method
        """
        super().__init__(reduction)
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1")
        self.quantile = quantile

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute pinball loss."""
        diff = targets - predictions
        loss = torch.where(
            diff >= 0,
            self.quantile * diff,
            (self.quantile - 1) * diff
        )
        return self._reduce(loss)


class DirectionalLoss(DataLoss):
    """
    Directional accuracy loss.

    Penalizes incorrect direction predictions more than magnitude errors.
    Useful for trading where direction matters more than exact value.
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Args:
            direction_weight: Weight for directional penalty
            magnitude_weight: Weight for magnitude error
            reduction: Reduction method
        """
        super().__init__(reduction)
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute directional loss."""
        # Direction agreement (1 if same sign, 0 otherwise)
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        direction_match = (pred_sign == target_sign).float()

        # Magnitude error (MSE)
        magnitude_loss = (predictions - targets) ** 2

        # Combined loss: penalize wrong direction more heavily
        # If direction is wrong, add penalty proportional to magnitude
        direction_penalty = (1 - direction_match) * torch.abs(predictions - targets)

        loss = (
            self.magnitude_weight * magnitude_loss
            + self.direction_weight * direction_penalty
        )
        return self._reduce(loss)


class WeightedMSELoss(DataLoss):
    """
    MSE with sample-specific weights.

    Useful for weighting recent samples more heavily
    or downweighting during volatile periods.
    """

    def __init__(
        self,
        default_weight: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            default_weight: Default weight when none provided
            reduction: Reduction method
        """
        super().__init__(reduction)
        self.default_weight = default_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MSE.

        Args:
            predictions: Model predictions
            targets: Ground truth
            weights: Optional per-sample weights
        """
        squared_error = (predictions - targets) ** 2

        if weights is None:
            weights = torch.full_like(squared_error, self.default_weight)

        weighted_loss = squared_error * weights

        if self.reduction == "mean":
            # Normalize by sum of weights
            return weighted_loss.sum() / (weights.sum() + 1e-8)
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class AsymmetricLoss(DataLoss):
    """
    Asymmetric loss that penalizes over/under-prediction differently.

    Useful when the cost of over-predicting differs from under-predicting.
    """

    def __init__(
        self,
        alpha_under: float = 1.0,
        alpha_over: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            alpha_under: Weight for under-predictions (pred < target)
            alpha_over: Weight for over-predictions (pred > target)
            reduction: Reduction method
        """
        super().__init__(reduction)
        self.alpha_under = alpha_under
        self.alpha_over = alpha_over

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute asymmetric loss."""
        diff = predictions - targets
        loss = torch.where(
            diff < 0,
            self.alpha_under * torch.abs(diff),
            self.alpha_over * torch.abs(diff)
        )
        return self._reduce(loss)


class FocalMSELoss(DataLoss):
    """
    Focal-weighted MSE loss.

    Down-weights easy samples (small errors) and focuses on hard samples.
    Inspired by Focal Loss for classification.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard samples)
            reduction: Reduction method
        """
        super().__init__(reduction)
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal MSE."""
        squared_error = (predictions - targets) ** 2

        # Normalize errors to [0, 1] for weighting
        max_error = squared_error.max() + 1e-8
        normalized_error = squared_error / max_error

        # Focal weight: (error^gamma) gives more weight to hard samples
        focal_weight = normalized_error ** self.gamma

        loss = squared_error * (1 + focal_weight)
        return self._reduce(loss)


def create_data_loss(
    loss_type: str,
    **kwargs
) -> DataLoss:
    """
    Factory function to create data loss by name.

    Args:
        loss_type: Type of loss ('mse', 'mae', 'huber', 'logcosh',
                   'quantile', 'directional', 'weighted_mse', 'focal')
        **kwargs: Additional arguments for the loss

    Returns:
        DataLoss instance

    Example:
        loss = create_data_loss('huber', delta=0.5)
        loss = create_data_loss('quantile', quantile=0.9)
    """
    loss_map = {
        'mse': MSELoss,
        'mae': MAELoss,
        'huber': HuberLoss,
        'logcosh': LogCoshLoss,
        'log_cosh': LogCoshLoss,
        'quantile': QuantileLoss,
        'pinball': QuantileLoss,
        'directional': DirectionalLoss,
        'weighted_mse': WeightedMSELoss,
        'asymmetric': AsymmetricLoss,
        'focal': FocalMSELoss,
        'focal_mse': FocalMSELoss,
    }

    loss_type = loss_type.lower()
    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_map.keys())}"
        )

    return loss_map[loss_type](**kwargs)
