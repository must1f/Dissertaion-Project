"""
Composite Loss Functions

Combines data losses and physics losses with configurable weighting.
Supports static, curriculum-based, and adaptive (GradNorm) weighting.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class WeightingStrategy(Enum):
    """Strategy for combining multiple losses."""
    STATIC = "static"          # Fixed weights
    CURRICULUM = "curriculum"  # Increasing physics weight over epochs
    GRADNORM = "gradnorm"      # GradNorm adaptive weighting
    UNCERTAINTY = "uncertainty"  # Uncertainty-based weighting


@dataclass
class LossWeight:
    """Configuration for a single loss weight."""
    name: str
    initial_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 10.0
    learnable: bool = False

    def __post_init__(self):
        if self.initial_weight < self.min_weight:
            raise ValueError(f"Initial weight {self.initial_weight} < min {self.min_weight}")
        if self.initial_weight > self.max_weight:
            raise ValueError(f"Initial weight {self.initial_weight} > max {self.max_weight}")


@dataclass
class LossConfig:
    """
    Configuration for composite loss.

    Example:
        config = LossConfig(
            data_loss='mse',
            physics_losses=['gbm', 'ou'],
            weights={
                'data': LossWeight('data', 1.0),
                'gbm': LossWeight('gbm', 0.1),
                'ou': LossWeight('ou', 0.1),
            },
            weighting_strategy=WeightingStrategy.CURRICULUM,
            curriculum_warmup_epochs=10,
            curriculum_ramp_epochs=20
        )
    """
    data_loss: str = "mse"
    physics_losses: List[str] = field(default_factory=list)
    weights: Dict[str, LossWeight] = field(default_factory=dict)
    weighting_strategy: WeightingStrategy = WeightingStrategy.STATIC

    # Curriculum learning settings
    curriculum_warmup_epochs: int = 10   # Epochs with data loss only
    curriculum_ramp_epochs: int = 20     # Epochs to ramp up physics
    curriculum_final_physics_scale: float = 1.0

    # GradNorm settings
    gradnorm_alpha: float = 1.5  # Restoring force strength

    def __post_init__(self):
        # Set default weights if not provided
        if 'data' not in self.weights:
            self.weights['data'] = LossWeight('data', 1.0)

        for physics_loss in self.physics_losses:
            if physics_loss not in self.weights:
                self.weights[physics_loss] = LossWeight(physics_loss, 0.1)


class CompositeLoss(nn.Module):
    """
    Composite loss combining data and physics losses.

    Supports multiple physics constraints with configurable weights.

    Example:
        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={
                'gbm': GBMResidual(weight=0.1),
                'ou': OUResidual(weight=0.1),
            }
        )

        total_loss, loss_dict = loss_fn(
            predictions=preds,
            targets=targets,
            physics_inputs={
                'gbm': {'prices': prices},
                'ou': {'values': returns}
            }
        )
    """

    def __init__(
        self,
        data_loss: nn.Module,
        physics_losses: Optional[Dict[str, nn.Module]] = None,
        data_weight: float = 1.0,
        enable_physics: bool = True
    ):
        """
        Args:
            data_loss: Primary data loss module
            physics_losses: Dict of physics loss modules
            data_weight: Weight for data loss
            enable_physics: Whether to compute physics losses
        """
        super().__init__()

        self.data_loss = data_loss
        self.physics_losses = nn.ModuleDict(physics_losses or {})
        self.data_weight = data_weight
        self.enable_physics = enable_physics

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_physics: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            physics_inputs: Dict mapping physics loss names to their inputs
            enable_physics: Override for physics computation

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Data loss
        data_loss_value = self.data_loss(predictions, targets)
        total_loss = self.data_weight * data_loss_value

        loss_dict = {
            'data_loss': data_loss_value.item(),
            'total_loss': total_loss.item()
        }

        # Physics losses
        use_physics = self.enable_physics if enable_physics is None else enable_physics

        if use_physics and physics_inputs:
            for name, physics_loss in self.physics_losses.items():
                if name in physics_inputs:
                    try:
                        inputs = physics_inputs[name]
                        physics_value = physics_loss(**inputs)
                        total_loss = total_loss + physics_value
                        loss_dict[f'{name}_loss'] = physics_value.item()
                    except Exception as e:
                        loss_dict[f'{name}_error'] = str(e)

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_physics_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get learned physics parameters."""
        params = {}
        for name, loss in self.physics_losses.items():
            loss_params = {}
            if hasattr(loss, 'theta'):
                loss_params['theta'] = loss.theta.item()
            if hasattr(loss, 'gamma'):
                loss_params['gamma'] = loss.gamma.item()
            if hasattr(loss, 'temperature'):
                loss_params['temperature'] = loss.temperature.item()
            if hasattr(loss, 'alpha'):
                loss_params['alpha'] = loss.alpha.item()
            if loss_params:
                params[name] = loss_params
        return params


class AdaptiveCompositeLoss(nn.Module):
    """
    Composite loss with adaptive weight learning.

    Implements GradNorm and curriculum learning strategies
    for automatic balancing of multiple loss terms.

    Reference:
        Chen et al., "GradNorm: Gradient Normalization for Adaptive
        Loss Balancing in Deep Multitask Networks", ICML 2018
    """

    def __init__(
        self,
        data_loss: nn.Module,
        physics_losses: Dict[str, nn.Module],
        config: Optional[LossConfig] = None
    ):
        """
        Args:
            data_loss: Primary data loss module
            physics_losses: Dict of physics loss modules
            config: Loss configuration
        """
        super().__init__()

        self.data_loss = data_loss
        self.physics_losses = nn.ModuleDict(physics_losses)
        self.config = config or LossConfig()

        # Number of tasks (data + physics)
        self.n_tasks = 1 + len(physics_losses)

        # Learnable weights for GradNorm
        if self.config.weighting_strategy == WeightingStrategy.GRADNORM:
            self.log_weights = nn.Parameter(
                torch.zeros(self.n_tasks)
            )
        else:
            self.register_buffer('log_weights', torch.zeros(self.n_tasks))

        # Task names in order
        self.task_names = ['data'] + list(physics_losses.keys())

        # Initial loss values for GradNorm normalization
        self.register_buffer('initial_losses', torch.zeros(self.n_tasks))
        self._initialized = False

        # Training state
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch

    def get_curriculum_scale(self) -> float:
        """Get physics weight scale based on curriculum."""
        warmup = self.config.curriculum_warmup_epochs
        ramp = self.config.curriculum_ramp_epochs
        final = self.config.curriculum_final_physics_scale

        if self.current_epoch < warmup:
            # Warmup phase: no physics
            return 0.0
        elif self.current_epoch < warmup + ramp:
            # Ramp phase: linear increase
            progress = (self.current_epoch - warmup) / ramp
            return progress * final
        else:
            # Full physics
            return final

    def get_weights(self) -> torch.Tensor:
        """Get current task weights."""
        if self.config.weighting_strategy == WeightingStrategy.GRADNORM:
            # Softmax normalization to ensure positive weights
            weights = torch.nn.functional.softmax(self.log_weights, dim=0)
            weights = weights * self.n_tasks  # Scale so mean is 1
        elif self.config.weighting_strategy == WeightingStrategy.CURRICULUM:
            physics_scale = self.get_curriculum_scale()
            weights = torch.ones(self.n_tasks, device=self.log_weights.device)
            weights[1:] = physics_scale  # Scale physics losses
        else:
            weights = torch.ones(self.n_tasks, device=self.log_weights.device)

        return weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        shared_params: Optional[List[nn.Parameter]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adaptive composite loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            physics_inputs: Dict mapping physics loss names to their inputs
            shared_params: Shared model parameters for GradNorm

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        physics_inputs = physics_inputs or {}
        losses = []

        # Data loss
        data_loss = self.data_loss(predictions, targets)
        losses.append(data_loss)

        # Physics losses
        for name in self.task_names[1:]:
            if name in self.physics_losses and name in physics_inputs:
                try:
                    physics_loss = self.physics_losses[name]
                    value = physics_loss(**physics_inputs[name])
                    losses.append(value)
                except Exception:
                    losses.append(torch.tensor(0.0, device=predictions.device))
            else:
                losses.append(torch.tensor(0.0, device=predictions.device))

        losses = torch.stack(losses)

        # Initialize reference losses for GradNorm
        if not self._initialized and losses[0].item() > 0:
            self.initial_losses.copy_(losses.detach())
            self._initialized = True

        # Get current weights
        weights = self.get_weights()

        # Weighted sum
        weighted_losses = weights * losses
        total_loss = weighted_losses.sum()

        # Build loss dict
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': losses[0].item()
        }
        for i, name in enumerate(self.task_names[1:], 1):
            loss_dict[f'{name}_loss'] = losses[i].item()
            loss_dict[f'{name}_weight'] = weights[i].item()

        return total_loss, loss_dict

    def gradnorm_update(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
        optimizer: torch.optim.Optimizer
    ):
        """
        Update weights using GradNorm algorithm.

        Should be called after backward() but before optimizer.step()

        Args:
            losses: Individual task losses
            shared_params: Parameters shared across tasks
            optimizer: Optimizer for weight updates
        """
        if self.config.weighting_strategy != WeightingStrategy.GRADNORM:
            return

        alpha = self.config.gradnorm_alpha

        # Compute gradient norms for each task
        grad_norms = []
        weights = self.get_weights()

        for i, loss in enumerate(losses):
            # Get gradients w.r.t. shared params
            if loss.requires_grad:
                grads = torch.autograd.grad(
                    weights[i] * loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True
                )
                grad_norm = torch.stack([
                    g.norm() for g in grads if g is not None
                ]).sum()
                grad_norms.append(grad_norm)
            else:
                grad_norms.append(torch.tensor(0.0, device=loss.device))

        grad_norms = torch.stack(grad_norms)

        # Mean gradient norm
        mean_norm = grad_norms.mean()

        # Relative inverse training rate
        if self._initialized:
            relative_loss = losses / (self.initial_losses + 1e-8)
            mean_relative_loss = relative_loss.mean()
            inverse_train_rate = relative_loss / (mean_relative_loss + 1e-8)
        else:
            inverse_train_rate = torch.ones_like(losses)

        # Target gradient norm
        target_grad_norm = mean_norm * (inverse_train_rate ** alpha)

        # GradNorm loss for weight update
        gradnorm_loss = torch.abs(grad_norms - target_grad_norm).sum()

        # Update log_weights
        self.log_weights.grad = torch.autograd.grad(
            gradnorm_loss,
            self.log_weights,
            retain_graph=True
        )[0]


def create_composite_loss(
    data_loss_type: str = "mse",
    physics_loss_types: Optional[List[str]] = None,
    config: Optional[LossConfig] = None,
    **kwargs
) -> Union[CompositeLoss, AdaptiveCompositeLoss]:
    """
    Factory function to create composite loss.

    Args:
        data_loss_type: Type of data loss
        physics_loss_types: List of physics loss types
        config: Optional loss configuration
        **kwargs: Additional arguments

    Returns:
        CompositeLoss or AdaptiveCompositeLoss

    Example:
        loss_fn = create_composite_loss(
            data_loss_type='mse',
            physics_loss_types=['gbm', 'ou'],
            config=LossConfig(weighting_strategy=WeightingStrategy.CURRICULUM)
        )
    """
    from .data_losses import create_data_loss
    from .physics_losses import create_physics_loss

    # Create data loss
    data_loss = create_data_loss(data_loss_type)

    # Create physics losses
    physics_losses = {}
    if physics_loss_types:
        for loss_type in physics_loss_types:
            physics_losses[loss_type] = create_physics_loss(loss_type)

    # Choose composite type based on config
    if config and config.weighting_strategy in [
        WeightingStrategy.GRADNORM,
        WeightingStrategy.CURRICULUM
    ]:
        return AdaptiveCompositeLoss(
            data_loss=data_loss,
            physics_losses=physics_losses,
            config=config
        )
    else:
        return CompositeLoss(
            data_loss=data_loss,
            physics_losses=physics_losses
        )
