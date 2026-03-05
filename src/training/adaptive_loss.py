"""
Adaptive Loss Weighting for PINN Training

Implements automatic loss balancing methods:
1. GradNorm - Gradient normalization (Chen et al., 2018)
2. Uncertainty Weighting - Homoscedastic uncertainty (Kendall et al., 2018)
3. Residual-based Reweighting - Based on PDE residual magnitudes

These methods address the optimization imbalance problem in multi-task
learning where different loss terms have vastly different gradient magnitudes.

References:
    - Chen et al. (2018). "GradNorm: Gradient Normalization for Adaptive
      Loss Balancing in Deep Multitask Networks"
    - Kendall et al. (2018). "Multi-Task Learning Using Uncertainty to
      Weigh Losses for Scene Geometry and Semantics"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WeightingMethod(Enum):
    """Available loss weighting methods"""
    NONE = "none"
    GRADNORM = "gradnorm"
    UNCERTAINTY = "uncertainty"
    RESIDUAL = "residual"
    SOFTADAPT = "softadapt"


@dataclass
class WeightHistory:
    """History of weight evolution"""
    steps: List[int]
    weights: Dict[str, List[float]]
    losses: Dict[str, List[float]]


class GradNormWeighter(nn.Module):
    """
    GradNorm: Adaptive loss weighting based on gradient magnitudes.

    Automatically balances gradients from different loss terms by
    adjusting their weights to achieve similar gradient norms.

    Reference:
        Chen et al. (2018). "GradNorm: Gradient Normalization for
        Adaptive Loss Balancing in Deep Multitask Networks"
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        lr: float = 0.025,
        initial_weights: Optional[List[float]] = None
    ):
        """
        Initialize GradNorm weighter.

        Args:
            num_tasks: Number of loss terms to balance
            alpha: Asymmetry parameter (higher = more aggressive balancing)
            lr: Learning rate for weight updates
            initial_weights: Initial weight values (default: all 1.0)
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = lr

        # Learnable weights (log scale for numerical stability)
        if initial_weights is not None:
            init_log_weights = torch.log(torch.tensor(initial_weights, dtype=torch.float32))
        else:
            init_log_weights = torch.zeros(num_tasks)

        self.log_weights = nn.Parameter(init_log_weights)

        # Track initial loss ratios for relative training rate
        self.initial_losses: Optional[torch.Tensor] = None
        self.step_count = 0

        # History for analysis
        self.weight_history: List[Dict[str, float]] = []
        self.loss_history: List[Dict[str, float]] = []

    @property
    def weights(self) -> torch.Tensor:
        """Get current weights (exponentiated for positivity)"""
        return torch.exp(self.log_weights)

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: List[nn.Parameter],
        task_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss and update weights.

        Args:
            losses: Dict of loss tensors
            shared_params: Shared model parameters for gradient computation
            task_names: Optional names for tasks (for logging)

        Returns:
            Tuple of (weighted_total_loss, weight_dict)
        """
        if task_names is None:
            task_names = list(losses.keys())

        loss_tensors = [losses[name] for name in task_names]
        loss_values = torch.stack(loss_tensors)

        # Initialize reference losses on first call
        if self.initial_losses is None:
            self.initial_losses = loss_values.detach().clone()
            logger.info(f"GradNorm initialized with losses: {[f'{l.item():.6f}' for l in loss_values]}")

        # Current weights
        weights = self.weights

        # Weighted losses
        weighted_losses = weights * loss_values
        total_loss = weighted_losses.sum()

        # ========== GRADNORM UPDATE ==========
        if self.training:
            self._update_weights(loss_values, shared_params, task_names)

        self.step_count += 1

        # Record history
        weight_dict = {name: weights[i].item() for i, name in enumerate(task_names)}
        loss_dict = {name: loss_values[i].item() for i, name in enumerate(task_names)}
        self.weight_history.append(weight_dict)
        self.loss_history.append(loss_dict)

        return total_loss, weight_dict

    def _update_weights(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
        task_names: List[str]
    ):
        """Update weights using GradNorm algorithm"""
        weights = self.weights

        # Compute gradient norms for each task
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute gradient of weighted loss
            weighted_loss = weights[i] * loss

            grads = torch.autograd.grad(
                weighted_loss,
                shared_params,
                retain_graph=True,
                allow_unused=True
            )

            # Total gradient norm
            total_norm = 0.0
            for g in grads:
                if g is not None:
                    total_norm += g.norm(2) ** 2
            grad_norms.append(torch.sqrt(total_norm))

        grad_norms = torch.stack(grad_norms)

        # Average gradient norm
        avg_grad_norm = grad_norms.mean()

        # Relative training rates (how fast each task is learning)
        loss_ratios = losses / (self.initial_losses + 1e-8)
        avg_loss_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (avg_loss_ratio + 1e-8)

        # Target gradient norms
        target_norms = avg_grad_norm * (relative_rates ** self.alpha)

        # Gradient norm loss (what we want to minimize)
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_norms))

        # Update weights
        self.log_weights.grad = torch.autograd.grad(
            gradnorm_loss,
            self.log_weights,
            retain_graph=True
        )[0]

        with torch.no_grad():
            self.log_weights -= self.lr * self.log_weights.grad

            # Renormalize weights to sum to num_tasks
            normalized_weights = self.weights * self.num_tasks / self.weights.sum()
            self.log_weights.data = torch.log(normalized_weights)

    def get_weight_history(self) -> WeightHistory:
        """Get weight evolution history"""
        steps = list(range(len(self.weight_history)))
        weights = {}
        losses = {}

        if self.weight_history:
            for key in self.weight_history[0].keys():
                weights[key] = [w[key] for w in self.weight_history]

        if self.loss_history:
            for key in self.loss_history[0].keys():
                losses[key] = [l[key] for l in self.loss_history]

        return WeightHistory(steps=steps, weights=weights, losses=losses)


class UncertaintyWeighter(nn.Module):
    """
    Uncertainty Weighting: Learn task weights via homoscedastic uncertainty.

    Each task has a learnable log-variance parameter that automatically
    balances the losses based on learned uncertainty.

    Reference:
        Kendall et al. (2018). "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(
        self,
        num_tasks: int,
        initial_log_vars: Optional[List[float]] = None
    ):
        """
        Initialize uncertainty weighter.

        Args:
            num_tasks: Number of loss terms
            initial_log_vars: Initial log variance values (default: 0)
        """
        super().__init__()

        self.num_tasks = num_tasks

        if initial_log_vars is not None:
            init_vals = torch.tensor(initial_log_vars, dtype=torch.float32)
        else:
            init_vals = torch.zeros(num_tasks)

        # Learnable log variances (log σ²)
        self.log_vars = nn.Parameter(init_vals)

        # History
        self.weight_history: List[Dict[str, float]] = []

    @property
    def precisions(self) -> torch.Tensor:
        """Get precisions (1/σ²)"""
        return torch.exp(-self.log_vars)

    @property
    def weights(self) -> torch.Tensor:
        """Get effective weights"""
        return self.precisions

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        task_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted loss.

        Loss = Σ (precision_i * loss_i + log_var_i)

        Args:
            losses: Dict of loss tensors
            task_names: Optional task names

        Returns:
            Tuple of (total_loss, weight_dict)
        """
        if task_names is None:
            task_names = list(losses.keys())

        loss_tensors = [losses[name] for name in task_names]

        total_loss = torch.tensor(0.0, device=loss_tensors[0].device)

        for i, loss in enumerate(loss_tensors):
            # precision * loss + log_variance (regularization)
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss + self.log_vars[i]

        weight_dict = {name: self.weights[i].item() for i, name in enumerate(task_names)}
        self.weight_history.append(weight_dict)

        return total_loss, weight_dict

    def get_uncertainties(self, task_names: List[str]) -> Dict[str, float]:
        """Get learned uncertainties (σ)"""
        return {name: np.exp(0.5 * self.log_vars[i].item())
                for i, name in enumerate(task_names)}


class ResidualWeighter(nn.Module):
    """
    Residual-based Weighting: Adjust weights based on PDE residual magnitudes.

    Tasks with larger residuals (less satisfied constraints) receive higher
    weights to encourage faster convergence.
    """

    def __init__(
        self,
        num_tasks: int,
        momentum: float = 0.9,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        """
        Initialize residual weighter.

        Args:
            num_tasks: Number of loss terms
            momentum: EMA momentum for residual tracking
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Running averages of residuals
        self.register_buffer('running_residuals', torch.ones(num_tasks))

        # History
        self.weight_history: List[Dict[str, float]] = []

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        residuals: Optional[Dict[str, torch.Tensor]] = None,
        task_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute residual-weighted loss.

        Args:
            losses: Dict of loss tensors
            residuals: Dict of residual tensors (optional, uses losses if not provided)
            task_names: Optional task names

        Returns:
            Tuple of (total_loss, weight_dict)
        """
        if task_names is None:
            task_names = list(losses.keys())

        loss_tensors = [losses[name] for name in task_names]

        # Use losses as residuals if not provided
        if residuals is None:
            residual_values = torch.stack([l.detach() for l in loss_tensors])
        else:
            residual_values = torch.stack([residuals.get(name, losses[name]).detach()
                                          for name in task_names])

        # Update running averages
        self.running_residuals = (self.momentum * self.running_residuals +
                                  (1 - self.momentum) * residual_values)

        # Compute weights (higher residual = higher weight)
        avg_residual = self.running_residuals.mean()
        weights = self.running_residuals / (avg_residual + 1e-8)

        # Clamp weights
        weights = torch.clamp(weights, self.min_weight, self.max_weight)

        # Weighted sum
        total_loss = torch.tensor(0.0, device=loss_tensors[0].device)
        for i, loss in enumerate(loss_tensors):
            total_loss = total_loss + weights[i] * loss

        weight_dict = {name: weights[i].item() for i, name in enumerate(task_names)}
        self.weight_history.append(weight_dict)

        return total_loss, weight_dict


class SoftAdaptWeighter(nn.Module):
    """
    SoftAdapt: Soft attention over loss changes.

    Weights tasks based on how much their losses are changing,
    giving more attention to tasks that are struggling.
    """

    def __init__(
        self,
        num_tasks: int,
        beta: float = 0.1,
        normalize: bool = True
    ):
        """
        Initialize SoftAdapt weighter.

        Args:
            num_tasks: Number of tasks
            beta: Temperature parameter for softmax
            normalize: Whether to normalize weights
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.beta = beta
        self.normalize = normalize

        # Previous losses for rate computation
        self.register_buffer('prev_losses', torch.zeros(num_tasks))
        self.step = 0

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        task_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SoftAdapt-weighted loss"""
        if task_names is None:
            task_names = list(losses.keys())

        loss_values = torch.stack([losses[name] for name in task_names])

        if self.step == 0:
            # First step: equal weights
            weights = torch.ones_like(loss_values)
        else:
            # Compute loss changes
            loss_changes = loss_values.detach() - self.prev_losses

            # Softmax over changes (higher change = lower weight)
            weights = torch.softmax(-loss_changes / self.beta, dim=0)

            if self.normalize:
                weights = weights * self.num_tasks

        # Update state
        self.prev_losses = loss_values.detach().clone()
        self.step += 1

        # Weighted sum
        total_loss = (weights * loss_values).sum()

        weight_dict = {name: weights[i].item() for i, name in enumerate(task_names)}

        return total_loss, weight_dict


class AdaptiveLossWeighter(nn.Module):
    """
    Unified interface for adaptive loss weighting.

    Supports multiple weighting methods with a consistent API.
    """

    def __init__(
        self,
        method: str = "gradnorm",
        num_tasks: int = 4,
        **kwargs
    ):
        """
        Initialize adaptive loss weighter.

        Args:
            method: Weighting method ('none', 'gradnorm', 'uncertainty', 'residual', 'softadapt')
            num_tasks: Number of loss terms
            **kwargs: Method-specific arguments
        """
        super().__init__()

        self.method = WeightingMethod(method.lower())
        self.num_tasks = num_tasks

        if self.method == WeightingMethod.GRADNORM:
            self.weighter = GradNormWeighter(
                num_tasks=num_tasks,
                alpha=kwargs.get('alpha', 1.5),
                lr=kwargs.get('lr', 0.025)
            )
        elif self.method == WeightingMethod.UNCERTAINTY:
            self.weighter = UncertaintyWeighter(num_tasks=num_tasks)
        elif self.method == WeightingMethod.RESIDUAL:
            self.weighter = ResidualWeighter(
                num_tasks=num_tasks,
                momentum=kwargs.get('momentum', 0.9)
            )
        elif self.method == WeightingMethod.SOFTADAPT:
            self.weighter = SoftAdaptWeighter(
                num_tasks=num_tasks,
                beta=kwargs.get('beta', 0.1)
            )
        else:
            self.weighter = None

        logger.info(f"Initialized AdaptiveLossWeighter with method={method}")

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[nn.Parameter]] = None,
        residuals: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adaptively weighted loss.

        Args:
            losses: Dict of loss tensors
            shared_params: Shared parameters (for GradNorm)
            residuals: PDE residuals (for residual-based)

        Returns:
            Tuple of (total_loss, weight_dict)
        """
        task_names = list(losses.keys())

        if self.method == WeightingMethod.NONE:
            total_loss = sum(losses.values())
            weight_dict = {name: 1.0 for name in task_names}
            return total_loss, weight_dict

        if self.method == WeightingMethod.GRADNORM:
            if shared_params is None:
                raise ValueError("GradNorm requires shared_params")
            return self.weighter(losses, shared_params, task_names)

        elif self.method == WeightingMethod.RESIDUAL:
            return self.weighter(losses, residuals, task_names)

        else:
            return self.weighter(losses, task_names)

    def get_weights(self) -> torch.Tensor:
        """Get current weights"""
        if self.weighter is not None and hasattr(self.weighter, 'weights'):
            return self.weighter.weights
        return torch.ones(self.num_tasks)

    def get_history(self) -> List[Dict[str, float]]:
        """Get weight history"""
        if self.weighter is not None and hasattr(self.weighter, 'weight_history'):
            return self.weighter.weight_history
        return []


# Convenience function
def create_adaptive_weighter(
    method: str,
    loss_names: List[str],
    **kwargs
) -> AdaptiveLossWeighter:
    """
    Create an adaptive loss weighter.

    Args:
        method: Weighting method
        loss_names: Names of loss terms
        **kwargs: Method-specific arguments

    Returns:
        AdaptiveLossWeighter instance
    """
    return AdaptiveLossWeighter(
        method=method,
        num_tasks=len(loss_names),
        **kwargs
    )
