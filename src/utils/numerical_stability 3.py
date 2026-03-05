"""
Numerical Stability Utilities for PINN Training

Provides safe mathematical operations and gradient handling:
- Safe log/exp/div to prevent NaN/Inf
- Gradient clipping and normalization
- Input/output normalization
- Mixed precision helpers

These utilities are critical for stable PINN training where
physics losses can produce extreme gradient values.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


# ========== SAFE MATHEMATICAL OPERATIONS ==========

def safe_log(x: torch.Tensor, eps: float = 1e-8, min_val: float = -100.0) -> torch.Tensor:
    """
    Safe logarithm that prevents NaN/Inf.

    Args:
        x: Input tensor
        eps: Small constant to prevent log(0)
        min_val: Minimum output value to prevent -Inf

    Returns:
        Clamped log(x)
    """
    x_safe = torch.clamp(x, min=eps)
    result = torch.log(x_safe)
    return torch.clamp(result, min=min_val)


def safe_exp(x: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
    """
    Safe exponential that prevents overflow.

    Args:
        x: Input tensor
        max_val: Maximum input value to prevent overflow

    Returns:
        Clamped exp(x)
    """
    x_clamped = torch.clamp(x, max=max_val)
    return torch.exp(x_clamped)


def safe_div(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Safe division that prevents division by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small constant added to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe square root that prevents NaN for negative inputs.

    Args:
        x: Input tensor
        eps: Small constant to prevent sqrt(0)

    Returns:
        sqrt(clamp(x, min=eps))
    """
    return torch.sqrt(torch.clamp(x, min=eps))


def safe_pow(
    base: torch.Tensor,
    exponent: Union[float, torch.Tensor],
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Safe power that handles negative bases.

    Args:
        base: Base tensor
        exponent: Exponent
        eps: Small constant

    Returns:
        Safe power computation
    """
    # For fractional exponents, ensure base is positive
    if isinstance(exponent, float) and exponent != int(exponent):
        base = torch.clamp(base.abs(), min=eps)
    return torch.pow(base + eps, exponent)


def safe_softmax(x: torch.Tensor, dim: int = -1, temp: float = 1.0) -> torch.Tensor:
    """
    Numerically stable softmax with temperature scaling.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        temp: Temperature parameter

    Returns:
        Softmax output
    """
    x_scaled = x / temp
    x_max = x_scaled.max(dim=dim, keepdim=True)[0]
    x_shifted = x_scaled - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# ========== GRADIENT UTILITIES ==========

@dataclass
class GradientStats:
    """Statistics about gradients"""
    total_norm: float
    max_norm: float
    min_norm: float
    mean_norm: float
    n_params: int
    n_nan: int
    n_inf: int


def compute_gradient_stats(model: nn.Module) -> GradientStats:
    """
    Compute statistics about model gradients.

    Args:
        model: PyTorch model

    Returns:
        GradientStats with gradient information
    """
    norms = []
    n_nan = 0
    n_inf = 0

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            norm = grad.norm(2).item()
            norms.append(norm)

            n_nan += torch.isnan(grad).sum().item()
            n_inf += torch.isinf(grad).sum().item()

    if not norms:
        return GradientStats(
            total_norm=0.0,
            max_norm=0.0,
            min_norm=0.0,
            mean_norm=0.0,
            n_params=0,
            n_nan=0,
            n_inf=0
        )

    total_norm = np.sqrt(sum(n ** 2 for n in norms))

    return GradientStats(
        total_norm=total_norm,
        max_norm=max(norms),
        min_norm=min(norms),
        mean_norm=np.mean(norms),
        n_params=len(norms),
        n_nan=n_nan,
        n_inf=n_inf
    )


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: L2)
        error_if_nonfinite: Whether to raise error on NaN/Inf

    Returns:
        Total gradient norm before clipping
    """
    params = [p for p in model.parameters() if p.grad is not None]

    if len(params) == 0:
        return 0.0

    # Check for NaN/Inf
    for p in params:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            if error_if_nonfinite:
                raise RuntimeError("Non-finite gradients detected")
            else:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                logger.warning("Non-finite gradients replaced with finite values")

    total_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm, norm_type, error_if_nonfinite=False
    )

    return total_norm.item()


def scale_gradients(
    model: nn.Module,
    scale: float
) -> None:
    """
    Scale all gradients by a factor.

    Args:
        model: PyTorch model
        scale: Scale factor
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scale)


def zero_nan_gradients(model: nn.Module) -> int:
    """
    Replace NaN gradients with zeros.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters with NaN gradients
    """
    n_nan = 0
    for p in model.parameters():
        if p.grad is not None:
            nan_mask = torch.isnan(p.grad)
            if nan_mask.any():
                p.grad[nan_mask] = 0.0
                n_nan += 1

    if n_nan > 0:
        logger.warning(f"Zeroed NaN gradients in {n_nan} parameters")

    return n_nan


# ========== INPUT/OUTPUT NORMALIZATION ==========

class RobustNormalizer:
    """
    Robust normalizer that handles outliers and missing values.

    Uses median and IQR for robust statistics.
    """

    def __init__(
        self,
        method: str = 'standard',
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        handle_nan: str = 'zero'
    ):
        """
        Initialize normalizer.

        Args:
            method: 'standard' (z-score) or 'robust' (median/IQR)
            clip_range: Output clipping range
            handle_nan: How to handle NaN ('zero', 'mean', 'raise')
        """
        self.method = method
        self.clip_range = clip_range
        self.handle_nan = handle_nan

        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.iqr_ = None

    def fit(self, X: torch.Tensor) -> 'RobustNormalizer':
        """Fit normalizer to data"""
        X_clean = self._handle_nan(X)

        if self.method == 'standard':
            self.mean_ = X_clean.mean(dim=0)
            self.std_ = X_clean.std(dim=0) + 1e-8
        elif self.method == 'robust':
            self.median_ = X_clean.median(dim=0)[0]
            q75 = torch.quantile(X_clean, 0.75, dim=0)
            q25 = torch.quantile(X_clean, 0.25, dim=0)
            self.iqr_ = (q75 - q25) + 1e-8

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data"""
        X_clean = self._handle_nan(X)

        if self.method == 'standard':
            X_norm = (X_clean - self.mean_) / self.std_
        elif self.method == 'robust':
            X_norm = (X_clean - self.median_) / self.iqr_

        # Clip to range
        X_norm = torch.clamp(X_norm, self.clip_range[0], self.clip_range[1])

        return X_norm

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform"""
        if self.method == 'standard':
            return X_norm * self.std_ + self.mean_
        elif self.method == 'robust':
            return X_norm * self.iqr_ + self.median_

    def _handle_nan(self, X: torch.Tensor) -> torch.Tensor:
        """Handle NaN values"""
        if not torch.isnan(X).any():
            return X

        if self.handle_nan == 'raise':
            raise ValueError("NaN values in input")
        elif self.handle_nan == 'zero':
            return torch.nan_to_num(X, nan=0.0)
        elif self.handle_nan == 'mean':
            col_means = torch.nanmean(X, dim=0)
            X_filled = X.clone()
            for i in range(X.shape[1]):
                nan_mask = torch.isnan(X[:, i])
                X_filled[nan_mask, i] = col_means[i]
            return X_filled


# ========== STABILITY CHECKS ==========

def check_tensor_health(
    tensor: torch.Tensor,
    name: str = "tensor"
) -> Dict[str, bool]:
    """
    Check tensor for numerical issues.

    Args:
        tensor: Tensor to check
        name: Name for logging

    Returns:
        Dict with health status
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    has_large = (tensor.abs() > 1e10).any().item()

    if has_nan:
        logger.warning(f"{name}: Contains NaN values")
    if has_inf:
        logger.warning(f"{name}: Contains Inf values")
    if has_large:
        logger.warning(f"{name}: Contains very large values (>1e10)")

    return {
        'healthy': not (has_nan or has_inf),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'has_large': has_large
    }


def check_loss_health(
    loss: torch.Tensor,
    max_loss: float = 1e6
) -> bool:
    """
    Check if loss is healthy (finite and reasonable).

    Args:
        loss: Loss tensor
        max_loss: Maximum acceptable loss value

    Returns:
        True if healthy
    """
    if torch.isnan(loss) or torch.isinf(loss):
        return False
    if loss.item() > max_loss:
        return False
    return True


# ========== MIXED PRECISION HELPERS ==========

class GradScalerWrapper:
    """
    Wrapper for gradient scaling in mixed precision training.

    Provides safe gradient scaling with automatic NaN handling.
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        """
        Initialize gradient scaler.

        Args:
            enabled: Whether to enable scaling
            init_scale: Initial scale factor
            growth_factor: Factor to increase scale
            backoff_factor: Factor to decrease scale on overflow
            growth_interval: Steps between scale increases
        """
        self.enabled = enabled

        if enabled and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Unscale and step optimizer"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def get_scale(self) -> float:
        """Get current scale factor"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


# ========== ACTIVATION FUNCTIONS WITH STABILITY ==========

def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid.

    Uses exp(-|x|) to avoid overflow.
    """
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(x))
    )


def stable_tanh(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """
    Numerically stable tanh.

    Clamps input to prevent overflow.
    """
    x_clamped = torch.clamp(x, -max_val, max_val)
    return torch.tanh(x_clamped)


def leaky_clamp(
    x: torch.Tensor,
    min_val: float,
    max_val: float,
    leak: float = 0.01
) -> torch.Tensor:
    """
    Soft clamping with leaky behavior at boundaries.

    Allows gradients to flow through clamped regions.
    """
    return torch.where(
        x < min_val,
        min_val + leak * (x - min_val),
        torch.where(
            x > max_val,
            max_val + leak * (x - max_val),
            x
        )
    )


# ========== LOSS MODIFICATION ==========

def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss) - more robust to outliers.

    Args:
        pred: Predictions
        target: Targets
        beta: Threshold for switching between L1 and L2

    Returns:
        Smooth L1 loss
    """
    diff = torch.abs(pred - target)
    return torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    ).mean()


def log_cosh_loss(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Log-cosh loss - smooth and robust to outliers.

    Similar to MSE for small errors, L1 for large errors.
    """
    diff = pred - target
    return torch.log(torch.cosh(diff)).mean()
