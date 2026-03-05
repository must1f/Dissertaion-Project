"""
Loss Diagnostics for PINN Training

Provides tools for monitoring and diagnosing PINN training:
- Gradient norm tracking per loss term
- Residual magnitude logging
- Loss imbalance detection
- Training stability analysis

These diagnostics help separate:
- PDE stiffness (problem property)
- Optimization imbalance (training issue)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GradientSnapshot:
    """Snapshot of gradient information at a training step"""
    step: int
    epoch: int
    grad_norms: Dict[str, float]
    loss_values: Dict[str, float]
    imbalance_ratio: float
    dominant_term: str


@dataclass
class ResidualSnapshot:
    """Snapshot of PDE residual information"""
    step: int
    epoch: int
    residual_means: Dict[str, float]
    residual_stds: Dict[str, float]
    residual_maxes: Dict[str, float]


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report for a training run"""
    total_steps: int
    total_epochs: int
    gradient_history: List[GradientSnapshot]
    residual_history: List[ResidualSnapshot]
    imbalance_warnings: List[str]
    stability_score: float  # 0-1, higher is more stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'gradient_history': [
                {
                    'step': g.step,
                    'epoch': g.epoch,
                    'grad_norms': g.grad_norms,
                    'loss_values': g.loss_values,
                    'imbalance_ratio': g.imbalance_ratio,
                    'dominant_term': g.dominant_term
                }
                for g in self.gradient_history
            ],
            'residual_history': [
                {
                    'step': r.step,
                    'epoch': r.epoch,
                    'residual_means': r.residual_means,
                    'residual_stds': r.residual_stds,
                    'residual_maxes': r.residual_maxes
                }
                for r in self.residual_history
            ],
            'imbalance_warnings': self.imbalance_warnings,
            'stability_score': self.stability_score
        }

    def save(self, path: Path):
        """Save report to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class LossDiagnostics:
    """
    Diagnostics for PINN training.

    Tracks:
    - Gradient norms per loss term
    - Residual magnitudes
    - Loss imbalance over time
    - Training stability metrics
    """

    def __init__(
        self,
        history_size: int = 1000,
        imbalance_threshold: float = 100.0,
        log_interval: int = 50
    ):
        """
        Initialize diagnostics.

        Args:
            history_size: Number of steps to keep in history
            imbalance_threshold: Ratio threshold for imbalance warning
            log_interval: How often to log diagnostics
        """
        self.history_size = history_size
        self.imbalance_threshold = imbalance_threshold
        self.log_interval = log_interval

        # History buffers
        self.gradient_history: deque = deque(maxlen=history_size)
        self.residual_history: deque = deque(maxlen=history_size)
        self.loss_history: deque = deque(maxlen=history_size)

        # Imbalance tracking
        self.imbalance_warnings: List[str] = []

        # Current state
        self.step = 0
        self.epoch = 0

    def reset(self):
        """Reset all diagnostics"""
        self.gradient_history.clear()
        self.residual_history.clear()
        self.loss_history.clear()
        self.imbalance_warnings.clear()
        self.step = 0
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch"""
        self.epoch = epoch

    def compute_gradient_norms(
        self,
        model: nn.Module,
        loss_terms: Dict[str, torch.Tensor],
        retain_graph: bool = True
    ) -> Dict[str, float]:
        """
        Compute gradient norms for each loss term.

        This is critical for diagnosing PINN training issues.
        Large imbalances in gradient norms indicate optimization problems.

        Args:
            model: Neural network model
            loss_terms: Dict mapping loss name to loss tensor
            retain_graph: Whether to retain computation graph

        Returns:
            Dict mapping loss name to gradient norm
        """
        grad_norms = {}

        # Get shared parameters (typically the base model parameters)
        shared_params = [p for p in model.parameters() if p.requires_grad]

        for name, loss in loss_terms.items():
            if loss is None or not loss.requires_grad:
                grad_norms[name] = 0.0
                continue

            # Zero gradients
            model.zero_grad()

            # Compute gradients
            try:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=shared_params,
                    retain_graph=retain_graph,
                    allow_unused=True
                )

                # Compute total gradient norm
                total_norm = 0.0
                for g in grads:
                    if g is not None:
                        total_norm += g.norm(2).item() ** 2
                total_norm = np.sqrt(total_norm)

                grad_norms[name] = total_norm

            except Exception as e:
                logger.debug(f"Failed to compute gradient norm for {name}: {e}")
                grad_norms[name] = 0.0

        return grad_norms

    def compute_residual_stats(
        self,
        residuals: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Compute statistics for PDE residuals.

        Args:
            residuals: Dict mapping residual name to residual tensor

        Returns:
            Tuple of (means, stds, maxes)
        """
        means = {}
        stds = {}
        maxes = {}

        for name, residual in residuals.items():
            if residual is None:
                means[name] = 0.0
                stds[name] = 0.0
                maxes[name] = 0.0
                continue

            residual = residual.detach()
            means[name] = float(residual.mean().item())
            stds[name] = float(residual.std().item())
            maxes[name] = float(residual.abs().max().item())

        return means, stds, maxes

    def detect_imbalance(
        self,
        grad_norms: Dict[str, float]
    ) -> Tuple[bool, float, str]:
        """
        Detect gradient imbalance between loss terms.

        Args:
            grad_norms: Dict of gradient norms per term

        Returns:
            Tuple of (is_imbalanced, ratio, dominant_term)
        """
        if len(grad_norms) < 2:
            return False, 1.0, ""

        # Filter out zero gradients
        valid_norms = {k: v for k, v in grad_norms.items() if v > 1e-10}

        if len(valid_norms) < 2:
            return False, 1.0, ""

        max_norm = max(valid_norms.values())
        min_norm = min(valid_norms.values())

        ratio = max_norm / (min_norm + 1e-10)
        dominant_term = max(valid_norms, key=valid_norms.get)

        is_imbalanced = ratio > self.imbalance_threshold

        return is_imbalanced, ratio, dominant_term

    def record_step(
        self,
        model: nn.Module,
        loss_terms: Dict[str, torch.Tensor],
        residuals: Optional[Dict[str, torch.Tensor]] = None,
        log: bool = True
    ) -> Dict[str, Any]:
        """
        Record diagnostics for a training step.

        Args:
            model: Neural network model
            loss_terms: Dict of loss tensors
            residuals: Optional dict of residual tensors
            log: Whether to log warnings

        Returns:
            Dict with diagnostic information
        """
        self.step += 1

        # Compute gradient norms
        grad_norms = self.compute_gradient_norms(model, loss_terms)

        # Get loss values
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v
                      for k, v in loss_terms.items()}

        # Detect imbalance
        is_imbalanced, ratio, dominant_term = self.detect_imbalance(grad_norms)

        # Create gradient snapshot
        grad_snapshot = GradientSnapshot(
            step=self.step,
            epoch=self.epoch,
            grad_norms=grad_norms,
            loss_values=loss_values,
            imbalance_ratio=ratio,
            dominant_term=dominant_term
        )
        self.gradient_history.append(grad_snapshot)
        self.loss_history.append(loss_values)

        # Log warning if imbalanced
        if is_imbalanced and log:
            warning = (f"Step {self.step}: Gradient imbalance detected! "
                      f"Ratio={ratio:.1f}x, dominant={dominant_term}")
            self.imbalance_warnings.append(warning)
            if self.step % self.log_interval == 0:
                logger.warning(warning)

        # Compute residual stats if provided
        if residuals is not None:
            means, stds, maxes = self.compute_residual_stats(residuals)
            residual_snapshot = ResidualSnapshot(
                step=self.step,
                epoch=self.epoch,
                residual_means=means,
                residual_stds=stds,
                residual_maxes=maxes
            )
            self.residual_history.append(residual_snapshot)

        # Periodic logging
        if log and self.step % self.log_interval == 0:
            self._log_diagnostics(grad_norms, loss_values, ratio)

        return {
            'grad_norms': grad_norms,
            'loss_values': loss_values,
            'imbalance_ratio': ratio,
            'is_imbalanced': is_imbalanced,
            'dominant_term': dominant_term
        }

    def _log_diagnostics(
        self,
        grad_norms: Dict[str, float],
        loss_values: Dict[str, float],
        imbalance_ratio: float
    ):
        """Log diagnostic information"""
        logger.info(f"Step {self.step} Diagnostics:")
        logger.info(f"  Losses: {', '.join(f'{k}={v:.6f}' for k, v in loss_values.items())}")
        logger.info(f"  Grad norms: {', '.join(f'{k}={v:.4f}' for k, v in grad_norms.items())}")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if len(self.gradient_history) == 0:
            return {'status': 'no_data'}

        # Average gradient norms
        all_grad_norms = [g.grad_norms for g in self.gradient_history]
        avg_grad_norms = {}
        for key in all_grad_norms[0].keys():
            values = [g[key] for g in all_grad_norms if key in g]
            avg_grad_norms[key] = np.mean(values)

        # Average imbalance ratio
        imbalance_ratios = [g.imbalance_ratio for g in self.gradient_history]

        # Dominant term frequency
        dominant_terms = [g.dominant_term for g in self.gradient_history]
        dominant_freq = {}
        for term in set(dominant_terms):
            if term:
                dominant_freq[term] = dominant_terms.count(term) / len(dominant_terms)

        return {
            'total_steps': self.step,
            'avg_grad_norms': avg_grad_norms,
            'avg_imbalance_ratio': np.mean(imbalance_ratios),
            'max_imbalance_ratio': np.max(imbalance_ratios),
            'imbalance_warnings': len(self.imbalance_warnings),
            'dominant_term_freq': dominant_freq
        }

    def compute_stability_score(self) -> float:
        """
        Compute overall training stability score (0-1).

        Higher score indicates more stable training.
        """
        if len(self.gradient_history) < 10:
            return 0.5  # Not enough data

        # Factor 1: Imbalance frequency (fewer imbalances = better)
        n_imbalanced = sum(1 for g in self.gradient_history
                         if g.imbalance_ratio > self.imbalance_threshold)
        imbalance_factor = 1.0 - (n_imbalanced / len(self.gradient_history))

        # Factor 2: Gradient norm stability (lower variance = better)
        all_norms = []
        for g in self.gradient_history:
            all_norms.extend(g.grad_norms.values())

        if len(all_norms) > 0 and np.mean(all_norms) > 0:
            norm_cv = np.std(all_norms) / np.mean(all_norms)
            stability_factor = 1.0 / (1.0 + norm_cv)
        else:
            stability_factor = 0.5

        # Factor 3: Loss convergence (decreasing trend = better)
        if len(self.loss_history) > 10:
            total_losses = []
            for loss_dict in self.loss_history:
                if 'total_loss' in loss_dict:
                    total_losses.append(loss_dict['total_loss'])
                elif 'data_loss' in loss_dict:
                    total_losses.append(sum(loss_dict.values()))

            if len(total_losses) > 0:
                # Check if losses are decreasing
                first_half = np.mean(total_losses[:len(total_losses)//2])
                second_half = np.mean(total_losses[len(total_losses)//2:])
                convergence_factor = 1.0 if second_half < first_half else 0.5
            else:
                convergence_factor = 0.5
        else:
            convergence_factor = 0.5

        # Combined score
        score = (imbalance_factor * 0.4 +
                stability_factor * 0.3 +
                convergence_factor * 0.3)

        return float(np.clip(score, 0.0, 1.0))

    def generate_report(self) -> DiagnosticsReport:
        """Generate complete diagnostics report"""
        return DiagnosticsReport(
            total_steps=self.step,
            total_epochs=self.epoch,
            gradient_history=list(self.gradient_history),
            residual_history=list(self.residual_history),
            imbalance_warnings=self.imbalance_warnings,
            stability_score=self.compute_stability_score()
        )


class ResidualTracker:
    """
    Track PDE residuals during training.

    Monitors:
    - GBM residual: dS/dt - μS
    - OU residual: dX/dt - θ(μ - X)
    - Black-Scholes residual: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV
    """

    def __init__(self, dt: float = 1.0/252.0):
        self.dt = dt
        self.history: Dict[str, List[float]] = {
            'gbm': [],
            'ou': [],
            'bs': [],
            'langevin': []
        }

    def compute_gbm_residual(
        self,
        prices: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        """Compute GBM residual"""
        if prices.shape[1] < 2:
            return torch.tensor(0.0, device=prices.device)

        S = prices[:, :-1]
        S_next = prices[:, 1:]
        dS_dt = (S_next - S) / self.dt

        residual = dS_dt - mu * S
        return residual

    def compute_ou_residual(
        self,
        returns: torch.Tensor,
        theta: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        """Compute OU residual"""
        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        X = returns[:, :-1]
        X_next = returns[:, 1:]
        dX_dt = (X_next - X) / self.dt

        residual = dX_dt - theta * (mu - X)
        return residual

    def record(self, name: str, residual: torch.Tensor):
        """Record a residual value"""
        if name in self.history:
            self.history[name].append(float(residual.mean().abs().item()))

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get residual summary statistics"""
        summary = {}
        for name, values in self.history.items():
            if values:
                summary[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'final': values[-1]
                }
        return summary


def create_diagnostics_callback(
    diagnostics: LossDiagnostics,
    model: nn.Module,
    log_interval: int = 50
):
    """
    Create a callback function for use with training loop.

    Args:
        diagnostics: LossDiagnostics instance
        model: Neural network model
        log_interval: How often to log

    Returns:
        Callback function
    """
    def callback(loss_terms: Dict[str, torch.Tensor], step: int):
        diagnostics.record_step(
            model=model,
            loss_terms=loss_terms,
            log=(step % log_interval == 0)
        )

    return callback
