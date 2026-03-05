"""
PDE Evaluator for Physics-Informed Neural Networks

Evaluation metrics for PDE solutions:
- Relative L2 error over the domain
- Time-resolved L2 error (error at each time slice)
- Point-wise error statistics
- Residual magnitude analysis

Includes reference solution computation for Burgers' equation
via the Hopf-Cole transformation.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..utils.sampling import generate_evaluation_grid

logger = get_logger(__name__)


@dataclass
class PDEMetrics:
    """Container for PDE evaluation metrics."""

    # L2 errors
    relative_l2_error: float = 0.0
    absolute_l2_error: float = 0.0

    # Time-resolved errors
    time_slices: List[float] = field(default_factory=list)
    time_resolved_l2: List[float] = field(default_factory=list)

    # Point-wise statistics
    max_error: float = 0.0
    mean_error: float = 0.0
    median_error: float = 0.0
    std_error: float = 0.0

    # Residual statistics
    mean_residual: float = 0.0
    max_residual: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "relative_l2_error": self.relative_l2_error,
            "absolute_l2_error": self.absolute_l2_error,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "median_error": self.median_error,
            "std_error": self.std_error,
            "mean_residual": self.mean_residual,
            "max_residual": self.max_residual,
        }


class PDEEvaluator:
    """
    Evaluator for PDE solutions.

    Computes various error metrics comparing PINN predictions
    against reference solutions.
    """

    def __init__(
        self,
        reference_solution: Callable,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        n_x: int = 256,
        n_t: int = 100,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PDEEvaluator.

        Args:
            reference_solution: Function (x, t) -> u_exact
            x_range: Spatial domain bounds
            t_range: Temporal domain bounds
            n_x: Number of spatial evaluation points
            n_t: Number of temporal evaluation points
            device: PyTorch device
        """
        self.reference_solution = reference_solution
        self.x_range = x_range
        self.t_range = t_range
        self.n_x = n_x
        self.n_t = n_t
        self.device = device or torch.device("cpu")

        # Generate evaluation grid
        self.grid = generate_evaluation_grid(
            n_x=n_x,
            n_t=n_t,
            x_range=x_range,
            t_range=t_range,
            device=self.device,
        )

        logger.info(
            f"PDEEvaluator initialized with {n_x}x{n_t} grid on {self.device}"
        )

    def relative_l2_error(
        self,
        model: nn.Module,
    ) -> float:
        """
        Compute relative L2 error over the domain.

        L2_rel = ||u_pred - u_exact||_2 / ||u_exact||_2

        Args:
            model: PINN model with forward(x, t) method

        Returns:
            Relative L2 error
        """
        model.eval()

        with torch.no_grad():
            x_flat = self.grid["x_flat"]
            t_flat = self.grid["t_flat"]

            # Get predictions
            u_pred = model(x_flat, t_flat)
            if u_pred.dim() == 2:
                u_pred = u_pred.squeeze(-1)

            # Get reference solution
            u_exact = self.reference_solution(x_flat, t_flat)
            if u_exact.dim() == 2:
                u_exact = u_exact.squeeze(-1)

            # Compute relative L2 error
            error = u_pred - u_exact
            l2_error = torch.norm(error, p=2)
            l2_exact = torch.norm(u_exact, p=2)

            relative_error = (l2_error / (l2_exact + 1e-10)).item()

        return relative_error

    def time_resolved_error(
        self,
        model: nn.Module,
        time_slices: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute L2 error at each time slice.

        Args:
            model: PINN model
            time_slices: Optional list of times to evaluate
                         (default: uniform grid)

        Returns:
            Tuple of (time_values, l2_errors)
        """
        model.eval()

        if time_slices is None:
            time_slices = self.grid["t_grid"].cpu().numpy().tolist()

        errors = []

        with torch.no_grad():
            x_grid = self.grid["x_grid"]

            for t_val in time_slices:
                t_tensor = torch.full_like(x_grid, t_val)

                # Get predictions
                u_pred = model(x_grid, t_tensor)
                if u_pred.dim() == 2:
                    u_pred = u_pred.squeeze(-1)

                # Get reference
                u_exact = self.reference_solution(x_grid, t_tensor)
                if u_exact.dim() == 2:
                    u_exact = u_exact.squeeze(-1)

                # L2 error at this time slice
                error = u_pred - u_exact
                l2_error = torch.norm(error, p=2) / (torch.norm(u_exact, p=2) + 1e-10)
                errors.append(l2_error.item())

        return time_slices, errors

    def evaluate_all(
        self,
        model: nn.Module,
        compute_residual: bool = True,
    ) -> PDEMetrics:
        """
        Compute all evaluation metrics.

        Args:
            model: PINN model
            compute_residual: Whether to compute PDE residual

        Returns:
            PDEMetrics object with all metrics
        """
        model.eval()
        metrics = PDEMetrics()

        with torch.no_grad():
            x_flat = self.grid["x_flat"]
            t_flat = self.grid["t_flat"]

            # Get predictions
            u_pred = model(x_flat, t_flat)
            if u_pred.dim() == 2:
                u_pred = u_pred.squeeze(-1)

            # Get reference solution
            u_exact = self.reference_solution(x_flat, t_flat)
            if u_exact.dim() == 2:
                u_exact = u_exact.squeeze(-1)

            # Point-wise error
            error = (u_pred - u_exact).abs()

            # L2 errors
            l2_error = torch.norm(u_pred - u_exact, p=2)
            l2_exact = torch.norm(u_exact, p=2)

            metrics.absolute_l2_error = l2_error.item()
            metrics.relative_l2_error = (l2_error / (l2_exact + 1e-10)).item()

            # Point-wise statistics
            metrics.max_error = error.max().item()
            metrics.mean_error = error.mean().item()
            metrics.median_error = error.median().item()
            metrics.std_error = error.std().item()

            # Time-resolved errors
            time_slices, time_errors = self.time_resolved_error(model)
            metrics.time_slices = time_slices
            metrics.time_resolved_l2 = time_errors

        # Compute residual if model supports it
        if compute_residual and hasattr(model, "compute_pde_residual"):
            # Use a subset of points for efficiency
            n_residual = min(10000, len(x_flat))
            indices = torch.randperm(len(x_flat))[:n_residual]

            residual = model.compute_pde_residual(
                x_flat[indices], t_flat[indices]
            )
            if residual.dim() == 2:
                residual = residual.squeeze(-1)

            metrics.mean_residual = residual.abs().mean().item()
            metrics.max_residual = residual.abs().max().item()

        logger.info(
            f"Evaluation complete: L2_rel={metrics.relative_l2_error:.6f}, "
            f"max_err={metrics.max_error:.6f}"
        )

        return metrics

    def get_prediction_grid(
        self,
        model: nn.Module,
    ) -> Dict[str, np.ndarray]:
        """
        Get model predictions on evaluation grid.

        Args:
            model: PINN model

        Returns:
            Dictionary with:
                - X, T: Meshgrid arrays [n_t, n_x]
                - u_pred: Predicted solution [n_t, n_x]
                - u_exact: Reference solution [n_t, n_x]
                - error: Absolute error [n_t, n_x]
        """
        model.eval()

        with torch.no_grad():
            x_flat = self.grid["x_flat"]
            t_flat = self.grid["t_flat"]

            # Get predictions
            u_pred = model(x_flat, t_flat)
            if u_pred.dim() == 2:
                u_pred = u_pred.squeeze(-1)

            # Get reference
            u_exact = self.reference_solution(x_flat, t_flat)
            if u_exact.dim() == 2:
                u_exact = u_exact.squeeze(-1)

            # Reshape to grid
            u_pred_grid = u_pred.reshape(self.n_t, self.n_x).cpu().numpy()
            u_exact_grid = u_exact.reshape(self.n_t, self.n_x).cpu().numpy()
            error_grid = np.abs(u_pred_grid - u_exact_grid)

        return {
            "X": self.grid["X"].cpu().numpy(),
            "T": self.grid["T"].cpu().numpy(),
            "x_grid": self.grid["x_grid"].cpu().numpy(),
            "t_grid": self.grid["t_grid"].cpu().numpy(),
            "u_pred": u_pred_grid,
            "u_exact": u_exact_grid,
            "error": error_grid,
        }


def burgers_exact_solution_hopf_cole(
    x: torch.Tensor,
    t: torch.Tensor,
    viscosity: float = 0.01 / math.pi,
    n_integration_points: int = 1000,
) -> torch.Tensor:
    """
    Compute exact solution of Burgers' equation via Hopf-Cole transformation.

    For the initial condition u(x, 0) = -sin(πx), the exact solution
    can be computed using numerical integration of the Hopf-Cole formula.

    The Hopf-Cole transformation:
        u = -2ν * (∂φ/∂x) / φ

    where φ satisfies the heat equation with transformed IC.

    Args:
        x: Spatial coordinates [batch]
        t: Temporal coordinates [batch]
        viscosity: Kinematic viscosity ν
        n_integration_points: Points for numerical integration

    Returns:
        Exact solution values [batch] or [batch, 1]
    """
    # Ensure proper shapes
    if x.dim() == 2:
        x = x.squeeze(-1)
    if t.dim() == 2:
        t = t.squeeze(-1)

    device = x.device
    dtype = x.dtype
    batch_size = x.shape[0]

    # For very small t, return IC
    eps = 1e-8
    result = torch.zeros(batch_size, dtype=dtype, device=device)

    small_t_mask = t.abs() < eps
    if small_t_mask.any():
        result[small_t_mask] = -torch.sin(math.pi * x[small_t_mask])

    # For t > 0, use numerical integration
    large_t_mask = ~small_t_mask
    if large_t_mask.any():
        x_eval = x[large_t_mask]
        t_eval = t[large_t_mask]
        n_eval = x_eval.shape[0]

        # Integration points in η ∈ [-1, 1]
        eta = torch.linspace(-1, 1, n_integration_points, device=device, dtype=dtype)
        deta = 2.0 / n_integration_points

        # Compute for each (x, t) point
        u_values = torch.zeros(n_eval, dtype=dtype, device=device)

        for i in range(n_eval):
            xi = x_eval[i].item()
            ti = t_eval[i].item()

            # Hopf-Cole integrand
            # φ = ∫ exp(-[F(η) + (x-η)²/(4νt)] / (2ν)) dη
            # where F(η) = ∫₀^η u(ξ, 0) dξ = (1/π)(cos(πη) - 1)

            F_eta = (1.0 / math.pi) * (torch.cos(math.pi * eta) - 1.0)
            exponent = (F_eta + (xi - eta) ** 2 / (4 * viscosity * ti)) / (2 * viscosity)

            # Numerical stability: shift by max
            exponent_shifted = exponent - exponent.max()
            integrand = torch.exp(-exponent_shifted)

            # Compute φ and ∂φ/∂x
            phi = torch.sum(integrand) * deta
            dphi_dx = torch.sum(-(xi - eta) / (2 * viscosity * ti) * integrand) * deta

            # u = -2ν * (∂φ/∂x) / φ
            u_values[i] = -2 * viscosity * dphi_dx / (phi + 1e-10)

        result[large_t_mask] = u_values

    return result


def create_burgers_evaluator(
    viscosity: float = 0.01 / math.pi,
    n_x: int = 256,
    n_t: int = 100,
    device: Optional[torch.device] = None,
) -> PDEEvaluator:
    """
    Create a PDEEvaluator for Burgers' equation.

    Args:
        viscosity: Kinematic viscosity
        n_x: Spatial resolution
        n_t: Temporal resolution
        device: PyTorch device

    Returns:
        PDEEvaluator configured for Burgers' equation
    """
    def reference_solution(x, t):
        return burgers_exact_solution_hopf_cole(x, t, viscosity=viscosity)

    return PDEEvaluator(
        reference_solution=reference_solution,
        n_x=n_x,
        n_t=n_t,
        device=device,
    )


def compare_models(
    models: Dict[str, nn.Module],
    evaluator: PDEEvaluator,
) -> Dict[str, PDEMetrics]:
    """
    Compare multiple models using the same evaluator.

    Args:
        models: Dictionary of model_name -> model
        evaluator: PDEEvaluator instance

    Returns:
        Dictionary of model_name -> PDEMetrics
    """
    results = {}

    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        metrics = evaluator.evaluate_all(model)
        results[name] = metrics

        logger.info(
            f"  {name}: L2_rel={metrics.relative_l2_error:.6f}, "
            f"max_err={metrics.max_error:.6f}"
        )

    # Log comparison summary
    logger.info("\n=== Model Comparison Summary ===")
    for name, metrics in sorted(results.items(), key=lambda x: x[1].relative_l2_error):
        logger.info(
            f"{name}: L2_rel={metrics.relative_l2_error:.6f}, "
            f"mean_err={metrics.mean_error:.6f}"
        )

    return results
