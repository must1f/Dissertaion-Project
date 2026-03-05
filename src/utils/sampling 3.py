"""
Sampling Utilities for Physics-Informed Neural Networks

Provides sampling strategies for PDE collocation points:
- Latin Hypercube Sampling (LHS) for efficient domain coverage
- Uniform random sampling
- Grid-based sampling
- Boundary and initial condition point generation

LHS provides better space-filling properties than random sampling,
leading to improved PINN training convergence.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .logger import get_logger

logger = get_logger(__name__)


def latin_hypercube_sampling(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate samples using Latin Hypercube Sampling (LHS).

    LHS divides each dimension into n_samples equally spaced intervals,
    then samples one point from each interval such that the projections
    onto each axis are uniformly distributed.

    This provides better space-filling properties than random sampling,
    reducing the number of samples needed for good domain coverage.

    Args:
        n_samples: Number of samples to generate
        bounds: List of (min, max) bounds for each dimension
        seed: Random seed for reproducibility
        device: PyTorch device for output tensor
        dtype: Data type for output tensor

    Returns:
        Tensor of samples [n_samples, n_dims]

    Example:
        >>> bounds = [(-1.0, 1.0), (0.0, 1.0)]  # x, t ranges
        >>> samples = latin_hypercube_sampling(1000, bounds, seed=42)
        >>> samples.shape
        torch.Size([1000, 2])
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)

    # Generate LHS samples in [0, 1]^d
    # Create n_samples intervals for each dimension
    samples = np.zeros((n_samples, n_dims))

    for dim in range(n_dims):
        # Generate random permutation of intervals
        permutation = np.random.permutation(n_samples)

        # Sample uniformly within each interval
        for i in range(n_samples):
            lower = permutation[i] / n_samples
            upper = (permutation[i] + 1) / n_samples
            samples[i, dim] = np.random.uniform(lower, upper)

    # Scale to actual bounds
    for dim, (low, high) in enumerate(bounds):
        samples[:, dim] = low + samples[:, dim] * (high - low)

    # Convert to tensor
    tensor = torch.tensor(samples, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def uniform_random_sampling(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate samples using uniform random sampling.

    Args:
        n_samples: Number of samples
        bounds: List of (min, max) bounds
        seed: Random seed
        device: Output device
        dtype: Output dtype

    Returns:
        Tensor of samples [n_samples, n_dims]
    """
    if seed is not None:
        torch.manual_seed(seed)

    samples = []
    for low, high in bounds:
        dim_samples = torch.rand(n_samples) * (high - low) + low
        samples.append(dim_samples)

    tensor = torch.stack(samples, dim=1).to(dtype)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def grid_sampling(
    n_points_per_dim: Union[int, List[int]],
    bounds: List[Tuple[float, float]],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate samples on a regular grid.

    Args:
        n_points_per_dim: Points per dimension (int or list)
        bounds: List of (min, max) bounds
        device: Output device
        dtype: Output dtype

    Returns:
        Tensor of samples [n_points_total, n_dims]
    """
    n_dims = len(bounds)

    if isinstance(n_points_per_dim, int):
        n_points_per_dim = [n_points_per_dim] * n_dims

    # Create 1D grids for each dimension
    grids = []
    for i, (low, high) in enumerate(bounds):
        grids.append(torch.linspace(low, high, n_points_per_dim[i], dtype=dtype))

    # Create meshgrid and flatten
    mesh = torch.meshgrid(*grids, indexing="ij")
    samples = torch.stack([m.flatten() for m in mesh], dim=1)

    if device is not None:
        samples = samples.to(device)

    return samples


def generate_burgers_training_data(
    n_collocation: int = 20000,
    n_boundary: int = 2000,
    n_initial: int = 2000,
    n_intermediate: int = 1000,
    t_switch: float = 0.4,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Generate complete training data for Burgers' equation PINN.

    Generates:
    - Collocation points: LHS samples over the full domain
    - Initial condition points: Points at t=0
    - Boundary condition points: Points at x=-1 and x=1
    - Intermediate points: Points at t=t_switch (for dual-phase)

    For dual-phase training, also generates phase-specific collocation points.

    Args:
        n_collocation: Number of collocation points (LHS)
        n_boundary: Number of boundary points (split between x=-1 and x=1)
        n_initial: Number of initial condition points at t=0
        n_intermediate: Number of intermediate constraint points at t_switch
        t_switch: Phase transition time for dual-phase PINN
        x_range: Spatial domain (x_min, x_max)
        t_range: Temporal domain (t_min, t_max)
        seed: Random seed for reproducibility
        device: PyTorch device
        dtype: Data type

    Returns:
        Dictionary with:
            - x_collocation, t_collocation: Full domain collocation
            - x_collocation_p1, t_collocation_p1: Phase 1 collocation
            - x_collocation_p2, t_collocation_p2: Phase 2 collocation
            - x_ic: Initial condition x points
            - t_bc: Boundary condition t points
            - x_intermediate: Intermediate constraint x points
    """
    logger.info(f"Generating Burgers' equation training data with seed={seed}")

    data = {}

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Full domain collocation points (LHS)
    collocation = latin_hypercube_sampling(
        n_samples=n_collocation,
        bounds=[x_range, t_range],
        seed=seed,
        device=device,
        dtype=dtype,
    )
    data["x_collocation"] = collocation[:, 0]
    data["t_collocation"] = collocation[:, 1]

    # Phase 1 collocation points: t ∈ [0, t_switch]
    collocation_p1 = latin_hypercube_sampling(
        n_samples=n_collocation // 2,
        bounds=[x_range, (t_range[0], t_switch)],
        seed=seed + 1,
        device=device,
        dtype=dtype,
    )
    data["x_collocation_p1"] = collocation_p1[:, 0]
    data["t_collocation_p1"] = collocation_p1[:, 1]

    # Phase 2 collocation points: t ∈ [t_switch, 1]
    collocation_p2 = latin_hypercube_sampling(
        n_samples=n_collocation // 2,
        bounds=[x_range, (t_switch, t_range[1])],
        seed=seed + 2,
        device=device,
        dtype=dtype,
    )
    data["x_collocation_p2"] = collocation_p2[:, 0]
    data["t_collocation_p2"] = collocation_p2[:, 1]

    # Initial condition points: x at t=0
    x_ic = latin_hypercube_sampling(
        n_samples=n_initial,
        bounds=[x_range],
        seed=seed + 3,
        device=device,
        dtype=dtype,
    )
    data["x_ic"] = x_ic.squeeze(-1)

    # Boundary condition points: t values for x=-1 and x=1
    t_bc = latin_hypercube_sampling(
        n_samples=n_boundary,
        bounds=[t_range],
        seed=seed + 4,
        device=device,
        dtype=dtype,
    )
    data["t_bc"] = t_bc.squeeze(-1)

    # Phase-specific BC points
    t_bc_p1 = latin_hypercube_sampling(
        n_samples=n_boundary // 2,
        bounds=[(t_range[0], t_switch)],
        seed=seed + 5,
        device=device,
        dtype=dtype,
    )
    data["t_bc_p1"] = t_bc_p1.squeeze(-1)

    t_bc_p2 = latin_hypercube_sampling(
        n_samples=n_boundary // 2,
        bounds=[(t_switch, t_range[1])],
        seed=seed + 6,
        device=device,
        dtype=dtype,
    )
    data["t_bc_p2"] = t_bc_p2.squeeze(-1)

    # Intermediate constraint points: x values at t=t_switch
    x_intermediate = latin_hypercube_sampling(
        n_samples=n_intermediate,
        bounds=[x_range],
        seed=seed + 7,
        device=device,
        dtype=dtype,
    )
    data["x_intermediate"] = x_intermediate.squeeze(-1)

    # Store t_switch value
    data["t_switch"] = torch.tensor(t_switch, device=device, dtype=dtype)

    logger.info(
        f"Generated training data: {n_collocation} collocation, "
        f"{n_initial} IC, {n_boundary} BC, {n_intermediate} intermediate"
    )

    return data


def generate_evaluation_grid(
    n_x: int = 256,
    n_t: int = 100,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Generate a regular grid for evaluation and visualization.

    Args:
        n_x: Number of spatial grid points
        n_t: Number of temporal grid points
        x_range: Spatial domain
        t_range: Temporal domain
        device: PyTorch device
        dtype: Data type

    Returns:
        Dictionary with:
            - x_grid: 1D spatial grid [n_x]
            - t_grid: 1D temporal grid [n_t]
            - X, T: 2D meshgrid arrays [n_t, n_x]
            - x_flat, t_flat: Flattened grid points [n_x * n_t]
    """
    x_grid = torch.linspace(x_range[0], x_range[1], n_x, dtype=dtype)
    t_grid = torch.linspace(t_range[0], t_range[1], n_t, dtype=dtype)

    # Create meshgrid (T varies along rows, X along columns)
    T, X = torch.meshgrid(t_grid, x_grid, indexing="ij")

    # Flatten for model evaluation
    x_flat = X.flatten()
    t_flat = T.flatten()

    if device is not None:
        x_grid = x_grid.to(device)
        t_grid = t_grid.to(device)
        X = X.to(device)
        T = T.to(device)
        x_flat = x_flat.to(device)
        t_flat = t_flat.to(device)

    return {
        "x_grid": x_grid,
        "t_grid": t_grid,
        "X": X,
        "T": T,
        "x_flat": x_flat,
        "t_flat": t_flat,
        "n_x": n_x,
        "n_t": n_t,
    }


def sobol_sampling(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate samples using Sobol sequence (quasi-random).

    Sobol sequences provide even better space-filling than LHS
    but require scipy.

    Args:
        n_samples: Number of samples
        bounds: List of (min, max) bounds
        seed: Random seed (actually scramble parameter for Sobol)
        device: Output device
        dtype: Output dtype

    Returns:
        Tensor of samples [n_samples, n_dims]
    """
    try:
        from scipy.stats import qmc

        n_dims = len(bounds)
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        samples = sampler.random(n_samples)

        # Scale to bounds
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]
        samples = qmc.scale(samples, l_bounds, u_bounds)

        tensor = torch.tensor(samples, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)

        return tensor

    except ImportError:
        logger.warning("scipy not available, falling back to LHS")
        return latin_hypercube_sampling(n_samples, bounds, seed, device, dtype)


def adaptive_sampling(
    model: torch.nn.Module,
    current_samples: torch.Tensor,
    bounds: List[Tuple[float, float]],
    n_new_samples: int,
    residual_fn: callable,
    percentile: float = 90.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate adaptive samples based on residual magnitude.

    Adds more samples in regions where the PDE residual is large,
    improving training efficiency for problems with steep gradients.

    Args:
        model: Current trained model
        current_samples: Existing sample points [n, n_dims]
        bounds: Domain bounds
        n_new_samples: Number of new samples to add
        residual_fn: Function that computes residual given (model, samples)
        percentile: High-residual percentile to sample around
        device: Output device
        dtype: Output dtype

    Returns:
        New samples to add [n_new_samples, n_dims]
    """
    with torch.no_grad():
        # Compute residuals at current points
        residuals = residual_fn(model, current_samples)
        residuals = residuals.abs().squeeze()

        # Find high-residual regions
        threshold = torch.quantile(residuals, percentile / 100.0)
        high_res_mask = residuals > threshold
        high_res_points = current_samples[high_res_mask]

    if len(high_res_points) == 0:
        # Fall back to LHS if no high-residual points
        return latin_hypercube_sampling(n_new_samples, bounds, device=device, dtype=dtype)

    # Sample new points around high-residual regions
    n_dims = len(bounds)
    new_samples = []

    for _ in range(n_new_samples):
        # Pick a random high-residual point
        idx = np.random.randint(len(high_res_points))
        center = high_res_points[idx]

        # Add Gaussian perturbation (scaled by domain size)
        scales = torch.tensor([(b[1] - b[0]) * 0.1 for b in bounds], dtype=dtype)
        if device is not None:
            scales = scales.to(device)

        perturbation = torch.randn(n_dims, dtype=dtype)
        if device is not None:
            perturbation = perturbation.to(device)

        new_point = center + perturbation * scales

        # Clip to bounds
        for dim, (low, high) in enumerate(bounds):
            new_point[dim] = torch.clamp(new_point[dim], low, high)

        new_samples.append(new_point)

    return torch.stack(new_samples)
