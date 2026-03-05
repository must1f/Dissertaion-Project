"""Utility modules"""

from .config import get_config, Config
from .logger import get_logger
from .reproducibility import set_seed, log_system_info, init_experiment
from .sampling import (
    latin_hypercube_sampling,
    uniform_random_sampling,
    grid_sampling,
    generate_burgers_training_data,
    generate_evaluation_grid,
)
from .numerical_stability import (
    # Safe operations
    safe_log,
    safe_exp,
    safe_div,
    safe_sqrt,
    safe_pow,
    safe_softmax,
    # Gradient utilities
    GradientStats,
    compute_gradient_stats,
    clip_gradients,
    scale_gradients,
    zero_nan_gradients,
    # Normalization
    RobustNormalizer,
    # Stability checks
    check_tensor_health,
    check_loss_health,
    # Mixed precision
    GradScalerWrapper,
    # Stable activations
    stable_sigmoid,
    stable_tanh,
    leaky_clamp,
    # Robust losses
    smooth_l1_loss,
    log_cosh_loss,
)

__all__ = [
    # Config
    "get_config",
    "Config",
    "get_logger",
    "set_seed",
    "log_system_info",
    "init_experiment",
    # Sampling utilities
    "latin_hypercube_sampling",
    "uniform_random_sampling",
    "grid_sampling",
    "generate_burgers_training_data",
    "generate_evaluation_grid",
    # Safe operations
    "safe_log",
    "safe_exp",
    "safe_div",
    "safe_sqrt",
    "safe_pow",
    "safe_softmax",
    # Gradient utilities
    "GradientStats",
    "compute_gradient_stats",
    "clip_gradients",
    "scale_gradients",
    "zero_nan_gradients",
    # Normalization
    "RobustNormalizer",
    # Stability checks
    "check_tensor_health",
    "check_loss_health",
    # Mixed precision
    "GradScalerWrapper",
    # Stable activations
    "stable_sigmoid",
    "stable_tanh",
    "leaky_clamp",
    # Robust losses
    "smooth_l1_loss",
    "log_cosh_loss",
]
