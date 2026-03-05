"""
Loss Functions Module

Centralized loss functions for PINN financial forecasting:
- Data losses (MSE, MAE, Huber, quantile)
- Physics losses (GBM, OU, Black-Scholes, Langevin)
- Composite loss builders with adaptive weighting
"""

from .data_losses import (
    DataLoss,
    MSELoss,
    MAELoss,
    HuberLoss,
    LogCoshLoss,
    QuantileLoss,
    DirectionalLoss,
    WeightedMSELoss,
    create_data_loss
)

from .physics_losses import (
    PhysicsResidual,
    GBMResidual,
    OUResidual,
    BlackScholesResidual,
    LangevinResidual,
    NoArbitrageResidual,
    create_physics_loss
)

from .composite import (
    CompositeLoss,
    AdaptiveCompositeLoss,
    LossConfig,
    LossWeight,
    WeightingStrategy,
    create_composite_loss
)

from .burgers_equation import (
    BurgersResidual,
    BurgersICLoss,
    BurgersBCLoss,
    BurgersIntermediateLoss,
    BurgersLossFunction,
    DualPhaseBurgersLoss,
    burgers_exact_solution,
)

from .spectral_loss import (
    SpectralResidual,
    AutocorrelationLoss,
    SpectralConsistencyLoss,
    SpectralEntropyLoss,
    CombinedSpectralLoss,
    create_spectral_loss,
)

__all__ = [
    # Data losses
    "DataLoss",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "LogCoshLoss",
    "QuantileLoss",
    "DirectionalLoss",
    "WeightedMSELoss",
    "create_data_loss",
    # Physics losses
    "PhysicsResidual",
    "GBMResidual",
    "OUResidual",
    "BlackScholesResidual",
    "LangevinResidual",
    "NoArbitrageResidual",
    "create_physics_loss",
    # Composite
    "CompositeLoss",
    "AdaptiveCompositeLoss",
    "LossConfig",
    "LossWeight",
    "WeightingStrategy",
    "create_composite_loss",
    # Burgers' equation losses
    "BurgersResidual",
    "BurgersICLoss",
    "BurgersBCLoss",
    "BurgersIntermediateLoss",
    "BurgersLossFunction",
    "DualPhaseBurgersLoss",
    "burgers_exact_solution",
    # Spectral losses
    "SpectralResidual",
    "AutocorrelationLoss",
    "SpectralConsistencyLoss",
    "SpectralEntropyLoss",
    "CombinedSpectralLoss",
    "create_spectral_loss",
]
