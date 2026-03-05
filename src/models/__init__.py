"""Model architectures"""

from .baseline import LSTMModel, GRUModel, BiLSTMModel
from .transformer import TransformerModel
from .pinn import PINNModel, PhysicsLoss
from .dp_pinn import BurgersPINN, DualPhasePINN, create_burgers_pinn

__all__ = [
    "LSTMModel",
    "GRUModel",
    "BiLSTMModel",
    "TransformerModel",
    "PINNModel",
    "PhysicsLoss",
    # Burgers' equation PINN models
    "BurgersPINN",
    "DualPhasePINN",
    "create_burgers_pinn",
]
