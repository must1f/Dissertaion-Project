"""
Dual-Phase Physics-Informed Neural Network (DP-PINN) for Burgers' Equation

Implements a two-phase PINN architecture for solving the viscous Burgers' equation,
which is a canonical stiff PDE benchmark:

    u_t + u * u_x - ν * u_xx = 0

on x ∈ [-1, 1], t ∈ [0, 1] with ν = 0.01/π

The dual-phase approach splits the time domain to better handle steep gradients
that develop during solution evolution.

Classes:
    BurgersPINN: Standard PINN for Burgers' equation
    DualPhasePINN: Two-phase architecture with intermediate constraint
"""

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BurgersPINN(nn.Module):
    """
    Physics-Informed Neural Network for the viscous Burgers' equation.

    Architecture: 8 fully-connected layers with 50 neurons each, tanh activation.

    The Burgers' equation models nonlinear advection-diffusion:
        u_t + u * u_x = ν * u_xx

    This creates steep gradients (shocks) that are challenging for standard PINNs.

    Attributes:
        viscosity: Kinematic viscosity ν (default 0.01/π)
        layers: Neural network layers
    """

    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 50,
        activation: str = "tanh",
        viscosity: float = 0.01 / math.pi,
        lambda_ic: float = 100.0,
        lambda_bc: float = 100.0,
        lambda_pde: float = 1.0,
    ):
        """
        Initialize BurgersPINN.

        Args:
            num_layers: Number of hidden layers (default 8)
            hidden_dim: Neurons per hidden layer (default 50)
            activation: Activation function ("tanh", "sin", "gelu")
            viscosity: Kinematic viscosity ν
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
            lambda_pde: Weight for PDE residual loss
        """
        super().__init__()

        self.viscosity = viscosity
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Select activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sin":
            self.activation = SinActivation()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network: input (x, t) -> output u
        layers = []

        # Input layer: 2 -> hidden_dim
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(self.activation)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)

        # Output layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

        logger.info(
            f"BurgersPINN initialized: {num_layers} layers, {hidden_dim} neurons, "
            f"ν={viscosity:.6f}, λ_ic={lambda_ic}, λ_bc={lambda_bc}, λ_pde={lambda_pde}"
        )

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Spatial coordinates [batch] or [batch, 1]
            t: Temporal coordinates [batch] or [batch, 1]

        Returns:
            u: Solution values [batch, 1]
        """
        # Ensure correct shapes
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Concatenate inputs
        inputs = torch.cat([x, t], dim=-1)

        return self.network(inputs)

    def forward_with_grad(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic differentiation for PDE residual.

        Computes u, u_t, u_x, u_xx using torch.autograd.grad with create_graph=True
        for integration into the training graph.

        Args:
            x: Spatial coordinates [batch] or [batch, 1]
            t: Temporal coordinates [batch] or [batch, 1]

        Returns:
            Tuple of (u, u_t, u_x, u_xx), each [batch, 1]
        """
        # Ensure gradient tracking
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        # Forward pass
        u = self.forward(x, t)

        # First derivatives
        grad_outputs = torch.ones_like(u)

        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        return u, u_t, u_x, u_xx

    def compute_pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Burgers' equation PDE residual.

        Residual: u_t + u * u_x - ν * u_xx = 0

        Args:
            x: Spatial coordinates [batch]
            t: Temporal coordinates [batch]

        Returns:
            residual: PDE residual [batch, 1]
        """
        u, u_t, u_x, u_xx = self.forward_with_grad(x, t)

        # Burgers' equation residual: u_t + u * u_x - ν * u_xx
        residual = u_t + u * u_x - self.viscosity * u_xx

        return residual

    def compute_ic_loss(
        self,
        x: torch.Tensor,
        t_zero: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute initial condition loss.

        IC: u(x, 0) = -sin(π * x)

        Args:
            x: Spatial coordinates at t=0 [batch]
            t_zero: Optional tensor of zeros (created if not provided)

        Returns:
            IC loss (scalar)
        """
        if t_zero is None:
            t_zero = torch.zeros_like(x)

        u_pred = self.forward(x, t_zero)
        u_exact = -torch.sin(math.pi * x)

        if u_exact.dim() == 1:
            u_exact = u_exact.unsqueeze(-1)

        return torch.mean((u_pred - u_exact) ** 2)

    def compute_bc_loss(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.

        BC: u(-1, t) = u(1, t) = 0

        Args:
            t: Temporal coordinates [batch]

        Returns:
            BC loss (scalar)
        """
        batch_size = t.shape[0]
        device = t.device
        dtype = t.dtype

        # Left boundary: x = -1
        x_left = torch.full((batch_size,), -1.0, device=device, dtype=dtype)
        u_left = self.forward(x_left, t)

        # Right boundary: x = 1
        x_right = torch.full((batch_size,), 1.0, device=device, dtype=dtype)
        u_right = self.forward(x_right, t)

        return torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    def compute_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_ic: torch.Tensor,
        t_bc: torch.Tensor,
        metadata: Optional[Dict] = None,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss for training.

        Total loss = λ_pde * L_pde + λ_ic * L_ic + λ_bc * L_bc

        Args:
            x_collocation: Spatial collocation points [n_colloc]
            t_collocation: Temporal collocation points [n_colloc]
            x_ic: Spatial points for IC [n_ic]
            t_bc: Temporal points for BC [n_bc]
            metadata: Optional metadata dict
            enable_physics: Whether to compute physics loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}

        # PDE residual loss
        residual = self.compute_pde_residual(x_collocation, t_collocation)
        pde_loss = torch.mean(residual ** 2)
        loss_dict["pde_loss"] = pde_loss.item()

        # Initial condition loss
        ic_loss = self.compute_ic_loss(x_ic)
        loss_dict["ic_loss"] = ic_loss.item()

        # Boundary condition loss
        bc_loss = self.compute_bc_loss(t_bc)
        loss_dict["bc_loss"] = bc_loss.item()

        # Total loss
        total_loss = (
            self.lambda_pde * pde_loss +
            self.lambda_ic * ic_loss +
            self.lambda_bc * bc_loss
        )
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["data_loss"] = (ic_loss + bc_loss).item()
        loss_dict["physics_loss"] = pde_loss.item()

        return total_loss, loss_dict


class SinActivation(nn.Module):
    """Sinusoidal activation function for periodic problems."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class DualPhasePINN(nn.Module):
    """
    Dual-Phase PINN for Burgers' equation.

    Splits the time domain into two phases:
    - Phase 1: t ∈ [0, t_switch] with initial condition constraint
    - Phase 2: t ∈ [t_switch, 1] with intermediate constraint

    This improves accuracy for stiff PDEs by preventing error accumulation
    across the full time domain.

    Attributes:
        t_switch: Time at which to switch phases (default 0.4)
        phase1_net: Network for phase 1
        phase2_net: Network for phase 2
        lambda_intermediate: Weight for intermediate constraint
    """

    def __init__(
        self,
        t_switch: float = 0.4,
        num_layers: int = 8,
        hidden_dim: int = 50,
        activation: str = "tanh",
        viscosity: float = 0.01 / math.pi,
        lambda_ic: float = 100.0,
        lambda_bc: float = 100.0,
        lambda_pde: float = 1.0,
        lambda_intermediate: float = 100.0,
    ):
        """
        Initialize DualPhasePINN.

        Args:
            t_switch: Phase transition time
            num_layers: Number of hidden layers per phase network
            hidden_dim: Neurons per hidden layer
            activation: Activation function
            viscosity: Kinematic viscosity ν
            lambda_ic: Weight for IC loss
            lambda_bc: Weight for BC loss
            lambda_pde: Weight for PDE loss
            lambda_intermediate: Weight for intermediate constraint
        """
        super().__init__()

        self.t_switch = t_switch
        self.viscosity = viscosity
        self.lambda_intermediate = lambda_intermediate

        # Phase 1 network: handles t ∈ [0, t_switch]
        self.phase1_net = BurgersPINN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            viscosity=viscosity,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
            lambda_pde=lambda_pde,
        )

        # Phase 2 network: handles t ∈ [t_switch, 1]
        self.phase2_net = BurgersPINN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            viscosity=viscosity,
            lambda_ic=0.0,  # No IC for phase 2
            lambda_bc=lambda_bc,
            lambda_pde=lambda_pde,
        )

        logger.info(
            f"DualPhasePINN initialized: t_switch={t_switch}, "
            f"λ_intermediate={lambda_intermediate}"
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass selecting appropriate phase network.

        Args:
            x: Spatial coordinates [batch]
            t: Temporal coordinates [batch]

        Returns:
            u: Solution values [batch, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Select network based on time
        mask_phase1 = (t <= self.t_switch).float()
        mask_phase2 = 1.0 - mask_phase1

        u1 = self.phase1_net(x, t)
        u2 = self.phase2_net(x, t)

        return mask_phase1 * u1 + mask_phase2 * u2

    def forward_phase1(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through phase 1 network only."""
        return self.phase1_net(x, t)

    def forward_phase2(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through phase 2 network only."""
        return self.phase2_net(x, t)

    def compute_intermediate_loss(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intermediate constraint loss at t = t_switch.

        Ensures continuity between phase 1 and phase 2:
        u1(x, t_switch) = u2(x, t_switch)

        Args:
            x: Spatial coordinates [batch]

        Returns:
            Intermediate loss (scalar)
        """
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        t_switch = torch.full(
            (batch_size,), self.t_switch, device=device, dtype=dtype
        )

        # Get predictions from both phases at t_switch
        u1 = self.phase1_net(x, t_switch)
        u2 = self.phase2_net(x, t_switch)

        return torch.mean((u1 - u2) ** 2)

    def compute_phase1_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_ic: torch.Tensor,
        t_bc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for phase 1 training.

        Phase 1 uses IC constraint and covers t ∈ [0, t_switch].

        Args:
            x_collocation: Spatial collocation points
            t_collocation: Temporal collocation points (should be ≤ t_switch)
            x_ic: Spatial points for IC
            t_bc: Temporal points for BC

        Returns:
            Tuple of (loss, loss_dict)
        """
        return self.phase1_net.compute_loss(
            x_collocation, t_collocation, x_ic, t_bc
        )

    def compute_phase2_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_intermediate: torch.Tensor,
        t_bc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for phase 2 training.

        Phase 2 uses intermediate constraint instead of IC.

        Args:
            x_collocation: Spatial collocation points
            t_collocation: Temporal collocation points (should be > t_switch)
            x_intermediate: Spatial points for intermediate constraint
            t_bc: Temporal points for BC

        Returns:
            Tuple of (loss, loss_dict)
        """
        loss_dict = {}

        # PDE residual loss
        residual = self.phase2_net.compute_pde_residual(x_collocation, t_collocation)
        pde_loss = torch.mean(residual ** 2)
        loss_dict["pde_loss"] = pde_loss.item()

        # Intermediate constraint loss (replaces IC)
        intermediate_loss = self.compute_intermediate_loss(x_intermediate)
        loss_dict["intermediate_loss"] = intermediate_loss.item()

        # Boundary condition loss
        bc_loss = self.phase2_net.compute_bc_loss(t_bc)
        loss_dict["bc_loss"] = bc_loss.item()

        # Total loss
        total_loss = (
            self.phase2_net.lambda_pde * pde_loss +
            self.lambda_intermediate * intermediate_loss +
            self.phase2_net.lambda_bc * bc_loss
        )
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["data_loss"] = (intermediate_loss + bc_loss).item()
        loss_dict["physics_loss"] = pde_loss.item()

        return total_loss, loss_dict

    def compute_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_ic: torch.Tensor,
        t_bc: torch.Tensor,
        metadata: Optional[Dict] = None,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss (for full-domain evaluation).

        This computes the loss across both phases, useful for monitoring
        during training. For actual two-phase training, use
        compute_phase1_loss and compute_phase2_loss separately.

        Args:
            x_collocation: Spatial collocation points
            t_collocation: Temporal collocation points
            x_ic: Spatial points for IC
            t_bc: Temporal points for BC
            metadata: Optional metadata
            enable_physics: Whether to compute physics loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}

        # Split collocation points by phase
        mask_phase1 = t_collocation <= self.t_switch
        mask_phase2 = t_collocation > self.t_switch

        total_loss = torch.tensor(0.0, device=x_collocation.device)

        # Phase 1 loss
        if mask_phase1.any():
            x_p1 = x_collocation[mask_phase1]
            t_p1 = t_collocation[mask_phase1]
            residual1 = self.phase1_net.compute_pde_residual(x_p1, t_p1)
            pde_loss1 = torch.mean(residual1 ** 2)
            total_loss = total_loss + self.phase1_net.lambda_pde * pde_loss1
            loss_dict["phase1_pde_loss"] = pde_loss1.item()

        # Phase 2 loss
        if mask_phase2.any():
            x_p2 = x_collocation[mask_phase2]
            t_p2 = t_collocation[mask_phase2]
            residual2 = self.phase2_net.compute_pde_residual(x_p2, t_p2)
            pde_loss2 = torch.mean(residual2 ** 2)
            total_loss = total_loss + self.phase2_net.lambda_pde * pde_loss2
            loss_dict["phase2_pde_loss"] = pde_loss2.item()

        # IC loss
        ic_loss = self.phase1_net.compute_ic_loss(x_ic)
        total_loss = total_loss + self.phase1_net.lambda_ic * ic_loss
        loss_dict["ic_loss"] = ic_loss.item()

        # Intermediate constraint loss
        intermediate_loss = self.compute_intermediate_loss(x_ic)
        total_loss = total_loss + self.lambda_intermediate * intermediate_loss
        loss_dict["intermediate_loss"] = intermediate_loss.item()

        # BC loss (both phases)
        bc_loss = (
            self.phase1_net.compute_bc_loss(t_bc[t_bc <= self.t_switch]) +
            self.phase2_net.compute_bc_loss(t_bc[t_bc > self.t_switch])
        )
        total_loss = total_loss + self.phase1_net.lambda_bc * bc_loss
        loss_dict["bc_loss"] = bc_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict

    def predict_field(
        self,
        x_grid: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Generate solution field for visualization (meshgrid inputs)."""
        x_flat = x_grid.flatten()
        t_flat = t_grid.flatten()
        with torch.enable_grad():
            u_flat = self.forward(x_flat, t_flat)
        return u_flat.view_as(x_grid)

    def continuity_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuity error across the phase boundary for diagnostics."""
        with torch.no_grad():
            return torch.sqrt(self.compute_intermediate_loss(x) + 1e-8)

    def freeze_phase1(self):
        """Freeze phase 1 network parameters."""
        for param in self.phase1_net.parameters():
            param.requires_grad = False
        logger.info("Phase 1 network frozen")

    def unfreeze_phase1(self):
        """Unfreeze phase 1 network parameters."""
        for param in self.phase1_net.parameters():
            param.requires_grad = True
        logger.info("Phase 1 network unfrozen")

    def get_trainable_params(self, phase: int = 1) -> List[torch.nn.Parameter]:
        """
        Get trainable parameters for specified phase.

        Args:
            phase: 1 or 2

        Returns:
            List of trainable parameters
        """
        if phase == 1:
            return list(self.phase1_net.parameters())
        elif phase == 2:
            return list(self.phase2_net.parameters())
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")


def create_burgers_pinn(
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create Burgers' equation PINN models.

    Args:
        model_type: "standard" for BurgersPINN, "dual_phase" for DualPhasePINN
        **kwargs: Additional arguments passed to the model

    Returns:
        PINN model instance
    """
    if model_type == "standard":
        return BurgersPINN(**kwargs)
    elif model_type == "dual_phase":
        return DualPhasePINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
