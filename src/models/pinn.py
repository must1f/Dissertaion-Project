"""
Physics-Informed Neural Network (PINN) for financial forecasting

Embeds quantitative finance equations into the loss function:
- Geometric Brownian Motion (GBM)
- Black-Scholes PDE
- Ornstein-Uhlenbeck (OU) process
- Langevin dynamics
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import math

from ..utils.logger import get_logger
from .baseline import LSTMModel

logger = get_logger(__name__)


class PhysicsLoss(nn.Module):
    """
    Physics-based loss functions for financial time series
    """

    def __init__(
        self,
        lambda_gbm: float = 0.1,
        lambda_bs: float = 0.1,
        lambda_ou: float = 0.1,
        lambda_langevin: float = 0.1,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        dt: float = 1.0 / 252.0  # Time step in years (1 trading day)
    ):
        """
        Initialize physics loss

        Args:
            lambda_gbm: Weight for GBM constraint
            lambda_bs: Weight for Black-Scholes constraint
            lambda_ou: Weight for Ornstein-Uhlenbeck constraint
            lambda_langevin: Weight for Langevin dynamics constraint
            risk_free_rate: Risk-free interest rate (annualized)
            dt: Time step in years
        """
        super(PhysicsLoss, self).__init__()

        self.lambda_gbm = lambda_gbm
        self.lambda_bs = lambda_bs
        self.lambda_ou = lambda_ou
        self.lambda_langevin = lambda_langevin
        self.risk_free_rate = risk_free_rate
        self.dt = dt

    def gbm_residual(
        self,
        S: torch.Tensor,
        dS_dt: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Geometric Brownian Motion residual: dS/dt - μS - σS·dW

        Args:
            S: Stock price
            dS_dt: Time derivative of S (approximated)
            mu: Drift parameter
            sigma: Volatility parameter

        Returns:
            GBM residual
        """
        # GBM equation: dS = μS dt + σS dW
        # Residual: dS/dt - μS (we can't directly model dW)
        residual = dS_dt - mu * S

        # L2 loss on residual
        return torch.mean(residual ** 2)

    def black_scholes_residual(
        self,
        V: torch.Tensor,
        S: torch.Tensor,
        dV_dt: torch.Tensor,
        dV_dS: torch.Tensor,
        d2V_dS2: torch.Tensor,
        sigma: torch.Tensor,
        r: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Black-Scholes PDE residual:
        ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

        Args:
            V: Option value (or predicted price)
            S: Stock price
            dV_dt: ∂V/∂t
            dV_dS: ∂V/∂S
            d2V_dS2: ∂²V/∂S²
            sigma: Volatility
            r: Risk-free rate (optional, uses default if None)

        Returns:
            Black-Scholes residual
        """
        if r is None:
            r = torch.tensor(self.risk_free_rate, device=V.device, dtype=V.dtype)

        # Black-Scholes PDE
        bs_pde = (
            dV_dt
            + 0.5 * sigma ** 2 * S ** 2 * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        return torch.mean(bs_pde ** 2)

    def ornstein_uhlenbeck_residual(
        self,
        X: torch.Tensor,
        dX_dt: torch.Tensor,
        theta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck mean reversion residual:
        dX = θ(μ - X)dt + σdW

        Args:
            X: Process value (e.g., log returns)
            dX_dt: Time derivative of X
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility

        Returns:
            OU residual
        """
        # OU equation: dX = θ(μ - X)dt + σdW
        # Residual: dX/dt - θ(μ - X)
        residual = dX_dt - theta * (mu - X)

        return torch.mean(residual ** 2)

    def langevin_residual(
        self,
        X: torch.Tensor,
        dX_dt: torch.Tensor,
        grad_U: torch.Tensor,
        gamma: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Langevin dynamics residual (for momentum modeling):
        dX = -γ∇U(X)dt + √(2γT)dW

        Args:
            X: State variable
            dX_dt: Time derivative of X
            grad_U: Gradient of potential function
            gamma: Friction coefficient
            T: Temperature parameter

        Returns:
            Langevin residual
        """
        # Langevin equation: dX/dt = -γ∇U(X) + noise
        residual = dX_dt + gamma * grad_U

        return torch.mean(residual ** 2)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prices: torch.Tensor,
        returns: torch.Tensor,
        volatilities: torch.Tensor,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with physics constraints

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            prices: Price sequences
            returns: Return sequences
            volatilities: Volatility sequences
            enable_physics: Whether to include physics losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Data loss (MSE)
        data_loss = nn.functional.mse_loss(predictions, targets)

        loss_dict = {
            'data_loss': data_loss.item(),
            'total_loss': data_loss.item()
        }

        if not enable_physics:
            return data_loss, loss_dict

        # Initialize physics losses
        physics_loss = torch.tensor(0.0, device=predictions.device)

        # GBM constraint
        if self.lambda_gbm > 0 and prices.shape[1] > 1:
            try:
                S = prices[:, :-1]  # Current prices
                S_next = prices[:, 1:]  # Next prices
                dS_dt = (S_next - S) / self.dt  # Approximate derivative

                # Estimate drift and volatility from returns
                mu = returns.mean(dim=1, keepdim=True)
                sigma = volatilities[:, :-1]

                gbm_loss = self.gbm_residual(S, dS_dt, mu, sigma)
                physics_loss = physics_loss + self.lambda_gbm * gbm_loss
                loss_dict['gbm_loss'] = gbm_loss.item()

            except Exception as e:
                logger.debug(f"GBM loss computation failed: {e}")

        # Ornstein-Uhlenbeck constraint (mean reversion in returns)
        if self.lambda_ou > 0 and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]  # Current returns
                X_next = returns[:, 1:]  # Next returns
                dX_dt = (X_next - X) / self.dt

                # Estimate OU parameters
                theta = torch.tensor(1.0, device=returns.device)  # Mean reversion speed
                mu = returns.mean(dim=1, keepdim=True)  # Long-term mean
                sigma = returns.std(dim=1, keepdim=True)  # Volatility

                ou_loss = self.ornstein_uhlenbeck_residual(X, dX_dt, theta, mu, sigma)
                physics_loss = physics_loss + self.lambda_ou * ou_loss
                loss_dict['ou_loss'] = ou_loss.item()

            except Exception as e:
                logger.debug(f"OU loss computation failed: {e}")

        # Langevin dynamics (momentum)
        if self.lambda_langevin > 0 and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]
                X_next = returns[:, 1:]
                dX_dt = (X_next - X) / self.dt

                # Approximate gradient of potential
                grad_U = -returns[:, :-1]  # Negative returns as potential gradient

                gamma = torch.tensor(0.5, device=returns.device)
                T = torch.tensor(0.1, device=returns.device)

                langevin_loss = self.langevin_residual(X, dX_dt, grad_U, gamma, T)
                physics_loss = physics_loss + self.lambda_langevin * langevin_loss
                loss_dict['langevin_loss'] = langevin_loss.item()

            except Exception as e:
                logger.debug(f"Langevin loss computation failed: {e}")

        # Total loss
        total_loss = data_loss + physics_loss

        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network combining neural architecture with physics constraints
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        base_model: str = 'lstm',
        lambda_gbm: float = 0.1,
        lambda_bs: float = 0.0,  # Disabled by default (requires automatic differentiation)
        lambda_ou: float = 0.1,
        lambda_langevin: float = 0.1,
        risk_free_rate: float = 0.02,
        dt: float = 1.0 / 252.0
    ):
        """
        Initialize PINN model

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            output_dim: Output dimension
            dropout: Dropout probability
            base_model: Base model type ('lstm', 'gru', 'transformer')
            lambda_gbm: GBM loss weight
            lambda_bs: Black-Scholes loss weight
            lambda_ou: OU loss weight
            lambda_langevin: Langevin loss weight
            risk_free_rate: Risk-free rate
            dt: Time step
        """
        super(PINNModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_model_type = base_model

        # Base neural network model
        if base_model == 'lstm':
            from .baseline import LSTMModel
            self.base_model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout,
                bidirectional=False
            )
        elif base_model == 'gru':
            from .baseline import GRUModel
            self.base_model = GRUModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout,
                bidirectional=False
            )
        elif base_model == 'transformer':
            from .transformer import TransformerModel
            self.base_model = TransformerModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=8,
                num_encoder_layers=num_layers,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        # Physics loss module
        self.physics_loss = PhysicsLoss(
            lambda_gbm=lambda_gbm,
            lambda_bs=lambda_bs,
            lambda_ou=lambda_ou,
            lambda_langevin=lambda_langevin,
            risk_free_rate=risk_free_rate,
            dt=dt
        )

        logger.info(f"Initialized PINN model with base={base_model}, "
                   f"λ_gbm={lambda_gbm}, λ_ou={lambda_ou}, λ_langevin={lambda_langevin}")

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through base model

        Args:
            x: Input tensor
            return_hidden: Whether to return hidden states

        Returns:
            Predictions
        """
        if self.base_model_type in ['lstm', 'gru']:
            output, hidden = self.base_model(x)
            if return_hidden:
                return output, hidden
            return output
        else:  # transformer
            return self.base_model(x)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with physics constraints

        Args:
            predictions: Model predictions
            targets: Ground truth
            metadata: Batch metadata containing prices, returns, volatilities
            enable_physics: Whether to enable physics losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Extract physics data from metadata
        prices = metadata.get('prices', None)
        returns = metadata.get('returns', None)
        volatilities = metadata.get('volatilities', None)

        # Compute physics-informed loss
        return self.physics_loss(
            predictions=predictions,
            targets=targets,
            prices=prices,
            returns=returns,
            volatilities=volatilities,
            enable_physics=enable_physics
        )
