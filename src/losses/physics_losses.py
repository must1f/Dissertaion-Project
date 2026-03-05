"""
Physics-Informed Loss Functions

Loss functions derived from quantitative finance equations:
- Geometric Brownian Motion (GBM) for price dynamics
- Ornstein-Uhlenbeck (OU) for mean reversion
- Black-Scholes for no-arbitrage constraint
- Langevin dynamics for momentum modeling
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from abc import ABC, abstractmethod
import math

from ..constants import (
    DAILY_TIME_STEP,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_LAMBDA_GBM,
    EPSILON,
)


class PhysicsResidual(nn.Module, ABC):
    """
    Base class for physics-informed residual losses.

    Each physics residual encodes a differential equation that
    financial time series should approximately satisfy.
    """

    def __init__(
        self,
        weight: float = DEFAULT_LAMBDA_GBM,
        dt: float = DAILY_TIME_STEP,
        eps: float = EPSILON
    ):
        """
        Args:
            weight: Weight for this physics constraint
            dt: Time step in years
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = weight
        self.dt = dt
        self.eps = eps

    @abstractmethod
    def compute_residual(self, **kwargs) -> torch.Tensor:
        """Compute the physics residual (should be ~0 if physics holds)."""
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute weighted residual loss."""
        residual = self.compute_residual(**kwargs)
        return self.weight * torch.mean(residual ** 2)


class GBMResidual(PhysicsResidual):
    """
    Geometric Brownian Motion residual.

    GBM equation: dS = μS dt + σS dW
    Residual: dS/dt - μS ≈ 0 (deterministic part)

    Used to encode trend-following behavior in prices.
    """

    def compute_residual(
        self,
        prices: torch.Tensor,
        drift: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute GBM residual.

        Args:
            prices: Price tensor [batch, seq_len] or [batch, seq_len, 1]
            drift: Optional drift parameter μ. If None, estimated from returns.
            returns: Optional returns for drift estimation.

        Returns:
            Residual tensor
        """
        # Ensure 2D
        if prices.dim() == 3:
            prices = prices.squeeze(-1)

        if prices.shape[1] < 2:
            return torch.tensor(0.0, device=prices.device)

        # Current and next prices
        S = prices[:, :-1]
        S_next = prices[:, 1:]

        # Approximate derivative
        dS_dt = (S_next - S) / self.dt

        # Estimate drift if not provided
        if drift is None:
            if returns is not None:
                if returns.dim() == 3:
                    returns = returns.squeeze(-1)
                drift = returns.mean(dim=1, keepdim=True)
            else:
                # Estimate from price changes
                drift = ((S_next - S) / (S + self.eps)).mean(dim=1, keepdim=True)

        # GBM residual: dS/dt - μS
        residual = dS_dt - drift * S

        return residual


class OUResidual(PhysicsResidual):
    """
    Ornstein-Uhlenbeck mean reversion residual.

    OU equation: dX = θ(μ - X)dt + σdW
    Residual: dX/dt - θ(μ - X) ≈ 0

    Used to encode mean-reverting behavior in returns or log-prices.
    """

    def __init__(
        self,
        weight: float = 0.1,
        theta_init: float = 1.0,
        learnable: bool = True,
        **kwargs
    ):
        """
        Args:
            weight: Weight for this constraint
            theta_init: Initial mean reversion speed
            learnable: If True, theta is a learnable parameter
        """
        super().__init__(weight=weight, **kwargs)

        if learnable:
            # Learnable parameter (constrained positive via softplus)
            self.theta_raw = nn.Parameter(torch.tensor(theta_init))
        else:
            self.register_buffer('theta_raw', torch.tensor(theta_init))

        self.learnable = learnable

    @property
    def theta(self) -> torch.Tensor:
        """Mean reversion speed (constrained positive)."""
        return torch.nn.functional.softplus(self.theta_raw)

    def compute_residual(
        self,
        values: torch.Tensor,
        long_term_mean: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute OU residual.

        Args:
            values: Process values [batch, seq_len] (e.g., returns)
            long_term_mean: Optional μ. If None, estimated from data.

        Returns:
            Residual tensor
        """
        if values.dim() == 3:
            values = values.squeeze(-1)

        if values.shape[1] < 2:
            return torch.tensor(0.0, device=values.device)

        X = values[:, :-1]
        X_next = values[:, 1:]

        # Approximate derivative
        dX_dt = (X_next - X) / self.dt

        # Long-term mean
        if long_term_mean is None:
            long_term_mean = values.mean(dim=1, keepdim=True)

        # OU residual: dX/dt - θ(μ - X)
        theta = self.theta.to(values.device)
        residual = dX_dt - theta * (long_term_mean - X)

        return residual


class BlackScholesResidual(PhysicsResidual):
    """
    Black-Scholes PDE residual.

    Black-Scholes: ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

    For prediction tasks, we use a simplified steady-state form:
    ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV ≈ 0

    Encodes no-arbitrage pricing relationship.
    """

    def __init__(
        self,
        weight: float = DEFAULT_LAMBDA_GBM,
        risk_free_rate: float = RISK_FREE_RATE,
        **kwargs
    ):
        """
        Args:
            weight: Weight for this constraint
            risk_free_rate: Annualized risk-free rate
        """
        super().__init__(weight=weight, **kwargs)
        self.risk_free_rate = risk_free_rate

    def compute_residual(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        volatility: torch.Tensor,
        price_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute Black-Scholes residual using automatic differentiation.

        Args:
            model: Neural network model
            inputs: Input tensor [batch, seq_len, features]
            volatility: Volatility tensor [batch] or [batch, 1]
            price_idx: Index of price feature

        Returns:
            BS residual
        """
        r = self.risk_free_rate

        # Enable gradients
        x = inputs.clone().detach().requires_grad_(True)

        # Forward pass
        V = model(x)
        if V.dim() == 1:
            V = V.unsqueeze(-1)

        # Extract price from last timestep
        S = inputs[:, -1, price_idx:price_idx + 1]

        # Ensure volatility shape
        if volatility.dim() == 0:
            sigma = volatility.expand(inputs.shape[0], 1)
        elif volatility.dim() == 1:
            sigma = volatility.unsqueeze(-1)
        else:
            sigma = volatility

        # First derivative dV/dS via autograd
        grad_outputs = torch.ones_like(V)
        dV_dx = torch.autograd.grad(
            outputs=V,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        dV_dS = dV_dx[:, -1, price_idx:price_idx + 1]

        # Second derivative d²V/dS²
        d2V_dx = torch.autograd.grad(
            outputs=dV_dS,
            inputs=x,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
            retain_graph=True
        )[0]
        d2V_dS2 = d2V_dx[:, -1, price_idx:price_idx + 1]

        # Black-Scholes residual (steady-state form)
        residual = (
            0.5 * (sigma ** 2) * (S ** 2) * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        return residual

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        volatility: torch.Tensor,
        price_idx: int = 0
    ) -> torch.Tensor:
        """Compute weighted BS loss."""
        residual = self.compute_residual(model, inputs, volatility, price_idx)
        return self.weight * torch.mean(residual ** 2)


class LangevinResidual(PhysicsResidual):
    """
    Langevin dynamics residual.

    Langevin: dX = -γ∇U(X)dt + √(2γT)dW
    Residual: dX/dt + γ∇U ≈ 0

    Models momentum effects with friction and noise.
    Useful for capturing market momentum dynamics.
    """

    def __init__(
        self,
        weight: float = 0.1,
        gamma_init: float = 0.5,
        temperature_init: float = 0.1,
        learnable: bool = True,
        **kwargs
    ):
        """
        Args:
            weight: Weight for this constraint
            gamma_init: Initial friction coefficient
            temperature_init: Initial temperature
            learnable: If True, parameters are learnable
        """
        super().__init__(weight=weight, **kwargs)

        if learnable:
            self.gamma_raw = nn.Parameter(torch.tensor(gamma_init))
            self.temperature_raw = nn.Parameter(torch.tensor(temperature_init))
        else:
            self.register_buffer('gamma_raw', torch.tensor(gamma_init))
            self.register_buffer('temperature_raw', torch.tensor(temperature_init))

        self.learnable = learnable

    @property
    def gamma(self) -> torch.Tensor:
        """Friction coefficient (constrained positive)."""
        return torch.nn.functional.softplus(self.gamma_raw)

    @property
    def temperature(self) -> torch.Tensor:
        """Temperature parameter (constrained positive)."""
        return torch.nn.functional.softplus(self.temperature_raw)

    def compute_residual(
        self,
        values: torch.Tensor,
        potential_gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Langevin residual.

        Args:
            values: Process values [batch, seq_len]
            potential_gradient: Optional ∇U. If None, approximated.

        Returns:
            Residual tensor
        """
        if values.dim() == 3:
            values = values.squeeze(-1)

        if values.shape[1] < 2:
            return torch.tensor(0.0, device=values.device)

        X = values[:, :-1]
        X_next = values[:, 1:]

        # Approximate derivative
        dX_dt = (X_next - X) / self.dt

        # Approximate potential gradient (negative returns as simple proxy)
        if potential_gradient is None:
            potential_gradient = -X

        # Langevin residual: dX/dt + γ∇U
        gamma = self.gamma.to(values.device)
        residual = dX_dt + gamma * potential_gradient

        return residual


class NoArbitrageResidual(PhysicsResidual):
    """
    No-arbitrage constraint residual.

    Enforces that expected returns equal the risk-free rate
    under risk-neutral measure: E[R] - r ≈ 0

    This is a weaker form of no-arbitrage that doesn't require
    full Black-Scholes PDE computation.
    """

    def __init__(
        self,
        weight: float = DEFAULT_LAMBDA_GBM,
        risk_free_rate: float = RISK_FREE_RATE,
        **kwargs
    ):
        """
        Args:
            weight: Weight for this constraint
            risk_free_rate: Annualized risk-free rate
        """
        super().__init__(weight=weight, **kwargs)
        self.risk_free_rate = risk_free_rate

    def compute_residual(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute no-arbitrage residual.

        Args:
            predicted_returns: Model's predicted returns
            actual_returns: Optional actual returns (for calibration)

        Returns:
            Residual tensor
        """
        if predicted_returns.dim() == 3:
            predicted_returns = predicted_returns.squeeze(-1)

        # Daily risk-free rate
        r_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR

        # Expected return should equal risk-free rate
        expected_return = predicted_returns.mean(dim=-1, keepdim=True)
        residual = expected_return - r_daily

        return residual


class MomentumResidual(PhysicsResidual):
    """
    Momentum constraint residual.

    Enforces autocorrelation structure in returns:
    R_t ≈ α * R_{t-1} + ε

    Captures empirical momentum effects in financial markets.
    """

    def __init__(
        self,
        weight: float = 0.1,
        momentum_coef_init: float = 0.1,
        learnable: bool = True,
        **kwargs
    ):
        """
        Args:
            weight: Weight for this constraint
            momentum_coef_init: Initial momentum coefficient α
            learnable: If True, coefficient is learnable
        """
        super().__init__(weight=weight, **kwargs)

        if learnable:
            self.alpha_raw = nn.Parameter(torch.tensor(momentum_coef_init))
        else:
            self.register_buffer('alpha_raw', torch.tensor(momentum_coef_init))

        self.learnable = learnable

    @property
    def alpha(self) -> torch.Tensor:
        """Momentum coefficient (constrained to [-1, 1])."""
        return torch.tanh(self.alpha_raw)

    def compute_residual(
        self,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute momentum residual.

        Args:
            returns: Return series [batch, seq_len]

        Returns:
            Residual tensor
        """
        if returns.dim() == 3:
            returns = returns.squeeze(-1)

        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        R_prev = returns[:, :-1]
        R_curr = returns[:, 1:]

        alpha = self.alpha.to(returns.device)
        residual = R_curr - alpha * R_prev

        return residual


def create_physics_loss(
    loss_type: str,
    **kwargs
) -> PhysicsResidual:
    """
    Factory function to create physics loss by name.

    Args:
        loss_type: Type of loss ('gbm', 'ou', 'black_scholes', 'langevin',
                   'no_arbitrage', 'momentum')
        **kwargs: Additional arguments for the loss

    Returns:
        PhysicsResidual instance

    Example:
        loss = create_physics_loss('gbm', weight=0.1)
        loss = create_physics_loss('ou', theta_init=1.5, learnable=True)
    """
    loss_map = {
        'gbm': GBMResidual,
        'geometric_brownian_motion': GBMResidual,
        'ou': OUResidual,
        'ornstein_uhlenbeck': OUResidual,
        'mean_reversion': OUResidual,
        'black_scholes': BlackScholesResidual,
        'bs': BlackScholesResidual,
        'langevin': LangevinResidual,
        'no_arbitrage': NoArbitrageResidual,
        'momentum': MomentumResidual,
    }

    loss_type = loss_type.lower().replace('-', '_')
    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown physics loss type: {loss_type}. "
            f"Available: {list(loss_map.keys())}"
        )

    return loss_map[loss_type](**kwargs)
