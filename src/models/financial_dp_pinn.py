"""
Financial Dual-Phase Physics-Informed Neural Network (Financial DP-PINN)

Implements a two-phase PINN architecture for financial time series forecasting,
embedding stochastic finance equations into the loss function:

1. Geometric Brownian Motion (GBM): dS = μS dt + σS dW
2. Ornstein-Uhlenbeck (OU): dX = θ(μ - X)dt + σdW
3. Black-Scholes PDE: ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

The dual-phase approach splits the time series into two phases:
- Phase 1: Historical data with initial condition constraint
- Phase 2: Recent data with intermediate constraint for continuity

This improves accuracy for regime changes and non-stationary financial data.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..constants import (
    DEFAULT_LAMBDA_GBM,
    DEFAULT_LAMBDA_OU,
    DEFAULT_LAMBDA_BS,
    RISK_FREE_RATE,
    DAILY_TIME_STEP,
)

logger = get_logger(__name__)


@dataclass
class FinancialPhysicsConfig:
    """Configuration for financial physics constraints."""
    lambda_gbm: float = DEFAULT_LAMBDA_GBM
    lambda_ou: float = DEFAULT_LAMBDA_OU
    lambda_bs: float = DEFAULT_LAMBDA_BS
    lambda_data: float = 1.0
    lambda_ic: float = 10.0  # Initial condition weight
    lambda_intermediate: float = 10.0  # Intermediate constraint weight
    risk_free_rate: float = RISK_FREE_RATE
    dt: float = DAILY_TIME_STEP
    # Learnable physics parameters
    theta_init: float = 1.0  # OU mean reversion speed
    mu_init: float = 0.0  # Long-term mean


class FinancialPhysicsLoss(nn.Module):
    """
    Financial physics loss with GBM, OU, and Black-Scholes constraints.

    Includes learnable physics parameters that are optimized during training.
    """

    def __init__(self, config: Optional[FinancialPhysicsConfig] = None):
        super().__init__()
        self.config = config or FinancialPhysicsConfig()

        # Learnable OU mean reversion speed (constrained positive via softplus)
        self.theta_raw = nn.Parameter(torch.tensor(self.config.theta_init))

        # Learnable long-term mean
        self.mu_raw = nn.Parameter(torch.tensor(self.config.mu_init))

        # Residual tracking for diagnostics
        self._residual_rms = {'gbm': 0.0, 'ou': 0.0, 'bs': 0.0}

        logger.info(
            f"FinancialPhysicsLoss initialized: λ_gbm={self.config.lambda_gbm}, "
            f"λ_ou={self.config.lambda_ou}, λ_bs={self.config.lambda_bs}"
        )

    @property
    def theta(self) -> torch.Tensor:
        """OU mean reversion speed (constrained positive)."""
        return torch.nn.functional.softplus(self.theta_raw)

    @property
    def mu(self) -> torch.Tensor:
        """Long-term mean."""
        return self.mu_raw

    def get_learned_params(self) -> Dict[str, float]:
        """Get current learned physics parameter values."""
        return {
            'theta': self.theta.item(),
            'mu': self.mu.item(),
        }

    def gbm_residual(
        self,
        S: torch.Tensor,
        S_next: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Geometric Brownian Motion residual.

        GBM: dS = μS dt + σS dW
        Residual: (S_next - S) / S - μ * dt (ignoring stochastic term)

        We check if returns follow expected drift pattern.
        """
        # Compute log returns
        log_returns = torch.log(S_next / (S + 1e-8))

        # Expected return under GBM: μ * dt - 0.5 * σ² * dt
        # For daily data, we estimate μ from mean return
        mu_est = log_returns.mean()
        sigma = volatility.mean()

        # GBM residual: actual return should be close to expected
        expected_return = mu_est - 0.5 * sigma ** 2 * self.config.dt
        residual = log_returns - expected_return

        # Normalize for consistent magnitude
        residual_std = residual.std() + 1e-8
        residual_norm = residual / residual_std
        self._residual_rms['gbm'] = float(residual_std.detach())

        return torch.mean(residual_norm ** 2)

    def ou_residual(
        self,
        X: torch.Tensor,
        X_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck mean reversion residual.

        OU: dX = θ(μ - X)dt + σdW
        Residual: dX/dt - θ(μ - X)

        Applied to log returns to model mean reversion.
        """
        theta = self.theta.to(X.device)
        mu = self.mu.to(X.device)

        # Approximate derivative
        dX_dt = (X_next - X) / self.config.dt

        # OU residual
        residual = dX_dt - theta * (mu - X)

        # Normalize
        residual_std = residual.std() + 1e-8
        residual_norm = residual / residual_std
        self._residual_rms['ou'] = float(residual_std.detach())

        return torch.mean(residual_norm ** 2)

    def black_scholes_residual(
        self,
        V: torch.Tensor,
        S: torch.Tensor,
        dV_dS: torch.Tensor,
        d2V_dS2: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified Black-Scholes PDE residual.

        BS: ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

        For price prediction, we use steady-state approximation.
        """
        r = self.config.risk_free_rate

        # Black-Scholes residual (steady-state)
        bs_residual = (
            0.5 * sigma ** 2 * S ** 2 * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        # Normalize
        residual_std = bs_residual.std() + 1e-8
        residual_norm = bs_residual / residual_std
        self._residual_rms['bs'] = float(residual_std.detach())

        return torch.mean(residual_norm ** 2)

    def compute_derivatives(
        self,
        model: nn.Module,
        x: torch.Tensor,
        price_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives using automatic differentiation.

        Returns: (V, dV_dS, d2V_dS2)
        """
        x_grad = x.clone().detach().requires_grad_(True)

        # Forward pass
        V = model(x_grad)
        if isinstance(V, tuple):
            V = V[0]
        if V.dim() == 1:
            V = V.unsqueeze(-1)

        # First derivative
        grad_outputs = torch.ones_like(V)
        dV_dx = torch.autograd.grad(
            outputs=V,
            inputs=x_grad,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        dV_dS = dV_dx[:, -1, price_idx:price_idx+1]

        # Second derivative
        d2V_dx = torch.autograd.grad(
            outputs=dV_dS,
            inputs=x_grad,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
            retain_graph=True,
        )[0]

        d2V_dS2 = d2V_dx[:, -1, price_idx:price_idx+1]

        return V, dV_dS, d2V_dS2


class FinancialPINNBase(nn.Module):
    """
    Base Financial PINN with LSTM backbone and financial physics constraints.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        config: Optional[FinancialPhysicsConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.config = config or FinancialPhysicsConfig()

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Physics loss
        self.physics_loss = FinancialPhysicsLoss(config)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"FinancialPINNBase initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}"
        )

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM and output layers."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with financial physics constraints.
        """
        # Data loss
        data_loss = nn.functional.mse_loss(predictions, targets)

        loss_dict = {
            'data_loss': data_loss.item(),
            'total_loss': data_loss.item(),
        }

        if not enable_physics:
            return data_loss, loss_dict

        physics_loss = torch.tensor(0.0, device=predictions.device)

        prices = metadata.get('prices')
        returns = metadata.get('returns')
        volatilities = metadata.get('volatilities')
        inputs = metadata.get('inputs')

        # GBM constraint
        if self.config.lambda_gbm > 0 and prices is not None and prices.shape[1] > 1:
            try:
                S = prices[:, :-1]
                S_next = prices[:, 1:]
                vol = volatilities[:, :-1] if volatilities is not None else torch.ones_like(S) * 0.2

                gbm_loss = self.physics_loss.gbm_residual(S, S_next, vol)
                physics_loss = physics_loss + self.config.lambda_gbm * gbm_loss
                loss_dict['gbm_loss'] = gbm_loss.item()
            except Exception as e:
                logger.debug(f"GBM loss failed: {e}")

        # OU constraint on returns
        if self.config.lambda_ou > 0 and returns is not None and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]
                X_next = returns[:, 1:]

                ou_loss = self.physics_loss.ou_residual(X, X_next)
                physics_loss = physics_loss + self.config.lambda_ou * ou_loss
                loss_dict['ou_loss'] = ou_loss.item()
            except Exception as e:
                logger.debug(f"OU loss failed: {e}")

        # Black-Scholes constraint
        if self.config.lambda_bs > 0 and inputs is not None and prices is not None:
            try:
                V, dV_dS, d2V_dS2 = self.physics_loss.compute_derivatives(
                    self, inputs, price_idx=0
                )
                S = prices[:, -1:]
                sigma = volatilities[:, -1:] if volatilities is not None else torch.ones_like(S) * 0.2

                bs_loss = self.physics_loss.black_scholes_residual(
                    V, S, dV_dS, d2V_dS2, sigma
                )
                physics_loss = physics_loss + self.config.lambda_bs * bs_loss
                loss_dict['bs_loss'] = bs_loss.item()
            except Exception as e:
                logger.debug(f"BS loss failed: {e}")

        total_loss = self.config.lambda_data * data_loss + physics_loss
        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class FinancialDualPhasePINN(nn.Module):
    """
    Dual-Phase Financial PINN for time series forecasting.

    Splits the time series into two phases:
    - Phase 1: Earlier data with initial condition constraint
    - Phase 2: Recent data with intermediate constraint

    This approach handles regime changes and non-stationarity better
    than single-phase models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        phase_split: float = 0.6,  # Fraction of data for phase 1
        config: Optional[FinancialPhysicsConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.phase_split = phase_split
        self.config = config or FinancialPhysicsConfig()

        # Phase 1 network (handles earlier data)
        self.phase1_net = FinancialPINNBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            config=config,
        )

        # Phase 2 network (handles recent data)
        self.phase2_net = FinancialPINNBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            config=config,
        )

        logger.info(
            f"FinancialDualPhasePINN initialized: phase_split={phase_split}"
        )

    def forward(self, x: torch.Tensor, phase: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]
            phase: Optional phase selection (1 or 2). If None, uses both.
        """
        if phase == 1:
            return self.phase1_net(x)
        elif phase == 2:
            return self.phase2_net(x)
        else:
            # Ensemble both phases (weighted average)
            out1 = self.phase1_net(x)
            out2 = self.phase2_net(x)
            return 0.5 * (out1 + out2)

    def compute_phase1_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        initial_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for phase 1 with initial condition constraint.
        """
        # Base physics loss
        total_loss, loss_dict = self.phase1_net.compute_loss(
            predictions, targets, metadata, enable_physics=True
        )

        # Initial condition constraint
        if initial_value is not None:
            ic_loss = nn.functional.mse_loss(
                predictions[:1], initial_value[:1]
            )
            total_loss = total_loss + self.config.lambda_ic * ic_loss
            loss_dict['ic_loss'] = ic_loss.item()
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def compute_phase2_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        phase1_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for phase 2 with intermediate constraint.

        The intermediate constraint ensures continuity between phases.
        """
        # Base physics loss
        total_loss, loss_dict = self.phase2_net.compute_loss(
            predictions, targets, metadata, enable_physics=True
        )

        # Intermediate constraint: phase 2 start should match phase 1 end
        if phase1_predictions is not None and len(predictions) > 0:
            intermediate_loss = nn.functional.mse_loss(
                predictions[:1], phase1_predictions[-1:]
            )
            total_loss = total_loss + self.config.lambda_intermediate * intermediate_loss
            loss_dict['intermediate_loss'] = intermediate_loss.item()
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss (for full training).
        """
        batch_size = predictions.shape[0]
        split_idx = int(batch_size * self.phase_split)

        # Split data by phase
        pred1, pred2 = predictions[:split_idx], predictions[split_idx:]
        targ1, targ2 = targets[:split_idx], targets[split_idx:]

        # Split metadata
        meta1 = self._split_metadata(metadata, 0, split_idx)
        meta2 = self._split_metadata(metadata, split_idx, batch_size)

        # Phase 1 loss
        loss1, dict1 = self.phase1_net.compute_loss(pred1, targ1, meta1, enable_physics)

        # Phase 2 loss with intermediate constraint
        loss2, dict2 = self.phase2_net.compute_loss(pred2, targ2, meta2, enable_physics)

        # Intermediate constraint
        if split_idx > 0 and split_idx < batch_size:
            intermediate_loss = nn.functional.mse_loss(
                pred2[:1], pred1[-1:]
            )
            loss2 = loss2 + self.config.lambda_intermediate * intermediate_loss
            dict2['intermediate_loss'] = intermediate_loss.item()

        # Combine losses
        total_loss = loss1 + loss2
        loss_dict = {
            'phase1_loss': loss1.item(),
            'phase2_loss': loss2.item(),
            'total_loss': total_loss.item(),
            'data_loss': dict1.get('data_loss', 0) + dict2.get('data_loss', 0),
            'physics_loss': dict1.get('physics_loss', 0) + dict2.get('physics_loss', 0),
        }

        # Include individual physics losses
        for key in ['gbm_loss', 'ou_loss', 'bs_loss']:
            if key in dict1 or key in dict2:
                loss_dict[key] = dict1.get(key, 0) + dict2.get(key, 0)

        return total_loss, loss_dict

    def _split_metadata(
        self,
        metadata: Dict,
        start: int,
        end: int,
    ) -> Dict:
        """Split metadata tensors by index range."""
        result = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor) and value.shape[0] > 0:
                result[key] = value[start:end]
            else:
                result[key] = value
        return result

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

    def get_trainable_params(self, phase: int) -> List[nn.Parameter]:
        """Get trainable parameters for specified phase."""
        if phase == 1:
            return list(self.phase1_net.parameters())
        elif phase == 2:
            return list(self.phase2_net.parameters())
        else:
            raise ValueError(f"Invalid phase: {phase}")

    def get_learned_physics_params(self) -> Dict[str, Dict[str, float]]:
        """Get learned physics parameters from both phases."""
        return {
            'phase1': self.phase1_net.physics_loss.get_learned_params(),
            'phase2': self.phase2_net.physics_loss.get_learned_params(),
        }


def create_financial_dp_pinn(
    model_type: str = "dual_phase",
    input_dim: int = 14,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Financial DP-PINN models.

    Args:
        model_type: "standard" for single-phase, "dual_phase" for two-phase
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        **kwargs: Additional configuration

    Returns:
        Financial PINN model instance
    """
    config = FinancialPhysicsConfig(
        lambda_gbm=kwargs.get('lambda_gbm', DEFAULT_LAMBDA_GBM),
        lambda_ou=kwargs.get('lambda_ou', DEFAULT_LAMBDA_OU),
        lambda_bs=kwargs.get('lambda_bs', DEFAULT_LAMBDA_BS),
    )

    if model_type == "standard":
        return FinancialPINNBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            config=config,
        )
    elif model_type == "dual_phase":
        return FinancialDualPhasePINN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            phase_split=kwargs.get('phase_split', 0.6),
            config=config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
