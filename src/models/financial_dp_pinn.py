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
from typing import Dict, List, Optional, Tuple, Any

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

        # Optional scaler to de-standardise price-based quantities
        self.price_mean: Optional[float] = None
        self.price_std: Optional[float] = None
        self.normalise_residuals: bool = True

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

    def set_scaler(self, price_mean: Optional[float], price_std: Optional[float]):
        """Register scaler stats used to de-standardise prices for physics terms."""
        if price_std is not None and abs(price_std) < 1e-12:
            raise ValueError("price_std must be non-zero for physics residuals")
        self.price_mean = price_mean
        self.price_std = price_std

    def _standardize(self, residual: torch.Tensor, key: str) -> torch.Tensor:
        """Standardise residual magnitude to keep λ interpretable at dt=1/252."""
        if not self.normalise_residuals:
            return residual
        residual_std = residual.std() + 1e-8
        self._residual_rms[key] = float(residual_std.detach())
        return residual / residual_std

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
        mu_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Geometric Brownian Motion residual.

        GBM: dS = μS dt + σS dW
        Residual: log(S_next/S) - (μ - 0.5σ²)·dt
        """
        log_returns = torch.log(torch.clamp(S_next / (S + 1e-8), min=1e-8))

        # Drift per step estimated in log space to avoid mixing domains
        if mu_override is None:
            mu_step = log_returns.mean(dim=1, keepdim=True)
        else:
            mu_step = mu_override

        expected = mu_step - 0.5 * (volatility ** 2) * self.config.dt
        residual = (log_returns - expected) / self.config.dt

        residual = self._standardize(residual, 'gbm')
        return torch.mean(residual ** 2)

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
        residual = self._standardize(residual, 'ou')

        return torch.mean(residual ** 2)

    def black_scholes_residual(
        self,
        V: torch.Tensor,
        S: torch.Tensor,
        dV_dS: torch.Tensor,
        d2V_dS2: torch.Tensor,
        sigma: torch.Tensor,
        price_mean: Optional[float] = None,
        price_std: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Simplified Black-Scholes PDE residual.

        BS: ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

        For price prediction, we use steady-state approximation.
        """
        r = self.config.risk_free_rate

        # De-normalise if scaler provided (avoid mixing z-scores with raw σ,r)
        price_mean = price_mean if price_mean is not None else self.price_mean
        price_std = price_std if price_std is not None else self.price_std
        if price_std is not None:
            S = S * price_std + (price_mean or 0.0)
            V = V * price_std + (price_mean or 0.0)

        bs_residual = (
            0.5 * sigma ** 2 * S ** 2 * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        bs_residual = self._standardize(bs_residual, 'bs')
        return torch.mean(bs_residual ** 2)

    def compute_derivatives(
        self,
        model: nn.Module,
        x: torch.Tensor,
        price_idx: int = 0,
        price_mean: Optional[float] = None,
        price_std: Optional[float] = None,
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

        # De-normalise outputs for price-based physics residuals when scalers are available
        price_mean = price_mean if price_mean is not None else self.price_mean
        price_std = price_std if price_std is not None else self.price_std
        if price_std is not None:
            V = V * price_std + (price_mean or 0.0)

            # Chain rule adjustment for derivatives: d/dS = (1/std) d/dS_norm
            dV_dS = dV_dS / (price_std + 1e-8)
            d2V_dS2 = d2V_dS2 / (price_std + 1e-8)

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
        price_mean = metadata.get('price_mean')
        price_std = metadata.get('price_std')
        if price_mean is not None or price_std is not None:
            try:
                self.physics_loss.set_scaler(price_mean, price_std)
            except Exception as e:
                logger.debug(f"Scaler registration failed: {e}")

        data_loss = nn.functional.mse_loss(predictions, targets)

        loss_dict: Dict[str, float] = {
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

        # GBM constraint (weak trend prior)
        if self.config.lambda_gbm > 0 and prices is not None and prices.shape[1] > 1:
            try:
                S = prices[:, :-1]
                S_next = prices[:, 1:]
                vol = volatilities[:, :-1] if volatilities is not None else torch.ones_like(S) * 0.2

                gbm_loss = self.physics_loss.gbm_residual(S, S_next, vol)
                physics_loss = physics_loss + self.config.lambda_gbm * gbm_loss
                loss_dict['gbm_loss'] = gbm_loss.item()
                loss_dict['gbm_residual_rms'] = self.physics_loss._residual_rms.get('gbm', 0.0)
            except Exception as e:
                logger.debug(f"GBM loss failed: {e}")

        # OU constraint on returns (primary regulariser)
        if self.config.lambda_ou > 0 and returns is not None and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]
                X_next = returns[:, 1:]

                ou_loss = self.physics_loss.ou_residual(X, X_next)
                physics_loss = physics_loss + self.config.lambda_ou * ou_loss
                loss_dict['ou_loss'] = ou_loss.item()
                loss_dict['ou_residual_rms'] = self.physics_loss._residual_rms.get('ou', 0.0)
                loss_dict['theta_ou'] = float(self.physics_loss.theta.detach())
                loss_dict['mu_ou'] = float(self.physics_loss.mu.detach())
            except Exception as e:
                logger.debug(f"OU loss failed: {e}")

        # Black-Scholes constraint (no-arbitrage inspired, small weight)
        if self.config.lambda_bs > 0 and inputs is not None and prices is not None:
            try:
                V, dV_dS, d2V_dS2 = self.physics_loss.compute_derivatives(
                    self, inputs, price_idx=0, price_mean=price_mean, price_std=price_std
                )
                S = prices[:, -1:]
                sigma = volatilities[:, -1:] if volatilities is not None else torch.ones_like(S) * 0.2

                bs_loss = self.physics_loss.black_scholes_residual(
                    V, S, dV_dS, d2V_dS2, sigma, price_mean=price_mean, price_std=price_std
                )
                physics_loss = physics_loss + self.config.lambda_bs * bs_loss
                loss_dict['bs_loss'] = bs_loss.item()
                loss_dict['bs_residual_rms'] = self.physics_loss._residual_rms.get('bs', 0.0)
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
        self._forward_cache: Dict[str, Any] = {}

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
        """Forward pass supporting per-phase evaluation and cached diagnostics."""
        batch_size, seq_len, _ = x.shape
        split_idx = max(2, min(seq_len - 1, int(seq_len * self.phase_split)))
        split_tensor = torch.tensor(split_idx, device=x.device)

        if phase == 1:
            return self.phase1_net(x[:, :split_idx, :])
        if phase == 2:
            return self.phase2_net(x[:, split_idx - 1:, :])

        x_phase1 = x[:, :split_idx, :]
        x_phase2 = x[:, split_idx - 1:, :]  # include boundary for continuity

        phase1_pred = self.phase1_net(x_phase1)
        phase2_pred = self.phase2_net(x_phase2)
        combined = 0.5 * (phase1_pred + phase2_pred)

        # Cache for loss/diagnostics
        self._forward_cache = {
            'split_idx': split_tensor,
            'phase1_pred': phase1_pred,
            'phase2_pred': phase2_pred,
            'combined_pred': combined,
            'phase1_input': x_phase1,
            'phase2_input': x_phase2,
        }

        return combined

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss using a fixed temporal split.
        """
        if not self._forward_cache:
            # Ensure forward cache exists (e.g., during evaluation-only calls)
            _ = self.forward(metadata['inputs'])

        split_idx_cache = self._forward_cache.get('split_idx')
        if split_idx_cache is None:
            _ = self.forward(metadata['inputs'])
            split_idx_cache = self._forward_cache.get('split_idx')

        if split_idx_cache is None:
            raise RuntimeError("Forward cache missing split index for dual-phase PINN")

        split_idx_tensor = torch.as_tensor(split_idx_cache)  # type: ignore[arg-type]
        split_idx = int(split_idx_tensor.item())

        phase1_pred = self._forward_cache.get('phase1_pred')
        phase2_pred = self._forward_cache.get('phase2_pred')

        if phase1_pred is None or phase2_pred is None:
            _ = self.forward(metadata['inputs'])
            phase1_pred = self._forward_cache.get('phase1_pred')
            phase2_pred = self._forward_cache.get('phase2_pred')
            split_idx_cache = self._forward_cache.get('split_idx')
            split_idx_tensor = torch.as_tensor(split_idx_cache)  # type: ignore[arg-type]
            split_idx = int(split_idx_tensor.item())

        assert phase1_pred is not None and phase2_pred is not None

        # Align targets for both phases (both forecast next-day price)
        phase1_data_loss = nn.functional.mse_loss(phase1_pred, targets)
        phase2_data_loss = nn.functional.mse_loss(phase2_pred, targets)
        data_loss = 0.5 * (phase1_data_loss + phase2_data_loss)

        # Physics losses on temporally split metadata
        meta_phase1 = self._split_metadata_time(metadata, end=split_idx)
        meta_phase2 = self._split_metadata_time(metadata, start=max(split_idx - 1, 0))

        physics_loss = torch.tensor(0.0, device=predictions.device)
        p1_loss, dict1 = self.phase1_net.compute_loss(
            phase1_pred, targets, meta_phase1, enable_physics
        )
        p2_loss, dict2 = self.phase2_net.compute_loss(
            phase2_pred, targets, meta_phase2, enable_physics
        )

        # Remove duplicated data contribution from sub-losses
        if enable_physics:
            physics_loss = (
                p1_loss - self.config.lambda_data * phase1_data_loss
                + p2_loss - self.config.lambda_data * phase2_data_loss
            )

        continuity_loss = nn.functional.mse_loss(phase1_pred, phase2_pred)
        total_loss = (
            self.config.lambda_data * data_loss
            + physics_loss
            + self.config.lambda_intermediate * continuity_loss
        )

        loss_dict: Dict[str, float] = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item() if enable_physics else 0.0,
            'continuity_loss': continuity_loss.item(),
            'phase_split_index': float(split_idx),
            'phase1_data_loss': phase1_data_loss.item(),
            'phase2_data_loss': phase2_data_loss.item(),
        }

        for key in ['gbm_loss', 'ou_loss', 'bs_loss', 'gbm_residual_rms', 'ou_residual_rms', 'bs_residual_rms']:
            if key in dict1 or key in dict2:
                loss_dict[key] = dict1.get(key, 0.0) + dict2.get(key, 0.0)

        return total_loss, loss_dict

    def _split_metadata_time(self, metadata: Dict, start: Optional[int] = 0, end: Optional[int] = None) -> Dict:
        """Temporal slice of sequence-like metadata for phase-specific physics losses."""
        result: Dict[str, Any] = {}
        start_idx = 0 if start is None else int(start)
        end_idx = None if end is None else int(end)

        for key, value in metadata.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] >= 2:
                result[key] = value[:, start_idx:end_idx, ...]
            else:
                result[key] = value
        return result


class AdaptiveFinancialDualPhasePINN(FinancialDualPhasePINN):
    """Dual-phase PINN with a learned, volatility-driven phase boundary."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        phase_split: float = 0.5,
        config: Optional[FinancialPhysicsConfig] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            phase_split=phase_split,
            config=config,
        )

        gate_hidden = max(hidden_dim // 4, 16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )

    def _compute_regime_score(self, x: torch.Tensor) -> torch.Tensor:
        """Use rolling volatility and residual magnitude as regime indicator."""
        price_feature = x[:, :, 0]
        diffs = price_feature[:, 1:] - price_feature[:, :-1]
        rolling_vol = diffs.std(dim=1, keepdim=True)
        tail = max(3, min(diffs.shape[1], 10))
        recent_vol = diffs[:, -tail:].std(dim=1, keepdim=True)
        residual_mag = diffs.abs().mean(dim=1, keepdim=True)
        features = torch.cat([rolling_vol, recent_vol, residual_mag], dim=-1)
        return self.gate_mlp(features)

    def forward(self, x: torch.Tensor, phase: Optional[int] = None) -> torch.Tensor:
        if phase is not None:
            return super().forward(x, phase=phase)

        gate_logits = self._compute_regime_score(x)
        gate = torch.sigmoid(gate_logits)  # weight for phase 1

        phase1_pred = self.phase1_net(x)
        phase2_pred = self.phase2_net(x)
        combined = gate * phase1_pred + (1.0 - gate) * phase2_pred

        self._forward_cache = {
            'gate': gate,
            'gate_logits': gate_logits,
            'phase1_pred': phase1_pred,
            'phase2_pred': phase2_pred,
            'combined_pred': combined,
            'split_idx': torch.tensor(int(x.shape[1] * self.phase_split), device=x.device),
            'phase1_input': x,
            'phase2_input': x,
        }

        return combined

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._forward_cache:
            _ = self.forward(metadata['inputs'])

        gate = self._forward_cache.get('gate')
        phase1_pred = self._forward_cache.get('phase1_pred')
        phase2_pred = self._forward_cache.get('phase2_pred')

        if gate is None or phase1_pred is None or phase2_pred is None:
            _ = self.forward(metadata['inputs'])
            gate = self._forward_cache.get('gate')
            phase1_pred = self._forward_cache.get('phase1_pred')
            phase2_pred = self._forward_cache.get('phase2_pred')

        if gate is None or phase1_pred is None or phase2_pred is None:
            raise RuntimeError("Adaptive gate not initialised before loss computation")

        # Data loss weighted by gate
        phase1_data = (gate * (phase1_pred - targets) ** 2).mean()
        phase2_data = ((1.0 - gate) * (phase2_pred - targets) ** 2).mean()
        data_loss = 0.5 * (phase1_data + phase2_data)

        # Physics losses (shared metadata, but weighted by expected regime occupancy)
        p1_loss, dict1 = self.phase1_net.compute_loss(
            phase1_pred, targets, metadata, enable_physics
        )
        p2_loss, dict2 = self.phase2_net.compute_loss(
            phase2_pred, targets, metadata, enable_physics
        )

        physics_loss = torch.tensor(0.0, device=predictions.device)
        if enable_physics:
            gate_mean = gate.mean()
            physics_loss = gate_mean * (p1_loss - self.config.lambda_data * nn.functional.mse_loss(phase1_pred, targets))
            physics_loss = physics_loss + (1 - gate_mean) * (p2_loss - self.config.lambda_data * nn.functional.mse_loss(phase2_pred, targets))

        continuity_loss = torch.mean(gate * (1.0 - gate) * (phase1_pred - phase2_pred) ** 2)
        total_loss = (
            self.config.lambda_data * data_loss
            + physics_loss
            + self.config.lambda_intermediate * continuity_loss
        )

        loss_dict: Dict[str, float] = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item() if enable_physics else 0.0,
            'continuity_loss': continuity_loss.item(),
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
        }

        for key in ['gbm_loss', 'ou_loss', 'bs_loss', 'gbm_residual_rms', 'ou_residual_rms', 'bs_residual_rms']:
            if key in dict1 or key in dict2:
                loss_dict[key] = dict1.get(key, 0.0) + dict2.get(key, 0.0)

        return total_loss, loss_dict

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
