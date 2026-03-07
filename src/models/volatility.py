"""
Volatility Forecasting Models for PINN-Based Financial Forecasting

This module implements neural network models specifically designed for volatility
(variance) prediction, embedding physics-informed constraints from stochastic
volatility theory.

Models:
    - VolatilityLSTM: Baseline LSTM for volatility forecasting
    - VolatilityGRU: Baseline GRU for volatility forecasting
    - VolatilityTransformer: Transformer for volatility forecasting
    - VolatilityPINN: PINN with variance mean-reversion and GARCH constraints
    - HestonPINN: PINN based on Heston stochastic volatility model

Physics Constraints:
    1. Variance Mean-Reversion (OU process for variance)
    2. GARCH Consistency Residual
    3. Feller Condition (variance positivity)
    4. Leverage Effect (negative correlation between returns and variance)

References:
    - Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
      Volatility with Applications to Bond and Currency Options."
    - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger
from ..constants import (
    TRADING_DAYS_PER_YEAR,
    DAILY_TIME_STEP,
    RISK_FREE_RATE,
)

logger = get_logger(__name__)


# =============================================================================
# POSITIONAL ENCODING FOR TRANSFORMER
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Transpose for positional encoding, then transpose back
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        return self.dropout(x)


# =============================================================================
# BASELINE VOLATILITY MODELS
# =============================================================================

class VolatilityLSTM(nn.Module):
    """
    LSTM-based volatility forecasting model.

    Predicts h-day ahead realized variance using LSTM architecture
    with Softplus output to ensure positive variance predictions.

    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        output_horizon: Forecast horizon (for multi-step predictions)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon),
            nn.Softplus(),  # Ensures positive variance output
        )

        self._init_weights()

        logger.info(f"Initialized VolatilityLSTM: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, layers={num_layers}")

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, features]
            return_hidden: Whether to return hidden states

        Returns:
            Variance predictions of shape [batch, output_horizon]
        """
        lstm_out, hidden = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        variance_pred = self.output_head(last_hidden)  # [batch, output_horizon]

        if return_hidden:
            return variance_pred, hidden
        return variance_pred


class VolatilityGRU(nn.Module):
    """
    GRU-based volatility forecasting model.

    Similar to VolatilityLSTM but uses GRU cells which are
    computationally more efficient with fewer parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon),
            nn.Softplus(),
        )

        logger.info(f"Initialized VolatilityGRU: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, layers={num_layers}")

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        gru_out, hidden = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        variance_pred = self.output_head(last_hidden)

        if return_hidden:
            return variance_pred, hidden
        return variance_pred


class VolatilityTransformer(nn.Module):
    """
    Transformer-based volatility forecasting model.

    Uses causal self-attention for time-series modeling
    with positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_horizon: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_horizon = output_horizon

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_horizon),
            nn.Softplus(),
        )

        logger.info(f"Initialized VolatilityTransformer: d_model={d_model}, "
                   f"nhead={nhead}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal attention.

        Args:
            x: Input tensor of shape [batch, seq_len, features]

        Returns:
            Variance predictions of shape [batch, output_horizon]
        """
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)

        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        x = self.transformer(x, mask=mask)
        variance_pred = self.output_head(x[:, -1, :])

        return variance_pred


# =============================================================================
# PHYSICS-INFORMED VOLATILITY LOSS
# =============================================================================

class VolatilityPhysicsLoss(nn.Module):
    """
    Physics-based loss functions for volatility forecasting.

    Implements:
        1. Variance Mean-Reversion (OU process): dσ² = θ(σ̄² - σ²)dt
        2. GARCH Consistency: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
        3. Feller Condition: Penalize negative variance predictions
        4. Leverage Effect: Corr(r_t, σ²_{t+1}) < 0

    Args:
        lambda_ou: Weight for OU mean-reversion residual
        lambda_garch: Weight for GARCH consistency residual
        lambda_feller: Weight for Feller condition (positivity)
        lambda_leverage: Weight for leverage effect constraint
        dt: Time step in years (default: 1/252 for daily)
    """

    def __init__(
        self,
        lambda_ou: float = 0.1,
        lambda_garch: float = 0.1,
        lambda_feller: float = 0.05,
        lambda_leverage: float = 0.05,
        lambda_heston: float = 0.0,
        enable_heston_constraint: bool = False,
        dt: float = DAILY_TIME_STEP,
        # Learnable physics parameters (initial values)
        theta_init: float = 1.0,      # OU mean reversion speed
        omega_init: float = 0.00001,  # GARCH intercept
        alpha_init: float = 0.1,      # GARCH alpha
        beta_init: float = 0.85,      # GARCH beta
        kappa_init: float = 1.0,      # Heston mean reversion speed
        theta_long_init: float = 0.04,  # Long-run variance
        xi_init: float = 0.2,         # Vol-of-vol
        rho_init: float = -0.5,       # Leverage correlation
    ):
        super().__init__()

        self.lambda_ou = lambda_ou
        self.lambda_garch = lambda_garch
        self.lambda_feller = lambda_feller
        self.lambda_leverage = lambda_leverage
        self.lambda_heston = lambda_heston
        self.enable_heston_constraint = enable_heston_constraint
        self.dt = dt

        # Learnable physics parameters
        # Use raw parameters that are transformed to constrained space
        self.theta_raw = nn.Parameter(torch.tensor(math.log(theta_init)))
        self.omega_raw = nn.Parameter(torch.tensor(math.log(omega_init)))
        self.alpha_raw = nn.Parameter(torch.tensor(self._logit(alpha_init)))
        self.beta_raw = nn.Parameter(torch.tensor(self._logit(beta_init)))
        self.kappa_raw = nn.Parameter(torch.tensor(math.log(kappa_init)))
        self.theta_long_raw = nn.Parameter(torch.tensor(math.log(theta_long_init)))
        self.xi_raw = nn.Parameter(torch.tensor(math.log(xi_init)))
        self.rho_raw = nn.Parameter(torch.tensor(math.atanh(max(min(rho_init, 0.999), -0.999))))

        logger.info(f"Volatility physics loss initialized: λ_ou={lambda_ou}, "
                   f"λ_garch={lambda_garch}, λ_feller={lambda_feller}, "
                   f"λ_leverage={lambda_leverage}, λ_heston={lambda_heston}, enable_heston={enable_heston_constraint}")

    @staticmethod
    def _logit(p: float) -> float:
        """Logit transformation for (0, 1) -> R."""
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    @property
    def theta(self) -> torch.Tensor:
        """OU mean reversion speed (positive)."""
        return torch.exp(self.theta_raw)

    @property
    def omega(self) -> torch.Tensor:
        """GARCH intercept (positive)."""
        return torch.exp(self.omega_raw)

    @property
    def alpha(self) -> torch.Tensor:
        """GARCH alpha in (0, 0.5)."""
        return torch.sigmoid(self.alpha_raw) * 0.5

    @property
    def beta(self) -> torch.Tensor:
        """GARCH beta in (0.3, 0.95)."""
        return torch.sigmoid(self.beta_raw) * 0.65 + 0.3

    @property
    def kappa(self) -> torch.Tensor:
        """Heston mean reversion speed κ > 0."""
        return torch.nn.functional.softplus(self.kappa_raw)

    @property
    def theta_long(self) -> torch.Tensor:
        """Heston long-run variance θ > 0."""
        return torch.nn.functional.softplus(self.theta_long_raw)

    @property
    def xi(self) -> torch.Tensor:
        """Vol-of-vol ξ > 0."""
        return torch.nn.functional.softplus(self.xi_raw)

    @property
    def rho(self) -> torch.Tensor:
        """Correlation ρ in (-1,1)."""
        return torch.tanh(self.rho_raw)

    def get_learned_params(self) -> Dict[str, float]:
        """Get current learned physics parameter values."""
        return {
            'theta': self.theta.item(),
            'omega': self.omega.item(),
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'persistence': (self.alpha + self.beta).item(),
            'kappa': self.kappa.item(),
            'theta_long': self.theta_long.item(),
            'xi': self.xi.item(),
            'rho': self.rho.item(),
        }

    def ou_residual(
        self,
        variance_pred: torch.Tensor,
        variance_prev: torch.Tensor,
        long_run_variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck residual for variance mean-reversion.

        dσ² = θ(σ̄² - σ²)dt

        Args:
            variance_pred: Predicted variance [batch, 1]
            variance_prev: Previous variance [batch, 1]
            long_run_variance: Long-run average variance [batch, 1] or scalar

        Returns:
            OU residual loss
        """
        # Variance change
        d_variance = variance_pred - variance_prev

        # Expected drift under OU
        expected_drift = self.theta * (long_run_variance - variance_prev) * self.dt

        # Residual: actual change vs expected drift
        residual = d_variance - expected_drift

        return torch.mean(residual ** 2)

    def garch_residual(
        self,
        variance_pred: torch.Tensor,
        variance_prev: torch.Tensor,
        returns_prev_sq: torch.Tensor,
    ) -> torch.Tensor:
        """
        GARCH(1,1) consistency residual.

        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

        Args:
            variance_pred: Predicted variance [batch, 1]
            variance_prev: Previous variance [batch, 1]
            returns_prev_sq: Squared previous returns [batch, 1]

        Returns:
            GARCH consistency loss
        """
        # GARCH-implied variance
        garch_implied = (
            self.omega
            + self.alpha * returns_prev_sq
            + self.beta * variance_prev
        )

        return F.mse_loss(variance_pred, garch_implied)

    def feller_residual(self, variance_pred: torch.Tensor) -> torch.Tensor:
        """
        Feller condition residual - penalize negative variance.

        While Softplus should prevent this, add explicit penalty for safety.
        """
        negative_penalty = torch.relu(-variance_pred)
        return torch.mean(negative_penalty)

    def leverage_residual(
        self,
        returns: torch.Tensor,
        variance_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Leverage effect constraint.

        Corr(r_t, σ²_{t+1}) should be negative (leverage effect).
        We penalize positive correlation.

        Args:
            returns: Returns tensor [batch]
            variance_pred: Predicted variance [batch, 1]

        Returns:
            Leverage effect loss (penalizes positive correlation)
        """
        if len(returns) < 10:
            return torch.tensor(0.0, device=variance_pred.device)

        returns_flat = returns.flatten()
        variance_flat = variance_pred.flatten()

        # Compute correlation
        returns_centered = returns_flat - returns_flat.mean()
        variance_centered = variance_flat - variance_flat.mean()

        returns_std = returns_centered.std()
        variance_std = variance_centered.std()

        if returns_std < 1e-8 or variance_std < 1e-8:
            return torch.tensor(0.0, device=variance_pred.device)

        correlation = torch.mean(returns_centered * variance_centered) / (returns_std * variance_std)

        # Penalize positive correlation (leverage effect should be negative)
        rho_penalty = torch.relu(self.rho)  # encourage negative
        return F.relu(correlation) + rho_penalty

    def heston_residual(
        self,
        variance_pred: torch.Tensor,
        variance_prev: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Heston variance drift residual: dV/dt - κ(θ - V)."""
        dt_t = torch.tensor(dt, device=variance_pred.device, dtype=variance_pred.dtype)
        dV_dt = (variance_pred - variance_prev) / (dt_t + 1e-8)
        expected = self.kappa * (self.theta_long - variance_prev)
        residual = dV_dt - expected
        residual_std = residual.std() + 1e-8
        residual = residual / residual_std
        return torch.mean(residual ** 2)

    def feller_penalty_heston(self) -> torch.Tensor:
        """Feller penalty for Heston parameters: max(0, ξ^2 - 2κθ)^2."""
        penalty = torch.relu(self.xi ** 2 - 2 * self.kappa * self.theta_long)
        return torch.mean(penalty ** 2)

    def forward(
        self,
        variance_pred: torch.Tensor,
        variance_target: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined data + physics loss.

        Args:
            variance_pred: Predicted variance [batch, 1]
            variance_target: Target variance [batch, 1]
            metadata: Dictionary containing:
                - variance_history: Previous variances [batch, seq_len]
                - returns: Returns sequence [batch, seq_len]
                - long_run_variance: Long-run variance estimate

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Data loss (MSE on variance)
        data_loss = F.mse_loss(variance_pred, variance_target)

        loss_dict = {
            'data_loss': data_loss.item(),
            'total_loss': data_loss.item(),
        }

        physics_loss = torch.tensor(0.0, device=variance_pred.device)

        # Extract metadata
        variance_history = metadata.get('variance_history')
        returns = metadata.get('returns')
        long_run_variance = metadata.get('long_run_variance')

        # OU mean-reversion residual
        if (self.lambda_ou > 0 and
            variance_history is not None and
            variance_history.shape[1] > 0):
            try:
                variance_prev = variance_history[:, -1:]
                if long_run_variance is None:
                    long_run_variance = variance_history.mean()

                ou_loss = self.ou_residual(variance_pred, variance_prev, long_run_variance)
                physics_loss = physics_loss + self.lambda_ou * ou_loss
                loss_dict['ou_loss'] = ou_loss.item()
                loss_dict['theta_learned'] = self.theta.item()
            except Exception as e:
                logger.debug(f"OU loss computation failed: {e}")

        # GARCH consistency residual
        if (self.lambda_garch > 0 and
            variance_history is not None and
            returns is not None and
            variance_history.shape[1] > 0 and
            returns.shape[1] > 0):
            try:
                variance_prev = variance_history[:, -1:]
                returns_prev_sq = returns[:, -1:] ** 2

                garch_loss = self.garch_residual(variance_pred, variance_prev, returns_prev_sq)
                physics_loss = physics_loss + self.lambda_garch * garch_loss
                loss_dict['garch_loss'] = garch_loss.item()
                loss_dict['alpha_learned'] = self.alpha.item()
                loss_dict['beta_learned'] = self.beta.item()
            except Exception as e:
                logger.debug(f"GARCH loss computation failed: {e}")

        # Feller condition (positivity) and Heston Feller penalty
        if self.lambda_feller > 0:
            feller_loss = self.feller_residual(variance_pred)
            physics_loss = physics_loss + self.lambda_feller * feller_loss
            loss_dict['feller_loss'] = feller_loss.item()

            if self.enable_heston_constraint or self.lambda_heston > 0:
                feller_heston = self.feller_penalty_heston()
                physics_loss = physics_loss + self.lambda_feller * feller_heston
                loss_dict['feller_heston_loss'] = feller_heston.item()

        # Leverage effect
        if self.lambda_leverage > 0 and returns is not None:
            try:
                returns_last = returns[:, -1]
                leverage_loss = self.leverage_residual(returns_last, variance_pred)
                physics_loss = physics_loss + self.lambda_leverage * leverage_loss
                loss_dict['leverage_loss'] = leverage_loss.item()
            except Exception as e:
                logger.debug(f"Leverage loss computation failed: {e}")

        # Optional Heston drift constraint on variance
        if (self.enable_heston_constraint or self.lambda_heston > 0) and variance_history is not None and variance_history.shape[1] > 0:
            try:
                variance_prev = variance_history[:, -1:]
                dt_val = metadata.get('dt', self.dt) if isinstance(metadata, dict) else self.dt
                heston_loss = self.heston_residual(variance_pred, variance_prev, dt_val)
                physics_loss = physics_loss + self.lambda_heston * heston_loss
                loss_dict['heston_loss'] = heston_loss.item()
                loss_dict['kappa_learned'] = self.kappa.item()
                loss_dict['theta_long_learned'] = self.theta_long.item()
                loss_dict['xi_learned'] = self.xi.item()
                loss_dict['rho_learned'] = self.rho.item()
            except Exception as e:
                logger.debug(f"Heston loss computation failed: {e}")

        # Total loss
        total_loss = data_loss + physics_loss

        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


# =============================================================================
# PHYSICS-INFORMED VOLATILITY MODELS
# =============================================================================

class VolatilityPINN(nn.Module):
    """
    Physics-Informed Neural Network for Volatility Forecasting.

    Combines neural network architecture with physics-based constraints:
        1. Variance mean-reversion (OU process)
        2. GARCH consistency
        3. Feller condition (positivity)
        4. Leverage effect

    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        base_model: Base model type ('lstm', 'gru', 'transformer')
        lambda_ou: Weight for OU residual
        lambda_garch: Weight for GARCH residual
        lambda_feller: Weight for Feller condition
        lambda_leverage: Weight for leverage effect
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        base_model: str = 'lstm',
        lambda_ou: float = 0.1,
        lambda_garch: float = 0.1,
        lambda_feller: float = 0.05,
        lambda_leverage: float = 0.05,
        lambda_heston: float = 0.0,
        enable_heston_constraint: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon
        self.base_model_type = base_model

        # Base neural network
        if base_model == 'lstm':
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif base_model == 'gru':
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        # Variance prediction head
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon),
            nn.Softplus(),  # Ensures positive output
        )

        # Physics loss module
        self.physics_loss = VolatilityPhysicsLoss(
            lambda_ou=lambda_ou,
            lambda_garch=lambda_garch,
            lambda_feller=lambda_feller,
            lambda_leverage=lambda_leverage,
            lambda_heston=lambda_heston,
            enable_heston_constraint=enable_heston_constraint,
        )

        logger.info(f"Initialized VolatilityPINN: base={base_model}, "
                   f"hidden_dim={hidden_dim}, λ_ou={lambda_ou}, "
                   f"λ_garch={lambda_garch}, λ_heston={lambda_heston}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Variance predictions [batch, output_horizon]
        """
        encoder_out, _ = self.encoder(x)
        last_hidden = encoder_out[:, -1, :]
        variance_pred = self.variance_head(last_hidden)
        return variance_pred

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with physics constraints.

        Args:
            predictions: Model predictions [batch, 1]
            targets: Ground truth variance [batch, 1]
            metadata: Dictionary with variance_history, returns, etc.
            enable_physics: Whether to enable physics losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if not enable_physics:
            data_loss = F.mse_loss(predictions, targets)
            return data_loss, {'data_loss': data_loss.item(), 'total_loss': data_loss.item()}

        return self.physics_loss(predictions, targets, metadata)

    def get_learned_physics_params(self) -> Dict[str, float]:
        """Get current learned physics parameter values."""
        return self.physics_loss.get_learned_params()

    def log_physics_params(self):
        """Log current learned physics parameters."""
        params = self.get_learned_physics_params()
        logger.info("Learned Volatility Physics Parameters:")
        logger.info(f"  θ (mean reversion speed): {params['theta']:.4f}")
        logger.info(f"  ω (GARCH intercept): {params['omega']:.6f}")
        logger.info(f"  α (GARCH alpha): {params['alpha']:.4f}")
        logger.info(f"  β (GARCH beta): {params['beta']:.4f}")
        logger.info(f"  Persistence (α + β): {params['persistence']:.4f}")


# =============================================================================
# HESTON STOCHASTIC VOLATILITY PINN
# =============================================================================

class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network based on Heston Stochastic Volatility Model.

    Heston Model:
        dS_t = μS_t dt + √V_t S_t dW_t^{(1)}
        dV_t = κ(θ - V_t)dt + ξ√V_t dW_t^{(2)}
        Corr(dW^{(1)}, dW^{(2)}) = ρ < 0 (leverage effect)

    The model learns:
        - κ (kappa): Mean reversion speed
        - θ (theta): Long-run variance
        - ξ (xi): Volatility of volatility (vol-of-vol)
        - ρ (rho): Correlation (leverage effect)

    Physics constraints:
        1. Variance dynamics follow Heston drift
        2. Feller condition: 2κθ > ξ² (ensures V_t > 0)
        3. Leverage effect: ρ < 0

    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        lambda_heston: Weight for Heston drift residual
        lambda_feller: Weight for Feller condition
        lambda_leverage: Weight for leverage effect
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        lambda_heston: float = 0.1,
        lambda_feller: float = 0.05,
        lambda_leverage: float = 0.05,
        # Initial Heston parameters (typical values)
        kappa_init: float = 2.0,    # Mean reversion speed
        theta_init: float = 0.04,   # Long-run variance (~20% annual vol)
        xi_init: float = 0.3,       # Vol-of-vol
        rho_init: float = -0.7,     # Leverage effect
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon

        self.lambda_heston = lambda_heston
        self.lambda_feller = lambda_feller
        self.lambda_leverage = lambda_leverage

        # Learnable Heston parameters
        self.kappa_raw = nn.Parameter(torch.tensor(math.log(kappa_init)))
        self.theta_raw = nn.Parameter(torch.tensor(math.log(theta_init)))
        self.xi_raw = nn.Parameter(torch.tensor(math.log(xi_init)))
        self.rho_raw = nn.Parameter(torch.tensor(math.atanh(rho_init)))

        # Neural network backbone
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon),
            nn.Softplus(),
        )

        logger.info(f"Initialized HestonPINN: hidden_dim={hidden_dim}, "
                   f"λ_heston={lambda_heston}, λ_feller={lambda_feller}")

    @property
    def kappa(self) -> torch.Tensor:
        """Mean reversion speed (positive)."""
        return torch.exp(self.kappa_raw)

    @property
    def theta(self) -> torch.Tensor:
        """Long-run variance (positive)."""
        return torch.exp(self.theta_raw)

    @property
    def xi(self) -> torch.Tensor:
        """Vol-of-vol (positive)."""
        return torch.exp(self.xi_raw)

    @property
    def rho(self) -> torch.Tensor:
        """Correlation (bounded in (-1, 1))."""
        return torch.tanh(self.rho_raw)

    def get_learned_params(self) -> Dict[str, float]:
        """Get current learned Heston parameter values."""
        return {
            'kappa': self.kappa.item(),
            'theta': self.theta.item(),
            'xi': self.xi.item(),
            'rho': self.rho.item(),
            'feller_ratio': (2 * self.kappa * self.theta / (self.xi ** 2)).item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoder_out, _ = self.encoder(x)
        last_hidden = encoder_out[:, -1, :]
        variance_pred = self.variance_head(last_hidden)
        return variance_pred

    def heston_drift_residual(
        self,
        variance_pred: torch.Tensor,
        variance_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Heston variance drift residual.

        dV = κ(θ - V)dt

        Args:
            variance_pred: Predicted variance [batch, 1]
            variance_prev: Previous variance [batch, 1]

        Returns:
            Heston drift residual loss
        """
        dt = DAILY_TIME_STEP

        # Variance change
        dV = variance_pred - variance_prev

        # Expected drift under Heston
        expected_drift = self.kappa * (self.theta - variance_prev) * dt

        # Residual
        residual = dV - expected_drift

        return torch.mean(residual ** 2)

    def feller_condition_loss(self) -> torch.Tensor:
        """
        Feller condition: 2κθ > ξ² ensures variance stays positive.

        We want feller_ratio = 2κθ/ξ² > 1
        """
        feller_ratio = 2 * self.kappa * self.theta / (self.xi ** 2 + 1e-8)
        # Penalize if ratio < 1
        violation = F.relu(1.0 - feller_ratio)
        return violation

    def leverage_effect_loss(self) -> torch.Tensor:
        """
        Leverage effect: ρ should be negative.
        Penalize positive correlation.
        """
        return F.relu(self.rho)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with Heston physics constraints.
        """
        # Data loss
        data_loss = F.mse_loss(predictions, targets)

        loss_dict = {
            'data_loss': data_loss.item(),
            'total_loss': data_loss.item(),
        }

        if not enable_physics:
            return data_loss, loss_dict

        physics_loss = torch.tensor(0.0, device=predictions.device)

        # Heston drift residual
        variance_history = metadata.get('variance_history')
        if self.lambda_heston > 0 and variance_history is not None:
            try:
                variance_prev = variance_history[:, -1:]
                heston_loss = self.heston_drift_residual(predictions, variance_prev)
                physics_loss = physics_loss + self.lambda_heston * heston_loss
                loss_dict['heston_loss'] = heston_loss.item()
            except Exception as e:
                logger.debug(f"Heston loss computation failed: {e}")

        # Feller condition
        if self.lambda_feller > 0:
            feller_loss = self.feller_condition_loss()
            physics_loss = physics_loss + self.lambda_feller * feller_loss
            loss_dict['feller_loss'] = feller_loss.item()

        # Leverage effect (parameter constraint)
        if self.lambda_leverage > 0:
            leverage_loss = self.leverage_effect_loss()
            physics_loss = physics_loss + self.lambda_leverage * leverage_loss
            loss_dict['leverage_loss'] = leverage_loss.item()

        # Log learned parameters
        params = self.get_learned_params()
        loss_dict['kappa_learned'] = params['kappa']
        loss_dict['theta_learned'] = params['theta']
        loss_dict['xi_learned'] = params['xi']
        loss_dict['rho_learned'] = params['rho']
        loss_dict['feller_ratio'] = params['feller_ratio']

        # Total loss
        total_loss = data_loss + physics_loss

        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def log_physics_params(self):
        """Log current learned Heston parameters."""
        params = self.get_learned_params()
        logger.info("Learned Heston Parameters:")
        logger.info(f"  κ (mean reversion): {params['kappa']:.4f}")
        logger.info(f"  θ (long-run var): {params['theta']:.6f} "
                   f"(≈{math.sqrt(params['theta'] * TRADING_DAYS_PER_YEAR) * 100:.1f}% annual vol)")
        logger.info(f"  ξ (vol-of-vol): {params['xi']:.4f}")
        logger.info(f"  ρ (leverage): {params['rho']:.4f}")
        logger.info(f"  Feller ratio: {params['feller_ratio']:.4f} "
                   f"({'OK' if params['feller_ratio'] > 1 else 'VIOLATION'})")


# =============================================================================
# STACKED VOLATILITY PINN
# =============================================================================

class StackedVolatilityPINN(nn.Module):
    """
    Advanced stacked architecture for volatility forecasting.

    Architecture:
        1. Physics-aware encoder (processes raw features)
        2. Parallel LSTM and GRU branches
        3. Attention-based combination
        4. Variance prediction with physics constraints

    This is analogous to the StackedPINN for returns but optimized
    for volatility prediction.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dim: int = 64,
        rnn_hidden_dim: int = 128,
        num_encoder_layers: int = 2,
        num_rnn_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        lambda_ou: float = 0.1,
        lambda_garch: float = 0.1,
        lambda_feller: float = 0.05,
        lambda_leverage: float = 0.05,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_horizon = output_horizon

        # Physics-aware encoder
        encoder_layers = []
        in_dim = input_dim
        for i in range(num_encoder_layers):
            out_dim = encoder_dim if i == num_encoder_layers - 1 else (input_dim + encoder_dim) // 2
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Parallel RNN branches
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
        )

        self.gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
        )

        # Attention weights for combining branches
        self.branch_attention = nn.Parameter(torch.ones(2))

        # Prediction head
        combined_dim = rnn_hidden_dim * 2
        self.variance_head = nn.Sequential(
            nn.Linear(combined_dim, rnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(rnn_hidden_dim // 2, output_horizon),
            nn.Softplus(),
        )

        # Physics loss
        self.physics_loss = VolatilityPhysicsLoss(
            lambda_ou=lambda_ou,
            lambda_garch=lambda_garch,
            lambda_feller=lambda_feller,
            lambda_leverage=lambda_leverage,
        )

        logger.info(f"Initialized StackedVolatilityPINN: encoder_dim={encoder_dim}, "
                   f"rnn_hidden_dim={rnn_hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked architecture."""
        # Encode features
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, seq_len, -1)

        # Parallel RNN branches
        lstm_out, _ = self.lstm(encoded)
        gru_out, _ = self.gru(encoded)

        # Attention-weighted combination
        weights = F.softmax(self.branch_attention, dim=0)
        combined = torch.cat([
            weights[0] * lstm_out[:, -1, :],
            weights[1] * gru_out[:, -1, :],
        ], dim=-1)

        # Variance prediction
        variance_pred = self.variance_head(combined)

        return variance_pred

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with physics constraints."""
        if not enable_physics:
            data_loss = F.mse_loss(predictions, targets)
            return data_loss, {'data_loss': data_loss.item(), 'total_loss': data_loss.item()}

        total_loss, loss_dict = self.physics_loss(predictions, targets, metadata)

        # Add branch attention info
        weights = F.softmax(self.branch_attention, dim=0)
        loss_dict['lstm_weight'] = weights[0].item()
        loss_dict['gru_weight'] = weights[1].item()

        return total_loss, loss_dict

    def get_learned_physics_params(self) -> Dict[str, float]:
        """Get learned physics parameters."""
        return self.physics_loss.get_learned_params()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_volatility_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    output_horizon: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create volatility forecasting models.

    Args:
        model_type: One of 'vol_lstm', 'vol_gru', 'vol_transformer',
                   'vol_pinn', 'heston_pinn', 'stacked_vol_pinn'
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout probability
        output_horizon: Forecast horizon
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated volatility model
    """
    model_type = model_type.lower()

    if model_type in ['vol_lstm', 'volatility_lstm']:
        return VolatilityLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
        )

    elif model_type in ['vol_gru', 'volatility_gru']:
        return VolatilityGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
        )

    elif model_type in ['vol_transformer', 'volatility_transformer']:
        return VolatilityTransformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=kwargs.get('nhead', 4),
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
        )

    elif model_type in ['vol_pinn', 'volatility_pinn']:
        return VolatilityPINN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
            base_model=kwargs.get('base_model', 'lstm'),
            lambda_ou=kwargs.get('lambda_ou', 0.1),
            lambda_garch=kwargs.get('lambda_garch', 0.1),
            lambda_feller=kwargs.get('lambda_feller', 0.05),
            lambda_leverage=kwargs.get('lambda_leverage', 0.05),
        )

    elif model_type in ['heston_pinn', 'heston']:
        return HestonPINN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
            lambda_heston=kwargs.get('lambda_heston', 0.1),
            lambda_feller=kwargs.get('lambda_feller', 0.05),
            lambda_leverage=kwargs.get('lambda_leverage', 0.05),
        )

    elif model_type in ['stacked_vol_pinn', 'stacked_volatility_pinn']:
        return StackedVolatilityPINN(
            input_dim=input_dim,
            encoder_dim=kwargs.get('encoder_dim', 64),
            rnn_hidden_dim=hidden_dim,
            num_encoder_layers=kwargs.get('num_encoder_layers', 2),
            num_rnn_layers=num_layers,
            dropout=dropout,
            output_horizon=output_horizon,
            lambda_ou=kwargs.get('lambda_ou', 0.1),
            lambda_garch=kwargs.get('lambda_garch', 0.1),
            lambda_feller=kwargs.get('lambda_feller', 0.05),
            lambda_leverage=kwargs.get('lambda_leverage', 0.05),
        )

    else:
        raise ValueError(f"Unknown volatility model type: {model_type}. "
                        f"Available: vol_lstm, vol_gru, vol_transformer, "
                        f"vol_pinn, heston_pinn, stacked_vol_pinn")
