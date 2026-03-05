"""
Stacked Physics-Informed Neural Networks for Financial Forecasting

Implements:
1. StackedPINN: Feature-level encoder + parallel LSTM/GRU + dense head
2. ResidualPINN: Physics-constrained correction to base model predictions
3. Physics losses on returns (GBM, OU) as soft constraints
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Literal
import math

from ..utils.logger import get_logger
from ..constants import DAILY_TIME_STEP

logger = get_logger(__name__)


class PhysicsEncoder(nn.Module):
    """
    Feature-level encoder that applies physics-aware transformations
    Learns representations that respect financial physics constraints
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(PhysicsEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature encoder
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Physics-aware projection
        self.physics_proj = nn.Linear(hidden_dim, hidden_dim)

        logger.info(f"PhysicsEncoder initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            encoded: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to process all timesteps
        x_flat = x.reshape(-1, self.input_dim)

        # Encode features
        encoded = self.encoder(x_flat)

        # Physics-aware projection
        physics_features = self.physics_proj(encoded)

        # Reshape back
        output = physics_features.reshape(batch_size, seq_len, self.hidden_dim)

        return output


class ParallelHeads(nn.Module):
    """
    Parallel LSTM and GRU heads for multi-perspective sequence modeling
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(ParallelHeads, self).__init__()

        self.hidden_dim = hidden_dim

        # LSTM head
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # GRU head
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention weights for combining heads
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        logger.info(f"ParallelHeads initialized: hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            combined: (batch, hidden_dim * 2)
            attention_weights: (batch, 2)
        """
        # LSTM forward
        lstm_out, (h_lstm, _) = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # GRU forward
        gru_out, h_gru = self.gru(x)
        gru_last = gru_out[:, -1, :]  # (batch, hidden_dim)

        # Concatenate
        combined = torch.cat([lstm_last, gru_last], dim=-1)  # (batch, hidden_dim * 2)

        # Compute attention weights
        attention_weights = self.attention(combined)  # (batch, 2)

        return combined, attention_weights


class PredictionHead(nn.Module):
    """
    Dense prediction head for both regression and classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2
    ):
        super(PredictionHead, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Regression head (return prediction)
        self.regression = nn.Linear(hidden_dim // 2, 1)

        # Classification head (direction: up/down)
        self.classification = nn.Linear(hidden_dim // 2, 2)

        logger.info(f"PredictionHead initialized: input_dim={input_dim}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            return_pred: (batch, 1) - predicted return
            direction_logits: (batch, 2) - direction classification logits
        """
        shared_features = self.shared(x)

        # Regression output
        return_pred = self.regression(shared_features)

        # Classification output
        direction_logits = self.classification(shared_features)

        return return_pred, direction_logits


class StackedPINN(nn.Module):
    """
    Stacked Physics-Informed Neural Network

    Architecture:
    1. Physics-aware feature encoder
    2. Parallel LSTM + GRU heads
    3. Concatenated outputs
    4. Dense prediction head (regression + classification)
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dim: int = 128,
        lstm_hidden_dim: int = 128,
        num_encoder_layers: int = 2,
        num_rnn_layers: int = 2,
        prediction_hidden_dim: int = 64,
        dropout: float = 0.2,
        lambda_gbm: float = 0.1,
        lambda_ou: float = 0.1
    ):
        """
        Args:
            input_dim: Number of input features
            encoder_dim: Physics encoder hidden dimension
            lstm_hidden_dim: LSTM/GRU hidden dimension
            num_encoder_layers: Number of encoder layers
            num_rnn_layers: Number of LSTM/GRU layers
            prediction_hidden_dim: Prediction head hidden dimension
            dropout: Dropout probability
            lambda_gbm: GBM physics loss weight
            lambda_ou: OU physics loss weight
        """
        super(StackedPINN, self).__init__()

        self.input_dim = input_dim
        self.lambda_gbm = lambda_gbm
        self.lambda_ou = lambda_ou

        # Physics-aware encoder
        self.encoder = PhysicsEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_dim,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        # Parallel LSTM/GRU heads
        self.parallel_heads = ParallelHeads(
            input_dim=encoder_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=num_rnn_layers,
            dropout=dropout
        )

        # Prediction head
        self.prediction_head = PredictionHead(
            input_dim=lstm_hidden_dim * 2,
            hidden_dim=prediction_hidden_dim,
            dropout=dropout
        )

        # Physics loss computation
        self.dt = DAILY_TIME_STEP  # Daily returns

        logger.info(f"StackedPINN initialized: input_dim={input_dim}, "
                   f"λ_gbm={lambda_gbm}, λ_ou={lambda_ou}")

    def forward(
        self,
        x: torch.Tensor,
        compute_physics: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
            compute_physics: Whether to compute attention weights

        Returns:
            return_pred: (batch, 1) - predicted return
            direction_logits: (batch, 2) - direction logits
            attention_weights: (batch, 2) or None - LSTM/GRU attention
        """
        # 1. Physics-aware encoding
        encoded = self.encoder(x)

        # 2. Parallel LSTM/GRU processing
        combined, attention_weights = self.parallel_heads(encoded)

        # 3. Prediction
        return_pred, direction_logits = self.prediction_head(combined)

        return return_pred, direction_logits, attention_weights if compute_physics else None

    def compute_physics_loss(
        self,
        x: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics losses on returns

        Args:
            x: Input features (batch, seq_len, input_dim)
            returns: Return sequence (batch, seq_len) - must be returns, not prices!

        Returns:
            physics_loss: Total physics loss
            loss_dict: Dictionary with individual losses
        """
        loss_dict = {}
        physics_loss = torch.tensor(0.0, device=x.device)

        if returns is None or returns.shape[1] < 2:
            return physics_loss, loss_dict

        # GBM loss on returns
        if self.lambda_gbm > 0:
            gbm_loss = self._gbm_residual(returns)
            physics_loss = physics_loss + self.lambda_gbm * gbm_loss
            loss_dict['gbm_loss'] = gbm_loss.item()

        # OU loss on returns (mean reversion)
        if self.lambda_ou > 0:
            ou_loss = self._ou_residual(returns)
            physics_loss = physics_loss + self.lambda_ou * ou_loss
            loss_dict['ou_loss'] = ou_loss.item()

        loss_dict['physics_loss'] = physics_loss.item()

        return physics_loss, loss_dict

    def _gbm_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """
        GBM residual on returns: dR/dt ≈ μ + σ·ε
        Enforce that return changes follow drift
        """
        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        # Return changes
        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt

        # Estimate drift from data
        mu = returns.mean(dim=1, keepdim=True)

        # Residual: how much return changes deviate from drift
        residual = dR_dt - mu

        return torch.mean(residual ** 2)

    def _ou_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck residual on returns
        dR = θ(μ - R)dt + σdW
        Enforce mean reversion in returns
        """
        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        # Return values
        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt

        # Estimate mean reversion parameters
        mu = returns.mean(dim=1, keepdim=True)  # Long-term mean
        theta = torch.tensor(1.0, device=returns.device)  # Mean reversion speed

        # OU residual: dR/dt should equal θ(μ - R)
        residual = dR_dt - theta * (mu - R_curr)

        return torch.mean(residual ** 2)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined data and physics loss for training.

        This method is required for the Trainer to recognize this as a PINN model.

        Args:
            predictions: Model predictions (return_pred from forward pass)
            targets: Ground truth targets
            metadata: Batch metadata containing 'returns' for physics loss
            enable_physics: Whether to apply physics constraints

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        import torch.nn.functional as F

        loss_dict = {}

        # Data loss (MSE)
        data_loss = F.mse_loss(predictions, targets)
        loss_dict['data_loss'] = data_loss.item()

        total_loss = data_loss

        # Physics loss if enabled
        if enable_physics:
            returns = metadata.get('returns', None)
            if returns is not None:
                physics_loss, physics_dict = self.compute_physics_loss(
                    metadata.get('inputs', None) or torch.zeros_like(returns),
                    returns
                )
                total_loss = total_loss + physics_loss
                loss_dict.update(physics_dict)
            else:
                loss_dict['physics_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class ResidualPINN(nn.Module):
    """
    Residual PINN: Physics-constrained correction to base model

    Architecture:
    1. Base model (LSTM or GRU) makes initial prediction
    2. Physics-informed correction network learns residual
    3. Final prediction = base_prediction + physics_correction
    """

    def __init__(
        self,
        input_dim: int,
        base_model_type: Literal['lstm', 'gru'] = 'lstm',
        base_hidden_dim: int = 128,
        correction_hidden_dim: int = 64,
        num_base_layers: int = 2,
        num_correction_layers: int = 2,
        dropout: float = 0.2,
        lambda_gbm: float = 0.1,
        lambda_ou: float = 0.1
    ):
        """
        Args:
            input_dim: Number of input features
            base_model_type: Type of base model ('lstm' or 'gru')
            base_hidden_dim: Base model hidden dimension
            correction_hidden_dim: Correction network hidden dimension
            num_base_layers: Number of base model layers
            num_correction_layers: Number of correction layers
            dropout: Dropout probability
            lambda_gbm: GBM physics loss weight
            lambda_ou: OU physics loss weight
        """
        super(ResidualPINN, self).__init__()

        self.base_model_type = base_model_type
        self.lambda_gbm = lambda_gbm
        self.lambda_ou = lambda_ou

        # Base model (LSTM or GRU)
        if base_model_type == 'lstm':
            self.base_model = nn.LSTM(
                input_size=input_dim,
                hidden_size=base_hidden_dim,
                num_layers=num_base_layers,
                batch_first=True,
                dropout=dropout if num_base_layers > 1 else 0
            )
        else:  # gru
            self.base_model = nn.GRU(
                input_size=input_dim,
                hidden_size=base_hidden_dim,
                num_layers=num_base_layers,
                batch_first=True,
                dropout=dropout if num_base_layers > 1 else 0
            )

        # Base prediction head
        self.base_head = nn.Linear(base_hidden_dim, 1)

        # Physics-informed correction network
        correction_layers = []
        in_dim = base_hidden_dim + 1  # Hidden state + base prediction
        for _ in range(num_correction_layers):
            correction_layers.extend([
                nn.Linear(in_dim, correction_hidden_dim),
                nn.LayerNorm(correction_hidden_dim),
                nn.Tanh(),  # Tanh for bounded corrections
                nn.Dropout(dropout)
            ])
            in_dim = correction_hidden_dim

        self.correction_network = nn.Sequential(*correction_layers)

        # Correction output (residual)
        self.correction_head = nn.Linear(correction_hidden_dim, 1)

        # Direction classification head (on corrected features)
        self.direction_head = nn.Sequential(
            nn.Linear(correction_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

        # Physics parameters
        self.dt = DAILY_TIME_STEP

        logger.info(f"ResidualPINN initialized: base={base_model_type}, "
                   f"λ_gbm={lambda_gbm}, λ_ou={lambda_ou}")

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
            return_components: Whether to return base and correction components

        Returns:
            return_pred: (batch, 1) - final corrected prediction
            direction_logits: (batch, 2) - direction logits
            components: Dict with base_pred and correction (if return_components=True)
        """
        # 1. Base model prediction
        rnn_out, _ = self.base_model(x)
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_dim)
        base_pred = self.base_head(last_hidden)  # (batch, 1)

        # 2. Physics-informed correction
        correction_input = torch.cat([last_hidden, base_pred], dim=-1)
        correction_features = self.correction_network(correction_input)
        correction = self.correction_head(correction_features)  # (batch, 1)

        # 3. Final prediction
        final_pred = base_pred + correction

        # 4. Direction classification
        direction_logits = self.direction_head(correction_features)

        components = None
        if return_components:
            components = {
                'base_pred': base_pred,
                'correction': correction,
                'correction_features': correction_features
            }

        return final_pred, direction_logits, components

    def compute_physics_loss(
        self,
        x: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics losses on returns
        Same implementation as StackedPINN
        """
        loss_dict = {}
        physics_loss = torch.tensor(0.0, device=x.device)

        if returns is None or returns.shape[1] < 2:
            return physics_loss, loss_dict

        # GBM loss
        if self.lambda_gbm > 0:
            gbm_loss = self._gbm_residual(returns)
            physics_loss = physics_loss + self.lambda_gbm * gbm_loss
            loss_dict['gbm_loss'] = gbm_loss.item()

        # OU loss
        if self.lambda_ou > 0:
            ou_loss = self._ou_residual(returns)
            physics_loss = physics_loss + self.lambda_ou * ou_loss
            loss_dict['ou_loss'] = ou_loss.item()

        loss_dict['physics_loss'] = physics_loss.item()

        return physics_loss, loss_dict

    def _gbm_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """GBM residual on returns"""
        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt
        mu = returns.mean(dim=1, keepdim=True)
        residual = dR_dt - mu

        return torch.mean(residual ** 2)

    def _ou_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """OU residual on returns"""
        if returns.shape[1] < 2:
            return torch.tensor(0.0, device=returns.device)

        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt
        mu = returns.mean(dim=1, keepdim=True)
        theta = torch.tensor(1.0, device=returns.device)
        residual = dR_dt - theta * (mu - R_curr)

        return torch.mean(residual ** 2)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined data and physics loss for training.

        This method is required for the Trainer to recognize this as a PINN model.

        Args:
            predictions: Model predictions (final_pred from forward pass)
            targets: Ground truth targets
            metadata: Batch metadata containing 'returns' for physics loss
            enable_physics: Whether to apply physics constraints

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        import torch.nn.functional as F

        loss_dict = {}

        # Data loss (MSE)
        data_loss = F.mse_loss(predictions, targets)
        loss_dict['data_loss'] = data_loss.item()

        total_loss = data_loss

        # Physics loss if enabled
        if enable_physics:
            returns = metadata.get('returns', None)
            if returns is not None:
                physics_loss, physics_dict = self.compute_physics_loss(
                    metadata.get('inputs', None) or torch.zeros_like(returns),
                    returns
                )
                total_loss = total_loss + physics_loss
                loss_dict.update(physics_dict)
            else:
                loss_dict['physics_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
