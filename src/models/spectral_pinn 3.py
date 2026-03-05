"""
Spectral Regime-Aware Physics-Informed Neural Network

Integrates:
1. SpectralEncoder: Frequency-domain feature extraction via FFT + attention
2. RegimeEncoder: Dense encoding of regime probabilities
3. LSTM: Temporal sequence processing
4. Physics losses: GBM + OU + Autocorrelation + Spectral consistency

This model addresses the limitation of standard PINNs that assume stationarity,
by explicitly conditioning on market regimes and incorporating frequency-domain
information that captures cyclical patterns.

Architecture:
=============
Input: [batch, seq_len, input_dim]
                |
                v
    +-------------------------+
    |    SpectralEncoder      |  <- FFT + attention over frequencies
    |  (freq_embed + attn)    |
    +-------------------------+
                |
                v
    +-------------------------+
    |     RegimeEncoder       |  <- Dense encoding of P(regime)
    +-------------------------+
                |
                v
    +-------------------------+
    |    Feature Fusion       |  <- Concatenate: input + spectral + regime
    +-------------------------+
                |
                v
    +-------------------------+
    |         LSTM            |  <- Temporal sequence modeling
    +-------------------------+
                |
                v
    +-------------------------+
    |    Prediction Head      |  <- Return prediction + regime prediction
    +-------------------------+

References:
    - Raissi, M. et al. (2019). "Physics-Informed Neural Networks."
    - Hamilton, J.D. (1989). "Regime Switching Models."
    - Granger, C.W.J. (1966). "Spectral Analysis of Economic Time Series."
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import math

from ..utils.logger import get_logger
from ..constants import DAILY_TIME_STEP

logger = get_logger(__name__)


class SpectralEncoder(nn.Module):
    """
    Encodes frequency-domain (spectral) features using FFT and attention.

    Computes FFT of input sequences and applies multi-head attention
    to learn which frequency components are most relevant for prediction.

    Key insight: Different market regimes have distinct spectral signatures:
    - Trending markets: Low-frequency power dominates
    - Mean-reverting: Mid-frequency power with characteristic period
    - Random walk: Uniform spectrum (high entropy)
    """

    def __init__(
        self,
        input_dim: int,
        n_fft: int = 32,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Number of input features
            n_fft: FFT size (number of frequency bins = n_fft // 2 + 1)
            embed_dim: Embedding dimension for frequency features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1  # Real FFT output size
        self.embed_dim = embed_dim

        # Embed each frequency bin
        self.freq_embed = nn.Linear(input_dim, embed_dim)

        # Position encoding for frequencies (learned)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, self.n_freq, embed_dim) * 0.02
        )

        # Multi-head self-attention over frequency bins
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        logger.info(
            f"SpectralEncoder initialized: n_fft={n_fft}, "
            f"embed_dim={embed_dim}, num_heads={num_heads}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral encoding.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Spectral embedding per timestep [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Use last n_fft samples for FFT
        fft_len = min(self.n_fft, seq_len)
        x_window = x[:, -fft_len:, :]  # [batch, fft_len, input_dim]

        # Zero-mean each feature
        x_centered = x_window - x_window.mean(dim=1, keepdim=True)

        # Compute FFT along time dimension for each feature
        # Result: [batch, n_freq, input_dim] (complex)
        x_fft = torch.fft.rfft(x_centered, n=self.n_fft, dim=1)

        # Get magnitude spectrum [batch, n_freq, input_dim]
        x_mag = torch.abs(x_fft)

        # Embed each frequency bin
        # [batch, n_freq, input_dim] -> [batch, n_freq, embed_dim]
        freq_embedded = self.freq_embed(x_mag)

        # Add positional embedding for frequencies
        freq_embedded = freq_embedded + self.freq_pos_embed

        # Self-attention over frequency bins
        attn_output, _ = self.attention(
            freq_embedded, freq_embedded, freq_embedded
        )

        # Pool over frequencies (mean pooling)
        pooled = attn_output.mean(dim=1)  # [batch, embed_dim]

        # Output projection
        pooled = self.output_proj(pooled)  # [batch, embed_dim]

        # Expand to sequence length for downstream concatenation
        output = pooled.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, embed_dim]

        return output


class RegimeEncoder(nn.Module):
    """
    Encodes regime probabilities into dense representation.

    Takes soft regime assignments (probabilities over regimes) and
    produces a fixed-size embedding that captures regime uncertainty.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_regimes: Number of market regimes
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.n_regimes = n_regimes
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_regimes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        logger.info(
            f"RegimeEncoder initialized: n_regimes={n_regimes}, "
            f"embed_dim={embed_dim}"
        )

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Encode regime probabilities.

        Args:
            regime_probs: Regime probability tensor [batch, n_regimes]

        Returns:
            Regime embedding [batch, embed_dim]
        """
        return self.encoder(regime_probs)


class SpectralRegimePINN(nn.Module):
    """
    Spectral Regime-Aware Physics-Informed Neural Network

    Combines:
    - SpectralEncoder: Frequency-domain features via FFT + attention
    - RegimeEncoder: Regime probability conditioning
    - LSTM: Temporal sequence modeling
    - Physics losses: GBM + OU + Autocorrelation + Spectral consistency

    This model captures:
    1. Non-stationarity through regime conditioning
    2. Cyclical patterns through spectral analysis
    3. Financial physics through PDE constraints
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_regimes: int = 3,
        n_fft: int = 32,
        spectral_embed_dim: int = 64,
        regime_embed_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        # Physics loss weights
        lambda_gbm: float = 0.1,
        lambda_ou: float = 0.1,
        lambda_autocorr: float = 0.05,
        lambda_spectral: float = 0.05,
        # Advanced options
        use_regime_gate: bool = True,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            n_regimes: Number of market regimes
            n_fft: FFT window size
            spectral_embed_dim: Spectral encoder output dimension
            regime_embed_dim: Regime encoder output dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            lambda_gbm: GBM physics loss weight
            lambda_ou: OU physics loss weight
            lambda_autocorr: Autocorrelation loss weight
            lambda_spectral: Spectral consistency loss weight
            use_regime_gate: Use regime-based gating mechanism
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.use_regime_gate = use_regime_gate

        # Physics loss weights
        self.lambda_gbm = lambda_gbm
        self.lambda_ou = lambda_ou
        self.lambda_autocorr = lambda_autocorr
        self.lambda_spectral = lambda_spectral
        self.dt = DAILY_TIME_STEP

        # Spectral encoder
        self.spectral_encoder = SpectralEncoder(
            input_dim=input_dim,
            n_fft=n_fft,
            embed_dim=spectral_embed_dim,
            dropout=dropout,
        )

        # Regime encoder
        self.regime_encoder = RegimeEncoder(
            n_regimes=n_regimes,
            embed_dim=regime_embed_dim,
        )

        # Combined input dimension
        # Original input + spectral embedding + regime embedding
        combined_dim = input_dim + spectral_embed_dim + regime_embed_dim

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Regime gating (optional)
        if use_regime_gate:
            self.regime_gate = nn.Sequential(
                nn.Linear(regime_embed_dim, lstm_output_dim),
                nn.Sigmoid(),
            )

        # Prediction head - return prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Regime prediction head (auxiliary task)
        self.regime_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_regimes),
        )

        # Map regime probabilities back into hidden space to ensure coupling/grad flow
        self.regime_to_hidden = nn.Linear(n_regimes, lstm_output_dim)

        logger.info(
            f"SpectralRegimePINN initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, n_regimes={n_regimes}, "
            f"λ_gbm={lambda_gbm}, λ_ou={lambda_ou}, "
            f"λ_autocorr={lambda_autocorr}, λ_spectral={lambda_spectral}"
        )

    def forward(
        self,
        x: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            regime_probs: Optional regime probabilities [batch, n_regimes]
                         If None, uses uniform distribution

        Returns:
            Tuple of:
                - prediction: Return prediction [batch, 1]
                - regime_logits: Regime classification logits [batch, n_regimes]
        """
        batch_size, seq_len, _ = x.shape

        # Default regime probabilities (uniform)
        if regime_probs is None:
            regime_probs = torch.ones(
                batch_size, self.n_regimes,
                device=x.device, dtype=x.dtype
            ) / self.n_regimes

        # 1. Compute spectral embedding
        spectral_embed = self.spectral_encoder(x)  # [batch, seq_len, spectral_embed_dim]

        # 2. Compute regime embedding
        regime_embed = self.regime_encoder(regime_probs)  # [batch, regime_embed_dim]

        # 3. Expand embeddings to match sequence length
        spectral_expanded = spectral_embed
        regime_expanded = regime_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # 4. Concatenate: input + spectral + regime
        combined = torch.cat([x, spectral_expanded, regime_expanded], dim=-1)

        # 5. Feature fusion
        fused = self.fusion(combined)  # [batch, seq_len, hidden_dim]

        # 6. LSTM processing
        lstm_out, _ = self.lstm(fused)  # [batch, seq_len, lstm_output_dim]
        last_hidden = lstm_out[:, -1, :]  # [batch, lstm_output_dim]

        # 7. Apply regime gating (if enabled)
        if self.use_regime_gate:
            gate = self.regime_gate(regime_embed)
            last_hidden = last_hidden * gate

        # 8. Regime prediction (auxiliary task)
        regime_logits = self.regime_head(last_hidden)  # [batch, n_regimes]
        regime_probs = torch.softmax(regime_logits, dim=-1)

        # Coupling for gradient flow: add regime context into hidden state
        regime_context = self.regime_to_hidden(regime_probs)
        last_hidden = last_hidden + regime_context

        # 9. Prediction
        prediction = self.prediction_head(last_hidden)  # [batch, 1]

        return prediction, regime_probs

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed losses.

        Includes:
        - GBM residual (trend dynamics)
        - OU residual (mean reversion)
        - Autocorrelation loss (realistic return structure)
        - Spectral consistency loss (frequency-domain matching)

        Args:
            predictions: Model predictions [batch, 1] or [batch, seq_len]
            targets: Ground truth [batch, 1] or [batch, seq_len]
            returns: Optional return series for physics computation

        Returns:
            Tuple of (physics_loss, loss_dict)
        """
        loss_dict = {}
        physics_loss = torch.tensor(0.0, device=predictions.device)

        # Use returns if provided, otherwise use predictions/targets
        if returns is None:
            returns = predictions if predictions.dim() > 1 else predictions.unsqueeze(1)

        # Ensure returns is 2D: [batch, seq_len]
        if returns.dim() == 3:
            returns = returns.squeeze(-1)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)

        # GBM loss
        if self.lambda_gbm > 0 and returns.shape[1] >= 2:
            gbm_loss = self._gbm_residual(returns)
            physics_loss = physics_loss + self.lambda_gbm * gbm_loss
            loss_dict['gbm_loss'] = gbm_loss.item()

        # OU loss
        if self.lambda_ou > 0 and returns.shape[1] >= 2:
            ou_loss = self._ou_residual(returns)
            physics_loss = physics_loss + self.lambda_ou * ou_loss
            loss_dict['ou_loss'] = ou_loss.item()

        # Autocorrelation loss
        if self.lambda_autocorr > 0 and returns.shape[1] >= 2:
            ac_loss = self._autocorrelation_loss(returns)
            physics_loss = physics_loss + self.lambda_autocorr * ac_loss
            loss_dict['autocorr_loss'] = ac_loss.item()

        # Spectral consistency loss
        if self.lambda_spectral > 0 and predictions.dim() > 1 and targets.dim() > 1:
            if predictions.shape[1] >= 4 and targets.shape[1] >= 4:
                spectral_loss = self._spectral_consistency_loss(
                    predictions.squeeze(-1) if predictions.dim() == 3 else predictions,
                    targets.squeeze(-1) if targets.dim() == 3 else targets
                )
                physics_loss = physics_loss + self.lambda_spectral * spectral_loss
                loss_dict['spectral_loss'] = spectral_loss.item()

        loss_dict['physics_loss'] = physics_loss.item()

        return physics_loss, loss_dict

    def _gbm_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """GBM residual: dR/dt ≈ μ"""
        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt
        mu = returns.mean(dim=1, keepdim=True)
        residual = dR_dt - mu
        return torch.mean(residual ** 2)

    def _ou_residual(self, returns: torch.Tensor) -> torch.Tensor:
        """OU residual: dR = θ(μ - R)dt"""
        R_curr = returns[:, :-1]
        R_next = returns[:, 1:]
        dR_dt = (R_next - R_curr) / self.dt
        mu = returns.mean(dim=1, keepdim=True)
        theta = 1.0  # Mean reversion speed
        residual = dR_dt - theta * (mu - R_curr)
        return torch.mean(residual ** 2)

    def _autocorrelation_loss(self, returns: torch.Tensor) -> torch.Tensor:
        """Penalize unrealistic autocorrelation in returns."""
        # Returns should have near-zero lag-1 autocorrelation
        r1 = returns[:, :-1]
        r2 = returns[:, 1:]

        mean1 = r1.mean(dim=1, keepdim=True)
        mean2 = r2.mean(dim=1, keepdim=True)
        std1 = r1.std(dim=1, keepdim=True) + 1e-8
        std2 = r2.std(dim=1, keepdim=True) + 1e-8

        corr = ((r1 - mean1) * (r2 - mean2)).mean(dim=1) / (std1.squeeze() * std2.squeeze())

        # Target: near-zero autocorrelation (EMH)
        return torch.mean(corr ** 2)

    def _spectral_consistency_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Frequency-domain consistency loss."""
        # Compute FFT
        pred_fft = torch.fft.rfft(predictions - predictions.mean(dim=1, keepdim=True), dim=1)
        target_fft = torch.fft.rfft(targets - targets.mean(dim=1, keepdim=True), dim=1)

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Normalize
        pred_norm = pred_mag / (pred_mag.sum(dim=1, keepdim=True) + 1e-8)
        target_norm = target_mag / (target_mag.sum(dim=1, keepdim=True) + 1e-8)

        # MSE in frequency domain
        return torch.mean((pred_norm - target_norm) ** 2)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for training.

        Required by Trainer to recognize this as a PINN model.

        Args:
            predictions: Model predictions
            targets: Ground truth
            metadata: Batch metadata with 'returns', 'regime_probs', etc.
            enable_physics: Whether to apply physics constraints

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict: Dict[str, torch.Tensor] = {}

        # Data loss (MSE)
        data_loss = nn.functional.mse_loss(predictions, targets)
        loss_dict['data_loss'] = data_loss

        total_loss = data_loss

        # Physics loss
        if enable_physics:
            returns = metadata.get('returns', None)
            physics_loss, physics_dict = self.compute_physics_loss(
                predictions, targets, returns
            )
            total_loss = total_loss + physics_loss
            # Store tensor versions
            for k, v in physics_dict.items():
                loss_dict[k] = torch.tensor(v, device=predictions.device) if not isinstance(v, torch.Tensor) else v

        # Regime prediction loss (if regime labels available)
        regime_labels = metadata.get('regime_labels', None)
        regime_logits = metadata.get('regime_logits', None)

        if regime_labels is not None and regime_logits is not None:
            regime_loss = nn.functional.cross_entropy(
                regime_logits, regime_labels
            )
            total_loss = total_loss + 0.1 * regime_loss
            loss_dict['regime_loss'] = regime_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict


def create_spectral_regime_pinn(
    input_dim: int,
    hidden_dim: int = 128,
    n_regimes: int = 3,
    **kwargs
) -> SpectralRegimePINN:
    """Factory function to create SpectralRegimePINN."""
    return SpectralRegimePINN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_regimes=n_regimes,
        **kwargs
    )
