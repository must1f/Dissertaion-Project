"""
Spectral (Frequency-Domain) Loss Functions

Physics-informed loss functions based on frequency-domain constraints:
- Autocorrelation Loss: Enforces realistic return autocorrelation structure
- Spectral Consistency Loss: Frequency-domain prediction consistency

These losses capture stylized facts of financial returns:
1. Return autocorrelation near zero (Efficient Market Hypothesis)
2. Absolute return autocorrelation positive (volatility clustering)
3. Consistent spectral structure between predictions and targets

References:
    - Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts."
    - Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices."
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod

from ..constants import EPSILON


class SpectralResidual(nn.Module, ABC):
    """Base class for spectral-domain physics losses."""

    def __init__(self, weight: float = 0.05, eps: float = EPSILON):
        """
        Args:
            weight: Weight for this constraint
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = weight
        self.eps = eps

    @abstractmethod
    def compute_residual(self, **kwargs) -> torch.Tensor:
        """Compute the spectral residual."""
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute weighted residual loss."""
        residual = self.compute_residual(**kwargs)
        return self.weight * residual


class AutocorrelationLoss(SpectralResidual):
    """
    Autocorrelation-based loss function.

    Enforces realistic autocorrelation structure in financial returns:
    - Raw returns: Near-zero lag-1 autocorrelation (EMH)
    - Absolute returns: Significant positive autocorrelation (volatility clustering)

    Financial Returns Stylized Facts:
    - ACF(returns, lag=1) ≈ 0 (no linear predictability)
    - ACF(|returns|, lag=1) > 0 (volatility persistence)

    Loss = (predicted_AC - target_AC)^2

    Args:
        weight: Loss weight (default: 0.05)
        target_ac_returns: Target autocorrelation for returns (default: 0.0)
        target_ac_abs_returns: Target autocorrelation for absolute returns (default: 0.2)
        enforce_abs_returns: Whether to also penalize absolute return autocorrelation
        max_lag: Maximum lag for autocorrelation calculation
    """

    def __init__(
        self,
        weight: float = 0.05,
        target_ac_returns: float = 0.0,
        target_ac_abs_returns: float = 0.2,
        enforce_abs_returns: bool = True,
        max_lag: int = 5,
        **kwargs
    ):
        super().__init__(weight=weight, **kwargs)
        self.target_ac_returns = target_ac_returns
        self.target_ac_abs_returns = target_ac_abs_returns
        self.enforce_abs_returns = enforce_abs_returns
        self.max_lag = max_lag

    def _compute_autocorrelation(
        self,
        x: torch.Tensor,
        lag: int = 1
    ) -> torch.Tensor:
        """
        Compute autocorrelation at specified lag.

        Args:
            x: Time series tensor [batch, seq_len] or [batch, seq_len, 1]
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation tensor [batch]
        """
        if x.dim() == 3:
            x = x.squeeze(-1)

        if x.shape[1] <= lag:
            return torch.zeros(x.shape[0], device=x.device)

        # Split into lagged series
        x1 = x[:, :-lag]
        x2 = x[:, lag:]

        # Mean and std
        mean1 = x1.mean(dim=1, keepdim=True)
        mean2 = x2.mean(dim=1, keepdim=True)
        std1 = x1.std(dim=1, keepdim=True) + self.eps
        std2 = x2.std(dim=1, keepdim=True) + self.eps

        # Correlation
        cov = ((x1 - mean1) * (x2 - mean2)).mean(dim=1)
        corr = cov / (std1.squeeze() * std2.squeeze())

        return torch.clamp(corr, -1.0, 1.0)

    def compute_residual(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute autocorrelation loss.

        Args:
            predicted_returns: Model predictions [batch, seq_len]
            actual_returns: Optional ground truth returns (for adaptive targeting)

        Returns:
            Loss value
        """
        if predicted_returns.dim() == 3:
            predicted_returns = predicted_returns.squeeze(-1)

        total_loss = torch.tensor(0.0, device=predicted_returns.device)

        # Loss on return autocorrelation
        for lag in range(1, min(self.max_lag + 1, predicted_returns.shape[1])):
            pred_ac = self._compute_autocorrelation(predicted_returns, lag)

            # Target: near-zero autocorrelation for returns (EMH)
            target_ac = self.target_ac_returns

            # If actual returns provided, use their autocorrelation as target
            if actual_returns is not None:
                if actual_returns.dim() == 3:
                    actual_returns = actual_returns.squeeze(-1)
                target_ac = self._compute_autocorrelation(actual_returns, lag)

            # Squared deviation from target
            ac_loss = (pred_ac - target_ac) ** 2
            total_loss = total_loss + ac_loss.mean()

        # Loss on absolute return autocorrelation (volatility clustering)
        if self.enforce_abs_returns:
            abs_returns = torch.abs(predicted_returns)
            pred_abs_ac = self._compute_autocorrelation(abs_returns, lag=1)

            # Target: positive autocorrelation for absolute returns
            target_abs_ac = self.target_ac_abs_returns

            if actual_returns is not None:
                actual_abs = torch.abs(actual_returns)
                target_abs_ac = self._compute_autocorrelation(actual_abs, lag=1)

            abs_ac_loss = (pred_abs_ac - target_abs_ac) ** 2
            total_loss = total_loss + abs_ac_loss.mean()

        return total_loss

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted autocorrelation loss."""
        residual = self.compute_residual(predicted_returns, actual_returns)
        return self.weight * residual


class SpectralConsistencyLoss(SpectralResidual):
    """
    Frequency-domain consistency loss.

    Enforces that predictions have similar spectral structure to targets.
    Low frequencies (trends) weighted higher than high frequencies (noise).

    Loss = sum_f w(f) * |FFT(pred)(f) - FFT(target)(f)|^2

    where w(f) weights low frequencies more heavily.

    This encourages the model to:
    1. Capture correct trend patterns (low frequency)
    2. Match cyclical patterns (mid frequency)
    3. Not overfit to noise (high frequency penalized less)
    """

    def __init__(
        self,
        weight: float = 0.05,
        n_fft: int = 64,
        low_freq_weight: float = 2.0,
        mid_freq_weight: float = 1.0,
        high_freq_weight: float = 0.5,
        low_freq_cutoff: float = 0.1,
        high_freq_cutoff: float = 0.25,
        use_phase: bool = False,
        **kwargs
    ):
        """
        Args:
            weight: Overall loss weight
            n_fft: FFT size (uses last n_fft points if sequence longer)
            low_freq_weight: Weight for low-frequency components
            mid_freq_weight: Weight for mid-frequency components
            high_freq_weight: Weight for high-frequency components
            low_freq_cutoff: Cutoff between low and mid bands (cycles/sample)
            high_freq_cutoff: Cutoff between mid and high bands
            use_phase: Whether to also match phase (not just magnitude)
        """
        super().__init__(weight=weight, **kwargs)
        self.n_fft = n_fft
        self.low_freq_weight = low_freq_weight
        self.mid_freq_weight = mid_freq_weight
        self.high_freq_weight = high_freq_weight
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.use_phase = use_phase

        # Precompute frequency weights
        self._freq_weights = None
        self._n_freq = None

    def _get_frequency_weights(self, n_freq: int, device: torch.device) -> torch.Tensor:
        """Get or compute frequency-dependent weights."""
        if self._freq_weights is not None and self._n_freq == n_freq:
            return self._freq_weights.to(device)

        # Normalized frequencies [0, 0.5]
        freqs = torch.linspace(0, 0.5, n_freq, device=device)

        # Create weight mask based on frequency bands
        weights = torch.ones(n_freq, device=device)

        low_mask = freqs < self.low_freq_cutoff
        mid_mask = (freqs >= self.low_freq_cutoff) & (freqs < self.high_freq_cutoff)
        high_mask = freqs >= self.high_freq_cutoff

        weights[low_mask] = self.low_freq_weight
        weights[mid_mask] = self.mid_freq_weight
        weights[high_mask] = self.high_freq_weight

        # Exclude DC component (index 0)
        weights[0] = 0.0

        self._freq_weights = weights
        self._n_freq = n_freq

        return weights

    def compute_residual(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral consistency loss.

        Args:
            predictions: Predicted values [batch, seq_len] or [batch, seq_len, 1]
            targets: Target values [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            Spectral consistency loss
        """
        # Ensure 2D
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        batch_size, seq_len = predictions.shape

        # Use last n_fft points or full sequence
        fft_len = min(self.n_fft, seq_len)
        pred_window = predictions[:, -fft_len:]
        target_window = targets[:, -fft_len:]

        # Zero-mean
        pred_centered = pred_window - pred_window.mean(dim=1, keepdim=True)
        target_centered = target_window - target_window.mean(dim=1, keepdim=True)

        # Compute FFT
        pred_fft = torch.fft.rfft(pred_centered, dim=1)
        target_fft = torch.fft.rfft(target_centered, dim=1)

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Normalize by DC-removed total power
        pred_power = (pred_mag[:, 1:] ** 2).sum(dim=1, keepdim=True) + self.eps
        target_power = (target_mag[:, 1:] ** 2).sum(dim=1, keepdim=True) + self.eps

        pred_mag_norm = pred_mag / torch.sqrt(pred_power)
        target_mag_norm = target_mag / torch.sqrt(target_power)

        # Get frequency weights
        n_freq = pred_fft.shape[1]
        freq_weights = self._get_frequency_weights(n_freq, predictions.device)

        # Weighted magnitude difference
        mag_diff = (pred_mag_norm - target_mag_norm) ** 2
        weighted_diff = mag_diff * freq_weights.unsqueeze(0)

        loss = weighted_diff.sum(dim=1).mean()

        # Optional phase consistency
        if self.use_phase:
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)

            # Phase difference (wrapped to [-pi, pi])
            phase_diff = pred_phase - target_phase
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

            # Weight phase loss by magnitude (phase matters more for strong components)
            phase_loss = (phase_diff ** 2) * target_mag_norm * freq_weights.unsqueeze(0)
            loss = loss + 0.1 * phase_loss.sum(dim=1).mean()

        return loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted spectral consistency loss."""
        residual = self.compute_residual(predictions, targets)
        return self.weight * residual


class SpectralEntropyLoss(SpectralResidual):
    """
    Spectral entropy regularization loss.

    Encourages predictions to have appropriate spectral entropy:
    - Not too low (overfitting to specific frequencies)
    - Not too high (pure noise)

    Financial returns typically have intermediate spectral entropy
    due to mix of trend, cyclical, and random components.
    """

    def __init__(
        self,
        weight: float = 0.02,
        target_entropy: float = 0.7,  # Normalized entropy [0, 1]
        entropy_mode: str = 'adaptive',  # 'fixed', 'adaptive', 'match'
        n_fft: int = 64,
        **kwargs
    ):
        """
        Args:
            weight: Loss weight
            target_entropy: Target normalized spectral entropy
            entropy_mode: How to compute target:
                - 'fixed': Use target_entropy directly
                - 'adaptive': Learn from actual returns
                - 'match': Match entropy of targets
            n_fft: FFT size
        """
        super().__init__(weight=weight, **kwargs)
        self.target_entropy = target_entropy
        self.entropy_mode = entropy_mode
        self.n_fft = n_fft

    def _compute_spectral_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized spectral entropy.

        Args:
            x: Time series [batch, seq_len]

        Returns:
            Entropy [batch]
        """
        batch_size, seq_len = x.shape

        # Use last n_fft points
        fft_len = min(self.n_fft, seq_len)
        x_window = x[:, -fft_len:]

        # Zero-mean and compute FFT
        x_centered = x_window - x_window.mean(dim=1, keepdim=True)
        x_fft = torch.fft.rfft(x_centered, dim=1)

        # Power spectrum (exclude DC)
        power = torch.abs(x_fft[:, 1:]) ** 2

        # Normalize to probability distribution
        total_power = power.sum(dim=1, keepdim=True) + self.eps
        p = power / total_power

        # Shannon entropy
        entropy = -torch.sum(p * torch.log(p + self.eps), dim=1)

        # Normalize by max entropy
        max_entropy = torch.log(torch.tensor(power.shape[1], device=x.device, dtype=x.dtype))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def compute_residual(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute spectral entropy loss.

        Args:
            predictions: Predicted values [batch, seq_len]
            targets: Target values (for 'match' mode)

        Returns:
            Entropy deviation loss
        """
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)

        pred_entropy = self._compute_spectral_entropy(predictions)

        if self.entropy_mode == 'fixed':
            target = self.target_entropy
        elif self.entropy_mode == 'match' and targets is not None:
            if targets.dim() == 3:
                targets = targets.squeeze(-1)
            target = self._compute_spectral_entropy(targets)
        else:  # adaptive
            target = self.target_entropy

        # Squared deviation from target entropy
        if isinstance(target, torch.Tensor):
            loss = (pred_entropy - target) ** 2
        else:
            loss = (pred_entropy - target) ** 2

        return loss.mean()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted entropy loss."""
        residual = self.compute_residual(predictions, targets)
        return self.weight * residual


class CombinedSpectralLoss(nn.Module):
    """
    Combined spectral loss for PINN training.

    Combines:
    - Autocorrelation loss (realistic return dynamics)
    - Spectral consistency loss (frequency-domain matching)
    - Optional spectral entropy regularization

    Example:
        spectral_loss = CombinedSpectralLoss(
            lambda_autocorr=0.05,
            lambda_spectral=0.05,
            lambda_entropy=0.02
        )
        loss, loss_dict = spectral_loss(predictions, targets)
    """

    def __init__(
        self,
        lambda_autocorr: float = 0.05,
        lambda_spectral: float = 0.05,
        lambda_entropy: float = 0.02,
        n_fft: int = 64,
        autocorr_kwargs: Optional[Dict] = None,
        spectral_kwargs: Optional[Dict] = None,
        entropy_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            lambda_autocorr: Weight for autocorrelation loss
            lambda_spectral: Weight for spectral consistency loss
            lambda_entropy: Weight for spectral entropy loss (0 to disable)
            n_fft: FFT size for spectral computations
            autocorr_kwargs: Additional kwargs for AutocorrelationLoss
            spectral_kwargs: Additional kwargs for SpectralConsistencyLoss
            entropy_kwargs: Additional kwargs for SpectralEntropyLoss
        """
        super().__init__()

        self.lambda_autocorr = lambda_autocorr
        self.lambda_spectral = lambda_spectral
        self.lambda_entropy = lambda_entropy

        # Initialize component losses
        self.autocorr_loss = AutocorrelationLoss(
            weight=1.0,  # Apply lambda externally
            **(autocorr_kwargs or {})
        )

        self.spectral_loss = SpectralConsistencyLoss(
            weight=1.0,
            n_fft=n_fft,
            **(spectral_kwargs or {})
        )

        if lambda_entropy > 0:
            self.entropy_loss = SpectralEntropyLoss(
                weight=1.0,
                n_fft=n_fft,
                **(entropy_kwargs or {})
            )
        else:
            self.entropy_loss = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        returns: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined spectral loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            returns: Optional return series (for autocorrelation)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=predictions.device)

        # Autocorrelation loss
        if self.lambda_autocorr > 0:
            # Use predictions as returns if not provided
            pred_returns = returns if returns is not None else predictions
            target_returns = returns if returns is not None else targets

            ac_loss = self.autocorr_loss(pred_returns, target_returns)
            total_loss = total_loss + self.lambda_autocorr * ac_loss
            loss_dict['autocorrelation_loss'] = ac_loss

        # Spectral consistency loss
        if self.lambda_spectral > 0:
            spec_loss = self.spectral_loss(predictions, targets)
            total_loss = total_loss + self.lambda_spectral * spec_loss
            loss_dict['spectral_consistency_loss'] = spec_loss
            loss_dict['spectral_loss'] = spec_loss

        # Entropy regularization
        if self.lambda_entropy > 0 and self.entropy_loss is not None:
            ent_loss = self.entropy_loss(predictions, targets)
            total_loss = total_loss + self.lambda_entropy * ent_loss
            loss_dict['entropy_loss'] = ent_loss

        loss_dict['total_spectral_loss'] = total_loss

        return loss_dict


def create_spectral_loss(
    loss_type: str,
    **kwargs
) -> SpectralResidual:
    """
    Factory function to create spectral loss by name.

    Args:
        loss_type: Type of loss ('autocorrelation', 'spectral_consistency',
                   'spectral_entropy', 'combined')
        **kwargs: Additional arguments for the loss

    Returns:
        SpectralResidual instance

    Example:
        loss = create_spectral_loss('autocorrelation', weight=0.1)
        loss = create_spectral_loss('combined', lambda_autocorr=0.05)
    """
    loss_map = {
        'autocorrelation': AutocorrelationLoss,
        'autocorr': AutocorrelationLoss,
        'ac': AutocorrelationLoss,
        'spectral_consistency': SpectralConsistencyLoss,
        'spectral': SpectralConsistencyLoss,
        'frequency': SpectralConsistencyLoss,
        'spectral_entropy': SpectralEntropyLoss,
        'entropy': SpectralEntropyLoss,
        'combined': CombinedSpectralLoss,
        'combined_spectral': CombinedSpectralLoss,
    }

    loss_type = loss_type.lower().replace('-', '_')
    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown spectral loss type: {loss_type}. "
            f"Available: {list(loss_map.keys())}"
        )

    return loss_map[loss_type](**kwargs)
