"""
PINN-Regime Integration: Physics-Informed Neural Networks with Regime Conditioning

This module demonstrates how market regime information can be integrated into
Physics-Informed Neural Networks (PINNs) for financial forecasting.

Theoretical Framework:
======================

Standard PINN embeds physics constraints (e.g., GBM, Black-Scholes) assuming
constant parameters. This is inconsistent with regime-switching markets where:
- Drift (μ) varies by regime
- Volatility (σ) varies by regime
- Mean-reversion strength varies by regime

Regime-Aware PINN Modifications:
================================

1. GBM Drift Term:
   Standard: dS/S = μdt + σdW
   Regime:   dS/S = μ(S_t)dt + σ(S_t)dW
   where μ(S_t) and σ(S_t) depend on current regime

2. Black-Scholes PDE:
   Standard: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
   Regime:   ∂V/∂t + ½σ(S_t)²S²∂²V/∂S² + rS∂V/∂S - rV = 0
   where σ(S_t) is regime-dependent volatility

3. Ornstein-Uhlenbeck:
   Standard: dX = θ(μ - X)dt + σdW
   Regime:   dX = θ(S_t)(μ(S_t) - X)dt + σ(S_t)dW
   where θ, μ, σ all vary by regime

4. Physics Loss Weighting:
   Different regimes may warrant different physics emphasis:
   - High vol regime: Lower GBM weight (trend less reliable)
   - Low vol regime: Higher OU weight (mean reversion stronger)

Implementation Approach:
========================
1. Regime features as neural network inputs
2. Regime-conditioned physics loss computation
3. Regime-dependent parameter estimation within PINN
4. Transition-aware temporal modeling

This addresses the research gap where standard PINNs assume stationarity
but financial markets exhibit clear regime dynamics.

Author: Dissertation Research Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Regime-Conditioned Physics Functions
# =============================================================================

def regime_conditioned_gbm_drift(
    S: torch.Tensor,
    regime_probs: torch.Tensor,
    regime_drifts: torch.Tensor
) -> torch.Tensor:
    """
    Compute GBM drift conditioned on regime probabilities

    The drift is a probability-weighted average of regime-specific drifts:
        μ_effective = Σ_k P(S_t = k) × μ_k

    This creates a smooth transition between regime-specific dynamics,
    avoiding discontinuities in the physics loss.

    Args:
        S: Stock prices [batch, seq_len]
        regime_probs: Probability of each regime [batch, n_regimes] or [batch, seq_len, n_regimes]
        regime_drifts: Drift for each regime [n_regimes] or [batch, n_regimes]

    Returns:
        Regime-conditioned drift [batch, seq_len] or [batch]
    """
    # Ensure regime_drifts is on same device
    regime_drifts = regime_drifts.to(S.device)

    # Handle different input shapes
    if regime_probs.dim() == 2:
        # [batch, n_regimes] - single probability per batch
        # μ_effective = Σ P(k) × μ_k
        drift = torch.sum(regime_probs * regime_drifts, dim=-1)  # [batch]
        # Expand to match S shape if needed
        if S.dim() == 2:
            drift = drift.unsqueeze(-1).expand_as(S)
    else:
        # [batch, seq_len, n_regimes] - time-varying probabilities
        # μ_effective(t) = Σ P(k, t) × μ_k
        drift = torch.sum(regime_probs * regime_drifts.unsqueeze(0).unsqueeze(0), dim=-1)

    return drift


def regime_conditioned_diffusion(
    S: torch.Tensor,
    regime_probs: torch.Tensor,
    regime_vols: torch.Tensor
) -> torch.Tensor:
    """
    Compute GBM diffusion coefficient conditioned on regime

    For the diffusion term, we use the square root of the variance:
        σ_effective = √(Σ_k P(S_t = k) × σ_k²)

    This ensures the effective volatility is always positive and
    properly accounts for the convexity of variance.

    Args:
        S: Stock prices [batch, seq_len]
        regime_probs: Probability of each regime
        regime_vols: Volatility for each regime [n_regimes]

    Returns:
        Regime-conditioned volatility
    """
    regime_vols = regime_vols.to(S.device)

    if regime_probs.dim() == 2:
        # Variance-weighted average: σ² = Σ P(k) × σ_k²
        variance = torch.sum(regime_probs * regime_vols**2, dim=-1)
        sigma = torch.sqrt(variance + 1e-8)  # Numerical stability

        if S.dim() == 2:
            sigma = sigma.unsqueeze(-1).expand_as(S)
    else:
        variance = torch.sum(regime_probs * (regime_vols**2).unsqueeze(0).unsqueeze(0), dim=-1)
        sigma = torch.sqrt(variance + 1e-8)

    return sigma


def regime_conditioned_mean_reversion(
    X: torch.Tensor,
    regime_probs: torch.Tensor,
    regime_thetas: torch.Tensor,
    regime_means: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OU mean-reversion parameters conditioned on regime

    The mean-reversion force is:
        Force = θ_effective × (μ_effective - X)

    where both θ and μ are regime-weighted.

    Args:
        X: Process values (e.g., log returns)
        regime_probs: Regime probabilities
        regime_thetas: Mean-reversion speeds per regime [n_regimes]
        regime_means: Long-term means per regime [n_regimes]

    Returns:
        Tuple of (theta_effective, mu_effective)
    """
    regime_thetas = regime_thetas.to(X.device)
    regime_means = regime_means.to(X.device)

    if regime_probs.dim() == 2:
        theta = torch.sum(regime_probs * regime_thetas, dim=-1)
        mu = torch.sum(regime_probs * regime_means, dim=-1)

        if X.dim() == 2:
            theta = theta.unsqueeze(-1).expand_as(X)
            mu = mu.unsqueeze(-1).expand_as(X)
    else:
        theta = torch.sum(regime_probs * regime_thetas.unsqueeze(0).unsqueeze(0), dim=-1)
        mu = torch.sum(regime_probs * regime_means.unsqueeze(0).unsqueeze(0), dim=-1)

    return theta, mu


# =============================================================================
# Regime-Conditioned Physics Loss
# =============================================================================

class RegimeConditionedLoss(nn.Module):
    """
    Physics-Informed Loss with Regime Conditioning

    This extends the standard PhysicsLoss to incorporate regime information:
    1. Regime-specific drift and volatility parameters
    2. Regime-dependent loss weighting
    3. Smooth transitions via probability weighting

    Key Innovation:
    Standard PINN assumes constant physics parameters, but financial markets
    exhibit regime-specific dynamics. This loss function:
    - Uses regime probabilities to weight physics parameters
    - Allows different physics emphasis per regime
    - Maintains differentiability for backpropagation
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lambda_gbm: float = 0.1,
        lambda_bs: float = 0.1,
        lambda_ou: float = 0.1,
        dt: float = 1/252,
        risk_free_rate: float = 0.02,
        # Regime-specific weight modifiers
        regime_weights: Optional[Dict[int, Dict[str, float]]] = None
    ):
        """
        Args:
            n_regimes: Number of market regimes
            lambda_gbm: Base GBM loss weight
            lambda_bs: Base Black-Scholes loss weight
            lambda_ou: Base OU loss weight
            dt: Time step
            risk_free_rate: Risk-free rate
            regime_weights: Dict mapping regime index to loss weight modifiers
                Example: {0: {'gbm': 1.2, 'ou': 0.8}, ...}
        """
        super().__init__()

        self.n_regimes = n_regimes
        self.lambda_gbm = lambda_gbm
        self.lambda_bs = lambda_bs
        self.lambda_ou = lambda_ou
        self.dt = dt
        self.risk_free_rate = risk_free_rate

        # Default regime-specific weights
        # Intuition:
        # - Low vol: GBM more reliable (trends), OU for mean reversion
        # - High vol: All physics less reliable, reduce weights
        if regime_weights is None:
            self.regime_weights = {
                0: {'gbm': 1.2, 'bs': 1.0, 'ou': 1.0},    # Low vol: trust GBM
                1: {'gbm': 1.0, 'bs': 1.0, 'ou': 1.0},    # Normal: balanced
                2: {'gbm': 0.6, 'bs': 0.8, 'ou': 0.8},    # High vol: reduce physics
            }
        else:
            self.regime_weights = regime_weights

        # Learnable regime-specific parameters
        # These are optimized during training
        self.regime_drifts = nn.Parameter(torch.tensor([0.0004, 0.0002, -0.0005]))
        self.regime_vols = nn.Parameter(torch.tensor([0.008, 0.012, 0.025]))
        self.regime_thetas = nn.Parameter(torch.tensor([0.5, 1.0, 2.0]))  # Mean reversion
        self.regime_means = nn.Parameter(torch.tensor([0.0003, 0.0001, -0.0002]))

        logger.info(f"RegimeConditionedLoss initialized with {n_regimes} regimes")

    def get_effective_weights(
        self,
        regime_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute effective loss weights based on regime probabilities

        Args:
            regime_probs: Regime probabilities [batch, n_regimes]

        Returns:
            Dict of effective weights for each loss component
        """
        effective_weights = {}

        for loss_type in ['gbm', 'bs', 'ou']:
            # Weight = base_lambda × Σ P(k) × regime_weight[k]
            regime_mods = torch.tensor(
                [self.regime_weights[k][loss_type] for k in range(self.n_regimes)],
                device=regime_probs.device,
                dtype=regime_probs.dtype
            )

            effective_weight = torch.sum(regime_probs * regime_mods, dim=-1)  # [batch]
            effective_weights[loss_type] = effective_weight

        return effective_weights

    def gbm_regime_residual(
        self,
        S: torch.Tensor,
        dS_dt: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        GBM residual with regime-conditioned parameters

        Residual: dS/S - μ(regime)

        Args:
            S: Prices [batch, seq_len]
            dS_dt: Price derivatives [batch, seq_len-1]
            regime_probs: Regime probabilities [batch, n_regimes]

        Returns:
            GBM residual loss (scalar)
        """
        # Get regime-conditioned drift and volatility
        mu = regime_conditioned_gbm_drift(S[:, :-1], regime_probs, self.regime_drifts)

        # GBM residual: dS/S/dt - μ
        S_safe = S[:, :-1] + 1e-8
        returns_actual = (S[:, 1:] - S[:, :-1]) / S_safe / self.dt
        residual = returns_actual - mu

        return torch.mean(residual ** 2)

    def ou_regime_residual(
        self,
        X: torch.Tensor,
        dX_dt: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        OU residual with regime-conditioned parameters

        Residual: dX/dt - θ(regime) × (μ(regime) - X)

        Args:
            X: Process values (returns) [batch, seq_len]
            dX_dt: Derivatives [batch, seq_len-1]
            regime_probs: Regime probabilities [batch, n_regimes]

        Returns:
            OU residual loss (scalar)
        """
        # Get regime-conditioned θ and μ
        theta, mu = regime_conditioned_mean_reversion(
            X[:, :-1], regime_probs, self.regime_thetas, self.regime_means
        )

        # OU residual: dX/dt - θ(μ - X)
        mean_reversion_force = theta * (mu - X[:, :-1])
        residual = dX_dt - mean_reversion_force

        return torch.mean(residual ** 2)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prices: torch.Tensor,
        returns: torch.Tensor,
        regime_probs: torch.Tensor,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute regime-conditioned physics loss

        Args:
            predictions: Model predictions [batch, 1]
            targets: Ground truth [batch, 1]
            prices: Price sequences [batch, seq_len]
            returns: Return sequences [batch, seq_len]
            regime_probs: Regime probabilities [batch, n_regimes]
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

        # Get effective weights based on current regime
        effective_weights = self.get_effective_weights(regime_probs)

        physics_loss = torch.tensor(0.0, device=predictions.device)

        # GBM loss (regime-conditioned)
        if self.lambda_gbm > 0 and prices is not None and prices.shape[1] > 1:
            try:
                dS_dt = (prices[:, 1:] - prices[:, :-1]) / self.dt
                gbm_residual = self.gbm_regime_residual(prices, dS_dt, regime_probs)

                # Apply regime-weighted lambda
                gbm_weight = self.lambda_gbm * effective_weights['gbm'].mean()
                gbm_loss = gbm_weight * gbm_residual

                physics_loss = physics_loss + gbm_loss
                loss_dict['gbm_loss'] = gbm_residual.item()
                loss_dict['gbm_weight'] = gbm_weight.item()

            except Exception as e:
                logger.debug(f"GBM regime loss failed: {e}")

        # OU loss (regime-conditioned)
        if self.lambda_ou > 0 and returns is not None and returns.shape[1] > 1:
            try:
                dX_dt = (returns[:, 1:] - returns[:, :-1]) / self.dt
                ou_residual = self.ou_regime_residual(returns, dX_dt, regime_probs)

                ou_weight = self.lambda_ou * effective_weights['ou'].mean()
                ou_loss = ou_weight * ou_residual

                physics_loss = physics_loss + ou_loss
                loss_dict['ou_loss'] = ou_residual.item()
                loss_dict['ou_weight'] = ou_weight.item()

            except Exception as e:
                logger.debug(f"OU regime loss failed: {e}")

        total_loss = data_loss + physics_loss
        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_learned_regime_params(self) -> Dict[str, np.ndarray]:
        """Get learned regime-specific parameters"""
        return {
            'drifts': self.regime_drifts.detach().cpu().numpy(),
            'volatilities': torch.nn.functional.softplus(self.regime_vols).detach().cpu().numpy(),
            'thetas': torch.nn.functional.softplus(self.regime_thetas).detach().cpu().numpy(),
            'means': self.regime_means.detach().cpu().numpy(),
        }


# =============================================================================
# Regime-Aware PINN Model
# =============================================================================

class RegimeAwarePINN(nn.Module):
    """
    Physics-Informed Neural Network with Regime Awareness

    This model extends the standard PINN architecture to:
    1. Accept regime information as additional input
    2. Use regime-conditioned physics losses
    3. Learn regime-specific physics parameters

    Architecture:
    - Input: [price_features, regime_probabilities]
    - Encoder: Process combined features
    - LSTM/GRU: Temporal modeling
    - Physics-informed output with regime conditioning

    Key Innovations:
    1. Regime Embedding: Soft regime probabilities as input features
    2. Regime-Gated Physics: Different physics emphasis per regime
    3. Learned Regime Parameters: μ, σ, θ are learned per regime
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        n_regimes: int = 3,
        dropout: float = 0.2,
        base_model: str = 'lstm',
        lambda_gbm: float = 0.1,
        lambda_ou: float = 0.1
    ):
        """
        Args:
            input_dim: Number of input features (excluding regime)
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            output_dim: Output dimension
            n_regimes: Number of market regimes
            dropout: Dropout probability
            base_model: 'lstm' or 'gru'
            lambda_gbm: GBM loss weight
            lambda_ou: OU loss weight
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim

        # Regime embedding layer
        # Maps regime probabilities to a learned representation
        self.regime_encoder = nn.Sequential(
            nn.Linear(n_regimes, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # Combined input dimension
        combined_dim = input_dim + hidden_dim // 4

        # Base temporal model
        if base_model == 'lstm':
            self.rnn = nn.LSTM(
                input_size=combined_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            self.rnn = nn.GRU(
                input_size=combined_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Regime-conditioned physics loss
        self.physics_loss = RegimeConditionedLoss(
            n_regimes=n_regimes,
            lambda_gbm=lambda_gbm,
            lambda_ou=lambda_ou
        )

        logger.info(
            f"RegimeAwarePINN initialized: input_dim={input_dim}, "
            f"n_regimes={n_regimes}, hidden_dim={hidden_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch, seq_len, input_dim]
            regime_probs: Regime probabilities [batch, n_regimes] or [batch, seq_len, n_regimes]

        Returns:
            Predictions [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Default to uniform regime probabilities if not provided
        if regime_probs is None:
            regime_probs = torch.ones(batch_size, self.n_regimes, device=x.device) / self.n_regimes

        # Encode regime information
        if regime_probs.dim() == 2:
            # [batch, n_regimes] -> expand to [batch, seq_len, regime_embed_dim]
            regime_embed = self.regime_encoder(regime_probs)  # [batch, embed_dim]
            regime_embed = regime_embed.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # [batch, seq_len, n_regimes] -> [batch, seq_len, regime_embed_dim]
            regime_embed = self.regime_encoder(regime_probs)

        # Concatenate input features with regime embedding
        x_combined = torch.cat([x, regime_embed], dim=-1)

        # Pass through RNN
        rnn_out, _ = self.rnn(x_combined)

        # Use last hidden state for prediction
        last_hidden = rnn_out[:, -1, :]

        # Output projection
        output = self.output_layer(last_hidden)

        return output

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
        enable_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute regime-conditioned physics loss

        Args:
            predictions: Model predictions
            targets: Ground truth
            metadata: Dict containing:
                - prices: Price sequences
                - returns: Return sequences
                - regime_probs: Regime probabilities
            enable_physics: Whether to enable physics losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        prices = metadata.get('prices', None)
        returns = metadata.get('returns', None)
        regime_probs = metadata.get('regime_probs', None)

        if regime_probs is None:
            # Default uniform
            batch_size = predictions.shape[0]
            regime_probs = torch.ones(
                batch_size, self.n_regimes,
                device=predictions.device
            ) / self.n_regimes

        return self.physics_loss(
            predictions=predictions,
            targets=targets,
            prices=prices,
            returns=returns,
            regime_probs=regime_probs,
            enable_physics=enable_physics
        )

    def get_regime_params(self) -> Dict[str, np.ndarray]:
        """Get learned regime parameters"""
        return self.physics_loss.get_learned_regime_params()


# =============================================================================
# Integration with Monte Carlo
# =============================================================================

def create_regime_features(
    regime_estimates,  # RegimeEstimates from regime_monte_carlo
    lookback: int = 30
) -> torch.Tensor:
    """
    Create regime features for PINN input

    Converts regime estimates into features suitable for neural network input.

    Args:
        regime_estimates: RegimeEstimates object
        lookback: Number of recent days to use

    Returns:
        Regime features [lookback, n_regimes + regime_summary_features]
    """
    n = len(regime_estimates.regime_labels)
    n_regimes = regime_estimates.n_regimes

    # Recent regime probabilities
    start_idx = max(0, n - lookback)
    recent_probs = regime_estimates.regime_probabilities[start_idx:]

    # Additional features: regime persistence, transition indicators
    recent_labels = regime_estimates.regime_labels[start_idx:]

    # Compute transition indicators
    transitions = np.diff(recent_labels) != 0
    transitions = np.concatenate([[0], transitions.astype(float)])

    # Compute regime persistence (time since last transition)
    persistence = np.zeros(len(recent_labels))
    days_since_transition = 0
    for i, label in enumerate(recent_labels):
        if i > 0 and recent_labels[i] != recent_labels[i-1]:
            days_since_transition = 0
        persistence[i] = days_since_transition / 21  # Normalize by month
        days_since_transition += 1

    # Combine features
    features = np.column_stack([
        recent_probs,
        transitions.reshape(-1, 1),
        persistence.reshape(-1, 1)
    ])

    return torch.tensor(features, dtype=torch.float32)


def estimate_pinn_regime_parameters(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Estimate initial PINN physics parameters from historical data

    These serve as good initial values for learnable parameters.

    Args:
        returns: Historical return series
        regime_labels: Regime labels
        n_regimes: Number of regimes

    Returns:
        Dict of parameter tensors for PINN initialization
    """
    drifts = []
    vols = []
    thetas = []
    means = []

    for k in range(n_regimes):
        mask = regime_labels == k
        if mask.sum() > 0:
            regime_returns = returns[mask]
            drifts.append(np.mean(regime_returns))
            vols.append(np.std(regime_returns))
            means.append(np.mean(regime_returns))

            # Estimate mean reversion speed (approximate)
            # Using autocorrelation: θ ≈ -log(ρ_1) / dt
            if len(regime_returns) > 10:
                autocorr = np.corrcoef(regime_returns[:-1], regime_returns[1:])[0, 1]
                theta = -np.log(max(abs(autocorr), 0.01)) * 252  # Annualized
                thetas.append(np.clip(theta, 0.1, 50))
            else:
                thetas.append(1.0)
        else:
            drifts.append(0.0)
            vols.append(0.01)
            thetas.append(1.0)
            means.append(0.0)

    return {
        'regime_drifts': torch.tensor(drifts, dtype=torch.float32),
        'regime_vols': torch.tensor(vols, dtype=torch.float32),
        'regime_thetas': torch.tensor(thetas, dtype=torch.float32),
        'regime_means': torch.tensor(means, dtype=torch.float32),
    }


# =============================================================================
# Documentation: How to Integrate with Existing PINN
# =============================================================================

"""
INTEGRATION GUIDE: Adding Regime Awareness to Existing PINN

The existing PINN implementation (src/models/pinn.py) can be extended for
regime awareness in several ways:

Method 1: Feature Augmentation (Simplest)
=========================================
Add regime probabilities to input features:

    # In data preparation
    regime_probs = regime_detector.predict_proba(returns)
    x_augmented = torch.cat([x_original, regime_probs], dim=-1)

Method 2: Loss Modification (Recommended)
=========================================
Replace PhysicsLoss with RegimeConditionedLoss:

    from src.simulation.pinn_regime_integration import RegimeConditionedLoss

    # In PINNModel.__init__
    self.physics_loss = RegimeConditionedLoss(
        n_regimes=3,
        lambda_gbm=lambda_gbm,
        lambda_ou=lambda_ou
    )

Method 3: Full Architecture (Most Expressive)
=============================================
Use RegimeAwarePINN which has built-in regime encoding:

    from src.simulation.pinn_regime_integration import RegimeAwarePINN

    model = RegimeAwarePINN(
        input_dim=5,
        hidden_dim=128,
        n_regimes=3
    )

Benefits of Regime Integration:
==============================
1. More accurate physics during crisis periods
2. Adaptive loss weighting by market condition
3. Learned regime-specific parameters (interpretable)
4. Better tail risk prediction

Limitations:
===========
1. Requires regime estimation (adds complexity)
2. More parameters to learn
3. May overfit if regimes are noisy
4. Historical regime labels may not predict future

Research Directions:
===================
1. Joint regime-prediction and forecasting
2. Online regime adaptation
3. Uncertainty quantification per regime
4. Multi-asset regime modeling
"""


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    """Demonstrate regime-aware PINN"""

    print("=" * 60)
    print("REGIME-AWARE PINN DEMONSTRATION")
    print("=" * 60)

    # Create synthetic data
    batch_size = 32
    seq_len = 30
    input_dim = 5
    n_regimes = 3

    # Random inputs
    x = torch.randn(batch_size, seq_len, input_dim)

    # Random regime probabilities (normalized)
    regime_probs = torch.rand(batch_size, n_regimes)
    regime_probs = regime_probs / regime_probs.sum(dim=-1, keepdim=True)

    # Create model
    model = RegimeAwarePINN(
        input_dim=input_dim,
        hidden_dim=64,
        n_regimes=n_regimes,
        lambda_gbm=0.1,
        lambda_ou=0.1
    )

    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim} + {n_regimes} regime features")
    print(f"  Hidden dim: 64")
    print(f"  Output dim: 1")

    # Forward pass
    output = model(x, regime_probs)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Regime probs shape: {regime_probs.shape}")
    print(f"  Output shape: {output.shape}")

    # Compute loss
    targets = torch.randn(batch_size, 1)
    prices = torch.exp(torch.cumsum(torch.randn(batch_size, seq_len) * 0.01, dim=1)) * 100
    returns = torch.diff(torch.log(prices), dim=1)

    metadata = {
        'prices': prices,
        'returns': returns,
        'regime_probs': regime_probs
    }

    loss, loss_dict = model.compute_loss(output, targets, metadata)

    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    # Get learned parameters
    params = model.get_regime_params()
    print(f"\nLearned regime parameters:")
    for key, values in params.items():
        print(f"  {key}: {values}")

    print("\n" + "=" * 60)
    print("Demo complete!")
