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
from typing import Dict, Tuple, Optional, Union
import math

from ..utils.logger import get_logger
from .baseline import LSTMModel
from ..constants import (
    DEFAULT_LAMBDA_GBM,
    DEFAULT_LAMBDA_OU,
    DEFAULT_LAMBDA_BS,
    DEFAULT_LAMBDA_LANGEVIN,
    RISK_FREE_RATE,
    DAILY_TIME_STEP,
)

logger = get_logger(__name__)


class PhysicsLoss(nn.Module):
    """
    Physics-based loss functions for financial time series

    DIMENSIONAL CONSISTENCY:
    Physics residuals must be computed on properly scaled data:
    - Black-Scholes: Requires real (de-normalised) prices S and option values V
    - GBM/OU/Langevin: Residuals are normalised by their std to ensure consistent magnitude

    This addresses the audit finding about mixing normalised and raw values.
    """

    def __init__(
        self,
        lambda_gbm: float = DEFAULT_LAMBDA_GBM,
        lambda_bs: float = DEFAULT_LAMBDA_BS,
        lambda_ou: float = DEFAULT_LAMBDA_OU,
        lambda_langevin: float = DEFAULT_LAMBDA_LANGEVIN,
        risk_free_rate: float = RISK_FREE_RATE,
        dt: float = DAILY_TIME_STEP,
        # Learnable physics parameters (initial values)
        theta_init: float = 1.0,      # OU mean reversion speed
        gamma_init: float = 0.5,      # Langevin friction coefficient
        temperature_init: float = 0.1,  # Langevin temperature
        # Scaler parameters for de-normalisation (set during training)
        price_mean: float = 0.0,
        price_std: float = 1.0,
        normalise_residuals: bool = True  # Normalise residuals for consistent magnitude
    ):
        """
        Initialize physics loss with LEARNABLE physics parameters

        Args:
            lambda_gbm: Weight for GBM constraint
            lambda_bs: Weight for Black-Scholes constraint
            lambda_ou: Weight for Ornstein-Uhlenbeck constraint
            lambda_langevin: Weight for Langevin dynamics constraint
            risk_free_rate: Risk-free interest rate (annualized)
            dt: Time step in years
            theta_init: Initial value for OU mean reversion speed (learnable)
            gamma_init: Initial value for Langevin friction (learnable)
            temperature_init: Initial value for Langevin temperature (learnable)
        """
        super(PhysicsLoss, self).__init__()

        self.lambda_gbm = lambda_gbm
        self.lambda_bs = lambda_bs
        self.lambda_ou = lambda_ou
        self.lambda_langevin = lambda_langevin
        self.risk_free_rate = risk_free_rate
        self.dt = dt

        # ===== DIMENSIONAL CONSISTENCY =====
        # Store scaler parameters for de-normalising prices in physics computations
        self.price_mean = price_mean
        self.price_std = price_std
        self.normalise_residuals = normalise_residuals
        # Lambda schedule (currently constant; warm-up not implemented yet)
        self.lambda_schedule = "constant"

        # Track residual magnitudes for diagnostics
        self._residual_rms = {
            'gbm': 0.0,
            'ou': 0.0,
            'langevin': 0.0,
            'langevin_diffusion': 0.0,
            'black_scholes': 0.0
        }

        # ========== LEARNABLE PHYSICS PARAMETERS ==========
        # These are registered as nn.Parameter so they're optimized during training
        # This addresses the audit finding about hardcoded parameters

        # Ornstein-Uhlenbeck mean reversion speed (θ)
        # Higher theta = faster mean reversion
        # Constrained positive via softplus in forward pass
        self.theta_raw = nn.Parameter(torch.tensor(theta_init))

        # Langevin friction coefficient (γ)
        # Represents market friction/resistance to momentum
        self.gamma_raw = nn.Parameter(torch.tensor(gamma_init))

        # Langevin temperature (T)
        # Represents noise/uncertainty level in the market
        self.temperature_raw = nn.Parameter(torch.tensor(temperature_init))

        logger.info(f"Physics parameters initialized: θ={theta_init:.3f}, γ={gamma_init:.3f}, T={temperature_init:.3f}")

    @property
    def theta(self) -> torch.Tensor:
        """Get learned OU mean reversion speed (constrained positive)"""
        return torch.nn.functional.softplus(self.theta_raw)

    @property
    def gamma(self) -> torch.Tensor:
        """Get learned Langevin friction coefficient (constrained positive)"""
        return torch.nn.functional.softplus(self.gamma_raw)

    @property
    def temperature(self) -> torch.Tensor:
        """Get learned Langevin temperature (constrained positive)"""
        return torch.nn.functional.softplus(self.temperature_raw)

    def get_learned_params(self) -> Dict[str, float]:
        """Get current learned physics parameter values"""
        return {
            'theta': self.theta.item(),
            'gamma': self.gamma.item(),
            'temperature': self.temperature.item()
        }

    def get_residual_rms(self) -> Dict[str, float]:
        """
        Get RMS magnitudes of physics residuals (for diagnostics).

        These values indicate the scale of each physics residual before normalisation.
        Large values may indicate dimensional inconsistency or poor constraint fit.

        Returns:
            Dict with gbm_residual_rms, ou_residual_rms, langevin_residual_rms, bs_residual_rms
        """
        return {
            'gbm_residual_rms': self._residual_rms.get('gbm', 0.0),
            'ou_residual_rms': self._residual_rms.get('ou', 0.0),
            'langevin_residual_rms': self._residual_rms.get('langevin', 0.0),
            'langevin_diffusion_rms': self._residual_rms.get('langevin_diffusion', 0.0),
            'bs_residual_rms': self._residual_rms.get('black_scholes', 0.0)
        }

    def set_scaler_params(self, price_mean: float, price_std: float):
        """
        Set scaler parameters for de-normalising physics computations.

        Should be called before training with the dataset's scaler values.

        Args:
            price_mean: Mean of the close price used for normalisation
            price_std: Std of the close price used for normalisation
        """
        self.price_mean = price_mean
        self.price_std = price_std
        logger.info(f"PhysicsLoss scaler params set: mean={price_mean:.4f}, std={price_std:.4f}")

    def gbm_residual(
        self,
        S: torch.Tensor,
        S_next: torch.Tensor,
        sigma: torch.Tensor,
        mu_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Geometric Brownian Motion residual in **log-price space**.

        This avoids mixing price-space and log-return dynamics. We compare the
        observed log return to a drift term μ·dt (optionally provided) and
        normalise by the running standard deviation to keep λ coefficients
        stable when dt=1/252.

        Args:
            S: Current prices
            S_next: Next-step prices
            sigma: Volatility tensor (same shape as S)
            mu_override: Optional drift to use instead of empirical drift
        """
        # Log-return over one step (dimensionless and numerically stable)
        log_ret = torch.log(torch.clamp(S_next / (S + 1e-8), min=1e-8))

        # Empirical drift per step (log space); keep dimensionally consistent
        if mu_override is None:
            mu_step = log_ret.mean(dim=1, keepdim=True)
        else:
            mu_step = mu_override

        expected = mu_step - 0.5 * (sigma ** 2) * self.dt
        residual = (log_ret - expected) / self.dt

        if self.normalise_residuals:
            residual_std = residual.std() + 1e-8
            residual = residual / residual_std
            self._residual_rms['gbm'] = float(residual_std.detach())

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
        Black-Scholes PDE residual (legacy - uses pre-computed derivatives):
        ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

        NOTE: For proper physics-informed learning, use black_scholes_autograd_residual
        which computes derivatives via automatic differentiation.

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

    def black_scholes_autograd_residual(
        self,
        model: nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,
        price_feature_idx: int = 0,
        r: Optional[torch.Tensor] = None,
        price_mean: Optional[float] = None,
        price_std: Optional[float] = None
    ) -> torch.Tensor:
        """
        Black-Scholes PDE residual using AUTOMATIC DIFFERENTIATION

        This is the correct physics-informed implementation that computes
        derivatives via torch.autograd.grad with create_graph=True.

        Black-Scholes PDE: ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

        DIMENSIONAL CONSISTENCY FIX:
        The audit identified that mixing normalised V,S with raw σ,r creates
        dimensionally inconsistent residuals. We now de-normalise S and V
        before computing the PDE.

        Args:
            model: Neural network model that produces predictions V
            x: Input tensor [batch, seq_len, features] with the price feature
            sigma: Volatility tensor [batch] or scalar (should be in real units)
            price_feature_idx: Index of the price feature in the feature dimension
            r: Risk-free rate (optional, uses default if None)
            price_mean: Mean for de-normalising prices (uses self.price_mean if None)
            price_std: Std for de-normalising prices (uses self.price_std if None)

        Returns:
            Black-Scholes PDE residual loss (normalised)
        """
        if r is None:
            r = torch.tensor(self.risk_free_rate, device=x.device, dtype=x.dtype)

        # Use stored scaler params if not provided
        if price_mean is None:
            price_mean = self.price_mean
        if price_std is None:
            price_std = self.price_std

        if price_std is None or float(price_std) == 0.0:
            raise ValueError("price_std must be provided and non-zero for Black-Scholes residual")

        price_mean_t = torch.as_tensor(price_mean, device=x.device, dtype=x.dtype)
        price_std_t = torch.as_tensor(price_std, device=x.device, dtype=x.dtype)

        # Clone input and enable gradient tracking
        x_grad = x.clone().detach().requires_grad_(True)

        # Forward pass through model
        V_norm = model(x_grad)
        # Handle models that return tuple (output, hidden_state)
        if isinstance(V_norm, tuple):
            V_norm = V_norm[0]
        if len(V_norm.shape) == 1:
            V_norm = V_norm.unsqueeze(-1)

        # Extract current price S from the last timestep (normalised)
        S_norm = x[:, -1, price_feature_idx:price_feature_idx+1]  # [batch, 1]

        # ===== DE-NORMALISE S AND V FOR DIMENSIONAL CONSISTENCY =====
        # Black-Scholes requires real price levels, not z-scores
        S = S_norm * price_std_t + price_mean_t
        V = V_norm * price_std_t + price_mean_t

        # Ensure sigma has correct shape
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)

        # ========== COMPUTE FIRST DERIVATIVE dV/dS via AUTOGRAD ==========
        # Note: derivatives computed w.r.t. normalised x, then scaled
        grad_outputs = torch.ones_like(V_norm)
        dV_dx = torch.autograd.grad(
            outputs=V_norm,
            inputs=x_grad,
            grad_outputs=grad_outputs,
            create_graph=True,  # CRITICAL: enables higher-order derivatives
            retain_graph=True
        )[0]

        # Extract gradient w.r.t. price feature at last timestep
        dV_dS_norm = dV_dx[:, -1, price_feature_idx:price_feature_idx+1]  # [batch, 1]
        # Chain rule: dV/dS = (dV_norm/dS_norm) / price_std
        dV_dS = dV_dS_norm / (price_std_t + 1e-8)

        # ========== COMPUTE SECOND DERIVATIVE d²V/dS² via AUTOGRAD ==========
        d2V_dx = torch.autograd.grad(
            outputs=dV_dS_norm,
            inputs=x_grad,
            grad_outputs=torch.ones_like(dV_dS_norm),
            create_graph=True,  # CRITICAL: integrates into training
            retain_graph=True
        )[0]

        d2V_dS2_norm = d2V_dx[:, -1, price_feature_idx:price_feature_idx+1]  # [batch, 1]
        # Chain rule: d²V/dS² scales with 1/std²
        d2V_dS2 = d2V_dS2_norm / ((price_std_t + 1e-8) ** 2)

        # ========== BLACK-SCHOLES PDE RESIDUAL ==========
        # Simplified steady-state form (without ∂V/∂t):
        # ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
        bs_residual = (
            0.5 * (sigma ** 2) * (S ** 2) * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        # ===== NORMALISE RESIDUAL FOR CONSISTENT MAGNITUDE =====
        if self.normalise_residuals:
            residual_std = bs_residual.std() + 1e-8
            bs_residual = bs_residual / residual_std
            self._residual_rms['black_scholes'] = float(residual_std.detach())

        return torch.mean(bs_residual ** 2)

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
            OU residual (normalised for dimensional consistency)
        """
        # OU equation: dX = θ(μ - X)dt + σdW
        # Residual: dX/dt - θ(μ - X)
        residual = dX_dt - theta * (mu - X)

        # ===== NORMALISE RESIDUAL FOR DIMENSIONAL CONSISTENCY =====
        if self.normalise_residuals:
            residual_std = residual.std() + 1e-8
            residual = residual / residual_std
            self._residual_rms['ou'] = float(residual_std.detach())

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
            Langevin residual (normalised for dimensional consistency)
        """
        # Langevin equation: dX/dt = -γ∇U(X) + noise
        residual = dX_dt + gamma * grad_U

        # Diffusion-consistency residual: target diffusion magnitude sqrt(2γT)
        diffusion_target = torch.sqrt(torch.clamp(2.0 * gamma * T, min=1e-8))
        diffusion_residual = torch.abs(dX_dt) - diffusion_target

        # ===== NORMALISE RESIDUALS FOR DIMENSIONAL CONSISTENCY =====
        if self.normalise_residuals:
            residual_std = residual.std() + 1e-8
            residual = residual / residual_std
            self._residual_rms['langevin'] = float(residual_std.detach())

            diffusion_std = diffusion_residual.std() + 1e-8
            diffusion_residual = diffusion_residual / diffusion_std
            self._residual_rms['langevin_diffusion'] = float(diffusion_std.detach())

        return torch.mean(residual ** 2 + diffusion_residual ** 2)

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
        if self.lambda_gbm > 0 and prices is not None and prices.shape[1] > 1:
            try:
                S = prices[:, :-1]  # Current prices
                S_next = prices[:, 1:]  # Next prices
                sigma = volatilities[:, :-1]

                # Optional drift override remains in log space for consistency
                gbm_loss = self.gbm_residual(S, S_next, sigma, mu_override=None)
                physics_loss = physics_loss + self.lambda_gbm * gbm_loss
                loss_dict['gbm_loss'] = gbm_loss.item()
                loss_dict['gbm_loss_weighted'] = (self.lambda_gbm * gbm_loss).item()

            except Exception as e:
                logger.debug(f"GBM loss computation failed: {e}")

        # Ornstein-Uhlenbeck constraint (mean reversion in returns)
        if self.lambda_ou > 0 and returns is not None and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]  # Current returns
                X_next = returns[:, 1:]  # Next returns
                dX_dt = (X_next - X) / self.dt

                # Use LEARNABLE theta (mean reversion speed)
                # self.theta is constrained positive via softplus
                theta = self.theta.to(returns.device)
                mu = returns.mean(dim=1, keepdim=True)  # Long-term mean
                sigma = returns.std(dim=1, keepdim=True)  # Volatility

                ou_loss = self.ornstein_uhlenbeck_residual(X, dX_dt, theta, mu, sigma)
                physics_loss = physics_loss + self.lambda_ou * ou_loss
                loss_dict['ou_loss'] = ou_loss.item()
                loss_dict['ou_loss_weighted'] = (self.lambda_ou * ou_loss).item()
                loss_dict['theta_learned'] = theta.item()  # Log learned value

            except Exception as e:
                logger.debug(f"OU loss computation failed: {e}")

        # Langevin dynamics (momentum)
        if self.lambda_langevin > 0 and returns is not None and returns.shape[1] > 1:
            try:
                X = returns[:, :-1]
                X_next = returns[:, 1:]
                dX_dt = (X_next - X) / self.dt

                # Approximate gradient of potential
                grad_U = -returns[:, :-1]  # Negative returns as potential gradient

                # Use LEARNABLE gamma (friction) and temperature
                # Both are constrained positive via softplus
                gamma = self.gamma.to(returns.device)
                T = self.temperature.to(returns.device)

                langevin_loss = self.langevin_residual(X, dX_dt, grad_U, gamma, T)
                physics_loss = physics_loss + self.lambda_langevin * langevin_loss
                loss_dict['langevin_loss'] = langevin_loss.item()
                loss_dict['langevin_loss_weighted'] = (self.lambda_langevin * langevin_loss).item()
                loss_dict['gamma_learned'] = gamma.item()  # Log learned values
                loss_dict['temperature_learned'] = T.item()

            except Exception as e:
                logger.debug(f"Langevin loss computation failed: {e}")

        # Total loss
        total_loss = data_loss + physics_loss

        loss_dict['physics_loss'] = physics_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        # ===== LOG RESIDUAL MAGNITUDES FOR DIAGNOSTICS =====
        # These help identify dimensional inconsistency issues
        residual_rms = self.get_residual_rms()
        loss_dict.update(residual_rms)

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
        lambda_bs: float = 0.1,  # Now enabled with proper autograd implementation
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
            lambda_bs: Black-Scholes loss weight (uses autograd for derivatives)
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
                   f"λ_gbm={lambda_gbm}, λ_bs={lambda_bs}, λ_ou={lambda_ou}, λ_langevin={lambda_langevin}")

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
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
            metadata: Batch metadata containing:
                - prices: Price sequences
                - returns: Return sequences
                - volatilities: Volatility sequences
                - inputs: Original input features (for Black-Scholes autograd)
                - price_feature_idx: Index of price feature (default: 0)
            enable_physics: Whether to enable physics losses

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Extract physics data from metadata
        prices = metadata.get('prices', None)
        returns = metadata.get('returns', None)
        volatilities = metadata.get('volatilities', None)
        inputs = metadata.get('inputs', None)  # For Black-Scholes autograd
        price_feature_idx = metadata.get('price_feature_idx', 0)

        # Compute base physics-informed loss (GBM, OU, Langevin)
        total_loss, loss_dict = self.physics_loss(
            predictions=predictions,
            targets=targets,
            prices=prices,
            returns=returns,
            volatilities=volatilities,
            enable_physics=enable_physics
        )

        # ========== BLACK-SCHOLES WITH AUTOMATIC DIFFERENTIATION ==========
        # This uses the new autograd implementation that computes derivatives
        # via torch.autograd.grad with create_graph=True
        if (enable_physics and
            self.physics_loss.lambda_bs > 0 and
            inputs is not None and
            volatilities is not None):
            try:
                # Get volatility (use mean volatility or last timestep)
                if volatilities.dim() > 1:
                    sigma = volatilities[:, -1]  # [batch]
                else:
                    sigma = volatilities

                # Compute Black-Scholes loss using autograd
                bs_loss = self.physics_loss.black_scholes_autograd_residual(
                    model=self,
                    x=inputs,
                    sigma=sigma,
                    price_feature_idx=price_feature_idx
                )

                # Add to total loss
                total_loss = total_loss + self.physics_loss.lambda_bs * bs_loss
                loss_dict['bs_loss'] = bs_loss.item()
                loss_dict['bs_loss_weighted'] = (self.physics_loss.lambda_bs * bs_loss).item()
                loss_dict['total_loss'] = total_loss.item()

                logger.debug(f"Black-Scholes autograd loss: {bs_loss.item():.6f}")

            except Exception as e:
                logger.debug(f"Black-Scholes autograd computation failed: {e}")

        return total_loss, loss_dict

    def get_learned_physics_params(self) -> Dict[str, float]:
        """
        Get current learned physics parameter values

        Returns:
            Dict with learned parameter values:
                - theta: OU mean reversion speed
                - gamma: Langevin friction coefficient
                - temperature: Langevin temperature
        """
        return self.physics_loss.get_learned_params()

    def log_physics_params(self):
        """Log current learned physics parameters"""
        params = self.get_learned_physics_params()
        logger.info("Learned Physics Parameters:")
        logger.info(f"  θ (OU mean reversion): {params['theta']:.4f}")
        logger.info(f"  γ (Langevin friction): {params['gamma']:.4f}")
        logger.info(f"  T (Langevin temperature): {params['temperature']:.4f}")

    def get_residual_rms(self) -> Dict[str, float]:
        """Get RMS magnitudes of physics residuals (for diagnostics)"""
        return self.physics_loss.get_residual_rms()

    def set_scaler_params(self, price_mean: float, price_std: float):
        """Set scaler parameters for de-normalising physics computations"""
        self.physics_loss.set_scaler_params(price_mean, price_std)
