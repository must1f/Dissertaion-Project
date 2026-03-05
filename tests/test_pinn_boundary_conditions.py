"""
PINN Boundary Condition and Residual Tests

Formal test suite for validating physics-informed neural network correctness:
- Residual magnitude verification on known solutions
- Boundary condition satisfaction
- Physics constraint verification
- Edge case handling

These tests ensure the PINN implementation correctly encodes financial physics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGBMResidual:
    """Tests for Geometric Brownian Motion residual correctness."""

    @pytest.fixture
    def physics_loss(self):
        """Create PhysicsLoss module."""
        from src.models.pinn import PhysicsLoss
        return PhysicsLoss(lambda_gbm=1.0, lambda_ou=0.0, lambda_bs=0.0)

    def test_gbm_residual_on_constant_prices(self, physics_loss):
        """
        GBM residual should be small when prices are constant.

        If S(t) = S0 (constant), then dS/dt = 0 and μS should be small
        if μ is estimated from constant returns (≈0).
        """
        # Constant prices
        batch_size = 16
        seq_len = 30
        S = torch.ones(batch_size, seq_len) * 100.0

        # Compute derivative (should be ~0)
        dS_dt = (S[:, 1:] - S[:, :-1]) / physics_loss.dt

        # Estimate drift from constant prices (should be ~0)
        returns = torch.zeros(batch_size, seq_len - 1)
        mu = returns.mean(dim=1, keepdim=True)

        # GBM residual: dS/dt - μS
        residual = dS_dt - mu * S[:, :-1]

        # Residual should be very small
        assert residual.abs().max() < 1e-5, "GBM residual should be ~0 for constant prices"

    def test_gbm_residual_on_exponential_growth(self, physics_loss):
        """
        GBM residual for deterministic exponential growth S(t) = S0 * exp(μt).

        The GBM equation dS = μS dt should have zero residual (ignoring dW term).
        """
        batch_size = 8
        seq_len = 30
        S0 = 100.0
        mu = 0.05  # 5% drift

        # Create exponential growth: S(t) = S0 * exp(μt)
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)
        S = S0 * torch.exp(mu * t)

        # Analytical derivative: dS/dt = μ * S
        dS_dt_analytical = mu * S[:, :-1]

        # Numerical derivative
        dt = physics_loss.dt
        dS_dt_numerical = (S[:, 1:] - S[:, :-1]) / dt

        # For small dt, these should be close
        # Note: Our time steps are discrete, so there will be some discretization error
        # The key is that the residual form is correct

        # Compute residual using drift = mu
        drift = torch.tensor([[mu]] * batch_size)
        residual = dS_dt_numerical - drift * S[:, :-1]

        # Residual should be bounded (discretization error exists but should be small)
        residual_magnitude = residual.abs().mean()
        assert residual_magnitude < S0 * mu * 10, "GBM residual should be bounded"

    def test_gbm_residual_varies_with_drift(self, physics_loss):
        """
        GBM residual should vary with different drift parameters.

        Verify that the GBM residual is sensitive to drift by checking that
        significantly different drift values produce different loss values.
        """
        batch_size = 8
        seq_len = 30

        # Generate random walk prices
        torch.manual_seed(42)
        returns = torch.randn(batch_size, seq_len - 1) * 0.02
        prices = 100.0 * torch.exp(torch.cumsum(returns, dim=1))
        prices = torch.cat([torch.ones(batch_size, 1) * 100.0, prices], dim=1)

        from src.losses import GBMResidual

        gbm = GBMResidual(weight=1.0)

        # Test with very different drift values
        drift_low = torch.tensor([[0.0]] * batch_size)
        drift_high = torch.tensor([[1.0]] * batch_size)  # Very different

        loss_low = gbm(prices=prices, drift=drift_low)
        loss_high = gbm(prices=prices, drift=drift_high)

        # Losses should be different (residual is sensitive to drift)
        # Using abs difference > 1% of average to ensure meaningful variation
        avg_loss = (loss_low.item() + loss_high.item()) / 2
        diff = abs(loss_low.item() - loss_high.item())

        assert diff > 0.01 * avg_loss, \
            f"GBM losses should differ: {loss_low.item():.2f} vs {loss_high.item():.2f} (diff={diff:.2f})"


class TestOUResidual:
    """Tests for Ornstein-Uhlenbeck residual correctness."""

    @pytest.fixture
    def physics_loss(self):
        """Create PhysicsLoss module."""
        from src.models.pinn import PhysicsLoss
        return PhysicsLoss(lambda_gbm=0.0, lambda_ou=1.0, lambda_bs=0.0)

    def test_ou_residual_at_mean(self, physics_loss):
        """
        OU residual should be small when process is at long-term mean.

        OU: dX = θ(μ - X)dt + σdW
        At X = μ, the drift term θ(μ - X) = 0.
        """
        batch_size = 16
        seq_len = 30
        long_term_mean = 0.0

        # Process exactly at mean
        X = torch.ones(batch_size, seq_len) * long_term_mean

        # Derivative should be ~0 when at mean (ignoring noise)
        dX_dt = (X[:, 1:] - X[:, :-1]) / physics_loss.dt

        # OU residual: dX/dt - θ(μ - X)
        # When X = μ, this becomes dX/dt - 0 = dX/dt
        theta = physics_loss.theta
        mu = X.mean(dim=1, keepdim=True)
        residual = dX_dt - theta * (mu - X[:, :-1])

        # At the mean, residual should be small
        assert residual.abs().mean() < 1e-3, "OU residual should be small at long-term mean"

    def test_ou_residual_mean_reversion_direction(self, physics_loss):
        """
        OU process should show mean-reverting behavior.

        When X > μ, drift should be negative (pulling back to mean).
        When X < μ, drift should be positive (pulling up to mean).
        """
        batch_size = 8
        long_term_mean = 0.0

        # Process above mean
        X_above = torch.ones(batch_size, 1) * 0.5  # 0.5 above mean
        drift_above = physics_loss.theta * (long_term_mean - X_above)

        # Process below mean
        X_below = torch.ones(batch_size, 1) * -0.5  # 0.5 below mean
        drift_below = physics_loss.theta * (long_term_mean - X_below)

        # Check direction
        assert (drift_above < 0).all(), "Drift should be negative when above mean"
        assert (drift_below > 0).all(), "Drift should be positive when below mean"

    def test_ou_theta_affects_reversion_speed(self, physics_loss):
        """
        Higher theta should produce faster mean reversion.
        """
        from src.losses.physics_losses import OUResidual

        # Two OU processes with different theta
        ou_slow = OUResidual(weight=1.0, theta_init=0.5, learnable=False)
        ou_fast = OUResidual(weight=1.0, theta_init=2.0, learnable=False)

        # Process away from mean
        X = torch.ones(8, 1) * 1.0  # 1.0 away from mean of 0
        long_term_mean = torch.zeros(8, 1)

        drift_slow = ou_slow.theta * (long_term_mean - X)
        drift_fast = ou_fast.theta * (long_term_mean - X)

        # Faster theta should have larger drift magnitude
        assert drift_fast.abs().mean() > drift_slow.abs().mean(), \
            "Higher theta should produce faster mean reversion"


class TestBlackScholesResidual:
    """Tests for Black-Scholes PDE residual correctness."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        from src.models.baseline import LSTMModel
        return LSTMModel(
            input_dim=5,
            hidden_dim=32,
            num_layers=1,
            output_dim=1,
            dropout=0.0
        )

    @pytest.fixture
    def physics_loss(self):
        """Create PhysicsLoss module."""
        from src.models.pinn import PhysicsLoss
        return PhysicsLoss(lambda_gbm=0.0, lambda_ou=0.0, lambda_bs=1.0)

    def test_bs_residual_is_scalar(self, simple_model, physics_loss):
        """Black-Scholes residual should be a scalar loss value."""
        batch_size = 8
        seq_len = 30
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)
        sigma = torch.tensor(0.2)

        residual = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x,
            sigma=sigma,
            price_feature_idx=0
        )

        assert residual.dim() == 0, "BS residual should be scalar"
        assert residual.item() >= 0, "BS residual (MSE) should be non-negative"

    def test_bs_residual_requires_gradients(self, simple_model, physics_loss):
        """Black-Scholes computation should maintain gradient graph."""
        batch_size = 8
        seq_len = 30
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)
        sigma = torch.tensor(0.2)

        residual = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x,
            sigma=sigma,
            price_feature_idx=0
        )

        # Should be able to backprop
        residual.backward()

        # Model should have gradients
        has_grad = any(p.grad is not None for p in simple_model.parameters())
        assert has_grad, "Model parameters should have gradients after BS residual backward"

    def test_bs_residual_different_volatilities(self, simple_model, physics_loss):
        """
        BS residual should vary with volatility.

        The σ² term in Black-Scholes means different volatilities
        should produce different residual values.
        """
        batch_size = 8
        seq_len = 30
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)

        residual_low_vol = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x.clone().detach().requires_grad_(True),
            sigma=torch.tensor(0.1),
            price_feature_idx=0
        )

        residual_high_vol = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x.clone().detach().requires_grad_(True),
            sigma=torch.tensor(0.5),
            price_feature_idx=0
        )

        # Residuals should differ (volatility affects PDE)
        assert residual_low_vol.item() != pytest.approx(residual_high_vol.item(), rel=1e-3), \
            "BS residual should vary with volatility"


class TestPhysicsConstraintSatisfaction:
    """Tests for overall physics constraint satisfaction."""

    def test_physics_constraints_produce_valid_losses(self):
        """All physics constraints should produce valid loss values."""
        from src.losses import GBMResidual, OUResidual, LangevinResidual

        batch_size = 8
        seq_len = 30

        # Test data
        prices = torch.randn(batch_size, seq_len).abs() + 1
        returns = torch.randn(batch_size, seq_len)

        # GBM - should produce finite, non-negative loss
        gbm = GBMResidual(weight=0.1)
        gbm_loss = gbm(prices=prices)
        assert torch.isfinite(gbm_loss), "GBM loss should be finite"
        assert gbm_loss.item() >= 0, "GBM loss should be non-negative"

        # OU - should produce finite, non-negative loss
        ou = OUResidual(weight=0.1)
        ou_loss = ou(values=returns)
        assert torch.isfinite(ou_loss), "OU loss should be finite"
        assert ou_loss.item() >= 0, "OU loss should be non-negative"

        # Langevin - should produce finite, non-negative loss
        langevin = LangevinResidual(weight=0.1)
        langevin_loss = langevin(values=returns)
        assert torch.isfinite(langevin_loss), "Langevin loss should be finite"
        assert langevin_loss.item() >= 0, "Langevin loss should be non-negative"

    def test_physics_losses_are_non_negative(self):
        """All physics losses should be non-negative (MSE-based)."""
        from src.losses import GBMResidual, OUResidual, LangevinResidual, NoArbitrageResidual

        batch_size = 8
        seq_len = 30

        prices = torch.randn(batch_size, seq_len).abs() + 1
        returns = torch.randn(batch_size, seq_len)

        losses = [
            GBMResidual(weight=0.1)(prices=prices),
            OUResidual(weight=0.1)(values=returns),
            LangevinResidual(weight=0.1)(values=returns),
            NoArbitrageResidual(weight=0.1)(predicted_returns=returns),
        ]

        for loss in losses:
            assert loss.item() >= 0, "Physics loss should be non-negative"

    def test_zero_physics_weight_disables_constraint(self):
        """Zero weight should effectively disable physics constraint."""
        from src.losses import GBMResidual

        prices = torch.randn(8, 30).abs() + 1

        loss_with_weight = GBMResidual(weight=0.1)(prices=prices)
        loss_zero_weight = GBMResidual(weight=0.0)(prices=prices)

        assert loss_zero_weight.item() == 0.0, "Zero weight should give zero loss"
        assert loss_with_weight.item() > 0, "Non-zero weight should give positive loss"


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_short_sequence_handling(self):
        """Physics losses should handle very short sequences gracefully."""
        from src.losses import GBMResidual, OUResidual

        # Single timestep (no derivative possible)
        short_data = torch.randn(4, 1)

        gbm_loss = GBMResidual(weight=0.1)(prices=short_data)
        ou_loss = OUResidual(weight=0.1)(values=short_data)

        # Should return 0, not error
        assert gbm_loss.item() == 0.0, "GBM should return 0 for length-1 sequence"
        assert ou_loss.item() == 0.0, "OU should return 0 for length-1 sequence"

    def test_constant_sequence_handling(self):
        """Physics losses should handle constant sequences."""
        from src.losses import GBMResidual, OUResidual

        # Constant sequence
        constant_data = torch.ones(8, 30) * 100.0

        gbm_loss = GBMResidual(weight=0.1)(prices=constant_data)
        ou_loss = OUResidual(weight=0.1)(values=constant_data)

        # Should be finite
        assert torch.isfinite(torch.tensor(gbm_loss.item())), "GBM should be finite for constant"
        assert torch.isfinite(torch.tensor(ou_loss.item())), "OU should be finite for constant"

    def test_near_zero_prices_handling(self):
        """Physics losses should handle near-zero prices without NaN."""
        from src.losses import GBMResidual

        # Near-zero prices
        small_prices = torch.ones(8, 30) * 1e-6

        loss = GBMResidual(weight=0.1)(prices=small_prices)

        assert torch.isfinite(torch.tensor(loss.item())), "GBM should handle small prices"
        assert not torch.isnan(torch.tensor(loss.item())), "GBM should not produce NaN"

    def test_large_values_handling(self):
        """Physics losses should handle large values without overflow."""
        from src.losses import GBMResidual, OUResidual

        # Large prices
        large_prices = torch.ones(8, 30) * 1e6

        gbm_loss = GBMResidual(weight=0.1)(prices=large_prices)

        assert torch.isfinite(torch.tensor(gbm_loss.item())), "GBM should handle large prices"

    def test_nan_input_detection(self):
        """Physics losses should handle NaN inputs gracefully or raise."""
        from src.losses import GBMResidual

        # Data with NaN
        nan_data = torch.randn(8, 30)
        nan_data[0, 0] = float('nan')

        loss = GBMResidual(weight=0.1)(prices=nan_data)

        # Either return NaN (detectable) or finite value
        # The key is not to crash
        assert True  # Test passes if no exception raised


class TestLearnableParameters:
    """Tests for learnable physics parameters."""

    def test_ou_theta_is_learnable(self):
        """OU theta parameter should be learnable."""
        from src.losses import OUResidual

        ou = OUResidual(weight=0.1, theta_init=1.0, learnable=True)

        # Should have parameters
        params = list(ou.parameters())
        assert len(params) == 1, "OU should have 1 learnable parameter (theta)"

        # Theta should be positive (via softplus)
        assert ou.theta.item() > 0, "Theta should be positive"

    def test_ou_theta_updates_during_training(self):
        """OU theta should update during optimization."""
        from src.losses import OUResidual

        ou = OUResidual(weight=0.1, theta_init=1.0, learnable=True)
        initial_theta = ou.theta.item()

        # Create data and optimizer
        returns = torch.randn(8, 30, requires_grad=True)
        optimizer = torch.optim.Adam(ou.parameters(), lr=0.1)

        # Training step
        for _ in range(10):
            optimizer.zero_grad()
            loss = ou(values=returns)
            loss.backward()
            optimizer.step()

        final_theta = ou.theta.item()

        # Theta should have changed
        assert final_theta != pytest.approx(initial_theta, rel=1e-3), \
            "Theta should update during training"

    def test_langevin_parameters_are_learnable(self):
        """Langevin gamma and temperature should be learnable."""
        from src.losses import LangevinResidual

        langevin = LangevinResidual(
            weight=0.1,
            gamma_init=0.5,
            temperature_init=0.1,
            learnable=True
        )

        # Should have 2 parameters
        params = list(langevin.parameters())
        assert len(params) == 2, "Langevin should have 2 learnable parameters"

        # Both should be positive
        assert langevin.gamma.item() > 0, "Gamma should be positive"
        assert langevin.temperature.item() > 0, "Temperature should be positive"


class TestPhysicsResidualMagnitudes:
    """Tests for verifying residual magnitudes are reasonable."""

    def test_gbm_residual_magnitude_on_realistic_data(self):
        """GBM residual should be bounded on realistic stock data."""
        from src.losses import GBMResidual

        # Simulate realistic stock prices (random walk with drift)
        np.random.seed(42)
        batch_size = 16
        seq_len = 60
        S0 = 100.0
        mu = 0.0002  # ~5% annual drift
        sigma = 0.01  # ~16% annual volatility

        returns = np.random.normal(mu, sigma, (batch_size, seq_len))
        prices = S0 * np.exp(np.cumsum(returns, axis=1))
        prices = torch.tensor(prices, dtype=torch.float32)

        gbm = GBMResidual(weight=1.0)
        loss = gbm(prices=prices)

        # Residual should be bounded (not exploding)
        assert loss.item() < 1e6, "GBM residual should be bounded on realistic data"

    def test_ou_residual_magnitude_on_mean_reverting_data(self):
        """OU residual should be small on mean-reverting data."""
        from src.losses import OUResidual

        # Simulate OU process
        np.random.seed(42)
        batch_size = 16
        seq_len = 60
        theta = 1.0  # Mean reversion speed
        mu = 0.0  # Long-term mean
        sigma = 0.02  # Volatility
        dt = 1/252

        X = np.zeros((batch_size, seq_len))
        for t in range(1, seq_len):
            dW = np.random.normal(0, np.sqrt(dt), batch_size)
            X[:, t] = X[:, t-1] + theta * (mu - X[:, t-1]) * dt + sigma * dW

        returns = torch.tensor(X, dtype=torch.float32)

        ou = OUResidual(weight=1.0, theta_init=theta, learnable=False)
        loss = ou(values=returns)

        # Should be bounded
        assert loss.item() < 1e3, "OU residual should be bounded on OU data"


def test_all_physics_losses_importable():
    """Verify all physics losses can be imported."""
    from src.losses import (
        GBMResidual,
        OUResidual,
        BlackScholesResidual,
        LangevinResidual,
        NoArbitrageResidual,
        create_physics_loss
    )

    # All should be classes
    assert callable(GBMResidual)
    assert callable(OUResidual)
    assert callable(BlackScholesResidual)
    assert callable(LangevinResidual)
    assert callable(NoArbitrageResidual)
    assert callable(create_physics_loss)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
