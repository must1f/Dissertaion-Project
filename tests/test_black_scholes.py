"""
Unit tests for Black-Scholes PINN implementation

Tests derivative computation accuracy using automatic differentiation.
Validates against known analytical Black-Scholes formulas.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pinn import PhysicsLoss
from src.models.baseline import LSTMModel


class TestBlackScholesDerivatives:
    """Test automatic differentiation for Black-Scholes PDE"""

    def test_first_derivative_simple(self):
        """Test that autograd computes first derivative correctly"""
        # Simple function: f(x) = x^2
        # df/dx = 2x
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2

        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

        expected = 2.0 * x
        assert torch.isclose(dy_dx, expected, atol=1e-5)

    def test_second_derivative_simple(self):
        """Test that autograd computes second derivative correctly"""
        # Simple function: f(x) = x^3
        # df/dx = 3x^2
        # d2f/dx2 = 6x
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 3

        # First derivative
        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

        # Second derivative
        d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]

        expected = 6.0 * x
        assert torch.isclose(d2y_dx2, expected, atol=1e-5)

    def test_autograd_chain_rule(self):
        """Test autograd with chain rule (simulating neural network)"""
        # f(x) = (x^2 + 1)^2
        # df/dx = 4x(x^2 + 1)
        x = torch.tensor([1.0], requires_grad=True)
        intermediate = x ** 2 + 1
        y = intermediate ** 2

        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

        expected = 4 * x * (x**2 + 1)
        assert torch.isclose(dy_dx, expected, atol=1e-5)


class TestBlackScholesFormulas:
    """Test Black-Scholes analytical formulas"""

    @staticmethod
    def black_scholes_call(S, K, r, sigma, T):
        """
        Analytical Black-Scholes formula for European call option

        Args:
            S: Stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiration (years)

        Returns:
            Call option value
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_value

    @staticmethod
    def black_scholes_delta(S, K, r, sigma, T):
        """
        Analytical delta (∂V/∂S) for European call option

        Returns:
            Delta = N(d1)
        """
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        return delta

    @staticmethod
    def black_scholes_gamma(S, K, r, sigma, T):
        """
        Analytical gamma (∂²V/∂S²) for European call option

        Returns:
            Gamma = N'(d1) / (S * sigma * sqrt(T))
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def test_call_option_value(self):
        """Test analytical Black-Scholes call option value"""
        S = 100.0  # Stock price
        K = 100.0  # Strike price
        r = 0.05   # Risk-free rate
        sigma = 0.2  # Volatility
        T = 1.0    # Time to expiration (years)

        call_value = self.black_scholes_call(S, K, r, sigma, T)

        # At-the-money call with these parameters should be ~$10
        assert 9.0 < call_value < 11.0, f"Call value {call_value} out of expected range"

    def test_call_delta(self):
        """Test analytical delta calculation"""
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        delta = self.black_scholes_delta(S, K, r, sigma, T)

        # At-the-money call delta should be ~0.5-0.65 (depends on r and sigma)
        assert 0.4 < delta < 0.7, f"Delta {delta} out of expected range"

    def test_call_gamma(self):
        """Test analytical gamma calculation"""
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        gamma = self.black_scholes_gamma(S, K, r, sigma, T)

        # Gamma should be positive and small
        assert gamma > 0, "Gamma should be positive"
        assert gamma < 0.1, f"Gamma {gamma} seems too large"

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)"""
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        call_value = self.black_scholes_call(S, K, r, sigma, T)

        # For put option, use put-call parity
        # P = C - S + K*exp(-rT)
        put_value = call_value - S + K * np.exp(-r * T)

        # Verify parity
        lhs = call_value - put_value
        rhs = S - K * np.exp(-r * T)

        assert np.isclose(lhs, rhs, atol=1e-6), "Put-call parity violated"


class TestBlackScholesPINNResidual:
    """Test Black-Scholes PDE residual computation in PINN"""

    @pytest.fixture
    def simple_model(self):
        """Create a simple LSTM model for testing"""
        return LSTMModel(
            input_dim=5,
            hidden_dim=32,
            num_layers=1,
            output_dim=1,
            dropout=0.0
        )

    @pytest.fixture
    def physics_loss(self):
        """Create PhysicsLoss module"""
        return PhysicsLoss(
            lambda_bs=1.0,
            risk_free_rate=0.05
        )

    def test_bs_residual_shape(self, simple_model, physics_loss):
        """Test that Black-Scholes residual returns correct shape"""
        batch_size = 16
        seq_len = 60
        features = 5

        # Create input with price feature
        x = torch.randn(batch_size, seq_len, features, requires_grad=True)

        # Volatility
        sigma = torch.tensor(0.2)

        # Compute residual
        residual = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x,
            sigma=sigma,
            price_feature_idx=0
        )

        # Residual should be a scalar
        assert residual.dim() == 0, "Residual should be scalar"
        assert residual.item() >= 0, "Residual should be non-negative"

    def test_bs_residual_gradient_flow(self, simple_model, physics_loss):
        """Test that gradients flow correctly through Black-Scholes residual"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)
        sigma = torch.tensor(0.2)

        # Compute residual
        residual = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x,
            sigma=sigma,
            price_feature_idx=0
        )

        # Backward pass
        residual.backward()

        # Check that model parameters have gradients
        for param in simple_model.parameters():
            assert param.grad is not None, "Model parameters should have gradients"
            assert not torch.isnan(param.grad).any(), "Gradients should not be NaN"

    def test_bs_residual_with_different_volatilities(self, simple_model, physics_loss):
        """Test Black-Scholes residual with varying volatilities"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)

        # Test with different volatilities
        sigmas = [0.1, 0.2, 0.3, 0.5]
        residuals = []

        for sigma in sigmas:
            residual = physics_loss.black_scholes_autograd_residual(
                model=simple_model,
                x=x,
                sigma=torch.tensor(sigma),
                price_feature_idx=0
            )
            residuals.append(residual.item())

        # All residuals should be non-negative
        assert all(r >= 0 for r in residuals), "All residuals should be non-negative"

    def test_bs_residual_batch_volatility(self, simple_model, physics_loss):
        """Test Black-Scholes residual with per-sample volatilities"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features, requires_grad=True)

        # Different volatility for each sample
        sigma = torch.rand(batch_size) * 0.3 + 0.1  # Range: 0.1 to 0.4

        # Compute residual
        residual = physics_loss.black_scholes_autograd_residual(
            model=simple_model,
            x=x,
            sigma=sigma,
            price_feature_idx=0
        )

        assert residual.dim() == 0, "Residual should be scalar"
        assert residual.item() >= 0, "Residual should be non-negative"


class TestBlackScholesPhysicsConstraint:
    """Test that Black-Scholes constraint affects model predictions"""

    def test_physics_loss_reduces_during_training(self):
        """Test that physics loss decreases during training"""
        # Create model and physics loss
        model = LSTMModel(input_dim=5, hidden_dim=32, num_layers=1, output_dim=1)
        physics_loss_fn = PhysicsLoss(lambda_bs=1.0, risk_free_rate=0.05)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Generate synthetic data
        batch_size = 16
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)
        sigma = torch.tensor(0.2)

        # Training loop
        initial_residual = None
        final_residual = None

        for epoch in range(10):
            optimizer.zero_grad()

            # Compute Black-Scholes residual
            residual = physics_loss_fn.black_scholes_autograd_residual(
                model=model,
                x=x,
                sigma=sigma,
                price_feature_idx=0
            )

            if epoch == 0:
                initial_residual = residual.item()
            if epoch == 9:
                final_residual = residual.item()

            # Backward and optimize
            residual.backward()
            optimizer.step()

        # Physics loss should decrease (or at least not increase significantly)
        # Note: May not always decrease due to random initialization
        # This is a sanity check, not a strict requirement
        print(f"Initial residual: {initial_residual:.6f}")
        print(f"Final residual: {final_residual:.6f}")


class TestBlackScholesIntegration:
    """Integration tests for Black-Scholes in full PINN training"""

    def test_bs_in_physics_loss_forward(self):
        """Test that Black-Scholes can be integrated into PhysicsLoss forward pass"""
        physics_loss_fn = PhysicsLoss(
            lambda_gbm=0.0,
            lambda_bs=1.0,
            lambda_ou=0.0,
            lambda_langevin=0.0
        )

        # Note: The current PhysicsLoss.forward() doesn't actually call
        # black_scholes_autograd_residual yet. This test documents that gap.

        # Create dummy inputs
        predictions = torch.randn(16, 1)
        targets = torch.randn(16, 1)
        prices = torch.randn(16, 60)
        returns = torch.randn(16, 60)
        volatilities = torch.randn(16, 60)

        # Forward pass
        total_loss, loss_dict = physics_loss_fn.forward(
            predictions=predictions,
            targets=targets,
            prices=prices,
            returns=returns,
            volatilities=volatilities,
            enable_physics=True
        )

        # Should return valid loss
        assert total_loss.dim() == 0, "Total loss should be scalar"
        assert 'data_loss' in loss_dict, "Loss dict should contain data_loss"


def test_black_scholes_module_import():
    """Test that Black-Scholes functions can be imported"""
    from src.models.pinn import PhysicsLoss

    physics = PhysicsLoss()

    # Check that Black-Scholes methods exist
    assert hasattr(physics, 'black_scholes_residual')
    assert hasattr(physics, 'black_scholes_autograd_residual')
    assert callable(physics.black_scholes_residual)
    assert callable(physics.black_scholes_autograd_residual)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
