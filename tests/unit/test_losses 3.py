"""
Unit Tests for Loss Functions

Tests for data losses, physics losses, and composite loss functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class TestDataLosses:
    """Tests for data loss functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets."""
        torch.manual_seed(42)
        predictions = torch.randn(32, 1)
        targets = torch.randn(32, 1)
        return predictions, targets

    def test_mse_loss_basic(self, sample_data):
        """Test MSE loss computation."""
        from src.losses.data_losses import MSELoss

        predictions, targets = sample_data
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Non-negative
        assert torch.isfinite(loss)  # No NaN/Inf

    def test_mse_loss_zero_error(self):
        """Test MSE loss is zero when predictions match targets."""
        from src.losses.data_losses import MSELoss

        targets = torch.tensor([1.0, 2.0, 3.0])
        predictions = targets.clone()
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_mae_loss_basic(self, sample_data):
        """Test MAE loss computation."""
        from src.losses.data_losses import MAELoss

        predictions, targets = sample_data
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_mae_loss_known_value(self):
        """Test MAE with known expected value."""
        from src.losses.data_losses import MAELoss

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 3.0, 4.0])
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets)

        # All errors are 1.0, so MAE = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-7)

    def test_huber_loss_basic(self, sample_data):
        """Test Huber loss computation."""
        from src.losses.data_losses import HuberLoss

        predictions, targets = sample_data
        loss_fn = HuberLoss(delta=1.0)

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_huber_loss_quadratic_region(self):
        """Test Huber loss in quadratic region (small errors)."""
        from src.losses.data_losses import HuberLoss

        # Small errors (|diff| < delta)
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 2.1, 3.1])
        loss_fn = HuberLoss(delta=1.0)

        loss = loss_fn(predictions, targets)

        # Should behave like 0.5 * MSE
        expected = 0.5 * torch.mean((predictions - targets) ** 2)
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_log_cosh_loss_basic(self, sample_data):
        """Test Log-Cosh loss computation."""
        from src.losses.data_losses import LogCoshLoss

        predictions, targets = sample_data
        loss_fn = LogCoshLoss()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_quantile_loss_basic(self, sample_data):
        """Test Quantile loss computation."""
        from src.losses.data_losses import QuantileLoss

        predictions, targets = sample_data

        # Test different quantiles
        for q in [0.1, 0.5, 0.9]:
            loss_fn = QuantileLoss(quantile=q)
            loss = loss_fn(predictions, targets)

            assert loss.ndim == 0
            assert loss >= 0
            assert torch.isfinite(loss)

    def test_quantile_loss_invalid_quantile(self):
        """Test Quantile loss raises error for invalid quantile."""
        from src.losses.data_losses import QuantileLoss

        with pytest.raises(ValueError):
            QuantileLoss(quantile=0.0)

        with pytest.raises(ValueError):
            QuantileLoss(quantile=1.0)

    def test_directional_loss_basic(self, sample_data):
        """Test Directional loss computation."""
        from src.losses.data_losses import DirectionalLoss

        predictions, targets = sample_data
        loss_fn = DirectionalLoss()

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_directional_loss_same_sign(self):
        """Test Directional loss when predictions have correct direction."""
        from src.losses.data_losses import DirectionalLoss

        # Same signs
        predictions = torch.tensor([0.1, 0.2, -0.1, -0.2])
        targets = torch.tensor([0.5, 0.3, -0.5, -0.3])
        loss_fn = DirectionalLoss(direction_weight=1.0, magnitude_weight=0.5)

        loss_correct = loss_fn(predictions, targets)

        # Opposite signs
        predictions_wrong = torch.tensor([-0.1, -0.2, 0.1, 0.2])
        loss_wrong = loss_fn(predictions_wrong, targets)

        # Wrong direction should have higher loss
        assert loss_wrong > loss_correct

    def test_weighted_mse_basic(self, sample_data):
        """Test Weighted MSE loss."""
        from src.losses.data_losses import WeightedMSELoss

        predictions, targets = sample_data
        weights = torch.ones_like(predictions)
        loss_fn = WeightedMSELoss()

        loss = loss_fn(predictions, targets, weights)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_weighted_mse_different_weights(self):
        """Test that weighting affects loss value."""
        from src.losses.data_losses import WeightedMSELoss

        predictions = torch.tensor([1.0, 2.0])
        targets = torch.tensor([0.0, 0.0])

        # Higher weight on first sample
        weights1 = torch.tensor([2.0, 1.0])
        # Higher weight on second sample
        weights2 = torch.tensor([1.0, 2.0])

        loss_fn = WeightedMSELoss()

        loss1 = loss_fn(predictions, targets, weights1)
        loss2 = loss_fn(predictions, targets, weights2)

        # Different weights should give different losses
        assert loss1.item() != pytest.approx(loss2.item(), rel=1e-3)

    def test_create_data_loss_factory(self):
        """Test data loss factory function."""
        from src.losses.data_losses import create_data_loss

        loss_types = ['mse', 'mae', 'huber', 'logcosh', 'directional']

        for loss_type in loss_types:
            loss_fn = create_data_loss(loss_type)
            assert loss_fn is not None

    def test_create_data_loss_invalid_type(self):
        """Test factory raises error for invalid loss type."""
        from src.losses.data_losses import create_data_loss

        with pytest.raises(ValueError):
            create_data_loss('invalid_loss')

    def test_reduction_modes(self, sample_data):
        """Test different reduction modes."""
        from src.losses.data_losses import MSELoss

        predictions, targets = sample_data

        loss_mean = MSELoss(reduction='mean')(predictions, targets)
        loss_sum = MSELoss(reduction='sum')(predictions, targets)
        loss_none = MSELoss(reduction='none')(predictions, targets)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == predictions.shape


class TestPhysicsLosses:
    """Tests for physics-informed loss functions."""

    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        torch.manual_seed(42)
        # Simulated prices with some trend
        batch_size = 16
        seq_len = 30
        prices = torch.cumsum(torch.randn(batch_size, seq_len) * 0.02 + 0.001, dim=1)
        prices = torch.abs(prices) + 100  # Ensure positive prices
        return prices

    @pytest.fixture
    def return_data(self, price_data):
        """Create return data from prices."""
        returns = (price_data[:, 1:] - price_data[:, :-1]) / price_data[:, :-1]
        return returns

    def test_gbm_residual_basic(self, price_data):
        """Test GBM residual computation."""
        from src.losses.physics_losses import GBMResidual

        loss_fn = GBMResidual(weight=0.1)
        loss = loss_fn(prices=price_data)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_gbm_residual_with_drift(self, price_data):
        """Test GBM residual with explicit drift."""
        from src.losses.physics_losses import GBMResidual

        drift = torch.tensor([[0.05]] * price_data.shape[0])
        loss_fn = GBMResidual(weight=0.1)

        loss = loss_fn(prices=price_data, drift=drift)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_ou_residual_basic(self, return_data):
        """Test OU residual computation."""
        from src.losses.physics_losses import OUResidual

        loss_fn = OUResidual(weight=0.1, theta_init=1.0, learnable=True)
        loss = loss_fn(values=return_data)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_ou_residual_learnable_theta(self, return_data):
        """Test that OU theta is learnable."""
        from src.losses.physics_losses import OUResidual

        loss_fn = OUResidual(weight=0.1, theta_init=1.0, learnable=True)

        # Check that theta is a parameter
        params = list(loss_fn.parameters())
        assert len(params) == 1

        # Check theta is positive (via softplus)
        assert loss_fn.theta.item() > 0

    def test_ou_residual_non_learnable(self, return_data):
        """Test OU with fixed theta."""
        from src.losses.physics_losses import OUResidual

        loss_fn = OUResidual(weight=0.1, theta_init=2.0, learnable=False)

        # Check that theta is a buffer, not parameter
        params = list(loss_fn.parameters())
        assert len(params) == 0

        # Theta should still be accessible
        assert loss_fn.theta.item() > 0

    def test_langevin_residual_basic(self, return_data):
        """Test Langevin residual computation."""
        from src.losses.physics_losses import LangevinResidual

        loss_fn = LangevinResidual(weight=0.1)
        loss = loss_fn(values=return_data)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_langevin_learnable_params(self):
        """Test Langevin learnable parameters."""
        from src.losses.physics_losses import LangevinResidual

        loss_fn = LangevinResidual(
            weight=0.1,
            gamma_init=0.5,
            temperature_init=0.1,
            learnable=True
        )

        # Check parameters
        params = list(loss_fn.parameters())
        assert len(params) == 2  # gamma and temperature

        # Check values are positive
        assert loss_fn.gamma.item() > 0
        assert loss_fn.temperature.item() > 0

    def test_no_arbitrage_residual(self):
        """Test no-arbitrage residual."""
        from src.losses.physics_losses import NoArbitrageResidual

        predicted_returns = torch.randn(16, 30) * 0.02
        loss_fn = NoArbitrageResidual(weight=0.1, risk_free_rate=0.02)

        loss = loss_fn(predicted_returns=predicted_returns)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_momentum_residual(self, return_data):
        """Test momentum residual."""
        from src.losses.physics_losses import MomentumResidual

        loss_fn = MomentumResidual(weight=0.1, momentum_coef_init=0.1)
        loss = loss_fn(returns=return_data)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_momentum_alpha_bounded(self):
        """Test momentum alpha is bounded to [-1, 1]."""
        from src.losses.physics_losses import MomentumResidual

        # Test extreme initial values
        for init_val in [-10.0, 0.0, 10.0]:
            loss_fn = MomentumResidual(momentum_coef_init=init_val)
            assert -1 <= loss_fn.alpha.item() <= 1

    def test_create_physics_loss_factory(self):
        """Test physics loss factory function."""
        from src.losses.physics_losses import create_physics_loss

        loss_types = ['gbm', 'ou', 'langevin', 'no_arbitrage', 'momentum']

        for loss_type in loss_types:
            loss_fn = create_physics_loss(loss_type, weight=0.1)
            assert loss_fn is not None

    def test_create_physics_loss_invalid_type(self):
        """Test factory raises error for invalid type."""
        from src.losses.physics_losses import create_physics_loss

        with pytest.raises(ValueError):
            create_physics_loss('invalid_physics_loss')

    def test_short_sequence_handling(self):
        """Test physics losses handle short sequences gracefully."""
        from src.losses.physics_losses import GBMResidual, OUResidual

        short_data = torch.randn(4, 1)  # Only 1 timestep

        gbm_loss = GBMResidual(weight=0.1)(prices=short_data)
        ou_loss = OUResidual(weight=0.1)(values=short_data)

        # Should return 0 for too-short sequences
        assert gbm_loss.item() == 0.0
        assert ou_loss.item() == 0.0


class TestCompositeLoss:
    """Tests for composite loss functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        torch.manual_seed(42)
        batch_size = 16
        seq_len = 30
        features = 5

        inputs = torch.randn(batch_size, seq_len, features)
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        prices = torch.abs(inputs[:, :, 0]) + 100
        returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

        return {
            'inputs': inputs,
            'predictions': predictions,
            'targets': targets,
            'prices': prices,
            'returns': returns
        }

    def test_composite_loss_basic(self, sample_batch):
        """Test basic composite loss."""
        from src.losses import CompositeLoss, MSELoss, GBMResidual

        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={'gbm': GBMResidual(weight=0.1)}
        )

        total_loss, loss_dict = loss_fn(
            predictions=sample_batch['predictions'],
            targets=sample_batch['targets'],
            physics_inputs={'gbm': {'prices': sample_batch['prices']}}
        )

        assert total_loss.ndim == 0
        assert torch.isfinite(total_loss)
        assert 'data_loss' in loss_dict
        assert 'gbm_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_composite_loss_no_physics(self, sample_batch):
        """Test composite loss with physics disabled."""
        from src.losses import CompositeLoss, MSELoss, GBMResidual

        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={'gbm': GBMResidual(weight=0.1)},
            enable_physics=False
        )

        total_loss, loss_dict = loss_fn(
            predictions=sample_batch['predictions'],
            targets=sample_batch['targets'],
            physics_inputs={'gbm': {'prices': sample_batch['prices']}}
        )

        # Should only have data loss
        assert 'gbm_loss' not in loss_dict

    def test_composite_loss_multiple_physics(self, sample_batch):
        """Test composite loss with multiple physics terms."""
        from src.losses import CompositeLoss, MSELoss, GBMResidual, OUResidual

        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={
                'gbm': GBMResidual(weight=0.1),
                'ou': OUResidual(weight=0.05)
            }
        )

        total_loss, loss_dict = loss_fn(
            predictions=sample_batch['predictions'],
            targets=sample_batch['targets'],
            physics_inputs={
                'gbm': {'prices': sample_batch['prices']},
                'ou': {'values': sample_batch['returns']}
            }
        )

        assert 'gbm_loss' in loss_dict
        assert 'ou_loss' in loss_dict

    def test_composite_loss_get_physics_params(self, sample_batch):
        """Test getting learned physics parameters."""
        from src.losses import CompositeLoss, MSELoss, OUResidual, LangevinResidual

        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={
                'ou': OUResidual(weight=0.1, learnable=True),
                'langevin': LangevinResidual(weight=0.05, learnable=True)
            }
        )

        params = loss_fn.get_physics_parameters()

        assert 'ou' in params
        assert 'theta' in params['ou']
        assert 'langevin' in params
        assert 'gamma' in params['langevin']
        assert 'temperature' in params['langevin']

    def test_adaptive_composite_loss(self, sample_batch):
        """Test adaptive composite loss."""
        from src.losses import (
            AdaptiveCompositeLoss,
            MSELoss,
            GBMResidual,
            LossConfig,
            WeightingStrategy
        )
        from src.losses.composite import WeightingStrategy

        config = LossConfig(
            weighting_strategy=WeightingStrategy.STATIC
        )

        loss_fn = AdaptiveCompositeLoss(
            data_loss=MSELoss(),
            physics_losses={'gbm': GBMResidual(weight=0.1)},
            config=config
        )

        total_loss, loss_dict = loss_fn(
            predictions=sample_batch['predictions'],
            targets=sample_batch['targets'],
            physics_inputs={'gbm': {'prices': sample_batch['prices']}}
        )

        assert total_loss.ndim == 0
        assert torch.isfinite(total_loss)

    def test_curriculum_weighting(self, sample_batch):
        """Test curriculum learning weight schedule."""
        from src.losses import AdaptiveCompositeLoss, MSELoss, GBMResidual, LossConfig
        from src.losses.composite import WeightingStrategy

        config = LossConfig(
            weighting_strategy=WeightingStrategy.CURRICULUM,
            curriculum_warmup_epochs=5,
            curriculum_ramp_epochs=10,
            curriculum_final_physics_scale=1.0
        )

        loss_fn = AdaptiveCompositeLoss(
            data_loss=MSELoss(),
            physics_losses={'gbm': GBMResidual(weight=0.1)},
            config=config
        )

        # During warmup (epoch 0-4), physics scale should be 0
        loss_fn.set_epoch(2)
        assert loss_fn.get_curriculum_scale() == 0.0

        # During ramp (epoch 5-14), physics scale increases
        loss_fn.set_epoch(10)  # Halfway through ramp
        scale = loss_fn.get_curriculum_scale()
        assert 0 < scale < 1

        # After ramp, full physics
        loss_fn.set_epoch(20)
        assert loss_fn.get_curriculum_scale() == 1.0

    def test_create_composite_loss_factory(self):
        """Test composite loss factory."""
        from src.losses import create_composite_loss

        loss_fn = create_composite_loss(
            data_loss_type='mse',
            physics_loss_types=['gbm', 'ou']
        )

        assert loss_fn is not None


class TestLossGradients:
    """Tests for gradient computation through losses."""

    def test_data_loss_gradients(self):
        """Test gradients flow through data losses."""
        from src.losses import MSELoss

        predictions = torch.randn(16, 1, requires_grad=True)
        targets = torch.randn(16, 1)

        loss_fn = MSELoss()
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()

    def test_physics_loss_gradients(self):
        """Test gradients flow through physics losses."""
        from src.losses import OUResidual

        values = torch.randn(16, 30, requires_grad=True)

        loss_fn = OUResidual(weight=0.1, learnable=True)
        loss = loss_fn(values=values)
        loss.backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()

        # Also check theta gradient
        assert loss_fn.theta_raw.grad is not None

    def test_composite_loss_gradients(self):
        """Test gradients flow through composite loss."""
        from src.losses import CompositeLoss, MSELoss, OUResidual

        predictions = torch.randn(16, 1, requires_grad=True)
        targets = torch.randn(16, 1)
        returns = torch.randn(16, 30, requires_grad=True)

        loss_fn = CompositeLoss(
            data_loss=MSELoss(),
            physics_losses={'ou': OUResidual(weight=0.1)}
        )

        total_loss, _ = loss_fn(
            predictions=predictions,
            targets=targets,
            physics_inputs={'ou': {'values': returns}}
        )
        total_loss.backward()

        assert predictions.grad is not None
        assert returns.grad is not None


class TestNumericalStability:
    """Tests for numerical stability of loss functions."""

    def test_mse_large_values(self):
        """Test MSE with large values."""
        from src.losses import MSELoss

        predictions = torch.tensor([1e10, 1e10])
        targets = torch.tensor([1e10 + 1, 1e10 + 1])

        loss = MSELoss()(predictions, targets)
        assert torch.isfinite(loss)

    def test_mse_small_values(self):
        """Test MSE with small values."""
        from src.losses import MSELoss

        predictions = torch.tensor([1e-10, 1e-10])
        targets = torch.tensor([1e-10 + 1e-11, 1e-10 + 1e-11])

        loss = MSELoss()(predictions, targets)
        assert torch.isfinite(loss)

    def test_physics_near_zero_prices(self):
        """Test physics losses handle near-zero prices."""
        from src.losses import GBMResidual

        # Near-zero prices (but positive)
        prices = torch.full((16, 30), 1e-8)

        loss = GBMResidual(weight=0.1)(prices=prices)
        assert torch.isfinite(loss)

    def test_log_cosh_moderate_values(self):
        """Test LogCosh handles moderate differences."""
        from src.losses import LogCoshLoss

        # Moderate difference (within numerical stability range)
        predictions = torch.tensor([10.0])
        targets = torch.tensor([0.0])

        loss = LogCoshLoss()(predictions, targets)
        assert torch.isfinite(loss)
        assert loss > 0
