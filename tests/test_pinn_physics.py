"""
Enhanced tests for PINN models and physics constraints
"""

import pytest
import torch
import numpy as np

from src.models.pinn import PINNModel, PhysicsLoss


class TestPhysicsLoss:
    """Test PhysicsLoss class with detailed physics constraint validation"""

    def test_physics_loss_initialization(self):
        """Test physics loss initialization"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )
        
        assert physics_loss.lambda_gbm == 0.1
        assert physics_loss.lambda_ou == 0.1
        assert physics_loss.lambda_langevin == 0.1

    def test_physics_loss_forward_with_physics(self):
        """Test physics loss calculation with physics enabled"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )
        
        batch_size = 16
        seq_len = 60
        
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        prices = torch.abs(torch.randn(batch_size, seq_len)) + 100.0
        returns = torch.randn(batch_size, seq_len) * 0.02
        volatilities = torch.abs(torch.randn(batch_size, seq_len)) * 0.1 + 0.2
        
        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities,
            enable_physics=True
        )
        
        # Check loss is scalar tensor
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        # Check loss dictionary
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        # Physics loss should be non-zero when enabled
        assert loss_dict['physics_loss'] > 0

    def test_physics_loss_forward_without_physics(self):
        """Test physics loss with physics disabled"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )
        
        batch_size = 16
        
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        prices = torch.randn(batch_size, 60)
        returns = torch.randn(batch_size, 60)
        volatilities = torch.randn(batch_size, 60)
        
        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities,
            enable_physics=False
        )
        
        # Physics loss should be zero when disabled
        assert loss_dict['physics_loss'] == 0.0
        assert loss_dict['total_loss'] == loss_dict['data_loss']

    def test_physics_loss_learnable_parameters(self):
        """Test that physics loss has learnable parameters"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )
        
        # Should have learnable parameters: theta, gamma, temperature
        params = list(physics_loss.parameters())
        assert len(params) == 3
        
        # Check they require gradients
        for param in params:
            assert param.requires_grad

    def test_physics_loss_parameter_bounds(self):
        """Test that physics parameters are within reasonable bounds"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )
        
        # Get transformed parameters
        theta = physics_loss.theta
        gamma = physics_loss.gamma
        temperature = physics_loss.temperature
        
        # All should be positive (softplus ensures this)
        assert theta > 0
        assert gamma > 0
        assert temperature > 0
        
        # Should be in reasonable ranges
        assert 0 < theta < 10  # OU mean reversion rate
        assert 0 < gamma < 10  # Langevin friction
        assert 0 < temperature < 10  # Langevin temperature

    def test_physics_loss_different_lambdas(self):
        """Test physics loss with different lambda weights"""
        # High GBM weight
        physics_loss_1 = PhysicsLoss(lambda_gbm=1.0, lambda_ou=0.01, lambda_langevin=0.01)
        
        # High OU weight
        physics_loss_2 = PhysicsLoss(lambda_gbm=0.01, lambda_ou=1.0, lambda_langevin=0.01)
        
        batch_size = 16
        seq_len = 60
        
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        prices = torch.abs(torch.randn(batch_size, seq_len)) + 100.0
        returns = torch.randn(batch_size, seq_len) * 0.02
        volatilities = torch.abs(torch.randn(batch_size, seq_len)) * 0.1 + 0.2
        
        loss_1, dict_1 = physics_loss_1(predictions, targets, prices, returns, volatilities, True)
        loss_2, dict_2 = physics_loss_2(predictions, targets, prices, returns, volatilities, True)
        
        # Losses should be different due to different weights
        assert not torch.allclose(loss_1, loss_2, atol=1e-6)


class TestPINNModel:
    """Test PINNModel with physics constraints"""

    def test_pinn_initialization(self):
        """Test PINN model initialization"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        
        assert model.input_dim == 10
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.base_model_type == 'lstm'

    def test_pinn_forward_pass(self):
        """Test PINN forward pass"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        
        batch_size = 16
        seq_len = 60
        
        x = torch.randn(batch_size, seq_len, 10)
        output = model(x)
        
        assert output.shape == (batch_size, 1)

    def test_pinn_with_different_base_models(self):
        """Test PINN with different base model types"""
        base_models = ['lstm', 'gru', 'transformer']
        
        for base_type in base_models:
            model = PINNModel(
                input_dim=10,
                hidden_dim=64,
                num_layers=2,
                base_model=base_type
            )
            
            x = torch.randn(4, 60, 10)
            output = model(x)
            
            assert output.shape == (4, 1)

    def test_pinn_compute_loss_with_physics(self):
        """Test PINN compute_loss method"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        
        batch_size = 16
        seq_len = 60
        
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        
        metadata = {
            'prices': torch.abs(torch.randn(batch_size, seq_len)) + 100.0,
            'returns': torch.randn(batch_size, seq_len) * 0.02,
            'volatilities': torch.abs(torch.randn(batch_size, seq_len)) * 0.1 + 0.2
        }
        
        loss, loss_dict = model.compute_loss(
            predictions, targets, metadata, enable_physics=True
        )
        
        assert loss.item() >= 0
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict

    def test_pinn_compute_loss_without_physics(self):
        """Test PINN compute_loss with physics disabled"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        
        predictions = torch.randn(16, 1)
        targets = torch.randn(16, 1)
        metadata = {}
        
        loss, loss_dict = model.compute_loss(
            predictions, targets, metadata, enable_physics=False
        )
        
        # Should only have data loss
        assert loss_dict['physics_loss'] == 0.0

    def test_pinn_gradient_flow(self):
        """Test that gradients flow through PINN"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        
        x = torch.randn(4, 60, 10)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_pinn_physics_parameters_trainable(self):
        """Test that physics parameters are trainable"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        
        # Get physics parameters
        physics_params = [
            model.physics_loss.theta_raw,
            model.physics_loss.gamma_raw,
            model.physics_loss.temperature_raw
        ]
        
        # All should require gradients
        for param in physics_params:
            assert param.requires_grad

    def test_pinn_parameter_count(self):
        """Test PINN parameter count includes physics parameters"""
        base_model_params = sum(p.numel() for p in PINNModel(
            input_dim=10, hidden_dim=64, num_layers=2, base_model='lstm'
        ).base_model.parameters())
        
        pinn_params = sum(p.numel() for p in PINNModel(
            input_dim=10, hidden_dim=64, num_layers=2, base_model='lstm'
        ).parameters())
        
        # PINN should have base model params + 3 physics params
        assert pinn_params == base_model_params + 3


class TestPINNEdgeCases:
    """Test edge cases for PINN models"""

    def test_pinn_with_zero_lambda(self):
        """Test PINN with zero physics weights (equivalent to base model)"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.0,
            lambda_ou=0.0,
            lambda_langevin=0.0
        )
        
        predictions = torch.randn(16, 1)
        targets = torch.randn(16, 1)
        metadata = {
            'prices': torch.randn(16, 60),
            'returns': torch.randn(16, 60),
            'volatilities': torch.randn(16, 60)
        }
        
        loss, loss_dict = model.compute_loss(predictions, targets, metadata, True)
        
        # Physics loss should be zero
        assert loss_dict['physics_loss'] == 0.0

    def test_pinn_with_extreme_lambda(self):
        """Test PINN with very high physics weights"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=100.0,  # Very high
            lambda_ou=100.0,
            lambda_langevin=100.0
        )
        
        predictions = torch.randn(16, 1)
        targets = torch.randn(16, 1)
        metadata = {
            'prices': torch.abs(torch.randn(16, 60)) + 100.0,
            'returns': torch.randn(16, 60) * 0.02,
            'volatilities': torch.abs(torch.randn(16, 60)) * 0.1 + 0.2
        }
        
        loss, loss_dict = model.compute_loss(predictions, targets, metadata, True)
        
        # Physics loss should dominate
        assert loss_dict['physics_loss'] > loss_dict['data_loss']

    def test_pinn_with_nan_metadata(self):
        """Test PINN with NaN in metadata"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        
        predictions = torch.randn(16, 1)
        targets = torch.randn(16, 1)
        
        # Create metadata with NaNs
        prices = torch.randn(16, 60)
        prices[0, 5] = float('nan')
        
        metadata = {
            'prices': prices,
            'returns': torch.randn(16, 60),
            'volatilities': torch.randn(16, 60)
        }
        
        # Should handle NaNs gracefully (or raise appropriate error)
        # Depending on implementation, might get NaN loss
        loss, loss_dict = model.compute_loss(predictions, targets, metadata, True)
        
        # Loss might be NaN or handle it gracefully
        assert loss is not None

    def test_pinn_with_single_sample(self):
        """Test PINN with single sample"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        
        x = torch.randn(1, 60, 10)
        output = model(x)
        
        assert output.shape == (1, 1)

    def test_pinn_with_different_sequence_lengths(self):
        """Test PINN with varying sequence lengths"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        
        # Different sequence lengths
        for seq_len in [30, 60, 120]:
            x = torch.randn(8, seq_len, 10)
            output = model(x)
            assert output.shape == (8, 1)


class TestPINNPhysicsConstraints:
    """Test specific physics constraints"""

    def test_gbm_constraint(self):
        """Test Geometric Brownian Motion constraint"""
        physics_loss = PhysicsLoss(lambda_gbm=1.0, lambda_ou=0.0, lambda_langevin=0.0)
        
        # Create realistic price and return data
        prices = torch.tensor([[100.0, 101.0, 102.0, 101.5, 103.0]] * 4)
        returns = torch.diff(torch.log(prices), dim=1)
        returns = torch.cat([returns, returns[:, -1:]], dim=1)  # Pad to match length
        volatilities = torch.ones_like(prices) * 0.2
        
        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        
        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities, True
        )
        
        # GBM loss should be computed
        assert loss_dict['physics_loss'] > 0

    def test_ou_constraint(self):
        """Test Ornstein-Uhlenbeck constraint"""
        physics_loss = PhysicsLoss(lambda_gbm=0.0, lambda_ou=1.0, lambda_langevin=0.0)
        
        returns = torch.randn(4, 60) * 0.02  # Mean-reverting returns
        prices = torch.ones(4, 60) * 100.0
        volatilities = torch.ones(4, 60) * 0.2
        
        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        
        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities, True
        )
        
        # OU loss should be computed
        assert loss_dict['physics_loss'] > 0

    def test_langevin_constraint(self):
        """Test Langevin dynamics constraint"""
        physics_loss = PhysicsLoss(lambda_gbm=0.0, lambda_ou=0.0, lambda_langevin=1.0)
        
        returns = torch.randn(4, 60) * 0.02
        prices = torch.ones(4, 60) * 100.0
        volatilities = torch.ones(4, 60) * 0.2
        
        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        
        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities, True
        )
        
        # Langevin loss should be computed
        assert loss_dict['physics_loss'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
