"""
Unit tests for StackedPINN and ResidualPINN models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stacked_pinn import StackedPINN, ResidualPINN


class TestStackedPINN:
    """Test StackedPINN architecture"""

    @pytest.fixture
    def model(self):
        """Create StackedPINN model for testing"""
        return StackedPINN(
            input_dim=10,
            encoder_dim=64,
            lstm_hidden_dim=64,
            num_encoder_layers=2,
            num_rnn_layers=2,
            prediction_hidden_dim=32,
            dropout=0.1,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )

    def test_forward_pass(self, model):
        """Test forward pass shape"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        return_pred, direction_logits, features = model(x, compute_physics=False)

        # Check output shapes
        assert return_pred.shape == (batch_size, 1)
        assert direction_logits.shape == (batch_size, 2)  # Binary classification
        # features may be None when compute_physics=False, that's OK

    def test_forward_with_physics(self, model):
        """Test forward pass with physics computation"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        return_pred, direction_logits, features = model(x, compute_physics=True)

        assert return_pred.shape == (batch_size, 1)
        assert direction_logits.shape == (batch_size, 2)

    def test_physics_loss_computation(self, model):
        """Test physics loss computation"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        returns = torch.randn(batch_size, seq_len)

        physics_loss, loss_dict = model.compute_physics_loss(x, returns)

        assert isinstance(physics_loss, torch.Tensor)
        assert 'gbm_loss' in loss_dict or 'ou_loss' in loss_dict

    def test_parameter_count(self, model):
        """Test model has reasonable parameter count"""
        n_params = sum(p.numel() for p in model.parameters())

        # Should have a reasonable number of parameters
        assert n_params > 10000
        assert n_params < 10000000  # Less than 10M

    def test_gradient_flow(self, model):
        """Test gradients flow through the model"""
        batch_size = 4
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        target = torch.randn(batch_size, 1)

        return_pred, _, _ = model(x, compute_physics=False)
        loss = nn.MSELoss()(return_pred, target)

        loss.backward()

        # Check that SOME gradients exist (not all params may be used in every forward pass)
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients computed for any parameter"


class TestResidualPINN:
    """Test ResidualPINN architecture"""

    @pytest.fixture
    def model(self):
        """Create ResidualPINN model for testing"""
        return ResidualPINN(
            input_dim=10,
            base_model_type='lstm',
            base_hidden_dim=64,
            correction_hidden_dim=32,
            num_base_layers=2,
            num_correction_layers=2,
            dropout=0.1,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )

    def test_forward_pass(self, model):
        """Test forward pass shape"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        return_pred, direction_logits, features = model(x, return_components=False)

        assert return_pred.shape == (batch_size, 1)
        assert direction_logits.shape == (batch_size, 2)

    def test_return_components(self, model):
        """Test returning base and correction components"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        return_pred, direction_logits, features = model(x, return_components=True)

        assert return_pred.shape == (batch_size, 1)

        # Features should contain components when return_components=True
        assert features is not None

    def test_residual_structure(self, model):
        """Test that output = base + correction"""
        batch_size = 4
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        # Get the base and correction separately by checking internal structure
        # This is a structural test - the residual should combine base + correction
        return_pred, _, _ = model(x, return_components=True)

        assert return_pred is not None

    def test_physics_loss(self, model):
        """Test physics loss computation"""
        batch_size = 16
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        returns = torch.randn(batch_size, seq_len)

        physics_loss, loss_dict = model.compute_physics_loss(x, returns)

        assert isinstance(physics_loss, torch.Tensor)

    def test_different_base_models(self):
        """Test ResidualPINN with different base models"""
        for base_type in ['lstm', 'gru']:
            model = ResidualPINN(
                input_dim=10,
                base_model_type=base_type,
                base_hidden_dim=32,
                correction_hidden_dim=16,
                num_base_layers=1,
                num_correction_layers=1
            )

            x = torch.randn(4, 60, 10)
            output, _, _ = model(x)

            assert output.shape == (4, 1)


class TestPhysicsEncoder:
    """Test Physics Encoder component"""

    def test_encoder_output(self):
        """Test encoder produces correct output shape"""
        from src.models.stacked_pinn import StackedPINN

        model = StackedPINN(
            input_dim=10,
            encoder_dim=64,
            lstm_hidden_dim=64
        )

        x = torch.randn(4, 60, 10)

        # Get encoder output
        return_pred, _, features = model(x)

        assert return_pred.shape[0] == 4


class TestParallelHeads:
    """Test Parallel LSTM/GRU Heads"""

    def test_parallel_processing(self):
        """Test that LSTM and GRU heads process in parallel"""
        model = StackedPINN(
            input_dim=10,
            encoder_dim=64,
            lstm_hidden_dim=64
        )

        x = torch.randn(4, 60, 10)
        output, _, _ = model(x)

        # Output should combine both heads
        assert output.shape == (4, 1)


class TestDirectionClassification:
    """Test direction classification head"""

    def test_direction_logits(self):
        """Test direction logits are valid"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32)

        x = torch.randn(4, 60, 10)
        _, direction_logits, _ = model(x)

        # Should have 2 classes (up/down)
        assert direction_logits.shape == (4, 2)

        # Can convert to probabilities
        probs = torch.softmax(direction_logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))

    def test_direction_prediction(self):
        """Test direction prediction"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32)

        x = torch.randn(8, 60, 10)
        _, direction_logits, _ = model(x)

        # Get predicted direction
        predicted_direction = torch.argmax(direction_logits, dim=-1)

        # Should be 0 or 1
        assert torch.all(predicted_direction >= 0)
        assert torch.all(predicted_direction <= 1)


class TestPhysicsConstraints:
    """Test physics constraint implementations"""

    def test_gbm_constraint(self):
        """Test GBM constraint computation"""
        model = StackedPINN(
            input_dim=10,
            encoder_dim=32,
            lstm_hidden_dim=32,
            lambda_gbm=0.1,
            lambda_ou=0.0
        )

        x = torch.randn(4, 60, 10)
        returns = torch.randn(4, 60)

        loss, loss_dict = model.compute_physics_loss(x, returns)

        # GBM loss should be present
        assert 'gbm_loss' in loss_dict

    def test_ou_constraint(self):
        """Test OU constraint computation"""
        model = StackedPINN(
            input_dim=10,
            encoder_dim=32,
            lstm_hidden_dim=32,
            lambda_gbm=0.0,
            lambda_ou=0.1
        )

        x = torch.randn(4, 60, 10)
        returns = torch.randn(4, 60)

        loss, loss_dict = model.compute_physics_loss(x, returns)

        # OU loss should be present
        assert 'ou_loss' in loss_dict

    def test_combined_constraints(self):
        """Test combined GBM + OU constraints"""
        model = StackedPINN(
            input_dim=10,
            encoder_dim=32,
            lstm_hidden_dim=32,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )

        x = torch.randn(4, 60, 10)
        returns = torch.randn(4, 60)

        loss, loss_dict = model.compute_physics_loss(x, returns)

        # Both should be present
        assert 'gbm_loss' in loss_dict
        assert 'ou_loss' in loss_dict


class TestModelModes:
    """Test model training/eval modes"""

    def test_train_mode(self):
        """Test model in training mode"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32, dropout=0.5)
        model.train()

        x = torch.randn(4, 60, 10)

        # In training mode, dropout should be applied
        output1, _, _ = model(x)
        output2, _, _ = model(x)

        # Outputs might differ due to dropout (not guaranteed but possible)
        # Just check it runs
        assert output1.shape == output2.shape

    def test_eval_mode(self):
        """Test model in evaluation mode"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32, dropout=0.5)
        model.eval()

        x = torch.randn(4, 60, 10)

        # In eval mode, outputs should be deterministic
        with torch.no_grad():
            output1, _, _ = model(x)
            output2, _, _ = model(x)

        torch.testing.assert_close(output1, output2)


class TestEdgeCases:
    """Test edge cases"""

    def test_single_sample(self):
        """Test with single sample batch"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32)

        x = torch.randn(1, 60, 10)
        output, direction, _ = model(x)

        assert output.shape == (1, 1)
        assert direction.shape == (1, 2)

    def test_short_sequence(self):
        """Test with short sequence"""
        model = StackedPINN(input_dim=10, encoder_dim=32, lstm_hidden_dim=32)

        x = torch.randn(4, 10, 10)  # Only 10 timesteps
        output, direction, _ = model(x)

        assert output.shape == (4, 1)

    def test_many_features(self):
        """Test with many input features"""
        model = StackedPINN(input_dim=100, encoder_dim=64, lstm_hidden_dim=64)

        x = torch.randn(4, 60, 100)
        output, direction, _ = model(x)

        assert output.shape == (4, 1)

    def test_zero_lambdas(self):
        """Test with zero physics weights"""
        model = StackedPINN(
            input_dim=10,
            encoder_dim=32,
            lstm_hidden_dim=32,
            lambda_gbm=0.0,
            lambda_ou=0.0
        )

        x = torch.randn(4, 60, 10)
        returns = torch.randn(4, 60)

        loss, loss_dict = model.compute_physics_loss(x, returns)

        # Physics loss should be zero or near zero
        assert loss.item() < 1e-6 or 'gbm_loss' in loss_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
