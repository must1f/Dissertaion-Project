"""
Unit tests for model architectures
"""

import pytest
import torch
import numpy as np

from src.models.baseline import LSTMModel, GRUModel, BiLSTMModel
from src.models.transformer import TransformerModel
from src.models.pinn import PINNModel, PhysicsLoss


class TestBaselineModels:
    """Test baseline LSTM/GRU models"""

    def test_lstm_forward(self):
        """Test LSTM forward pass"""
        batch_size = 32
        seq_len = 60
        input_dim = 10

        model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, seq_len, input_dim)
        output, hidden = model(x)

        assert output.shape == (batch_size, 1)
        assert hidden[0].shape == (2, batch_size, 64)  # h_n
        assert hidden[1].shape == (2, batch_size, 64)  # c_n

    def test_gru_forward(self):
        """Test GRU forward pass"""
        batch_size = 32
        seq_len = 60
        input_dim = 10

        model = GRUModel(input_dim=input_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, seq_len, input_dim)
        output, hidden = model(x)

        assert output.shape == (batch_size, 1)
        assert hidden.shape == (2, batch_size, 64)

    def test_bilstm_forward(self):
        """Test BiLSTM forward pass"""
        batch_size = 32
        seq_len = 60
        input_dim = 10

        model = BiLSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, seq_len, input_dim)
        output, hidden = model(x)

        assert output.shape == (batch_size, 1)


class TestTransformer:
    """Test Transformer model"""

    def test_transformer_forward(self):
        """Test Transformer forward pass"""
        batch_size = 32
        seq_len = 60
        input_dim = 10

        model = TransformerModel(
            input_dim=input_dim,
            d_model=64,
            nhead=8,
            num_encoder_layers=3
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        assert output.shape == (batch_size, 1)


class TestPINN:
    """Test Physics-Informed Neural Network"""

    def test_pinn_forward(self):
        """Test PINN forward pass"""
        batch_size = 32
        seq_len = 60
        input_dim = 10

        model = PINNModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_physics_loss(self):
        """Test physics loss computation"""
        physics_loss = PhysicsLoss(
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_langevin=0.1
        )

        batch_size = 32
        seq_len = 60

        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        prices = torch.randn(batch_size, seq_len)
        returns = torch.randn(batch_size, seq_len)
        volatilities = torch.abs(torch.randn(batch_size, seq_len))

        loss, loss_dict = physics_loss(
            predictions, targets, prices, returns, volatilities,
            enable_physics=True
        )

        assert loss.item() >= 0
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_pinn_compute_loss(self):
        """Test PINN loss computation"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm',
            lambda_gbm=0.1,
            lambda_ou=0.1
        )

        batch_size = 32
        seq_len = 60

        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)

        metadata = {
            'prices': torch.randn(batch_size, seq_len),
            'returns': torch.randn(batch_size, seq_len),
            'volatilities': torch.abs(torch.randn(batch_size, seq_len))
        }

        loss, loss_dict = model.compute_loss(
            predictions, targets, metadata, enable_physics=True
        )

        assert loss.item() >= 0
        assert 'data_loss' in loss_dict


class TestModelSizes:
    """Test model parameter counts"""

    def test_lstm_parameters(self):
        """Test LSTM parameter count"""
        model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2)
        n_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert n_params > 0
        assert n_params < 1_000_000  # Less than 1M params

    def test_pinn_parameters(self):
        """Test PINN parameter count"""
        model = PINNModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            base_model='lstm'
        )
        n_params = sum(p.numel() for p in model.parameters())

        # PINN should have same params as base model (physics is in loss, not architecture)
        base_model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2)
        base_params = sum(p.numel() for p in base_model.parameters())

        assert n_params == base_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
