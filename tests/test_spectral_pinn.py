"""Unit tests for SpectralRegimePINN model."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.spectral_pinn import (
    SpectralEncoder,
    RegimeEncoder,
    SpectralRegimePINN,
)
from src.losses.spectral_loss import (
    AutocorrelationLoss,
    SpectralConsistencyLoss,
    SpectralEntropyLoss,
    CombinedSpectralLoss,
)


class TestSpectralEncoder:
    """Test suite for SpectralEncoder module."""

    @pytest.fixture
    def encoder(self) -> SpectralEncoder:
        """Create a SpectralEncoder with default parameters."""
        return SpectralEncoder(input_dim=5, n_fft=32, embed_dim=64)

    def test_output_shape(self, encoder: SpectralEncoder):
        """Test that output has correct shape."""
        batch_size = 4
        seq_len = 30
        input_dim = 5

        x = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(x)

        assert output.shape == (batch_size, seq_len, encoder.embed_dim)

    def test_forward_pass(self, encoder: SpectralEncoder):
        """Test that forward pass runs without error."""
        x = torch.randn(2, 20, 5)
        output = encoder(x)

        assert output is not None
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, encoder: SpectralEncoder):
        """Test that gradients flow through encoder."""
        x = torch.randn(2, 20, 5, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sequence_lengths(self, encoder: SpectralEncoder):
        """Test with different sequence lengths."""
        for seq_len in [10, 30, 50, 100]:
            x = torch.randn(2, seq_len, 5)
            output = encoder(x)
            assert output.shape == (2, seq_len, encoder.embed_dim)

    def test_batch_independence(self, encoder: SpectralEncoder):
        """Test that batch samples are processed independently."""
        x1 = torch.randn(1, 20, 5)
        x2 = torch.randn(1, 20, 5)
        x_combined = torch.cat([x1, x2], dim=0)

        out1 = encoder(x1)
        out2 = encoder(x2)
        out_combined = encoder(x_combined)

        assert torch.allclose(out1, out_combined[:1], atol=1e-5)
        assert torch.allclose(out2, out_combined[1:], atol=1e-5)


class TestRegimeEncoder:
    """Test suite for RegimeEncoder module."""

    @pytest.fixture
    def encoder(self) -> RegimeEncoder:
        """Create a RegimeEncoder with default parameters."""
        return RegimeEncoder(n_regimes=3, embed_dim=32)

    def test_output_shape(self, encoder: RegimeEncoder):
        """Test that output has correct shape."""
        batch_size = 4
        n_regimes = 3

        probs = torch.softmax(torch.randn(batch_size, n_regimes), dim=-1)
        output = encoder(probs)

        assert output.shape == (batch_size, encoder.embed_dim)

    def test_valid_probability_input(self, encoder: RegimeEncoder):
        """Test with valid probability distributions."""
        probs = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ])
        output = encoder(probs)

        assert not torch.isnan(output).any()

    def test_one_hot_input(self, encoder: RegimeEncoder):
        """Test with one-hot encoded regimes."""
        probs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        output = encoder(probs)

        assert not torch.isnan(output).any()
        # Different regimes should produce different embeddings
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])


class TestSpectralRegimePINN:
    """Test suite for SpectralRegimePINN model."""

    @pytest.fixture
    def model(self) -> SpectralRegimePINN:
        """Create a SpectralRegimePINN with default parameters."""
        return SpectralRegimePINN(
            input_dim=5,
            hidden_dim=64,
            n_regimes=3,
            n_fft=32,
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_autocorr=0.05,
            lambda_spectral=0.05,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 4
        seq_len = 30
        input_dim = 5
        n_regimes = 3

        x = torch.randn(batch_size, seq_len, input_dim)
        regime_probs = torch.softmax(torch.randn(batch_size, n_regimes), dim=-1)
        targets = torch.randn(batch_size, 1)

        return x, regime_probs, targets

    def test_output_shape(self, model: SpectralRegimePINN, sample_batch):
        """Test that outputs have correct shapes."""
        x, regime_probs, _ = sample_batch
        predictions, regime_preds = model(x, regime_probs)

        assert predictions.shape == (x.shape[0], 1)
        assert regime_preds.shape == (x.shape[0], model.n_regimes)

    def test_forward_without_regimes(self, model: SpectralRegimePINN, sample_batch):
        """Test forward pass without regime probabilities."""
        x, _, _ = sample_batch
        predictions, regime_preds = model(x)

        assert predictions.shape == (x.shape[0], 1)
        assert regime_preds.shape == (x.shape[0], model.n_regimes)

    def test_forward_with_regimes(self, model: SpectralRegimePINN, sample_batch):
        """Test forward pass with regime probabilities."""
        x, regime_probs, _ = sample_batch
        predictions, regime_preds = model(x, regime_probs)

        assert predictions.shape == (x.shape[0], 1)
        assert not torch.isnan(predictions).any()

    def test_regime_predictions_sum_to_one(self, model: SpectralRegimePINN, sample_batch):
        """Test that regime predictions are valid probabilities."""
        x, regime_probs, _ = sample_batch
        _, regime_preds = model(x, regime_probs)

        # After softmax, should sum to 1
        sums = regime_preds.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_compute_loss(self, model: SpectralRegimePINN, sample_batch):
        """Test loss computation."""
        x, regime_probs, targets = sample_batch
        predictions, _ = model(x, regime_probs)

        metadata = {
            "prices": torch.rand(x.shape[0], x.shape[1]) * 100 + 50,
            "returns": torch.randn(x.shape[0], x.shape[1]) * 0.02,
            "dt": 1.0 / 252,
        }

        loss_dict = model.compute_loss(predictions, targets, metadata)

        assert "total_loss" in loss_dict
        assert "data_loss" in loss_dict
        assert loss_dict["total_loss"].item() >= 0

    def test_compute_loss_physics_disabled(self, model: SpectralRegimePINN, sample_batch):
        """Test loss computation with physics disabled."""
        x, regime_probs, targets = sample_batch
        predictions, _ = model(x, regime_probs)

        metadata = {
            "prices": torch.rand(x.shape[0], x.shape[1]) * 100 + 50,
            "returns": torch.randn(x.shape[0], x.shape[1]) * 0.02,
            "dt": 1.0 / 252,
        }

        loss_dict = model.compute_loss(predictions, targets, metadata, enable_physics=False)

        # Without physics, only data loss
        assert "data_loss" in loss_dict
        assert loss_dict["total_loss"] == loss_dict["data_loss"]

    def test_gradient_flow(self, model: SpectralRegimePINN, sample_batch):
        """Test that gradients flow through the model."""
        x, regime_probs, targets = sample_batch
        predictions, _ = model(x, regime_probs)

        loss = nn.MSELoss()(predictions, targets)
        loss.backward()

        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self, model: SpectralRegimePINN, sample_batch):
        """Test model in eval mode."""
        model.eval()
        x, regime_probs, _ = sample_batch

        with torch.no_grad():
            predictions, regime_preds = model(x, regime_probs)

        assert predictions.shape == (x.shape[0], 1)
        assert not torch.isnan(predictions).any()

    def test_different_batch_sizes(self, model: SpectralRegimePINN):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 30, 5)
            regime_probs = torch.softmax(torch.randn(batch_size, 3), dim=-1)

            predictions, _ = model(x, regime_probs)
            assert predictions.shape == (batch_size, 1)

    def test_model_state_dict(self, model: SpectralRegimePINN):
        """Test that model state dict can be saved and loaded."""
        state_dict = model.state_dict()

        new_model = SpectralRegimePINN(input_dim=5, hidden_dim=64, n_regimes=3)
        new_model.load_state_dict(state_dict)

        # Check that loaded model produces same output
        x = torch.randn(2, 30, 5)
        regime_probs = torch.softmax(torch.randn(2, 3), dim=-1)

        model.eval()
        new_model.eval()

        with torch.no_grad():
            out1, _ = model(x, regime_probs)
            out2, _ = new_model(x, regime_probs)

        assert torch.allclose(out1, out2)


class TestAutocorrelationLoss:
    """Test suite for AutocorrelationLoss."""

    @pytest.fixture
    def loss_fn(self) -> AutocorrelationLoss:
        """Create AutocorrelationLoss with default parameters."""
        return AutocorrelationLoss(weight=0.05)

    def test_loss_value(self, loss_fn: AutocorrelationLoss):
        """Test that loss returns a scalar."""
        returns = torch.randn(4, 30) * 0.02
        loss = loss_fn(returns)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_zero_autocorrelation(self, loss_fn: AutocorrelationLoss):
        """Test that perfect white noise has low loss."""
        torch.manual_seed(42)
        # Large sample for good statistics
        returns = torch.randn(100, 1000) * 0.02
        loss = loss_fn(returns)

        # White noise should have low autocorrelation loss
        assert loss.item() < 0.1

    def test_high_autocorrelation(self, loss_fn: AutocorrelationLoss):
        """Test that persistent series has higher loss."""
        # Create series with high autocorrelation
        returns = torch.zeros(4, 100)
        for i in range(1, 100):
            returns[:, i] = 0.8 * returns[:, i - 1] + 0.2 * torch.randn(4)

        loss = loss_fn(returns)
        assert loss.item() > 0.01  # Should have non-trivial loss

    def test_gradient_flow(self, loss_fn: AutocorrelationLoss):
        """Test that gradients flow through loss."""
        returns = torch.randn(4, 30, requires_grad=True)
        loss = loss_fn(returns)
        loss.backward()

        assert returns.grad is not None


class TestSpectralConsistencyLoss:
    """Test suite for SpectralConsistencyLoss."""

    @pytest.fixture
    def loss_fn(self) -> SpectralConsistencyLoss:
        """Create SpectralConsistencyLoss with default parameters."""
        return SpectralConsistencyLoss(weight=0.05, n_fft=32)

    def test_loss_shape(self, loss_fn: SpectralConsistencyLoss):
        """Test that loss returns a scalar."""
        predictions = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_signals(self, loss_fn: SpectralConsistencyLoss):
        """Test that identical signals have zero loss."""
        signal = torch.randn(4, 64)
        loss = loss_fn(signal, signal)

        assert loss.item() < 1e-5

    def test_different_signals(self, loss_fn: SpectralConsistencyLoss):
        """Test that different signals have non-zero loss."""
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        loss = loss_fn(pred, target)

        assert loss.item() > 0

    def test_gradient_flow(self, loss_fn: SpectralConsistencyLoss):
        """Test that gradients flow through loss."""
        predictions = torch.randn(4, 64, requires_grad=True)
        targets = torch.randn(4, 64)
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None


class TestSpectralEntropyLoss:
    """Test suite for SpectralEntropyLoss."""

    @pytest.fixture
    def loss_fn(self) -> SpectralEntropyLoss:
        """Create SpectralEntropyLoss with default parameters."""
        return SpectralEntropyLoss(weight=0.01, target_entropy=0.7)

    def test_loss_shape(self, loss_fn: SpectralEntropyLoss):
        """Test that loss returns a scalar."""
        returns = torch.randn(4, 64)
        loss = loss_fn(returns)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_gradient_flow(self, loss_fn: SpectralEntropyLoss):
        """Test that gradients flow through loss."""
        returns = torch.randn(4, 64, requires_grad=True)
        loss = loss_fn(returns)
        loss.backward()

        assert returns.grad is not None


class TestCombinedSpectralLoss:
    """Test suite for CombinedSpectralLoss."""

    @pytest.fixture
    def loss_fn(self) -> CombinedSpectralLoss:
        """Create CombinedSpectralLoss with default parameters."""
        return CombinedSpectralLoss(
            lambda_autocorr=0.05,
            lambda_spectral=0.05,
            lambda_entropy=0.01,
        )

    def test_loss_dict(self, loss_fn: CombinedSpectralLoss):
        """Test that loss returns a dictionary."""
        predictions = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        loss_dict = loss_fn(predictions, targets)

        assert isinstance(loss_dict, dict)
        assert "spectral_loss" in loss_dict
        assert "autocorrelation_loss" in loss_dict
        assert "spectral_consistency_loss" in loss_dict
        assert "entropy_loss" in loss_dict

    def test_total_loss(self, loss_fn: CombinedSpectralLoss):
        """Test that total spectral loss is computed."""
        predictions = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        loss_dict = loss_fn(predictions, targets)

        total = loss_dict["spectral_loss"]
        assert total.item() >= 0

    def test_gradient_flow(self, loss_fn: CombinedSpectralLoss):
        """Test that gradients flow through combined loss."""
        predictions = torch.randn(4, 64, requires_grad=True)
        targets = torch.randn(4, 64)
        loss_dict = loss_fn(predictions, targets)
        loss_dict["spectral_loss"].backward()

        assert predictions.grad is not None


class TestModelRegistryIntegration:
    """Test SpectralRegimePINN integration with model registry."""

    def test_model_creation_from_registry(self):
        """Test creating SpectralRegimePINN through model registry."""
        from pathlib import Path
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry(Path("."))
        model = registry.create_model("spectral_pinn", input_dim=5)

        assert model is not None
        assert isinstance(model, SpectralRegimePINN)

    def test_model_info_in_registry(self):
        """Test that spectral_pinn is registered correctly."""
        from pathlib import Path
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry(Path("."))
        models = registry.list_available_models()

        # Find spectral_pinn in list
        spectral_info = None
        for m in models:
            if m["model_key"] == "spectral_pinn":
                spectral_info = m
                break

        assert spectral_info is not None
        assert spectral_info["model_type"] == "advanced"
        assert "physics_constraints" in spectral_info

    def test_forward_pass_from_registry(self):
        """Test forward pass with model from registry."""
        from pathlib import Path
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry(Path("."))
        model = registry.create_model("spectral_pinn", input_dim=5)

        x = torch.randn(2, 30, 5)
        regime_probs = torch.softmax(torch.randn(2, 3), dim=-1)

        predictions, regime_preds = model(x, regime_probs)

        assert predictions.shape == (2, 1)
        assert regime_preds.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
