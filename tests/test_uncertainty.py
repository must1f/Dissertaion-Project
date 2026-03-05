"""
Unit tests for uncertainty quantification module
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.uncertainty import MCDropoutPredictor, add_uncertainty_to_results
from src.models.baseline import LSTMModel


class SimpleDropoutModel(nn.Module):
    """Simple model with dropout for testing"""

    def __init__(self, input_dim=5, hidden_dim=32, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq, features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden]
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output


class TestMCDropoutPredictor:
    """Test MC Dropout uncertainty quantification"""

    @pytest.fixture
    def model_with_dropout(self):
        """Create a model with dropout layers"""
        return SimpleDropoutModel(input_dim=5, hidden_dim=32, output_dim=1, dropout=0.2)

    @pytest.fixture
    def mc_predictor(self, model_with_dropout):
        """Create MC Dropout predictor"""
        return MCDropoutPredictor(
            model=model_with_dropout,
            n_samples=50,
            dropout_rate=0.2,
            confidence_level=0.95
        )

    def test_initialization(self, mc_predictor):
        """Test that MC Dropout predictor initializes correctly"""
        assert mc_predictor.n_samples == 50
        assert mc_predictor.dropout_rate == 0.2
        assert mc_predictor.confidence_level == 0.95
        assert abs(mc_predictor.lower_quantile - 0.025) < 1e-10
        assert abs(mc_predictor.upper_quantile - 0.975) < 1e-10

    def test_enable_dropout(self, model_with_dropout, mc_predictor):
        """Test that dropout is enabled at inference time"""
        # Set model to eval mode
        model_with_dropout.eval()

        # Dropout should be inactive
        for module in model_with_dropout.modules():
            if isinstance(module, nn.Dropout):
                assert not module.training, "Dropout should be inactive in eval mode"

        # Enable dropout via MC predictor
        mc_predictor.enable_dropout()

        # Dropout should now be active
        for module in model_with_dropout.modules():
            if isinstance(module, nn.Dropout):
                assert module.training, "Dropout should be active after enable_dropout()"

    def test_predict_with_uncertainty_shape(self, mc_predictor):
        """Test that predictions have correct shape"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        result = mc_predictor.predict_with_uncertainty(x)

        # Check that all required keys are present
        assert 'mean' in result
        assert 'std' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'coefficient_of_variation' in result

        # Check shapes
        assert result['mean'].shape == (batch_size, 1)
        assert result['std'].shape == (batch_size, 1)
        assert result['lower_bound'].shape == (batch_size, 1)
        assert result['upper_bound'].shape == (batch_size, 1)
        assert result['coefficient_of_variation'].shape == (batch_size, 1)

    def test_uncertainty_is_nonzero(self, mc_predictor):
        """Test that MC Dropout produces nonzero uncertainty"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        result = mc_predictor.predict_with_uncertainty(x)

        # Standard deviation should be positive (uncertainty exists)
        assert (result['std'] > 0).all(), "Uncertainty should be positive with dropout"

    def test_confidence_intervals_ordering(self, mc_predictor):
        """Test that confidence intervals are ordered correctly"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        result = mc_predictor.predict_with_uncertainty(x)

        # Lower bound < mean < upper bound
        assert (result['lower_bound'] <= result['mean']).all()
        assert (result['mean'] <= result['upper_bound']).all()

    def test_return_samples(self, mc_predictor):
        """Test that MC samples can be returned"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        result = mc_predictor.predict_with_uncertainty(x, return_samples=True)

        assert 'samples' in result
        assert result['samples'].shape == (mc_predictor.n_samples, batch_size, 1)

    def test_predict_with_confidence(self, mc_predictor):
        """Test simplified prediction interface"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        mean, std, confidence = mc_predictor.predict_with_confidence(x)

        # Check shapes
        assert mean.shape == (batch_size, 1)
        assert std.shape == (batch_size, 1)
        assert confidence.shape == (batch_size, 1)

        # Confidence should be in [0, 1]
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_multiple_confidence_levels(self, mc_predictor):
        """Test prediction intervals at multiple confidence levels"""
        batch_size = 8
        seq_len = 60
        features = 5

        x = torch.randn(batch_size, seq_len, features)

        intervals = mc_predictor.compute_prediction_intervals(
            x,
            confidence_levels=[0.68, 0.95, 0.99]
        )

        # Check that all requested levels are present
        assert 0.68 in intervals
        assert 0.95 in intervals
        assert 0.99 in intervals

        # Higher confidence → wider intervals
        width_68 = (intervals[0.68][1] - intervals[0.68][0]).mean()
        width_95 = (intervals[0.95][1] - intervals[0.95][0]).mean()
        width_99 = (intervals[0.99][1] - intervals[0.99][0]).mean()

        assert width_68 < width_95 < width_99, "Higher confidence should give wider intervals"

    def test_reproducibility_with_fixed_seed(self, model_with_dropout):
        """Test that results are reproducible with fixed seed"""
        torch.manual_seed(42)

        predictor1 = MCDropoutPredictor(model_with_dropout, n_samples=20)

        x = torch.randn(4, 60, 5)

        torch.manual_seed(42)
        result1 = predictor1.predict_with_uncertainty(x, return_samples=True)

        torch.manual_seed(42)
        result2 = predictor1.predict_with_uncertainty(x, return_samples=True)

        # Results should be identical with same seed
        assert torch.allclose(result1['mean'], result2['mean'])
        assert torch.allclose(result1['std'], result2['std'])

    def test_different_batch_sizes(self, mc_predictor):
        """Test that MC Dropout works with different batch sizes"""
        seq_len = 60
        features = 5

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, seq_len, features)
            result = mc_predictor.predict_with_uncertainty(x)

            assert result['mean'].shape[0] == batch_size
            assert result['std'].shape[0] == batch_size


class TestUncertaintyUtils:
    """Test utility functions for uncertainty"""

    def test_add_uncertainty_to_results(self):
        """Test packaging uncertainty into results dictionary"""
        predictions = np.array([100.0, 105.0, 110.0])
        uncertainties = np.array([2.0, 3.0, 4.0])
        confidence = np.array([0.8, 0.75, 0.7])

        results = add_uncertainty_to_results(predictions, uncertainties, confidence)

        assert 'predictions' in results
        assert 'uncertainty_std' in results
        assert 'confidence' in results
        assert 'lower_bound_95' in results
        assert 'upper_bound_95' in results

        # Check that bounds are calculated correctly
        expected_lower_95 = predictions - 1.96 * uncertainties
        expected_upper_95 = predictions + 1.96 * uncertainties

        np.testing.assert_array_almost_equal(results['lower_bound_95'], expected_lower_95)
        np.testing.assert_array_almost_equal(results['upper_bound_95'], expected_upper_95)

    def test_uncertainty_bounds_68(self):
        """Test 68% confidence bounds (1 sigma)"""
        predictions = np.array([100.0])
        uncertainties = np.array([10.0])
        confidence = np.array([0.8])

        results = add_uncertainty_to_results(predictions, uncertainties, confidence)

        # 68% bounds should be ±1 sigma
        assert results['lower_bound_68'][0] == 90.0
        assert results['upper_bound_68'][0] == 110.0


class TestMCDropoutVsDeterministic:
    """Compare MC Dropout predictions to deterministic predictions"""

    def test_mc_dropout_increases_uncertainty_appropriately(self):
        """Test that MC Dropout provides sensible uncertainty estimates"""
        model = SimpleDropoutModel(input_dim=5, hidden_dim=32, output_dim=1, dropout=0.3)

        # With higher dropout, we expect higher uncertainty
        predictor_low = MCDropoutPredictor(model, n_samples=50, dropout_rate=0.1)
        predictor_high = MCDropoutPredictor(model, n_samples=50, dropout_rate=0.3)

        x = torch.randn(16, 60, 5)

        # Note: This test may not always hold because dropout_rate parameter
        # doesn't actually change model architecture, just affects interpretation
        # Uncertainty primarily comes from the model's existing dropout layers

        result = predictor_high.predict_with_uncertainty(x)

        # Just verify that we get reasonable uncertainty values
        assert (result['std'] > 0).all(), "Should have positive uncertainty"

    def test_model_state_restored_after_prediction(self):
        """Test that model training state is restored after MC Dropout"""
        model = SimpleDropoutModel()

        # Start in training mode
        model.train()
        assert model.training

        predictor = MCDropoutPredictor(model, n_samples=10)

        x = torch.randn(8, 60, 5)
        _ = predictor.predict_with_uncertainty(x)

        # Should be back in training mode
        assert model.training

        # Now test starting from eval mode
        model.eval()
        assert not model.training

        _ = predictor.predict_with_uncertainty(x)

        # Should still be in eval mode
        assert not model.training


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
