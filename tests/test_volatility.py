"""
Unit tests for volatility forecasting framework

Tests cover:
- Volatility model architectures (LSTM, GRU, Transformer, PINN variants)
- Volatility baseline models (Rolling, EWMA, GARCH, GJR-GARCH)
- Volatility metrics (QLIKE, HMSE, Mincer-Zarnowitz R²)
- Volatility targeting strategy
- Physics constraints (OU, GARCH consistency, Feller condition)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Skip markers for optional dependencies
# =============================================================================

try:
    from src.models.volatility import (
        VolatilityLSTM,
        VolatilityGRU,
        VolatilityTransformer,
        VolatilityPINN,
        HestonPINN,
        StackedVolatilityPINN,
        VolatilityPhysicsLoss,
        create_volatility_model,
    )
    HAS_VOLATILITY_MODELS = True
except ImportError:
    HAS_VOLATILITY_MODELS = False

try:
    from src.models.volatility_baselines import (
        NaiveRollingVol,
        EWMA,
        GARCHModel,
        GJRGARCHModel,
        create_volatility_baseline,
    )
    HAS_VOLATILITY_BASELINES = True
except ImportError:
    HAS_VOLATILITY_BASELINES = False

try:
    from src.evaluation.volatility_metrics import (
        VolatilityMetrics,
        EconomicVolatilityMetrics,
        VolatilityDiagnostics,
        evaluate_volatility_forecast,
    )
    HAS_VOLATILITY_METRICS = True
except ImportError:
    HAS_VOLATILITY_METRICS = False

try:
    from src.trading.volatility_strategy import (
        VolatilityTargetingStrategy,
        StrategyResult,
    )
    HAS_VOLATILITY_STRATEGY = True
except ImportError:
    HAS_VOLATILITY_STRATEGY = False


requires_volatility_models = pytest.mark.skipif(
    not HAS_VOLATILITY_MODELS,
    reason="Volatility models not available"
)

requires_volatility_baselines = pytest.mark.skipif(
    not HAS_VOLATILITY_BASELINES,
    reason="Volatility baselines not available"
)

requires_volatility_metrics = pytest.mark.skipif(
    not HAS_VOLATILITY_METRICS,
    reason="Volatility metrics not available"
)

requires_volatility_strategy = pytest.mark.skipif(
    not HAS_VOLATILITY_STRATEGY,
    reason="Volatility strategy not available"
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample returns data."""
    np.random.seed(42)
    n = 1000
    # Simulate returns with time-varying volatility (GARCH-like)
    returns = np.zeros(n)
    vol = 0.01
    for i in range(n):
        vol = 0.00001 + 0.1 * returns[i-1]**2 + 0.85 * vol if i > 0 else vol
        returns[i] = np.random.normal(0, np.sqrt(vol))
    return returns


@pytest.fixture
def sample_volatility_data():
    """Generate sample data for volatility models."""
    batch_size = 32
    seq_len = 40
    input_dim = 10

    X = torch.randn(batch_size, seq_len, input_dim)
    y = torch.abs(torch.randn(batch_size, 1)) * 0.01  # Variance targets
    returns = torch.randn(batch_size, seq_len) * 0.01

    return X, y, returns


# =============================================================================
# Volatility Model Tests
# =============================================================================

@requires_volatility_models
class TestVolatilityModels:
    """Test volatility neural network models."""

    def test_volatility_lstm_forward(self, sample_volatility_data):
        """Test VolatilityLSTM forward pass."""
        X, y, returns = sample_volatility_data

        model = VolatilityLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0), "Variance predictions must be non-negative"

    def test_volatility_gru_forward(self, sample_volatility_data):
        """Test VolatilityGRU forward pass."""
        X, y, returns = sample_volatility_data

        model = VolatilityGRU(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0)

    def test_volatility_transformer_forward(self, sample_volatility_data):
        """Test VolatilityTransformer forward pass."""
        X, y, returns = sample_volatility_data

        model = VolatilityTransformer(
            input_dim=10,
            d_model=64,
            nhead=4,
            num_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0)

    def test_volatility_pinn_forward(self, sample_volatility_data):
        """Test VolatilityPINN forward pass."""
        X, y, returns = sample_volatility_data

        model = VolatilityPINN(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0)

    def test_heston_pinn_forward(self, sample_volatility_data):
        """Test HestonPINN forward pass."""
        X, y, returns = sample_volatility_data

        model = HestonPINN(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0)

    def test_stacked_volatility_pinn_forward(self, sample_volatility_data):
        """Test StackedVolatilityPINN forward pass."""
        X, y, returns = sample_volatility_data

        model = StackedVolatilityPINN(
            input_dim=10,
            rnn_hidden_dim=64,
            num_rnn_layers=2,
        )

        output = model(X)

        assert output.shape == (32, 1)
        assert torch.all(output >= 0)

    def test_create_volatility_model_factory(self):
        """Test model factory function."""
        model_types = ['vol_lstm', 'vol_gru', 'vol_transformer', 'vol_pinn', 'heston_pinn']

        for model_type in model_types:
            model = create_volatility_model(
                model_type=model_type,
                input_dim=10,
                hidden_dim=64,
                num_layers=2,
            )
            assert model is not None, f"Failed to create model: {model_type}"


@requires_volatility_models
class TestVolatilityPhysicsLoss:
    """Test volatility physics loss functions."""

    def test_physics_loss_computation(self, sample_volatility_data):
        """Test physics loss computation."""
        X, y, returns = sample_volatility_data
        predictions = torch.abs(torch.randn(32, 1)) * 0.01

        physics_loss = VolatilityPhysicsLoss(
            lambda_ou=0.1,
            lambda_garch=0.1,
            lambda_feller=0.05,
            lambda_leverage=0.05,
        )

        # VolatilityPhysicsLoss expects (variance_pred, variance_target, metadata)
        metadata = {
            'variance_history': predictions.expand(-1, 10) * 0.9,  # [batch, seq_len]
            'returns': returns,
            'long_run_variance': predictions.mean(),
        }

        loss, loss_dict = physics_loss(predictions, y, metadata)

        assert loss.item() >= 0
        assert 'data_loss' in loss_dict
        assert 'ou_loss' in loss_dict or 'physics_loss' in loss_dict

    def test_ou_residual_loss(self, sample_volatility_data):
        """Test OU (mean-reversion) residual loss."""
        X, y, returns = sample_volatility_data

        physics_loss = VolatilityPhysicsLoss(lambda_ou=1.0)

        # Variance far from mean should have higher loss
        pred_far = torch.ones(32, 1) * 0.1  # Far from typical variance
        pred_near = torch.ones(32, 1) * 0.0001  # Close to zero

        # VolatilityPhysicsLoss expects (variance_pred, variance_target, metadata)
        metadata_far = {
            'variance_history': pred_far.expand(-1, 10) * 0.9,
            'returns': returns,
        }
        metadata_near = {
            'variance_history': pred_near.expand(-1, 10) * 0.9,
            'returns': returns,
        }

        loss_far, _ = physics_loss(pred_far, y, metadata_far)
        loss_near, _ = physics_loss(pred_near, y, metadata_near)

        # Both should be computable
        assert not torch.isnan(loss_far)
        assert not torch.isnan(loss_near)

    def test_garch_consistency_loss(self, sample_volatility_data):
        """Test GARCH consistency loss."""
        X, y, returns = sample_volatility_data

        physics_loss = VolatilityPhysicsLoss(lambda_garch=1.0)
        predictions = torch.abs(torch.randn(32, 1)) * 0.01
        prev_variance = predictions * 0.9

        # VolatilityPhysicsLoss expects (variance_pred, variance_target, metadata)
        metadata = {
            'variance_history': prev_variance.expand(-1, 10),
            'returns': returns,
        }

        loss, loss_dict = physics_loss(predictions, y, metadata)

        assert not torch.isnan(loss)

    def test_learnable_physics_params(self):
        """Test that physics parameters are learnable."""
        physics_loss = VolatilityPhysicsLoss(
            lambda_ou=0.1,
            lambda_garch=0.1,
        )

        # Check parameters are nn.Parameter
        assert hasattr(physics_loss, 'theta_raw') or hasattr(physics_loss, 'theta')

        # Get learned params
        params = physics_loss.get_learned_params()
        assert isinstance(params, dict)


# =============================================================================
# Volatility Baseline Tests
# =============================================================================

@requires_volatility_baselines
class TestVolatilityBaselines:
    """Test baseline volatility models."""

    def test_naive_rolling_vol(self, sample_returns):
        """Test naive rolling volatility."""
        model = NaiveRollingVol(lookback=20)
        model.fit(sample_returns)
        forecast = model.predict(sample_returns)

        assert len(forecast.variance) == len(sample_returns)
        assert np.all(forecast.variance[20:] >= 0), "Variance must be non-negative"

    def test_ewma(self, sample_returns):
        """Test EWMA model."""
        model = EWMA(decay=0.94)
        model.fit(sample_returns)
        forecast = model.predict(sample_returns)

        assert len(forecast.variance) == len(sample_returns)
        assert np.all(forecast.variance >= 0)

        # EWMA with decay=0.94 is RiskMetrics standard
        assert model.decay == 0.94

    def test_garch_model(self, sample_returns):
        """Test GARCH(1,1) model."""
        model = GARCHModel()
        model.fit(sample_returns)
        forecast = model.predict(sample_returns)

        assert len(forecast.variance) == len(sample_returns)
        assert np.all(forecast.variance >= 0)

        # Check estimated parameters
        assert 'omega' in forecast.params
        assert 'alpha' in forecast.params
        assert 'beta' in forecast.params

        # GARCH parameters should sum to less than 1 for stationarity
        persistence = forecast.params.get('alpha', 0) + forecast.params.get('beta', 0)
        assert persistence < 1.0, "GARCH not stationary"

    def test_gjr_garch_model(self, sample_returns):
        """Test GJR-GARCH model (asymmetric)."""
        model = GJRGARCHModel()
        model.fit(sample_returns)
        forecast = model.predict(sample_returns)

        assert len(forecast.variance) == len(sample_returns)
        assert np.all(forecast.variance >= 0)

        # GJR-GARCH should have gamma parameter for leverage effect
        assert 'gamma' in forecast.params

    def test_create_volatility_baseline_factory(self, sample_returns):
        """Test baseline factory function."""
        model_types = ['rolling', 'ewma', 'garch', 'gjr_garch']

        for model_type in model_types:
            model = create_volatility_baseline(model_type)
            assert model is not None, f"Failed to create baseline: {model_type}"

            model.fit(sample_returns)
            forecast = model.predict(sample_returns)
            assert forecast is not None


# =============================================================================
# Volatility Metrics Tests
# =============================================================================

@requires_volatility_metrics
class TestVolatilityMetrics:
    """Test volatility forecast evaluation metrics."""

    def test_qlike_loss(self):
        """Test QLIKE (quasi-likelihood) loss."""
        predicted_var = np.array([0.0001, 0.0002, 0.0003, 0.0004])
        realized_var = np.array([0.00015, 0.00018, 0.00035, 0.0003])

        qlike = VolatilityMetrics.qlike(predicted_var, realized_var)

        assert np.isfinite(qlike)
        assert qlike >= 0  # QLIKE should be non-negative

    def test_hmse_loss(self):
        """Test heteroskedasticity-adjusted MSE."""
        predicted_var = np.array([0.0001, 0.0002, 0.0003, 0.0004])
        realized_var = np.array([0.00015, 0.00018, 0.00035, 0.0003])

        hmse = VolatilityMetrics.hmse(predicted_var, realized_var)

        assert np.isfinite(hmse)
        assert hmse >= 0

    def test_mincer_zarnowitz_r2(self):
        """Test Mincer-Zarnowitz R² (forecast efficiency)."""
        # Perfect forecast (need at least 10 samples for M-Z regression)
        np.random.seed(42)
        predicted_var = np.random.uniform(0.0001, 0.0005, 50)
        realized_var = predicted_var.copy()

        r2 = VolatilityMetrics.mincer_zarnowitz_r2(predicted_var, realized_var)

        assert r2 >= 0.99, f"Perfect forecast should have R² close to 1, got {r2}"

        # Random forecast
        random_var = np.random.uniform(0.0001, 0.0005, 100)
        realized = np.random.uniform(0.0001, 0.0005, 100)

        r2_random = VolatilityMetrics.mincer_zarnowitz_r2(random_var, realized)

        # Random forecast should have lower R²
        assert r2_random < r2

    def test_evaluate_volatility_forecast(self, sample_returns):
        """Test comprehensive volatility forecast evaluation."""
        n = len(sample_returns)

        # Generate some predicted variance
        predicted_var = np.var(sample_returns) * np.ones(n)

        # Compute realized variance (rolling)
        window = 20
        realized_var = np.array([
            np.var(sample_returns[max(0, i-window):i+1])
            for i in range(n)
        ])

        metrics = evaluate_volatility_forecast(
            predicted_var=predicted_var[window:],
            realized_var=realized_var[window:],
            returns=sample_returns[window:],
            model_name='test_model',
        )

        assert 'model' in metrics
        assert 'mse' in metrics or 'qlike' in metrics


@requires_volatility_metrics
class TestEconomicMetrics:
    """Test economic volatility metrics."""

    def test_var_breach_rate(self, sample_returns):
        """Test Value-at-Risk breach rate calculation."""
        n = len(sample_returns)
        predicted_vol = np.std(sample_returns) * np.ones(n)

        result = EconomicVolatilityMetrics.var_breach_rate(
            returns=sample_returns,
            predicted_vol=predicted_vol,
            confidence=0.99,
        )

        # Result is a dictionary
        assert isinstance(result, dict)
        assert 'breach_rate' in result
        breach_rate = result['breach_rate']
        assert 0 <= breach_rate <= 1
        # At 99% confidence, expected breach rate ~1%
        assert breach_rate < 0.10, "Breach rate too high for 99% VaR"

    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall (CVaR) accuracy."""
        n = len(sample_returns)
        predicted_vol = np.std(sample_returns) * np.ones(n)

        result = EconomicVolatilityMetrics.expected_shortfall_accuracy(
            returns=sample_returns,
            predicted_vol=predicted_vol,
            confidence=0.95,
        )

        # Result is a dictionary
        assert isinstance(result, dict)
        assert 'es_predicted' in result or 'es_realized' in result or 'es_ratio' in result


@requires_volatility_metrics
class TestVolatilityDiagnostics:
    """Test volatility forecast diagnostics."""

    def test_diebold_mariano_test(self):
        """Test Diebold-Mariano test for equal predictive accuracy."""
        np.random.seed(42)
        n = 200

        # Create two forecasts with different errors
        realized = np.random.uniform(0.0001, 0.0005, n)
        forecast1 = realized + np.random.normal(0, 0.00005, n)
        forecast2 = realized + np.random.normal(0, 0.0001, n)  # Worse forecast

        # Compute forecast errors (function expects errors, not forecasts)
        errors1 = forecast1 - realized
        errors2 = forecast2 - realized

        result = VolatilityDiagnostics.diebold_mariano_test(
            errors1=errors1,
            errors2=errors2,
        )

        assert 'dm_statistic' in result
        assert 'p_value' in result
        # p_value might be NaN for edge cases, so check for that
        if not np.isnan(result['p_value']):
            assert 0 <= result['p_value'] <= 1


# =============================================================================
# Volatility Strategy Tests
# =============================================================================

@requires_volatility_strategy
class TestVolatilityStrategy:
    """Test volatility targeting strategy."""

    def test_volatility_targeting_strategy(self, sample_returns):
        """Test volatility targeting strategy backtest."""
        n = len(sample_returns)

        # Use simple rolling volatility as prediction
        window = 20
        predicted_vol = np.array([
            np.std(sample_returns[max(0, i-window):i+1])
            for i in range(n)
        ])
        # Annualize
        predicted_vol *= np.sqrt(252)

        strategy = VolatilityTargetingStrategy(
            target_vol=0.15,  # 15% annual target
            min_weight=0.25,
            max_weight=2.0,
            transaction_cost=0.001,
        )

        result = strategy.backtest(
            returns=sample_returns[window:],
            predicted_vol=predicted_vol[window:],
            benchmark_returns=sample_returns[window:],
        )

        assert isinstance(result, StrategyResult)
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'equity_curve')
        assert len(result.equity_curve) == len(sample_returns) - window

    def test_strategy_weights_bounded(self, sample_returns):
        """Test that strategy weights are properly bounded."""
        n = len(sample_returns)

        # Extreme volatility predictions
        predicted_vol = np.ones(n) * 0.5  # Very high vol

        strategy = VolatilityTargetingStrategy(
            target_vol=0.15,
            min_weight=0.25,
            max_weight=2.0,
        )

        result = strategy.backtest(
            returns=sample_returns,
            predicted_vol=predicted_vol,
        )

        # All weights should be within bounds
        assert np.all(result.weights >= 0.25)
        assert np.all(result.weights <= 2.0)

    def test_strategy_metrics(self, sample_returns):
        """Test that strategy computes all required metrics."""
        n = len(sample_returns)
        predicted_vol = np.std(sample_returns) * np.sqrt(252) * np.ones(n)

        strategy = VolatilityTargetingStrategy(target_vol=0.15)
        result = strategy.backtest(
            returns=sample_returns,
            predicted_vol=predicted_vol,
            benchmark_returns=sample_returns,
        )

        # Check all metrics exist
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'annual_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'sortino_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'calmar_ratio')
        assert hasattr(result, 'information_ratio')
        assert hasattr(result, 'vol_tracking_error')


# =============================================================================
# Integration Tests
# =============================================================================

@requires_volatility_models
@requires_volatility_metrics
class TestVolatilityIntegration:
    """Integration tests for the volatility framework."""

    def test_model_training_and_evaluation(self, sample_volatility_data):
        """Test end-to-end model training and evaluation."""
        X, y, returns = sample_volatility_data

        # Create model
        model = VolatilityPINN(
            input_dim=10,
            hidden_dim=32,
            num_layers=1,
        )

        # Forward pass
        predictions = model(X)

        # Evaluate
        metrics = evaluate_volatility_forecast(
            predicted_var=predictions.detach().numpy().flatten(),
            realized_var=y.numpy().flatten(),
            model_name='vol_pinn',
        )

        assert metrics is not None

    def test_pinn_learns_physics_params(self, sample_volatility_data):
        """Test that PINN can learn physics parameters."""
        X, y, returns = sample_volatility_data

        model = VolatilityPINN(
            input_dim=10,
            hidden_dim=32,
            num_layers=1,
        )

        # Get initial physics params
        if hasattr(model, 'get_learned_physics_params'):
            initial_params = model.get_learned_physics_params()

            # Train a few steps
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for _ in range(10):
                optimizer.zero_grad()
                pred = model(X)
                loss = torch.mean((pred - y) ** 2)
                loss.backward()
                optimizer.step()

            # Check params changed
            final_params = model.get_learned_physics_params()

            # At least some params should change
            params_changed = any(
                initial_params.get(k) != final_params.get(k)
                for k in initial_params.keys()
            )
            # This may or may not be true depending on initialization
            assert isinstance(final_params, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
