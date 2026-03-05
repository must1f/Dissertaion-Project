"""Unit tests for SpectralAnalyzer module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from src.data.spectral_analyzer import SpectralAnalyzer, SpectralFeatures


class TestSpectralAnalyzer:
    """Test suite for SpectralAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> SpectralAnalyzer:
        """Create a SpectralAnalyzer with default parameters."""
        return SpectralAnalyzer(window_size=32, sampling_rate=252.0)

    @pytest.fixture
    def sample_returns(self) -> np.ndarray:
        """Generate synthetic returns data."""
        np.random.seed(42)
        n = 200
        # Generate returns with some structure
        trend = 0.0001 * np.arange(n)
        noise = 0.02 * np.random.randn(n)
        # Add a periodic component (weekly cycle ~5 days)
        cycle = 0.01 * np.sin(2 * np.pi * np.arange(n) / 5)
        returns = trend + noise + cycle
        return returns

    @pytest.fixture
    def pure_noise(self) -> np.ndarray:
        """Generate pure white noise returns."""
        np.random.seed(123)
        return 0.02 * np.random.randn(200)

    # --- Power Spectrum Tests ---

    def test_power_spectrum_shape(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that power spectrum has correct shape."""
        window = sample_returns[:32]
        frequencies, power = analyzer.compute_power_spectrum(window)

        # For window_size=32, rfft gives 17 frequency bins (n//2 + 1)
        assert len(frequencies) == 17
        assert len(power) == 17
        assert frequencies.shape == power.shape

    def test_power_spectrum_normalization(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that power spectrum sums to approximately 1."""
        window = sample_returns[:32]
        _, power = analyzer.compute_power_spectrum(window)

        # Power should be normalized (sum ~= 1)
        assert_allclose(np.sum(power), 1.0, atol=0.01)

    def test_power_spectrum_nonnegative(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that all power values are non-negative."""
        window = sample_returns[:32]
        _, power = analyzer.compute_power_spectrum(window)

        assert np.all(power >= 0)

    def test_power_spectrum_frequencies(self, analyzer: SpectralAnalyzer):
        """Test that frequency bins are correctly computed."""
        window = np.random.randn(32)
        frequencies, _ = analyzer.compute_power_spectrum(window)

        # DC component should be at 0
        assert frequencies[0] == 0.0
        # Nyquist frequency should be sampling_rate / 2
        expected_nyquist = analyzer.sampling_rate / 2
        assert_allclose(frequencies[-1], expected_nyquist, rtol=0.01)

    # --- Spectral Entropy Tests ---

    def test_spectral_entropy_range(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that spectral entropy is in [0, 1] range."""
        window = sample_returns[:32]
        _, power = analyzer.compute_power_spectrum(window)
        entropy = analyzer.compute_spectral_entropy(power)

        assert 0.0 <= entropy <= 1.0

    def test_spectral_entropy_white_noise(self, analyzer: SpectralAnalyzer, pure_noise: np.ndarray):
        """Test that white noise has high spectral entropy."""
        window = pure_noise[:32]
        _, power = analyzer.compute_power_spectrum(window)
        entropy = analyzer.compute_spectral_entropy(power)

        # White noise should have high entropy (close to 1)
        assert entropy > 0.8

    def test_spectral_entropy_sine_wave(self, analyzer: SpectralAnalyzer):
        """Test that pure sine wave has low spectral entropy."""
        # Generate pure sine wave
        t = np.linspace(0, 1, 32)
        sine = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
        _, power = analyzer.compute_power_spectrum(sine)
        entropy = analyzer.compute_spectral_entropy(power)

        # Pure sine should have low entropy (concentrated at one frequency)
        assert entropy < 0.5

    # --- Feature Computation Tests ---

    def test_compute_features_shape(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that computed features have correct shape."""
        features = analyzer.compute_features(sample_returns)

        # All feature arrays should have same length as input
        # (NaN values for first window_size-1 elements)
        expected_len = len(sample_returns)
        assert len(features.spectral_entropy) == expected_len
        assert len(features.dominant_frequency) == expected_len
        assert len(features.power_low) == expected_len
        assert len(features.power_mid) == expected_len
        assert len(features.power_high) == expected_len
        assert len(features.power_ratio) == expected_len
        assert len(features.autocorrelation_lag1) == expected_len
        assert len(features.spectral_slope) == expected_len

        # First window_size-1 values should be NaN
        n_nan = analyzer.window_size - 1
        assert np.all(np.isnan(features.spectral_entropy[:n_nan]))

    def test_compute_features_no_nan_after_warmup(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that computed features contain no NaN values after warmup period."""
        features = analyzer.compute_features(sample_returns)

        # Skip the first window_size-1 values which are NaN by design
        start_idx = analyzer.window_size - 1

        assert not np.any(np.isnan(features.spectral_entropy[start_idx:]))
        assert not np.any(np.isnan(features.dominant_frequency[start_idx:]))
        assert not np.any(np.isnan(features.power_low[start_idx:]))
        assert not np.any(np.isnan(features.power_mid[start_idx:]))
        assert not np.any(np.isnan(features.power_high[start_idx:]))
        assert not np.any(np.isnan(features.power_ratio[start_idx:]))
        assert not np.any(np.isnan(features.autocorrelation_lag1[start_idx:]))
        assert not np.any(np.isnan(features.spectral_slope[start_idx:]))

    def test_power_bands_sum(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that power bands sum to approximately 1."""
        features = analyzer.compute_features(sample_returns)

        # Only check valid (non-NaN) values
        start_idx = analyzer.window_size - 1
        total_power = (
            features.power_low[start_idx:] +
            features.power_mid[start_idx:] +
            features.power_high[start_idx:]
        )
        assert_allclose(total_power, np.ones_like(total_power), atol=0.05)

    def test_power_bands_nonnegative(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that all power bands are non-negative."""
        features = analyzer.compute_features(sample_returns)

        # Only check valid (non-NaN) values
        start_idx = analyzer.window_size - 1
        assert np.all(features.power_low[start_idx:] >= 0)
        assert np.all(features.power_mid[start_idx:] >= 0)
        assert np.all(features.power_high[start_idx:] >= 0)

    def test_autocorrelation_range(self, analyzer: SpectralAnalyzer, sample_returns: np.ndarray):
        """Test that autocorrelation is in [-1, 1] range."""
        features = analyzer.compute_features(sample_returns)

        # Only check valid (non-NaN) values
        start_idx = analyzer.window_size - 1
        valid_ac = features.autocorrelation_lag1[start_idx:]
        assert np.all(valid_ac >= -1.0)
        assert np.all(valid_ac <= 1.0)

    # --- Edge Cases ---

    def test_short_series(self, analyzer: SpectralAnalyzer):
        """Test behavior with series shorter than window."""
        short_returns = np.random.randn(10)  # Shorter than window_size=32

        # Should return all NaN values for series shorter than window
        features = analyzer.compute_features(short_returns)
        assert len(features.spectral_entropy) == 10
        assert np.all(np.isnan(features.spectral_entropy))

    def test_constant_series(self, analyzer: SpectralAnalyzer):
        """Test behavior with constant series."""
        constant = np.ones(64) * 0.01

        # Should handle gracefully (may produce warnings but not crash)
        features = analyzer.compute_features(constant)
        assert isinstance(features, SpectralFeatures)

    def test_zero_series(self, analyzer: SpectralAnalyzer):
        """Test behavior with all-zero series."""
        zeros = np.zeros(64)

        # Should handle gracefully
        features = analyzer.compute_features(zeros)
        assert isinstance(features, SpectralFeatures)

    # --- Feature Analysis Tests ---

    def test_dominant_frequency_periodic_signal(self, analyzer: SpectralAnalyzer):
        """Test that dominant frequency is correctly identified for periodic signal."""
        # Generate signal with known dominant frequency
        t = np.arange(100) / analyzer.sampling_rate
        freq = 10.0  # 10 cycles per day
        signal = np.sin(2 * np.pi * freq * t)

        features = analyzer.compute_features(signal)

        # Dominant frequency should be close to 10 for most windows
        # Filter out NaN values before computing median
        valid_freqs = features.dominant_frequency[~np.isnan(features.dominant_frequency)]
        median_dom_freq = np.median(valid_freqs)
        assert abs(median_dom_freq - freq) < 5.0  # Within 5 cycles/day

    def test_power_ratio_signal_vs_noise(self):
        """Test that signal has higher power ratio than noise."""
        np.random.seed(42)

        # Use sampling_rate=1.0 for proper frequency bands
        analyzer = SpectralAnalyzer(window_size=32, sampling_rate=1.0)

        # Create signal with strong trend (longer series for more valid points)
        n = 200
        trend = 0.05 * np.arange(n) / n  # Stronger trend
        signal = trend + 0.001 * np.random.randn(n)

        # Pure noise
        noise = 0.02 * np.random.randn(n)

        features_signal = analyzer.compute_features(signal)
        features_noise = analyzer.compute_features(noise)

        # Signal should have higher low-frequency power
        # Filter out NaN values before computing mean using nanmean
        mean_signal = np.nanmean(features_signal.power_low)
        mean_noise = np.nanmean(features_noise.power_low)

        # Verify we got valid data
        assert not np.isnan(mean_signal), "Signal power_low should have valid values"
        assert not np.isnan(mean_noise), "Noise power_low should have valid values"
        assert mean_signal > mean_noise


class TestSpectralFeatures:
    """Test suite for SpectralFeatures dataclass."""

    def test_dataclass_creation(self):
        """Test that SpectralFeatures can be created."""
        n = 50
        features = SpectralFeatures(
            spectral_entropy=np.random.rand(n),
            dominant_frequency=np.random.rand(n) * 10,
            power_low=np.random.rand(n) * 0.3,
            power_mid=np.random.rand(n) * 0.3,
            power_high=np.random.rand(n) * 0.4,
            power_ratio=np.random.rand(n),
            autocorrelation_lag1=np.random.rand(n) * 0.2 - 0.1,
            spectral_slope=np.random.rand(n) * 0.5 - 0.25,
        )

        assert len(features.spectral_entropy) == n
        assert len(features.dominant_frequency) == n

    def test_dataclass_to_dict(self):
        """Test conversion to dictionary if method exists."""
        n = 10
        features = SpectralFeatures(
            spectral_entropy=np.zeros(n),
            dominant_frequency=np.zeros(n),
            power_low=np.zeros(n),
            power_mid=np.zeros(n),
            power_high=np.zeros(n),
            power_ratio=np.zeros(n),
            autocorrelation_lag1=np.zeros(n),
            spectral_slope=np.zeros(n),
        )

        # Check all attributes are accessible
        assert hasattr(features, "spectral_entropy")
        assert hasattr(features, "dominant_frequency")
        assert hasattr(features, "power_low")


class TestSpectralAnalyzerIntegration:
    """Integration tests for SpectralAnalyzer."""

    def test_realistic_stock_returns(self):
        """Test with realistic stock return characteristics."""
        np.random.seed(42)
        n = 500

        # Simulate realistic returns: near-zero mean, clustered volatility
        returns = np.zeros(n)
        vol = 0.01
        for i in range(1, n):
            # GARCH-like volatility
            vol = 0.9 * vol + 0.1 * abs(returns[i - 1]) + 0.001
            returns[i] = vol * np.random.randn()

        analyzer = SpectralAnalyzer(window_size=64)
        features = analyzer.compute_features(returns)

        # Realistic returns should have:
        # - Low autocorrelation in returns (EMH) - use nanmean to ignore NaN
        mean_ac = np.nanmean(np.abs(features.autocorrelation_lag1))
        assert mean_ac < 0.3  # Should be small

        # - Moderate spectral entropy (not pure noise, not pure signal)
        mean_entropy = np.nanmean(features.spectral_entropy)
        assert 0.3 < mean_entropy < 0.95

    def test_different_window_sizes(self):
        """Test with different window sizes."""
        np.random.seed(42)
        returns = 0.02 * np.random.randn(200)

        for window_size in [16, 32, 64, 128]:
            analyzer = SpectralAnalyzer(window_size=window_size)
            features = analyzer.compute_features(returns)

            # Output length matches input length (with NaN for warmup period)
            assert len(features.spectral_entropy) == len(returns)

            # Valid values start after warmup period
            n_valid = len(returns) - window_size + 1
            valid_values = features.spectral_entropy[~np.isnan(features.spectral_entropy)]
            assert len(valid_values) == n_valid

    def test_different_sampling_rates(self):
        """Test with different sampling rates."""
        np.random.seed(42)
        returns = 0.02 * np.random.randn(100)

        for sampling_rate in [1.0, 252.0, 365.0]:
            analyzer = SpectralAnalyzer(window_size=32, sampling_rate=sampling_rate)
            features = analyzer.compute_features(returns)

            # Valid values (after warmup) should not be NaN
            valid_entropy = features.spectral_entropy[31:]  # After window_size-1
            assert not np.any(np.isnan(valid_entropy))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
