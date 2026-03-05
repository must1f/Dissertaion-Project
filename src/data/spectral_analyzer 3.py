"""
Spectral Analysis Module for Financial Time Series

Provides frequency-domain feature extraction for financial data:
- FFT-based power spectrum analysis
- Spectral entropy for regime detection
- Rolling window spectral features
- Frequency band decomposition (trend, cycles, noise)

The spectral features capture cyclical patterns and regime characteristics
that are not visible in time-domain analysis alone.

Mathematical Background:
=======================
For a return series r_t, the power spectrum P(f) captures the variance
contribution at each frequency f:

    P(f) = |FFT(r)|^2 / N

Spectral entropy H measures the "uniformity" of the spectrum:
    H = -sum(p_i * log(p_i)) where p_i = P(f_i) / sum(P)

Low entropy → dominant frequencies (trending/cyclical)
High entropy → uniform spectrum (random/efficient market)

References:
    - Granger, C.W.J. (1966). "The Typical Spectral Shape of an Economic Variable."
    - Ramsey, J.B. (2002). "Wavelets in Economics and Finance: Past and Future."
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass
import warnings

# Optional scipy.fft for better performance
try:
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import welch
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)


@dataclass
class SpectralFeatures:
    """
    Container for spectral features extracted from a return series.

    All features are computed using rolling windows to ensure
    no lookahead bias - each value at time t uses only data from [t-window, t].

    Attributes:
        spectral_entropy: Shannon entropy of power spectrum (0 = concentrated, high = uniform)
        dominant_frequency: Peak frequency in cycles per day
        power_low: Power in low-frequency band (< 0.1 cpd) - captures trends
        power_mid: Power in mid-frequency band (0.1-0.25 cpd) - captures weekly cycles
        power_high: Power in high-frequency band (> 0.25 cpd) - captures daily noise
        power_ratio: Signal-to-noise ratio (low+mid) / high
        autocorrelation_lag1: Lag-1 autocorrelation of returns
        spectral_slope: Slope of log-log spectrum (captures self-similarity)
    """
    spectral_entropy: np.ndarray
    dominant_frequency: np.ndarray
    power_low: np.ndarray
    power_mid: np.ndarray
    power_high: np.ndarray
    power_ratio: np.ndarray
    autocorrelation_lag1: np.ndarray
    spectral_slope: np.ndarray

    def to_array(self) -> np.ndarray:
        """Convert to feature matrix for ML models (n_samples, n_features)."""
        # Stack all features, handling potential shape mismatches
        n_samples = len(self.spectral_entropy)
        features = np.column_stack([
            self.spectral_entropy.reshape(-1),
            self.dominant_frequency.reshape(-1),
            self.power_low.reshape(-1),
            self.power_mid.reshape(-1),
            self.power_high.reshape(-1),
            self.power_ratio.reshape(-1),
            self.autocorrelation_lag1.reshape(-1),
            self.spectral_slope.reshape(-1),
        ])
        return features

    @staticmethod
    def feature_names() -> list:
        """Return list of feature names in order."""
        return [
            'spectral_entropy',
            'dominant_frequency',
            'power_low',
            'power_mid',
            'power_high',
            'power_ratio',
            'autocorrelation_lag1',
            'spectral_slope',
        ]


class SpectralAnalyzer:
    """
    Computes spectral (frequency-domain) features for financial time series.

    All features use ONLY historical data with proper lag handling to avoid
    lookahead bias. The window_size parameter controls the FFT window.

    Typical financial time series frequency bands (for daily data):
    - Low frequency (< 0.1 cpd): Trends lasting > 10 days
    - Mid frequency (0.1 - 0.25 cpd): Weekly to bi-weekly cycles
    - High frequency (> 0.25 cpd): Daily noise and microstructure

    Example usage:
        analyzer = SpectralAnalyzer(window_size=64)
        features = analyzer.compute_features(returns)
        entropy = features.spectral_entropy  # Shape: (n_samples,)
    """

    def __init__(
        self,
        window_size: int = 64,
        sampling_rate: float = 1.0,  # 1 sample per trading day
        low_freq_cutoff: float = 0.1,
        high_freq_cutoff: float = 0.25,
        use_welch: bool = False,
        welch_nperseg: int = 32,
    ):
        """
        Initialize SpectralAnalyzer.

        Args:
            window_size: Size of FFT window (must be power of 2 for efficiency)
            sampling_rate: Samples per day (1.0 for daily data)
            low_freq_cutoff: Frequency cutoff for low band (cycles/day)
            high_freq_cutoff: Frequency cutoff between mid and high bands
            use_welch: Use Welch's method for smoother spectrum (more robust)
            welch_nperseg: Segment length for Welch's method
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.use_welch = use_welch
        self.welch_nperseg = min(welch_nperseg, window_size // 2)

        # Precompute frequency bins
        if HAS_SCIPY_FFT:
            self._freqs = rfftfreq(window_size, d=1.0/sampling_rate)
        else:
            self._freqs = np.fft.rfftfreq(window_size, d=1.0/sampling_rate)

        # Create frequency band masks
        self._low_mask = self._freqs < low_freq_cutoff
        self._mid_mask = (self._freqs >= low_freq_cutoff) & (self._freqs < high_freq_cutoff)
        self._high_mask = self._freqs >= high_freq_cutoff

        # Exclude DC component (freq = 0) from low band for entropy
        self._low_mask[0] = False

        logger.info(
            f"SpectralAnalyzer initialized: window={window_size}, "
            f"freq_bands=[0-{low_freq_cutoff}, {low_freq_cutoff}-{high_freq_cutoff}, "
            f"{high_freq_cutoff}+] cpd"
        )

    def compute_power_spectrum(
        self,
        returns: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of a return series using FFT.

        Args:
            returns: 1D array of returns (length should match window_size)
            normalize: Whether to normalize spectrum to sum to 1

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        n = len(returns)

        # Zero-mean the returns (remove DC component)
        returns_centered = returns - np.mean(returns)

        if self.use_welch and HAS_SCIPY_FFT:
            # Welch's method for smoother spectrum
            freqs, power = welch(
                returns_centered,
                fs=self.sampling_rate,
                nperseg=min(self.welch_nperseg, n),
                noverlap=self.welch_nperseg // 2,
            )
        else:
            # Standard periodogram via FFT
            if HAS_SCIPY_FFT:
                fft_result = rfft(returns_centered)
                freqs = rfftfreq(n, d=1.0/self.sampling_rate)
            else:
                fft_result = np.fft.rfft(returns_centered)
                freqs = np.fft.rfftfreq(n, d=1.0/self.sampling_rate)

            # Power spectrum = |FFT|^2 / N
            power = np.abs(fft_result) ** 2 / n

        if normalize:
            total_power = np.sum(power)
            if total_power > 0:
                power = power / total_power

        return freqs, power

    def compute_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """
        Compute Shannon entropy of power spectrum.

        Higher entropy indicates more uniform frequency distribution (random),
        lower entropy indicates concentrated power (trending/cyclical).

        Args:
            power_spectrum: Normalized power spectrum (should sum to 1)

        Returns:
            Spectral entropy value (0 to log(N))
        """
        # Remove zero/negative values for log safety
        p = power_spectrum[power_spectrum > 1e-10]

        if len(p) == 0:
            return 0.0

        # Normalize in case it doesn't sum to 1
        p = p / np.sum(p)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(p * np.log(p))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(p))
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return float(entropy)

    def compute_dominant_frequency(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray
    ) -> float:
        """
        Find the dominant (peak) frequency in the spectrum.

        Args:
            freqs: Frequency array
            power_spectrum: Power spectrum

        Returns:
            Dominant frequency (cycles per day)
        """
        if len(freqs) == 0 or len(power_spectrum) == 0:
            return 0.0

        # Skip DC component (index 0)
        if len(freqs) > 1:
            peak_idx = np.argmax(power_spectrum[1:]) + 1
        else:
            peak_idx = 0

        return float(freqs[peak_idx])

    def compute_band_powers(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute power in low, mid, and high frequency bands.

        Args:
            freqs: Frequency array
            power_spectrum: Power spectrum

        Returns:
            Tuple of (power_low, power_mid, power_high)
        """
        # Create masks for this specific frequency array
        low_mask = freqs < self.low_freq_cutoff
        mid_mask = (freqs >= self.low_freq_cutoff) & (freqs < self.high_freq_cutoff)
        high_mask = freqs >= self.high_freq_cutoff

        # Exclude DC component from low band
        if len(low_mask) > 0:
            low_mask[0] = False

        power_low = float(np.sum(power_spectrum[low_mask])) if np.any(low_mask) else 0.0
        power_mid = float(np.sum(power_spectrum[mid_mask])) if np.any(mid_mask) else 0.0
        power_high = float(np.sum(power_spectrum[high_mask])) if np.any(high_mask) else 0.0

        return power_low, power_mid, power_high

    def compute_spectral_slope(
        self,
        freqs: np.ndarray,
        power_spectrum: np.ndarray
    ) -> float:
        """
        Compute the slope of the log-log spectrum (spectral exponent).

        A slope of -2 indicates Brownian motion (random walk).
        Steeper slopes indicate more persistence (trending).
        Flatter slopes indicate anti-persistence (mean-reverting).

        Args:
            freqs: Frequency array
            power_spectrum: Power spectrum

        Returns:
            Spectral slope (beta)
        """
        # Remove DC and zero power values
        mask = (freqs > 0) & (power_spectrum > 1e-10)

        if np.sum(mask) < 3:
            return -2.0  # Default to random walk

        log_freq = np.log(freqs[mask])
        log_power = np.log(power_spectrum[mask])

        # Linear regression in log-log space
        try:
            slope, _ = np.polyfit(log_freq, log_power, 1)
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return -2.0

    def compute_autocorrelation(
        self,
        returns: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Compute autocorrelation at specified lag.

        Args:
            returns: Return series
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation coefficient
        """
        if len(returns) <= lag:
            return 0.0

        r1 = returns[:-lag]
        r2 = returns[lag:]

        # Pearson correlation
        mean1, mean2 = np.mean(r1), np.mean(r2)
        std1, std2 = np.std(r1), np.std(r2)

        if std1 < 1e-10 or std2 < 1e-10:
            return 0.0

        correlation = np.mean((r1 - mean1) * (r2 - mean2)) / (std1 * std2)

        return float(np.clip(correlation, -1.0, 1.0))

    def compute_features(
        self,
        returns: Union[np.ndarray, 'pd.Series'],
        min_periods: Optional[int] = None
    ) -> SpectralFeatures:
        """
        Compute all spectral features using rolling windows.

        Features are computed using ONLY historical data at each point.
        The first (window_size - 1) values will be NaN.

        Args:
            returns: Return series (1D array or pandas Series)
            min_periods: Minimum periods required (default: window_size)

        Returns:
            SpectralFeatures object with all computed features
        """
        # Convert pandas to numpy if needed
        if hasattr(returns, 'values'):
            returns = returns.values

        returns = np.asarray(returns, dtype=np.float64).flatten()
        n = len(returns)

        if min_periods is None:
            min_periods = self.window_size

        # Initialize output arrays with NaN
        spectral_entropy = np.full(n, np.nan)
        dominant_frequency = np.full(n, np.nan)
        power_low = np.full(n, np.nan)
        power_mid = np.full(n, np.nan)
        power_high = np.full(n, np.nan)
        power_ratio = np.full(n, np.nan)
        autocorrelation_lag1 = np.full(n, np.nan)
        spectral_slope = np.full(n, np.nan)

        # Rolling window computation
        for i in range(min_periods - 1, n):
            # Extract window: [i - window_size + 1, i] inclusive
            start_idx = max(0, i - self.window_size + 1)
            window = returns[start_idx:i + 1]

            if len(window) < min_periods:
                continue

            # Compute power spectrum
            try:
                freqs, power = self.compute_power_spectrum(window, normalize=True)

                # Compute features
                spectral_entropy[i] = self.compute_spectral_entropy(power)
                dominant_frequency[i] = self.compute_dominant_frequency(freqs, power)

                pl, pm, ph = self.compute_band_powers(freqs, power)
                power_low[i] = pl
                power_mid[i] = pm
                power_high[i] = ph

                # Signal-to-noise ratio
                if ph > 1e-10:
                    power_ratio[i] = (pl + pm) / ph
                else:
                    power_ratio[i] = 10.0  # High ratio when no noise

                autocorrelation_lag1[i] = self.compute_autocorrelation(window, lag=1)
                spectral_slope[i] = self.compute_spectral_slope(freqs, power)

            except Exception as e:
                logger.debug(f"Spectral computation failed at index {i}: {e}")
                continue

        return SpectralFeatures(
            spectral_entropy=spectral_entropy,
            dominant_frequency=dominant_frequency,
            power_low=power_low,
            power_mid=power_mid,
            power_high=power_high,
            power_ratio=power_ratio,
            autocorrelation_lag1=autocorrelation_lag1,
            spectral_slope=spectral_slope,
        )

    def compute_single_window(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral features for a single window (no rolling).

        Useful for real-time analysis or when you have a fixed window.

        Args:
            returns: Return series (should be window_size length)

        Returns:
            Dictionary of spectral features
        """
        if len(returns) < 2:
            return {name: 0.0 for name in SpectralFeatures.feature_names()}

        # Compute power spectrum
        freqs, power = self.compute_power_spectrum(returns, normalize=True)

        # Compute features
        pl, pm, ph = self.compute_band_powers(freqs, power)

        return {
            'spectral_entropy': self.compute_spectral_entropy(power),
            'dominant_frequency': self.compute_dominant_frequency(freqs, power),
            'power_low': pl,
            'power_mid': pm,
            'power_high': ph,
            'power_ratio': (pl + pm) / max(ph, 1e-10),
            'autocorrelation_lag1': self.compute_autocorrelation(returns, lag=1),
            'spectral_slope': self.compute_spectral_slope(freqs, power),
        }


def get_spectral_analyzer(
    window_size: int = 64,
    **kwargs
) -> SpectralAnalyzer:
    """Factory function to create SpectralAnalyzer with common defaults."""
    return SpectralAnalyzer(window_size=window_size, **kwargs)


# Convenience functions for quick analysis
def compute_spectral_entropy(returns: np.ndarray, window_size: int = 64) -> float:
    """Compute spectral entropy for a single window of returns."""
    analyzer = SpectralAnalyzer(window_size=window_size)
    _, power = analyzer.compute_power_spectrum(returns[-window_size:], normalize=True)
    return analyzer.compute_spectral_entropy(power)


def compute_dominant_cycle(returns: np.ndarray, window_size: int = 64) -> float:
    """Find dominant cycle period in trading days."""
    analyzer = SpectralAnalyzer(window_size=window_size)
    freqs, power = analyzer.compute_power_spectrum(returns[-window_size:], normalize=True)
    dominant_freq = analyzer.compute_dominant_frequency(freqs, power)

    if dominant_freq > 0:
        return 1.0 / dominant_freq  # Convert frequency to period
    return float('inf')


if __name__ == "__main__":
    """Demo of spectral analysis capabilities."""
    print("=" * 60)
    print("SPECTRAL ANALYZER DEMO")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic returns with different characteristics
    n = 500

    # 1. Random walk (white noise returns)
    random_returns = np.random.randn(n) * 0.01

    # 2. Trending (low frequency dominant)
    trend = np.cumsum(np.random.randn(n) * 0.001)
    trending_returns = np.diff(np.concatenate([[0], trend]))

    # 3. Cyclical (specific frequency)
    t = np.arange(n)
    cycle = 0.005 * np.sin(2 * np.pi * t / 20)  # 20-day cycle
    cyclical_returns = cycle + np.random.randn(n) * 0.002

    # Create analyzer
    analyzer = SpectralAnalyzer(window_size=64)

    print("\nAnalyzing different market regimes:")
    print("-" * 40)

    for name, returns in [
        ("Random (Efficient)", random_returns),
        ("Trending", trending_returns),
        ("Cyclical (20-day)", cyclical_returns),
    ]:
        features = analyzer.compute_single_window(returns[-64:])
        print(f"\n{name}:")
        print(f"  Spectral Entropy: {features['spectral_entropy']:.3f}")
        print(f"  Dominant Freq: {features['dominant_frequency']:.4f} cpd")
        print(f"  Power Ratio (S/N): {features['power_ratio']:.2f}")
        print(f"  Spectral Slope: {features['spectral_slope']:.2f}")
        print(f"  Autocorr(1): {features['autocorrelation_lag1']:.3f}")

    print("\n" + "=" * 60)
    print("Rolling spectral features:")
    rolling_features = analyzer.compute_features(random_returns)
    valid_entropy = rolling_features.spectral_entropy[~np.isnan(rolling_features.spectral_entropy)]
    print(f"  Mean entropy over time: {np.mean(valid_entropy):.3f}")
    print(f"  Entropy std: {np.std(valid_entropy):.3f}")
    print("=" * 60)
