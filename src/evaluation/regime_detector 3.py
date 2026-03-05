"""
Market Regime Detection

Implements multiple regime detection methods:
- Hidden Markov Model (HMM) for regime classification
- K-means clustering on volatility
- Rolling volatility-based detection

Used for:
- Conditional performance analysis
- Regime-aware trading strategies
- Risk management during different market conditions

References:
    - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
      Nonstationary Time Series." Econometrica.
    - Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates." NBER.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import datetime as dt
from scipy import stats
import warnings

# Optional YAML for stress-window config
try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:  # pragma: no cover
    HAS_YAML = False

warnings.filterwarnings('ignore')

# Try to import hmmlearn - optional dependency
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# Try to import sklearn for clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RegimeLabel(Enum):
    """Market regime labels"""
    LOW_VOL = 0
    NORMAL = 1
    HIGH_VOL = 2


@dataclass
class RegimeState:
    """Current regime state with metadata"""
    regime: RegimeLabel
    probability: float  # Probability of current regime
    regime_probabilities: Dict[str, float]  # All regime probabilities
    volatility: float  # Current volatility estimate
    transition_prob: float  # Probability of regime change


@dataclass
class RegimeHistory:
    """History of regime classifications"""
    timestamps: List
    regimes: np.ndarray
    probabilities: np.ndarray  # Shape: (n_samples, n_regimes)
    volatility: np.ndarray
    transitions: List[int]  # Indices where regime changed


# ---------------------------------------------------------------------------
# Stress windows (config-driven)
# ---------------------------------------------------------------------------

@dataclass
class StressWindow:
    name: str
    start: dt.date
    end: dt.date
    type: str = "crisis"


def load_stress_windows(config_path: Union[str, Path]) -> List[StressWindow]:
    """Load stress windows from YAML; returns empty list on failure."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Stress window config not found: {path}")
        return []
    if not HAS_YAML:
        logger.warning("PyYAML not installed; cannot load stress_windows.yaml")
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
        windows: List[StressWindow] = []
        for item in data.get("windows", []):
            try:
                windows.append(
                    StressWindow(
                        name=item["name"],
                        start=dt.date.fromisoformat(item["start"]),
                        end=dt.date.fromisoformat(item["end"]),
                        type=item.get("type", "crisis"),
                    )
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Skipping stress window entry {item}: {exc}")
        return windows
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to parse stress windows from {path}: {exc}")
        return []


def label_stress_windows(dates: pd.DatetimeIndex, windows: List[StressWindow]) -> List[Optional[str]]:
    """Return a label for each date indicating which stress window (if any)."""
    labels: List[Optional[str]] = []
    for d in dates:
        label = None
        for w in windows:
            if w.start <= d.date() <= w.end:
                label = w.name
                break
        labels.append(label)
    return labels


class HMMRegimeDetector:
    """
    Hidden Markov Model Regime Detector

    Uses a 3-state Gaussian HMM to classify market regimes:
    - State 0: Low volatility (calm markets)
    - State 1: Normal volatility
    - State 2: High volatility (stress/crisis)

    The HMM captures:
    - Different volatility levels per regime
    - Regime persistence (diagonal of transition matrix)
    - Probabilistic classification
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 100,
        covariance_type: str = 'diag',
        random_state: int = 42,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            n_iter: Maximum EM iterations
            covariance_type: Type of covariance ('diag', 'full', 'spherical')
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.model = None
        self.is_fitted = False
        self._regime_mapping: Dict[int, RegimeLabel] = {}

        if not HAS_HMMLEARN:
            logger.warning(
                "hmmlearn not installed. HMM regime detection unavailable. "
                "Install with: pip install hmmlearn"
            )

    def _prepare_features(
        self,
        returns: Union[np.ndarray, pd.Series],
        include_abs_returns: bool = True,
    ) -> np.ndarray:
        """
        Prepare features for HMM.

        Args:
            returns: Return series
            include_abs_returns: Include absolute returns as feature

        Returns:
            Feature matrix (n_samples, n_features)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if include_abs_returns:
            # Use both returns and absolute returns
            features = np.column_stack([
                returns,
                np.abs(returns)
            ])
        else:
            features = returns.reshape(-1, 1)

        return features

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        verbose: bool = False
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM to return series.

        Args:
            returns: Historical return series
            verbose: Print fitting progress

        Returns:
            Self for method chaining
        """
        if not HAS_HMMLEARN:
            logger.error("hmmlearn not installed")
            return self

        features = self._prepare_features(returns)

        if len(features) < 50:
            logger.warning("Insufficient data for HMM fitting")
            return self

        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # Fit model
        try:
            self.model.fit(features)
            self.is_fitted = True

            # Map states to volatility levels
            self._map_regimes_to_labels(features)

            if verbose:
                logger.info(
                    f"HMM fitted successfully. "
                    f"Converged: {self.model.monitor_.converged}"
                )

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            self.is_fitted = False

        return self

    def _map_regimes_to_labels(self, features: np.ndarray):
        """
        Map HMM states to meaningful regime labels.

        States are ordered by mean absolute return (proxy for volatility).
        """
        if not self.is_fitted:
            return

        # Get mean absolute return for each state
        states = self.model.predict(features)
        state_vols = {}

        for state in range(self.n_regimes):
            mask = states == state
            if np.sum(mask) > 0:
                state_vols[state] = np.mean(np.abs(features[mask, 0]))
            else:
                state_vols[state] = 0.0

        # Sort states by volatility
        sorted_states = sorted(state_vols.items(), key=lambda x: x[1])

        # Map to labels (LOW_VOL = lowest vol state, HIGH_VOL = highest)
        labels = [RegimeLabel.LOW_VOL, RegimeLabel.NORMAL, RegimeLabel.HIGH_VOL]
        for i, (state, _) in enumerate(sorted_states):
            if i < len(labels):
                self._regime_mapping[state] = labels[i]

        logger.debug(f"Regime mapping: {self._regime_mapping}")

    def predict(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Predict regime for each observation.

        Args:
            returns: Return series

        Returns:
            Array of regime labels (0, 1, 2)
        """
        if not self.is_fitted:
            logger.warning("HMM not fitted, returning default regimes")
            n = len(returns) if hasattr(returns, '__len__') else 1
            return np.ones(n, dtype=int)  # Default to NORMAL

        features = self._prepare_features(returns)
        states = self.model.predict(features)

        # Map to regime labels
        regimes = np.array([
            self._regime_mapping.get(s, RegimeLabel.NORMAL).value
            for s in states
        ])

        return regimes

    def predict_proba(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Get probability distribution over regimes.

        Args:
            returns: Return series

        Returns:
            Array of probabilities (n_samples, n_regimes)
        """
        if not self.is_fitted:
            n = len(returns) if hasattr(returns, '__len__') else 1
            probs = np.zeros((n, self.n_regimes))
            probs[:, 1] = 1.0  # Default to NORMAL
            return probs

        features = self._prepare_features(returns)
        probs = self.model.predict_proba(features)

        # Reorder columns to match regime labels
        reordered = np.zeros_like(probs)
        for state, label in self._regime_mapping.items():
            reordered[:, label.value] = probs[:, state]

        return reordered

    def get_current_state(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> RegimeState:
        """
        Get current regime state with full information.

        Args:
            returns: Recent return series

        Returns:
            RegimeState object
        """
        regimes = self.predict(returns)
        probs = self.predict_proba(returns)

        current_regime = RegimeLabel(regimes[-1])
        current_probs = probs[-1]

        # Calculate volatility
        if isinstance(returns, pd.Series):
            returns = returns.values
        volatility = np.std(returns[-21:]) * np.sqrt(252) if len(returns) >= 21 else 0.15

        # Calculate transition probability from transition matrix
        if self.is_fitted and self.model is not None:
            trans_matrix = self.model.transmat_
            current_state = self.model.predict(self._prepare_features(returns))[-1]
            transition_prob = 1 - trans_matrix[current_state, current_state]
        else:
            transition_prob = 0.1

        return RegimeState(
            regime=current_regime,
            probability=current_probs[current_regime.value],
            regime_probabilities={
                'low_vol': current_probs[0],
                'normal': current_probs[1],
                'high_vol': current_probs[2],
            },
            volatility=volatility,
            transition_prob=transition_prob,
        )

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Get the regime transition probability matrix."""
        if not self.is_fitted or self.model is None:
            return None

        # Reorder transition matrix to match regime labels
        trans = self.model.transmat_.copy()
        n = self.n_regimes
        reordered = np.zeros((n, n))

        for from_state, from_label in self._regime_mapping.items():
            for to_state, to_label in self._regime_mapping.items():
                reordered[from_label.value, to_label.value] = trans[from_state, to_state]

        return reordered


class VolatilityClusterDetector:
    """
    K-means Clustering on Volatility

    Uses rolling volatility and K-means clustering to detect regimes.
    Simpler than HMM but can be effective for regime detection.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        vol_window: int = 21,
        random_state: int = 42,
    ):
        """
        Initialize volatility cluster detector.

        Args:
            n_clusters: Number of volatility clusters
            vol_window: Window for rolling volatility
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.vol_window = vol_window
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._cluster_to_regime: Dict[int, RegimeLabel] = {}

        if not HAS_SKLEARN:
            logger.warning(
                "sklearn not installed. Volatility clustering unavailable. "
                "Install with: pip install scikit-learn"
            )

    def _compute_features(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Compute volatility-based features.

        Args:
            returns: Return series

        Returns:
            Feature matrix
        """
        if isinstance(returns, pd.Series):
            series = returns
        else:
            series = pd.Series(returns)

        # Rolling volatility
        rolling_vol = series.rolling(self.vol_window).std() * np.sqrt(252)

        # Rolling absolute return
        rolling_abs = series.abs().rolling(self.vol_window).mean() * np.sqrt(252)

        # Combine features
        features = pd.DataFrame({
            'rolling_vol': rolling_vol,
            'rolling_abs': rolling_abs,
        }).dropna()

        return features.values

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> 'VolatilityClusterDetector':
        """
        Fit clustering model.

        Args:
            returns: Historical return series

        Returns:
            Self for method chaining
        """
        if not HAS_SKLEARN:
            logger.error("sklearn not installed")
            return self

        features = self._compute_features(returns)

        if len(features) < 50:
            logger.warning("Insufficient data for clustering")
            return self

        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        # Fit K-means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(scaled_features)
        self.is_fitted = True

        # Map clusters to regime labels
        self._map_clusters_to_regimes(features)

        logger.info(f"Volatility clustering fitted with {self.n_clusters} clusters")

        return self

    def _map_clusters_to_regimes(self, features: np.ndarray):
        """Map clusters to regime labels based on volatility levels."""
        if self.scaler is None or self.model is None:
            return

        scaled_features = self.scaler.transform(features)
        clusters = self.model.predict(scaled_features)

        # Get mean volatility for each cluster
        cluster_vols = {}
        for cluster in range(self.n_clusters):
            mask = clusters == cluster
            if np.sum(mask) > 0:
                cluster_vols[cluster] = np.mean(features[mask, 0])
            else:
                cluster_vols[cluster] = 0.0

        # Sort and map
        sorted_clusters = sorted(cluster_vols.items(), key=lambda x: x[1])
        labels = [RegimeLabel.LOW_VOL, RegimeLabel.NORMAL, RegimeLabel.HIGH_VOL]

        for i, (cluster, _) in enumerate(sorted_clusters):
            if i < len(labels):
                self._cluster_to_regime[cluster] = labels[i]

    def predict(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Predict regime for each observation.

        Args:
            returns: Return series

        Returns:
            Array of regime labels
        """
        if not self.is_fitted:
            n = len(returns) if hasattr(returns, '__len__') else 1
            return np.ones(n, dtype=int)

        features = self._compute_features(returns)
        scaled_features = self.scaler.transform(features)
        clusters = self.model.predict(scaled_features)

        # Map to regimes
        regimes = np.array([
            self._cluster_to_regime.get(c, RegimeLabel.NORMAL).value
            for c in clusters
        ])

        # Pad beginning (where rolling window wasn't available)
        n_padding = len(returns) - len(regimes) if hasattr(returns, '__len__') else 0
        if n_padding > 0:
            regimes = np.concatenate([
                np.ones(n_padding, dtype=int),  # Default to NORMAL
                regimes
            ])

        return regimes


class RollingVolatilityDetector:
    """
    Simple Rolling Volatility-Based Regime Detector

    Uses percentile thresholds on rolling volatility.
    No ML required - purely rule-based.
    """

    def __init__(
        self,
        vol_window: int = 21,
        low_vol_percentile: float = 0.25,
        high_vol_percentile: float = 0.75,
    ):
        """
        Initialize rolling volatility detector.

        Args:
            vol_window: Window for rolling volatility
            low_vol_percentile: Percentile below which is LOW_VOL
            high_vol_percentile: Percentile above which is HIGH_VOL
        """
        self.vol_window = vol_window
        self.low_vol_percentile = low_vol_percentile
        self.high_vol_percentile = high_vol_percentile

        self._low_threshold: Optional[float] = None
        self._high_threshold: Optional[float] = None
        self.is_fitted = False

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> 'RollingVolatilityDetector':
        """
        Calibrate thresholds from historical data.

        Args:
            returns: Historical return series

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            series = returns
        else:
            series = pd.Series(returns)

        rolling_vol = series.rolling(self.vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        self._low_threshold = np.percentile(rolling_vol, self.low_vol_percentile * 100)
        self._high_threshold = np.percentile(rolling_vol, self.high_vol_percentile * 100)
        self.is_fitted = True

        logger.info(
            f"Volatility thresholds: "
            f"low<{self._low_threshold:.2%}, high>{self._high_threshold:.2%}"
        )

        return self

    def predict(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Predict regime for each observation.

        Args:
            returns: Return series

        Returns:
            Array of regime labels
        """
        if isinstance(returns, pd.Series):
            series = returns
        else:
            series = pd.Series(returns)

        rolling_vol = series.rolling(self.vol_window).std() * np.sqrt(252)

        # Use defaults if not fitted
        low_thresh = self._low_threshold or 0.10
        high_thresh = self._high_threshold or 0.25

        regimes = np.ones(len(series), dtype=int)  # Default to NORMAL

        for i, vol in enumerate(rolling_vol):
            if pd.isna(vol):
                continue
            if vol < low_thresh:
                regimes[i] = RegimeLabel.LOW_VOL.value
            elif vol > high_thresh:
                regimes[i] = RegimeLabel.HIGH_VOL.value

        return regimes

    def get_current_volatility(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """Get current rolling volatility."""
        if isinstance(returns, pd.Series):
            series = returns
        else:
            series = pd.Series(returns)

        rolling_vol = series.rolling(self.vol_window).std() * np.sqrt(252)
        return float(rolling_vol.iloc[-1]) if not pd.isna(rolling_vol.iloc[-1]) else 0.15


class SpectralHMMRegimeDetector(HMMRegimeDetector):
    """
    Enhanced HMM Regime Detector with Spectral Features

    Extends the base HMM detector with frequency-domain features:
    - Spectral entropy (measure of randomness)
    - Dominant frequency (cyclical patterns)
    - Power ratio (signal-to-noise)

    This captures regime characteristics that are not visible in
    time-domain features alone:
    - Low volatility regimes often have higher spectral entropy
    - Crisis regimes may show different dominant frequencies
    - Trending markets have lower spectral entropy

    References:
        - Granger, C.W.J. (1966). "The Typical Spectral Shape of an Economic Variable."
    """

    def __init__(
        self,
        n_regimes: int = 3,
        spectral_window: int = 64,
        n_iter: int = 100,
        covariance_type: str = 'diag',
        random_state: int = 42,
        use_spectral_features: bool = True,
    ):
        """
        Initialize Spectral HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            spectral_window: Window size for spectral feature computation
            n_iter: Maximum EM iterations
            covariance_type: Type of covariance ('diag', 'full', 'spherical')
            random_state: Random seed for reproducibility
            use_spectral_features: Whether to include spectral features
        """
        super().__init__(
            n_regimes=n_regimes,
            n_iter=n_iter,
            covariance_type=covariance_type,
            random_state=random_state,
        )

        self.spectral_window = spectral_window
        self.use_spectral_features = use_spectral_features
        self._spectral_analyzer = None

        if use_spectral_features:
            try:
                from ..data.spectral_analyzer import SpectralAnalyzer
                self._spectral_analyzer = SpectralAnalyzer(
                    window_size=spectral_window
                )
                logger.info(
                    f"SpectralHMMRegimeDetector initialized with "
                    f"spectral_window={spectral_window}"
                )
            except ImportError:
                logger.warning(
                    "SpectralAnalyzer not available. "
                    "Falling back to standard HMM features."
                )
                self.use_spectral_features = False

    def _prepare_features(
        self,
        returns: Union[np.ndarray, pd.Series],
        include_abs_returns: bool = True,
    ) -> np.ndarray:
        """
        Prepare features for HMM including spectral features.

        Observation vector includes:
        - returns (captures mean differences across regimes)
        - |returns| (captures volatility differences)
        - spectral_entropy (captures randomness)
        - power_ratio (captures signal-to-noise)

        Args:
            returns: Return series
            include_abs_returns: Include absolute returns as feature

        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Get base features
        base_features = super()._prepare_features(returns, include_abs_returns)

        if not self.use_spectral_features or self._spectral_analyzer is None:
            return base_features

        # Compute spectral features
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = returns

        returns_array = returns_array[~np.isnan(returns_array)]

        try:
            spectral_feats = self._spectral_analyzer.compute_features(returns_array)

            # Extract key spectral features (avoiding NaN rows)
            spectral_entropy = spectral_feats.spectral_entropy
            power_ratio = spectral_feats.power_ratio
            autocorr = spectral_feats.autocorrelation_lag1

            # Find valid indices (where spectral features are not NaN)
            valid_mask = ~np.isnan(spectral_entropy)

            if np.sum(valid_mask) < 50:
                logger.warning(
                    "Insufficient valid spectral features. "
                    "Using base features only."
                )
                return base_features

            # Trim base features to match spectral features
            n_valid = np.sum(valid_mask)
            base_trimmed = base_features[-n_valid:]

            # Combine features
            # Normalize spectral features to similar scale
            spectral_entropy_valid = spectral_entropy[valid_mask]
            power_ratio_valid = np.clip(power_ratio[valid_mask], 0, 10)  # Clip outliers
            autocorr_valid = autocorr[valid_mask]

            # Stack features
            combined = np.column_stack([
                base_trimmed,
                spectral_entropy_valid.reshape(-1, 1),
                power_ratio_valid.reshape(-1, 1),
                autocorr_valid.reshape(-1, 1),
            ])

            logger.debug(
                f"Spectral HMM features: {combined.shape[1]} features, "
                f"{combined.shape[0]} samples"
            )

            return combined

        except Exception as e:
            logger.warning(f"Spectral feature computation failed: {e}")
            return base_features

    def get_spectral_regime_characteristics(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> Dict[int, Dict[str, float]]:
        """
        Get spectral characteristics for each regime.

        Returns:
            Dict mapping regime index to spectral statistics
        """
        if not self.is_fitted:
            logger.warning("Model not fitted")
            return {}

        regimes = self.predict(returns)

        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        # Align with regimes (account for spectral window lag)
        if len(returns) != len(regimes):
            min_len = min(len(returns), len(regimes))
            returns = returns[-min_len:]
            regimes = regimes[-min_len:]

        characteristics = {}

        for regime_idx in range(self.n_regimes):
            mask = regimes == regime_idx

            if np.sum(mask) < self.spectral_window:
                characteristics[regime_idx] = {
                    'mean_return': 0.0,
                    'volatility': 0.0,
                    'spectral_entropy': 0.5,
                    'dominant_frequency': 0.0,
                    'power_ratio': 1.0,
                }
                continue

            regime_returns = returns[mask]

            # Compute spectral features for this regime
            if self._spectral_analyzer is not None and len(regime_returns) >= self.spectral_window:
                feats = self._spectral_analyzer.compute_single_window(
                    regime_returns[-self.spectral_window:]
                )
                spectral_entropy = feats.get('spectral_entropy', 0.5)
                dominant_freq = feats.get('dominant_frequency', 0.0)
                power_ratio = feats.get('power_ratio', 1.0)
            else:
                spectral_entropy = 0.5
                dominant_freq = 0.0
                power_ratio = 1.0

            characteristics[regime_idx] = {
                'mean_return': float(np.mean(regime_returns)),
                'volatility': float(np.std(regime_returns) * np.sqrt(252)),
                'spectral_entropy': spectral_entropy,
                'dominant_frequency': dominant_freq,
                'power_ratio': power_ratio,
                'sample_count': int(np.sum(mask)),
            }

        return characteristics


def get_regime_detector(
    method: str = 'hmm',
    **kwargs
) -> Union[HMMRegimeDetector, SpectralHMMRegimeDetector, VolatilityClusterDetector, RollingVolatilityDetector]:
    """
    Factory function to get appropriate regime detector.

    Args:
        method: Detection method ('hmm', 'spectral_hmm', 'kmeans', 'rolling')
        **kwargs: Passed to detector constructor

    Returns:
        Regime detector instance
    """
    if method == 'hmm':
        if not HAS_HMMLEARN:
            logger.warning("HMM unavailable, falling back to rolling volatility")
            return RollingVolatilityDetector(**kwargs)
        return HMMRegimeDetector(**kwargs)
    elif method == 'spectral_hmm' or method == 'spectral':
        if not HAS_HMMLEARN:
            logger.warning("HMM unavailable, falling back to rolling volatility")
            return RollingVolatilityDetector(**kwargs)
        return SpectralHMMRegimeDetector(**kwargs)
    elif method == 'kmeans':
        if not HAS_SKLEARN:
            logger.warning("K-means unavailable, falling back to rolling volatility")
            return RollingVolatilityDetector(**kwargs)
        return VolatilityClusterDetector(**kwargs)
    else:
        return RollingVolatilityDetector(**kwargs)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Regime Detection Demo")
    print("=" * 60)

    # Generate synthetic data with regime changes
    np.random.seed(42)

    # Simulate returns with different volatility regimes
    low_vol = np.random.randn(100) * 0.01   # 10% annualized
    normal_vol = np.random.randn(100) * 0.015  # 15% annualized
    high_vol = np.random.randn(100) * 0.03   # 30% annualized

    returns = np.concatenate([low_vol, normal_vol, high_vol, normal_vol, low_vol])
    print(f"\nGenerated {len(returns)} days with known regime changes")

    # Test each detector
    print("\n" + "-" * 40)
    print("Testing Rolling Volatility Detector:")
    rolling_detector = RollingVolatilityDetector()
    rolling_detector.fit(returns)
    rolling_regimes = rolling_detector.predict(returns)
    print(f"  Regime distribution: LOW={np.sum(rolling_regimes==0)}, "
          f"NORMAL={np.sum(rolling_regimes==1)}, HIGH={np.sum(rolling_regimes==2)}")

    if HAS_SKLEARN:
        print("\n" + "-" * 40)
        print("Testing Volatility Cluster Detector:")
        cluster_detector = VolatilityClusterDetector()
        cluster_detector.fit(returns)
        cluster_regimes = cluster_detector.predict(returns)
        print(f"  Regime distribution: LOW={np.sum(cluster_regimes==0)}, "
              f"NORMAL={np.sum(cluster_regimes==1)}, HIGH={np.sum(cluster_regimes==2)}")

    if HAS_HMMLEARN:
        print("\n" + "-" * 40)
        print("Testing HMM Regime Detector:")
        hmm_detector = HMMRegimeDetector()
        hmm_detector.fit(returns, verbose=True)
        hmm_regimes = hmm_detector.predict(returns)
        print(f"  Regime distribution: LOW={np.sum(hmm_regimes==0)}, "
              f"NORMAL={np.sum(hmm_regimes==1)}, HIGH={np.sum(hmm_regimes==2)}")

        # Get current state
        current_state = hmm_detector.get_current_state(returns)
        print(f"\n  Current state: {current_state.regime.name}")
        print(f"  Probability: {current_state.probability:.2%}")
        print(f"  Transition prob: {current_state.transition_prob:.2%}")

        # Transition matrix
        trans = hmm_detector.get_transition_matrix()
        if trans is not None:
            print("\n  Transition matrix:")
            print("       LOW   NORM  HIGH")
            for i, label in enumerate(['LOW ', 'NORM', 'HIGH']):
                print(f"    {label} {trans[i, 0]:.2f}  {trans[i, 1]:.2f}  {trans[i, 2]:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
