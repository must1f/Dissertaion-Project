"""
Regime-Switching Monte Carlo Simulation Framework

This module implements a Markov regime-switching Monte Carlo model for asset returns,
providing a significant improvement over standard IID Monte Carlo simulation.

Mathematical Framework:
=======================

Standard Monte Carlo assumes:
    r_t ~ N(μ, σ²)  (IID returns with constant parameters)

Regime-Switching Monte Carlo assumes:
    r_t | S_t = k ~ N(μ_k, σ_k²)  (returns conditional on regime k)
    P(S_{t+1} = j | S_t = i) = π_{ij}  (Markov transition probabilities)

Where:
    - S_t ∈ {0, 1, 2, ...} is the latent regime state
    - μ_k, σ_k are regime-specific return parameters
    - π_{ij} is the transition probability from regime i to j

Why Standard MC Underestimates Tail Risk:
=========================================
1. Volatility clustering: Real markets exhibit persistent high/low vol periods
2. Regime persistence: States are "sticky" (diagonal of transition matrix > 0.5)
3. Fat tails: Regime mixing naturally produces excess kurtosis
4. Non-Gaussianity: Mixture of normals has heavier tails than a single normal

Key Features:
- Gaussian Mixture Model (GMM) for regime identification
- Hidden Markov Model (HMM) for transition estimation
- GBM simulation within each regime
- Optional Ornstein-Uhlenbeck dynamics for mean-reversion
- Comprehensive risk metrics comparison

Author: Dissertation Research Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Scientific computing
from scipy import stats
from scipy.special import logsumexp

# Optional dependencies with fallback
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HAS_GMM = True
except ImportError:
    HAS_GMM = False

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS, DAILY_TIME_STEP

logger = get_logger(__name__)


# =============================================================================
# Data Classes for Configuration and Results
# =============================================================================

class RegimeType(Enum):
    """Market regime labels with financial interpretation"""
    LOW_VOL = 0      # Bull market, calm conditions
    NORMAL = 1       # Typical market conditions
    HIGH_VOL = 2     # Bear market, crisis, stress


@dataclass
class RegimeParameters:
    """
    Parameters for each market regime

    Attributes:
        mean_return: Expected daily return (annualized drift / 252)
        volatility: Daily volatility (annualized vol / sqrt(252))
        mean_reversion: OU mean reversion speed (optional, for mean-reverting dynamics)
        long_term_mean: Long-term mean for OU process

    Financial Interpretation:
        - LOW_VOL: μ ≈ 10-15% ann., σ ≈ 10-12% ann. (bull market)
        - NORMAL: μ ≈ 5-10% ann., σ ≈ 15-18% ann. (typical)
        - HIGH_VOL: μ ≈ -10-20% ann., σ ≈ 30-50% ann. (crisis)
    """
    mean_return: float           # Daily mean return
    volatility: float            # Daily volatility
    mean_reversion: float = 0.0  # OU theta (0 = pure GBM)
    long_term_mean: float = 0.0  # OU mu

    @property
    def annualized_return(self) -> float:
        """Convert daily return to annualized"""
        return self.mean_return * TRADING_DAYS_PER_YEAR

    @property
    def annualized_volatility(self) -> float:
        """Convert daily vol to annualized"""
        return self.volatility * SQRT_TRADING_DAYS


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    n_paths: int = 10000                    # Number of simulated paths
    horizon: int = TRADING_DAYS_PER_YEAR    # Simulation horizon (trading days)
    dt: float = DAILY_TIME_STEP             # Time step (1 trading day)
    seed: int = 42                          # Random seed for reproducibility
    use_antithetic: bool = True             # Variance reduction via antithetic variates
    use_gbm: bool = True                    # Use GBM dynamics (vs simple returns)
    use_ou: bool = False                    # Use OU mean-reversion within regimes


@dataclass
class RegimeEstimates:
    """
    Estimated regime parameters from historical data

    Attributes:
        regime_params: Dict mapping regime index to RegimeParameters
        transition_matrix: n_regimes x n_regimes transition probability matrix
        stationary_dist: Stationary distribution over regimes
        regime_labels: Time series of regime labels
        regime_probabilities: Time series of regime probabilities
        log_likelihood: Model log-likelihood (for model selection)
        bic: Bayesian Information Criterion
        n_regimes: Number of regimes
    """
    regime_params: Dict[int, RegimeParameters]
    transition_matrix: np.ndarray
    stationary_dist: np.ndarray
    regime_labels: np.ndarray
    regime_probabilities: np.ndarray
    log_likelihood: float
    bic: float
    n_regimes: int

    def get_current_regime(self) -> int:
        """Get the most recent regime"""
        return int(self.regime_labels[-1])

    def get_regime_durations(self) -> Dict[int, List[int]]:
        """
        Compute duration statistics for each regime

        Returns:
            Dict mapping regime to list of consecutive durations
        """
        durations = {i: [] for i in range(self.n_regimes)}

        if len(self.regime_labels) == 0:
            return durations

        current_regime = self.regime_labels[0]
        current_duration = 1

        for label in self.regime_labels[1:]:
            if label == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = label
                current_duration = 1

        # Add final duration
        durations[current_regime].append(current_duration)

        return durations


@dataclass
class SimulationResult:
    """
    Results from Monte Carlo simulation

    Attributes:
        paths: Simulated price paths [n_paths, horizon+1]
        returns: Simulated returns [n_paths, horizon]
        terminal_prices: Terminal prices [n_paths]
        terminal_returns: Log returns over full horizon [n_paths]
        regime_paths: Regime evolution for each path [n_paths, horizon] (if regime-switching)
        config: Simulation configuration
        initial_price: Starting price
    """
    paths: np.ndarray
    returns: np.ndarray
    terminal_prices: np.ndarray
    terminal_returns: np.ndarray
    regime_paths: Optional[np.ndarray]
    config: SimulationConfig
    initial_price: float

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def horizon(self) -> int:
        return self.returns.shape[1]

    def get_path_statistics(self) -> Dict[str, float]:
        """Compute summary statistics across all paths"""
        return {
            'mean_terminal_price': float(np.mean(self.terminal_prices)),
            'std_terminal_price': float(np.std(self.terminal_prices)),
            'mean_terminal_return': float(np.mean(self.terminal_returns)),
            'std_terminal_return': float(np.std(self.terminal_returns)),
            'min_terminal_price': float(np.min(self.terminal_prices)),
            'max_terminal_price': float(np.max(self.terminal_prices)),
            'median_terminal_price': float(np.median(self.terminal_prices)),
        }


# =============================================================================
# Regime Identification
# =============================================================================

class GMMRegimeIdentifier:
    """
    Gaussian Mixture Model for Regime Identification

    Why GMM is appropriate:
    1. Non-parametric: Doesn't assume specific regime count a priori
    2. Probabilistic: Provides soft regime assignments
    3. Captures volatility clustering: High vol returns cluster together
    4. Computationally efficient: EM algorithm converges quickly

    Alternatives considered:
    - HMM: Better for sequential data but more complex
    - K-means: Deterministic, doesn't capture transition dynamics
    - Threshold-based: Simple but arbitrary cutoffs
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_init: int = 10,
        max_iter: int = 200,
        random_state: int = 42,
        covariance_type: str = 'full'
    ):
        """
        Initialize GMM regime identifier

        Args:
            n_regimes: Number of regimes to identify
            n_init: Number of EM initializations (best is kept)
            max_iter: Maximum EM iterations per initialization
            random_state: Random seed
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
        """
        if not HAS_GMM:
            raise ImportError(
                "scikit-learn required for GMM. Install: pip install scikit-learn"
            )

        self.n_regimes = n_regimes
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.covariance_type = covariance_type

        self.gmm = None
        self.scaler = None
        self._regime_order = None  # Maps GMM component to volatility-ordered regime

    def _prepare_features(self, returns: np.ndarray) -> np.ndarray:
        """
        Prepare features for GMM fitting

        We use both returns and absolute returns as features:
        - Returns capture mean differences across regimes
        - |Returns| capture volatility differences (key discriminator)
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]

        # Feature 1: Raw returns (captures mean)
        # Feature 2: Absolute returns (captures volatility)
        features = np.column_stack([
            returns,
            np.abs(returns)
        ])

        return features

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> 'GMMRegimeIdentifier':
        """
        Fit GMM to return series

        Args:
            returns: Historical return series

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        features = self._prepare_features(returns)

        if len(features) < 50:
            logger.warning("Insufficient data for reliable regime identification")

        # Standardize features for numerical stability
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            covariance_type=self.covariance_type
        )
        self.gmm.fit(features_scaled)

        # Order regimes by volatility (low to high)
        self._order_regimes_by_volatility(features)

        logger.info(
            f"GMM fitted: {self.n_regimes} regimes, "
            f"converged={self.gmm.converged_}, "
            f"BIC={self.gmm.bic(features_scaled):.2f}"
        )

        return self

    def _order_regimes_by_volatility(self, features: np.ndarray):
        """
        Order GMM components from lowest to highest volatility

        This ensures consistent regime labeling:
        - Regime 0: Low volatility (bull market)
        - Regime 1: Normal volatility
        - Regime 2: High volatility (crisis)
        """
        features_scaled = self.scaler.transform(features)
        labels = self.gmm.predict(features_scaled)

        # Compute mean absolute return for each component
        component_vols = {}
        for k in range(self.n_regimes):
            mask = labels == k
            if np.sum(mask) > 0:
                # Use mean |return| as volatility proxy
                component_vols[k] = np.mean(np.abs(features[mask, 0]))
            else:
                component_vols[k] = 0.0

        # Sort components by volatility
        sorted_components = sorted(component_vols.items(), key=lambda x: x[1])

        # Create mapping: GMM component -> volatility-ordered regime
        self._regime_order = {
            component: regime_idx
            for regime_idx, (component, _) in enumerate(sorted_components)
        }

        logger.debug(f"Regime ordering: {self._regime_order}")

    def predict(self, returns: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict regime labels for return series

        Args:
            returns: Return series

        Returns:
            Array of regime labels (0=low vol, 1=normal, 2=high vol)
        """
        if self.gmm is None:
            raise ValueError("Must call fit() before predict()")

        if isinstance(returns, pd.Series):
            returns = returns.values

        features = self._prepare_features(returns)
        features_scaled = self.scaler.transform(features)

        # Get GMM predictions
        gmm_labels = self.gmm.predict(features_scaled)

        # Map to volatility-ordered regimes
        ordered_labels = np.array([
            self._regime_order.get(label, 1) for label in gmm_labels
        ])

        return ordered_labels

    def predict_proba(self, returns: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Get probability distribution over regimes

        Args:
            returns: Return series

        Returns:
            Array of probabilities [n_samples, n_regimes]
        """
        if self.gmm is None:
            raise ValueError("Must call fit() before predict_proba()")

        if isinstance(returns, pd.Series):
            returns = returns.values

        features = self._prepare_features(returns)
        features_scaled = self.scaler.transform(features)

        # Get GMM probabilities
        gmm_probs = self.gmm.predict_proba(features_scaled)

        # Reorder columns to match volatility ordering
        reordered = np.zeros_like(gmm_probs)
        for gmm_comp, ordered_regime in self._regime_order.items():
            reordered[:, ordered_regime] = gmm_probs[:, gmm_comp]

        return reordered


class TransitionMatrixEstimator:
    """
    Estimate Markov transition matrix from regime labels

    The transition matrix π where π[i,j] = P(S_{t+1}=j | S_t=i) captures:
    - Regime persistence (diagonal elements)
    - Regime switching probabilities (off-diagonal)

    Typical values for financial markets:
    - Diagonal ~ 0.95-0.99 (regimes are "sticky")
    - Off-diagonal ~ 0.01-0.05 (rare transitions)
    """

    def __init__(self, n_regimes: int = 3, add_smoothing: float = 1.0):
        """
        Args:
            n_regimes: Number of regimes
            add_smoothing: Additive smoothing (Laplace) to avoid zero probabilities
        """
        self.n_regimes = n_regimes
        self.add_smoothing = add_smoothing
        self.transition_matrix = None
        self.transition_counts = None

    def fit(self, regime_labels: np.ndarray) -> 'TransitionMatrixEstimator':
        """
        Estimate transition matrix from regime sequence

        Args:
            regime_labels: Array of regime labels [T]

        Returns:
            Self for method chaining
        """
        # Count transitions
        self.transition_counts = np.zeros((self.n_regimes, self.n_regimes))

        for t in range(len(regime_labels) - 1):
            i = int(regime_labels[t])
            j = int(regime_labels[t + 1])
            if 0 <= i < self.n_regimes and 0 <= j < self.n_regimes:
                self.transition_counts[i, j] += 1

        # Apply additive smoothing and normalize
        smoothed_counts = self.transition_counts + self.add_smoothing
        row_sums = smoothed_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = smoothed_counts / row_sums

        logger.info("Transition matrix estimated:")
        for i in range(self.n_regimes):
            logger.info(f"  State {i}: {self.transition_matrix[i]}")

        return self

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution π* where π* = π* @ P

        The stationary distribution tells us the long-run proportion
        of time spent in each regime.
        """
        if self.transition_matrix is None:
            raise ValueError("Must call fit() first")

        # Solve for left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1
        stationary = np.abs(stationary) / np.sum(np.abs(stationary))

        return stationary

    def expected_regime_duration(self) -> Dict[int, float]:
        """
        Compute expected duration in each regime

        If p_ii is the probability of staying in regime i,
        then expected duration = 1 / (1 - p_ii) days
        """
        if self.transition_matrix is None:
            raise ValueError("Must call fit() first")

        durations = {}
        for i in range(self.n_regimes):
            p_stay = self.transition_matrix[i, i]
            durations[i] = 1.0 / (1.0 - p_stay + 1e-10)

        return durations


# =============================================================================
# Monte Carlo Simulators
# =============================================================================

class StandardMC:
    """
    Standard (IID) Monte Carlo Simulation

    Assumes returns are independently and identically distributed:
        r_t ~ N(μ, σ²)

    This serves as a baseline to demonstrate the importance of
    regime-switching for accurate tail risk estimation.

    Limitations:
    - Ignores volatility clustering (GARCH effects)
    - Underestimates tail risk during crisis periods
    - Cannot capture regime persistence
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()

    def estimate_parameters(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> Tuple[float, float]:
        """
        Estimate mean and volatility from historical returns

        Args:
            returns: Historical return series

        Returns:
            Tuple of (daily_mean, daily_volatility)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        daily_mean = np.mean(returns)
        daily_vol = np.std(returns)

        logger.info(
            f"Standard MC parameters: μ={daily_mean*252:.2%} ann., "
            f"σ={daily_vol*np.sqrt(252):.2%} ann."
        )

        return daily_mean, daily_vol

    def simulate(
        self,
        initial_price: float,
        mean_return: float,
        volatility: float,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate price paths using standard Monte Carlo

        Uses Geometric Brownian Motion:
            dS = μS dt + σS dW
            S_t = S_0 * exp((μ - σ²/2)t + σW_t)

        Args:
            initial_price: Starting price S_0
            mean_return: Daily mean return μ
            volatility: Daily volatility σ
            n_paths: Number of paths (uses config if None)
            horizon: Simulation horizon (uses config if None)

        Returns:
            SimulationResult with paths, returns, and terminal values
        """
        n_paths = n_paths or self.config.n_paths
        horizon = horizon or self.config.horizon
        dt = self.config.dt

        np.random.seed(self.config.seed)

        # Generate random shocks
        if self.config.use_antithetic:
            # Antithetic variates for variance reduction
            n_half = n_paths // 2
            Z = np.random.standard_normal((n_half, horizon))
            Z = np.vstack([Z, -Z])  # Mirror the shocks
            if n_paths % 2 == 1:
                Z = np.vstack([Z, np.random.standard_normal((1, horizon))])
        else:
            Z = np.random.standard_normal((n_paths, horizon))

        # GBM simulation
        if self.config.use_gbm:
            # Log return: r_t = (μ - σ²/2)dt + σ√dt Z
            log_returns = (mean_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z

            # Cumulative log returns
            cumulative_log_returns = np.cumsum(log_returns, axis=1)

            # Price paths
            log_price_paths = np.log(initial_price) + np.hstack([
                np.zeros((n_paths, 1)),
                cumulative_log_returns
            ])
            paths = np.exp(log_price_paths)
        else:
            # Simple returns (less accurate but faster)
            returns = mean_return * dt + volatility * np.sqrt(dt) * Z
            cumulative_returns = np.cumsum(returns, axis=1)
            paths = initial_price * (1 + np.hstack([
                np.zeros((n_paths, 1)),
                cumulative_returns
            ]))

        # Compute terminal values
        terminal_prices = paths[:, -1]
        terminal_returns = np.log(terminal_prices / initial_price)

        # Extract returns from paths
        simulated_returns = np.diff(np.log(paths), axis=1)

        return SimulationResult(
            paths=paths,
            returns=simulated_returns,
            terminal_prices=terminal_prices,
            terminal_returns=terminal_returns,
            regime_paths=None,  # Standard MC has no regimes
            config=self.config,
            initial_price=initial_price
        )


class RegimeSwitchingMC:
    """
    Regime-Switching Monte Carlo Simulation

    This is the core contribution: Monte Carlo simulation where:
    1. Regime evolves according to a Markov chain
    2. Returns are sampled from regime-specific distributions
    3. Volatility clustering emerges naturally from regime persistence

    Mathematical Model:
    ===================
    Let S_t ∈ {0, 1, ..., K-1} be the regime at time t.

    Regime transition:
        P(S_{t+1} = j | S_t = i) = π_{ij}

    Return given regime (GBM within regime):
        r_t | S_t = k ~ N(μ_k - σ_k²/2, σ_k²) × dt + σ_k √dt Z_t

    Price evolution:
        P_{t+1} = P_t exp(r_t)

    Optional OU dynamics within regime:
        dr_t = θ_k (μ_k - r_{t-1}) dt + σ_k dW_t
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        n_regimes: int = 3
    ):
        """
        Args:
            config: Simulation configuration
            n_regimes: Number of market regimes
        """
        self.config = config or SimulationConfig()
        self.n_regimes = n_regimes

        # Regime identification
        self.regime_identifier = None
        self.transition_estimator = None

        # Estimated parameters
        self.regime_estimates: Optional[RegimeEstimates] = None

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        method: str = 'gmm'
    ) -> 'RegimeSwitchingMC':
        """
        Fit regime model to historical returns

        Args:
            returns: Historical return series
            method: 'gmm' or 'hmm' for regime identification

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        logger.info(f"Fitting regime model with {len(returns)} observations")

        # Step 1: Identify regimes
        if method == 'gmm':
            self.regime_identifier = GMMRegimeIdentifier(
                n_regimes=self.n_regimes,
                random_state=self.config.seed
            )
            self.regime_identifier.fit(returns)
            regime_labels = self.regime_identifier.predict(returns)
            regime_probs = self.regime_identifier.predict_proba(returns)
            log_likelihood = self.regime_identifier.gmm.score(
                self.regime_identifier.scaler.transform(
                    self.regime_identifier._prepare_features(returns)
                )
            ) * len(returns)
        elif method == 'hmm':
            if not HAS_HMM:
                raise ImportError("hmmlearn required for HMM. Install: pip install hmmlearn")
            # HMM implementation
            features = np.column_stack([returns, np.abs(returns)])
            hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='diag',
                n_iter=100,
                random_state=self.config.seed
            )
            hmm_model.fit(features)
            regime_labels = hmm_model.predict(features)
            regime_probs = hmm_model.predict_proba(features)
            log_likelihood = hmm_model.score(features)

            # Order by volatility
            state_vols = {}
            for k in range(self.n_regimes):
                mask = regime_labels == k
                state_vols[k] = np.std(returns[mask]) if mask.sum() > 0 else 0

            order = sorted(state_vols.keys(), key=lambda x: state_vols[x])
            mapping = {old: new for new, old in enumerate(order)}
            regime_labels = np.array([mapping[l] for l in regime_labels])
            new_probs = np.zeros_like(regime_probs)
            for old, new in mapping.items():
                new_probs[:, new] = regime_probs[:, old]
            regime_probs = new_probs
        else:
            raise ValueError(f"Unknown method: {method}")

        # Step 2: Estimate transition matrix
        self.transition_estimator = TransitionMatrixEstimator(self.n_regimes)
        self.transition_estimator.fit(regime_labels)

        # Step 3: Estimate regime-specific parameters
        regime_params = {}
        for k in range(self.n_regimes):
            mask = regime_labels == k
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_params[k] = RegimeParameters(
                    mean_return=float(np.mean(regime_returns)),
                    volatility=float(np.std(regime_returns)),
                    mean_reversion=0.0,  # Can be estimated if using OU
                    long_term_mean=float(np.mean(regime_returns))
                )
            else:
                # Default parameters if no observations
                regime_params[k] = RegimeParameters(
                    mean_return=0.0,
                    volatility=0.02,
                    mean_reversion=0.0,
                    long_term_mean=0.0
                )

        # Compute BIC
        n_params = (self.n_regimes * 2 +  # means and variances
                    self.n_regimes * (self.n_regimes - 1))  # transition probs
        bic = -2 * log_likelihood + n_params * np.log(len(returns))

        # Store estimates
        self.regime_estimates = RegimeEstimates(
            regime_params=regime_params,
            transition_matrix=self.transition_estimator.transition_matrix,
            stationary_dist=self.transition_estimator.get_stationary_distribution(),
            regime_labels=regime_labels,
            regime_probabilities=regime_probs,
            log_likelihood=log_likelihood,
            bic=bic,
            n_regimes=self.n_regimes
        )

        # Log summary
        self._log_regime_summary()

        return self

    def _log_regime_summary(self):
        """Log summary of estimated regime parameters"""
        if self.regime_estimates is None:
            return

        logger.info("=" * 60)
        logger.info("REGIME-SWITCHING MODEL SUMMARY")
        logger.info("=" * 60)

        for k, params in self.regime_estimates.regime_params.items():
            regime_name = ['LOW_VOL (Bull)', 'NORMAL', 'HIGH_VOL (Crisis)'][k]
            logger.info(f"\nRegime {k} ({regime_name}):")
            logger.info(f"  Annualized Return: {params.annualized_return:+.2%}")
            logger.info(f"  Annualized Volatility: {params.annualized_volatility:.2%}")
            logger.info(f"  Stationary Prob: {self.regime_estimates.stationary_dist[k]:.2%}")

        logger.info(f"\nTransition Matrix:")
        logger.info("       LOW   NORM  HIGH")
        labels = ['LOW ', 'NORM', 'HIGH']
        for i in range(self.n_regimes):
            row = self.regime_estimates.transition_matrix[i]
            logger.info(f"  {labels[i]} {row[0]:.3f} {row[1]:.3f} {row[2]:.3f}")

        expected_durations = self.transition_estimator.expected_regime_duration()
        logger.info(f"\nExpected Regime Durations (days):")
        for k, duration in expected_durations.items():
            logger.info(f"  Regime {k}: {duration:.1f}")

        logger.info(f"\nModel Fit: BIC = {self.regime_estimates.bic:.2f}")
        logger.info("=" * 60)

    def simulate(
        self,
        initial_price: float,
        initial_regime: Optional[int] = None,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate price paths with regime switching

        Args:
            initial_price: Starting price
            initial_regime: Starting regime (uses current if None)
            n_paths: Number of paths
            horizon: Simulation horizon

        Returns:
            SimulationResult with paths, returns, regime evolution
        """
        if self.regime_estimates is None:
            raise ValueError("Must call fit() before simulate()")

        n_paths = n_paths or self.config.n_paths
        horizon = horizon or self.config.horizon
        dt = self.config.dt

        np.random.seed(self.config.seed)

        # Initialize regime paths
        if initial_regime is None:
            initial_regime = self.regime_estimates.get_current_regime()

        regime_paths = np.zeros((n_paths, horizon), dtype=int)
        regime_paths[:, 0] = initial_regime

        # Simulate regime evolution
        trans_matrix = self.regime_estimates.transition_matrix

        for t in range(1, horizon):
            for p in range(n_paths):
                current_regime = regime_paths[p, t-1]
                # Sample next regime from transition probabilities
                regime_paths[p, t] = np.random.choice(
                    self.n_regimes,
                    p=trans_matrix[current_regime]
                )

        # Generate random shocks
        if self.config.use_antithetic and n_paths > 1:
            n_half = n_paths // 2
            Z = np.random.standard_normal((n_half, horizon))
            Z = np.vstack([Z, -Z])
            if n_paths % 2 == 1:
                Z = np.vstack([Z, np.random.standard_normal((1, horizon))])
        else:
            Z = np.random.standard_normal((n_paths, horizon))

        # Simulate returns conditional on regime
        returns = np.zeros((n_paths, horizon))

        for k in range(self.n_regimes):
            params = self.regime_estimates.regime_params[k]
            mask = regime_paths == k

            if self.config.use_ou and params.mean_reversion > 0:
                # Ornstein-Uhlenbeck dynamics
                # dr = θ(μ - r_{t-1})dt + σ dW
                # This requires sequential computation
                for p in range(n_paths):
                    for t in range(horizon):
                        if regime_paths[p, t] == k:
                            if t == 0:
                                prev_return = params.long_term_mean
                            else:
                                prev_return = returns[p, t-1]

                            mean_revert = params.mean_reversion * (
                                params.long_term_mean - prev_return
                            ) * dt
                            diffusion = params.volatility * np.sqrt(dt) * Z[p, t]
                            returns[p, t] = prev_return + mean_revert + diffusion
            else:
                # GBM dynamics within regime
                drift = (params.mean_return - 0.5 * params.volatility**2) * dt
                diffusion = params.volatility * np.sqrt(dt) * Z[mask]
                returns[mask] = drift + diffusion

        # Compute price paths
        cumulative_returns = np.cumsum(returns, axis=1)
        log_price_paths = np.log(initial_price) + np.hstack([
            np.zeros((n_paths, 1)),
            cumulative_returns
        ])
        paths = np.exp(log_price_paths)

        # Terminal values
        terminal_prices = paths[:, -1]
        terminal_returns = np.log(terminal_prices / initial_price)

        return SimulationResult(
            paths=paths,
            returns=returns,
            terminal_prices=terminal_prices,
            terminal_returns=terminal_returns,
            regime_paths=regime_paths,
            config=self.config,
            initial_price=initial_price
        )

    def get_regime_estimates(self) -> Optional[RegimeEstimates]:
        """Get estimated regime parameters"""
        return self.regime_estimates

    def generate_fan_chart(
        self,
        initial_price: float,
        n_paths: int = 1000,
        horizon: int = 252,
        percentiles: Optional[List[int]] = None,
        initial_regime: Optional[int] = None,
    ) -> 'FanChartData':
        """
        Generate fan chart data with regime-colored confidence bands.

        Fan charts visualize forecast uncertainty by showing percentile bands
        that widen over time. This implementation colors regions by dominant
        regime to show when high-volatility conditions are likely.

        Args:
            initial_price: Starting price
            n_paths: Number of Monte Carlo paths
            horizon: Simulation horizon in trading days
            percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95])
            initial_regime: Starting regime (uses current if None)

        Returns:
            FanChartData with percentile bands and regime information
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        # Run simulation
        result = self.simulate(
            initial_price=initial_price,
            n_paths=n_paths,
            horizon=horizon,
            initial_regime=initial_regime,
        )

        # Compute percentile bands over time
        bands = {}
        for p in percentiles:
            bands[p] = np.percentile(result.paths, p, axis=0)

        # Compute dominant regime per timestep
        regime_probs = np.zeros((horizon, self.n_regimes))
        for t in range(horizon):
            counts = np.bincount(
                result.regime_paths[:, t],
                minlength=self.n_regimes
            )
            regime_probs[t] = counts / n_paths

        dominant_regimes = np.argmax(regime_probs, axis=1)

        # Generate dates (simple index if no actual dates)
        dates = np.arange(horizon + 1)

        return FanChartData(
            dates=dates,
            percentiles=bands,
            dominant_regime=dominant_regimes,
            regime_probs=regime_probs,
            initial_price=initial_price,
            n_paths=n_paths,
            horizon=horizon,
        )


@dataclass
class FanChartData:
    """
    Data for rendering fan charts with regime coloring.

    Attributes:
        dates: Array of time indices or dates [horizon + 1]
        percentiles: Dict mapping percentile to price array {p: [horizon + 1]}
        dominant_regime: Most likely regime at each timestep [horizon]
        regime_probs: Probability of each regime over time [horizon, n_regimes]
        initial_price: Starting price
        n_paths: Number of simulation paths used
        horizon: Simulation horizon
    """
    dates: np.ndarray
    percentiles: Dict[int, np.ndarray]
    dominant_regime: np.ndarray
    regime_probs: np.ndarray
    initial_price: float
    n_paths: int
    horizon: int

    def get_confidence_interval(
        self,
        level: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence interval for a given level.

        Args:
            level: Confidence level (e.g., 0.90 for 90% CI)

        Returns:
            Tuple of (lower_band, upper_band)
        """
        lower_pct = int((1 - level) / 2 * 100)
        upper_pct = int((1 + level) / 2 * 100)

        # Find closest available percentiles
        available = sorted(self.percentiles.keys())

        lower_key = min(available, key=lambda x: abs(x - lower_pct))
        upper_key = min(available, key=lambda x: abs(x - upper_pct))

        return self.percentiles[lower_key], self.percentiles[upper_key]

    def get_median(self) -> np.ndarray:
        """Get median projection."""
        if 50 in self.percentiles:
            return self.percentiles[50]
        # Fallback: average of 25th and 75th
        if 25 in self.percentiles and 75 in self.percentiles:
            return (self.percentiles[25] + self.percentiles[75]) / 2
        return np.zeros(self.horizon + 1)

    def get_regime_periods(self) -> List[Dict]:
        """
        Get contiguous regime periods for visualization.

        Returns:
            List of dicts with 'start', 'end', 'regime' keys
        """
        periods = []
        if len(self.dominant_regime) == 0:
            return periods

        current_regime = self.dominant_regime[0]
        start_idx = 0

        for i, regime in enumerate(self.dominant_regime[1:], start=1):
            if regime != current_regime:
                periods.append({
                    'start': start_idx,
                    'end': i,
                    'regime': int(current_regime),
                })
                current_regime = regime
                start_idx = i

        # Add final period
        periods.append({
            'start': start_idx,
            'end': len(self.dominant_regime),
            'regime': int(current_regime),
        })

        return periods

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dates': self.dates.tolist(),
            'percentiles': {str(k): v.tolist() for k, v in self.percentiles.items()},
            'dominant_regime': self.dominant_regime.tolist(),
            'regime_probs': self.regime_probs.tolist(),
            'initial_price': self.initial_price,
            'n_paths': self.n_paths,
            'horizon': self.horizon,
            'regime_periods': self.get_regime_periods(),
        }


# =============================================================================
# Monte Carlo Comparison Framework
# =============================================================================

class MonteCarloComparison:
    """
    Framework for comparing standard vs regime-switching Monte Carlo

    This class provides a unified interface for:
    1. Fitting both models to historical data
    2. Running simulations
    3. Computing risk metrics
    4. Generating comparative analysis

    Key Insight:
    Standard MC uses unconditional (pooled) estimates of μ and σ,
    which averages across regimes and underestimates tail risk during crises.
    Regime-switching MC conditions on the current regime, capturing:
    - Higher volatility during stress periods
    - Regime persistence (volatility clustering)
    - Heavier tails from regime mixing
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        n_regimes: int = 3
    ):
        """
        Args:
            config: Simulation configuration
            n_regimes: Number of regimes for switching model
        """
        self.config = config or SimulationConfig()
        self.n_regimes = n_regimes

        self.standard_mc = StandardMC(self.config)
        self.regime_mc = RegimeSwitchingMC(self.config, n_regimes)

        self._historical_returns = None
        self._standard_params = None
        self._is_fitted = False

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        method: str = 'gmm'
    ) -> 'MonteCarloComparison':
        """
        Fit both standard and regime-switching models

        Args:
            returns: Historical return series
            method: Regime identification method ('gmm' or 'hmm')

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        self._historical_returns = returns[~np.isnan(returns)]

        # Fit standard MC (just estimates mean and vol)
        self._standard_params = self.standard_mc.estimate_parameters(
            self._historical_returns
        )

        # Fit regime-switching MC
        self.regime_mc.fit(self._historical_returns, method=method)

        self._is_fitted = True

        return self

    def simulate_both(
        self,
        initial_price: float,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> Tuple[SimulationResult, SimulationResult]:
        """
        Run both standard and regime-switching simulations

        Args:
            initial_price: Starting price
            n_paths: Number of paths
            horizon: Simulation horizon

        Returns:
            Tuple of (standard_result, regime_result)
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before simulate_both()")

        # Standard MC
        mean_return, volatility = self._standard_params
        standard_result = self.standard_mc.simulate(
            initial_price=initial_price,
            mean_return=mean_return,
            volatility=volatility,
            n_paths=n_paths,
            horizon=horizon
        )

        # Regime-switching MC
        regime_result = self.regime_mc.simulate(
            initial_price=initial_price,
            n_paths=n_paths,
            horizon=horizon
        )

        return standard_result, regime_result

    def compare_terminal_distributions(
        self,
        standard_result: SimulationResult,
        regime_result: SimulationResult
    ) -> Dict[str, Any]:
        """
        Compare terminal return distributions

        Args:
            standard_result: Standard MC simulation result
            regime_result: Regime-switching MC simulation result

        Returns:
            Dict with comparison statistics
        """
        std_returns = standard_result.terminal_returns
        reg_returns = regime_result.terminal_returns

        comparison = {
            'standard': {
                'mean': float(np.mean(std_returns)),
                'std': float(np.std(std_returns)),
                'skew': float(stats.skew(std_returns)),
                'kurtosis': float(stats.kurtosis(std_returns)),
                'var_95': float(np.percentile(std_returns, 5)),
                'var_99': float(np.percentile(std_returns, 1)),
                'min': float(np.min(std_returns)),
                'max': float(np.max(std_returns)),
            },
            'regime_switching': {
                'mean': float(np.mean(reg_returns)),
                'std': float(np.std(reg_returns)),
                'skew': float(stats.skew(reg_returns)),
                'kurtosis': float(stats.kurtosis(reg_returns)),
                'var_95': float(np.percentile(reg_returns, 5)),
                'var_99': float(np.percentile(reg_returns, 1)),
                'min': float(np.min(reg_returns)),
                'max': float(np.max(reg_returns)),
            }
        }

        # Compute relative differences
        comparison['differences'] = {
            'var_95_diff': (comparison['regime_switching']['var_95'] -
                           comparison['standard']['var_95']),
            'var_99_diff': (comparison['regime_switching']['var_99'] -
                           comparison['standard']['var_99']),
            'kurtosis_diff': (comparison['regime_switching']['kurtosis'] -
                             comparison['standard']['kurtosis']),
            'tail_ratio_5pct': (comparison['regime_switching']['var_95'] /
                               comparison['standard']['var_95']
                               if comparison['standard']['var_95'] != 0 else 1.0),
        }

        return comparison

    def get_regime_estimates(self) -> Optional[RegimeEstimates]:
        """Get regime parameter estimates"""
        return self.regime_mc.get_regime_estimates()


# =============================================================================
# Demonstration / Test
# =============================================================================

if __name__ == "__main__":
    """Demonstration of regime-switching Monte Carlo"""

    print("=" * 70)
    print("REGIME-SWITCHING MONTE CARLO DEMONSTRATION")
    print("=" * 70)

    # Generate synthetic data with regime changes
    np.random.seed(42)

    # Simulate 5 years of daily returns with regime changes
    n_days = 252 * 5

    # Define true regime parameters
    true_params = {
        0: {'mean': 0.0005, 'vol': 0.008},   # Low vol (12% ann.)
        1: {'mean': 0.0002, 'vol': 0.012},   # Normal (19% ann.)
        2: {'mean': -0.001, 'vol': 0.025},   # High vol (40% ann.)
    }

    # True transition matrix (regimes are sticky)
    true_trans = np.array([
        [0.98, 0.015, 0.005],  # From low vol
        [0.02, 0.96, 0.02],    # From normal
        [0.01, 0.04, 0.95],    # From high vol
    ])

    # Simulate regime path
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = 1  # Start in normal regime

    for t in range(1, n_days):
        regimes[t] = np.random.choice(3, p=true_trans[regimes[t-1]])

    # Simulate returns conditional on regime
    returns = np.zeros(n_days)
    for t in range(n_days):
        k = regimes[t]
        returns[t] = np.random.normal(
            true_params[k]['mean'],
            true_params[k]['vol']
        )

    print(f"\nGenerated {n_days} days of synthetic returns")
    print(f"True regime distribution: {np.bincount(regimes) / n_days}")

    # Create comparison framework
    config = SimulationConfig(
        n_paths=10000,
        horizon=252,
        seed=42
    )

    comparison = MonteCarloComparison(config, n_regimes=3)
    comparison.fit(returns, method='gmm')

    # Run simulations
    initial_price = 100.0
    standard_result, regime_result = comparison.simulate_both(initial_price)

    # Compare results
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    stats_comparison = comparison.compare_terminal_distributions(
        standard_result, regime_result
    )

    print("\nTerminal Return Distribution:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Standard':<15} {'Regime-Switch':<15}")
    print("-" * 40)

    for metric in ['mean', 'std', 'skew', 'kurtosis', 'var_95', 'var_99']:
        std_val = stats_comparison['standard'][metric]
        reg_val = stats_comparison['regime_switching'][metric]
        print(f"{metric:<20} {std_val:>12.4f}   {reg_val:>12.4f}")

    print("\n" + "-" * 40)
    print("KEY FINDING: Regime-switching model shows:")
    print(f"  - {abs(stats_comparison['differences']['var_95_diff'])*100:.2f}% more extreme VaR(95%)")
    print(f"  - {abs(stats_comparison['differences']['var_99_diff'])*100:.2f}% more extreme VaR(99%)")
    print(f"  - {stats_comparison['differences']['kurtosis_diff']:.2f} higher kurtosis")
    print("=" * 70)
