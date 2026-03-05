"""
Simulation Module for Advanced Monte Carlo Methods

This module implements regime-switching Monte Carlo simulation for financial
asset returns, providing a significant improvement over standard IID Monte Carlo.

Key Components:
    - RegimeSwitchingMC: Markov regime-switching Monte Carlo simulator
    - StandardMC: Baseline IID Monte Carlo for comparison
    - RiskMetrics: Comprehensive tail risk computation
    - RegimeVisualizer: Publication-quality plotting

Research Background:
    Standard Monte Carlo assumes IID returns with constant volatility, which
    underestimates tail risk due to:
    1. Volatility clustering (GARCH effects)
    2. Regime persistence (HMM structure)
    3. Fat tails in return distributions

    Regime-switching models address this by:
    1. Identifying distinct market states (low/normal/high volatility)
    2. Modeling regime transitions via Markov chains
    3. Sampling returns conditional on the active regime

References:
    - Hamilton, J.D. (1989). "A New Approach to Economic Analysis of
      Nonstationary Time Series." Econometrica.
    - Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates." NBER.
    - Hardy, M.R. (2001). "A Regime-Switching Model of Long-Term Stock Returns."

Author: Dissertation Research Project
"""

from .regime_monte_carlo import (
    RegimeSwitchingMC,
    StandardMC,
    MonteCarloComparison,
    RegimeParameters,
    SimulationConfig,
)

from .risk_metrics import (
    RiskMetricsCalculator,
    RiskMetricsResult,
    compute_var,
    compute_expected_shortfall,
    compute_maximum_drawdown,
)

# Visualization imports are optional (requires matplotlib)
try:
    from .visualizations import (
        RegimeVisualizer,
        plot_simulation_comparison,
        plot_regime_evolution,
        plot_transition_matrix,
        plot_tail_comparison,
    )
except ImportError:
    RegimeVisualizer = None
    plot_simulation_comparison = None
    plot_regime_evolution = None
    plot_transition_matrix = None
    plot_tail_comparison = None

from .pinn_regime_integration import (
    RegimeAwarePINN,
    RegimeConditionedLoss,
    regime_conditioned_gbm_drift,
    regime_conditioned_diffusion,
)

__all__ = [
    # Monte Carlo
    'RegimeSwitchingMC',
    'StandardMC',
    'MonteCarloComparison',
    'RegimeParameters',
    'SimulationConfig',
    # Risk Metrics
    'RiskMetricsCalculator',
    'RiskMetricsResult',
    'compute_var',
    'compute_expected_shortfall',
    'compute_maximum_drawdown',
    # Visualizations
    'RegimeVisualizer',
    'plot_simulation_comparison',
    'plot_regime_evolution',
    'plot_transition_matrix',
    'plot_tail_comparison',
    # PINN Integration
    'RegimeAwarePINN',
    'RegimeConditionedLoss',
    'regime_conditioned_gbm_drift',
    'regime_conditioned_diffusion',
]
