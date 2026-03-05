"""Evaluation modules"""

from .metrics import (
    calculate_metrics,
    calculate_financial_metrics,
    MetricsCalculator
)
from .backtester import Backtester, BacktestResults
from .backtesting_platform import (
    BacktestingPlatform,
    BacktestConfig,
    StrategyResult,
    Strategy,
    BuyAndHoldStrategy,
    SMACrossoverStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    ModelBasedStrategy,
    run_comprehensive_backtest
)
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    compute_var_cvar,
    monte_carlo_price_path
)
from .naive_baselines import (
    NaiveBaselines,
    BaselineResults,
    evaluate_naive_baselines
)
from .financial_metrics import (
    FinancialMetrics,
    compute_strategy_returns,
    validate_metrics
)
from .statistical_tests import (
    StatisticalTests,
    ModelComparator,
    BootstrapResult,
    DieboldMarianoResult,
    PairedTestResult,
    bootstrap_sharpe_ci,
    compare_forecasts
)
from .evaluation_harness import (
    EvaluationHarness,
    EvaluationResult,
    AggregatedResult,
    ComparisonResult,
    evaluate_model
)
from .split_manager import SplitManager, SplitConfig, SplitStrategy
from .metrics_registry import MetricsRegistry
from .leaderboard import ResultsDatabase
# Optional imports that require heavier deps (e.g., scipy). Guard so lightweight
# CLI usage like `python -m src.evaluation` doesn't fail.
try:  # pragma: no cover - optional dependency path
    from .error_analyzer import (
        ErrorAnalyzer,
        ErrorStatistics,
        RegimeErrorAnalysis,
        EventErrorAnalysis,
        ErrorReport,
        ErrorMetric,
        MarketRegime,
        analyze_forecast_errors
    )
except ImportError:  # pragma: no cover
    ErrorAnalyzer = None  # type: ignore
    ErrorStatistics = None  # type: ignore
    RegimeErrorAnalysis = None  # type: ignore
    EventErrorAnalysis = None  # type: ignore
    ErrorReport = None  # type: ignore
    ErrorMetric = None  # type: ignore
    MarketRegime = None  # type: ignore
    analyze_forecast_errors = None  # type: ignore
from .leaderboard import (
    ResultsDatabase as LeaderboardDB,
    LeaderboardGenerator,
    ExperimentEntry,
    LeaderboardEntry,
    Leaderboard,
    RankingMetric,
    create_experiment_entry
)
from .stress_test_evaluator import (
    StressTestEvaluator,
    StressTestResult,
    StressTestReport,
    RegimeAnalysis,
    CrisisPeriod,
    CrisisCalendar,
    CrisisType,
    VolatilityRegime,
    run_stress_tests
)
from .pde_evaluator import (
    PDEEvaluator,
    PDEMetrics,
    burgers_exact_solution_hopf_cole,
    create_burgers_evaluator,
    compare_models as compare_pde_models,
)
from .strategy_engine import StrategyEngine, StrategyConfig
from .pipeline import EvaluationPipeline, PipelineConfig, PipelineResult, run_pipeline_evaluation
try:
    from .plot_diagnostics import DiagnosticPlotter
except ImportError:
    DiagnosticPlotter = None  # matplotlib may not be available in all environments

__all__ = [
    # Core metrics
    "calculate_metrics",
    "calculate_financial_metrics",
    "MetricsCalculator",
    # Backtesting
    "Backtester",
    "BacktestResults",
    "BacktestingPlatform",
    "BacktestConfig",
    "StrategyResult",
    "Strategy",
    "BuyAndHoldStrategy",
    "SMACrossoverStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "ModelBasedStrategy",
    "run_comprehensive_backtest",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloResults",
    "compute_var_cvar",
    "monte_carlo_price_path",
    # Baselines
    "NaiveBaselines",
    "BaselineResults",
    "evaluate_naive_baselines",
    # Financial metrics
    "FinancialMetrics",
    "compute_strategy_returns",
    "validate_metrics",
    # Statistical tests
    "StatisticalTests",
    "ModelComparator",
    "BootstrapResult",
    "DieboldMarianoResult",
    "PairedTestResult",
    "bootstrap_sharpe_ci",
    "compare_forecasts",
    # Evaluation harness
    "EvaluationHarness",
    "EvaluationResult",
    "AggregatedResult",
    "ComparisonResult",
    "SplitManager",
    "SplitConfig",
    "SplitStrategy",
    "MetricsRegistry",
    "ResultsDatabase",
    "evaluate_model",
    # Error analysis
    "ErrorAnalyzer",
    "ErrorStatistics",
    "RegimeErrorAnalysis",
    "EventErrorAnalysis",
    "ErrorReport",
    "ErrorMetric",
    "MarketRegime",
    "analyze_forecast_errors",
    # Leaderboard
    "LeaderboardDB",
    "LeaderboardGenerator",
    "ExperimentEntry",
    "LeaderboardEntry",
    "Leaderboard",
    "RankingMetric",
    "create_experiment_entry",
    # Stress testing
    "StressTestEvaluator",
    "StressTestResult",
    "StressTestReport",
    "RegimeAnalysis",
    "CrisisPeriod",
    "CrisisCalendar",
    "CrisisType",
    "VolatilityRegime",
    "run_stress_tests",
    # PDE evaluation
    "PDEEvaluator",
    "PDEMetrics",
    "burgers_exact_solution_hopf_cole",
    "create_burgers_evaluator",
    "compare_pde_models",
    # Strategy engine & pipeline
    "StrategyEngine",
    "StrategyConfig",
    "EvaluationPipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline_evaluation",
    "DiagnosticPlotter",
]
