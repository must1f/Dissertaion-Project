"""Trading modules"""

from .agent import TradingAgent, SignalGenerator
from .position_sizing import (
    PositionSizer,
    FixedRiskSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
    ConfidenceBasedSizer,
)
from .realtime_data import RealTimeDataService, SimulatedDataService, MarketQuote
from .realtime_agent import RealTimeTradingAgent
from .strategy_evaluator import (
    TradingStrategy,
    StrategyConfig,
    StrategyResult,
    TradeSignal,
    PositionType,
    ThresholdStrategy,
    RankingStrategy,
    VolatilityScaledStrategy,
    ConfidenceWeightedStrategy,
    StrategyEvaluator,
    StrategyFactory,
    evaluate_model_trading_value
)
from .benchmark_strategies import (
    BenchmarkStrategy,
    BuyAndHoldStrategy,
    NaiveLastValueStrategy,
    MomentumBenchmark,
    MeanReversionBenchmark,
    MACrossoverBenchmark,
    RandomStrategy,
    BenchmarkComparison,
    BenchmarkEvaluator,
    evaluate_against_benchmarks
)

__all__ = [
    "TradingAgent",
    "SignalGenerator",
    "PositionSizer",
    "FixedRiskSizer",
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    "ConfidenceBasedSizer",
    "RealTimeDataService",
    "SimulatedDataService",
    "MarketQuote",
    "RealTimeTradingAgent",
    # Strategy evaluation
    "TradingStrategy",
    "StrategyConfig",
    "StrategyResult",
    "TradeSignal",
    "PositionType",
    "ThresholdStrategy",
    "RankingStrategy",
    "VolatilityScaledStrategy",
    "ConfidenceWeightedStrategy",
    "StrategyEvaluator",
    "StrategyFactory",
    "evaluate_model_trading_value",
    # Benchmarks
    "BenchmarkStrategy",
    "BuyAndHoldStrategy",
    "NaiveLastValueStrategy",
    "MomentumBenchmark",
    "MeanReversionBenchmark",
    "MACrossoverBenchmark",
    "RandomStrategy",
    "BenchmarkComparison",
    "BenchmarkEvaluator",
    "evaluate_against_benchmarks",
]
