"""
Trading Strategy Evaluation Layer

Converts model forecasts into tradeable signals and evaluates performance
against various benchmarks. Provides a rigorous framework for assessing
the economic value of predictions.

Key Components:
- StrategyConverter: Transform predictions to position signals
- Multiple strategy types: threshold, ranking, volatility-scaled
- Transaction cost modeling
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from ..evaluation.financial_metrics import FinancialMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    """Types of positions"""
    LONG = 1
    SHORT = -1
    FLAT = 0


class RebalanceFrequency(Enum):
    """Rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TradeSignal:
    """A single trade signal"""
    timestamp: datetime
    ticker: str
    position: PositionType
    confidence: float  # 0-1 confidence in signal
    target_weight: float  # Target portfolio weight
    predicted_return: float
    actual_return: Optional[float] = None


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    max_position_size: float = 0.2  # Max 20% in single position
    min_holding_period: int = 1  # Minimum days to hold
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    allow_short: bool = False
    leverage: float = 1.0


@dataclass
class StrategyResult:
    """Results from strategy evaluation"""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    n_trades: int
    turnover: float
    transaction_costs: float
    net_return: float
    daily_returns: np.ndarray
    cumulative_returns: np.ndarray
    positions: np.ndarray
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        """Generate trading signals from predictions"""
        pass

    def calculate_positions(
        self,
        signals: List[TradeSignal]
    ) -> np.ndarray:
        """Convert signals to position weights"""
        positions = np.zeros(len(signals))
        for i, signal in enumerate(signals):
            positions[i] = signal.target_weight * signal.position.value
        return positions


class ThresholdStrategy(TradingStrategy):
    """
    Threshold-based strategy.

    Goes long if predicted return > buy_threshold
    Goes short if predicted return < sell_threshold (if allowed)
    """

    def __init__(
        self,
        config: StrategyConfig,
        buy_threshold: float = 0.001,
        sell_threshold: float = -0.001
    ):
        super().__init__(config)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        signals = []

        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred > self.buy_threshold:
                position = PositionType.LONG
                confidence = min(1.0, pred / (self.buy_threshold * 5))
                weight = min(self.config.max_position_size, confidence)
            elif pred < self.sell_threshold and self.config.allow_short:
                position = PositionType.SHORT
                confidence = min(1.0, abs(pred) / (abs(self.sell_threshold) * 5))
                weight = min(self.config.max_position_size, confidence)
            else:
                position = PositionType.FLAT
                confidence = 0.0
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=confidence,
                target_weight=weight,
                predicted_return=pred
            ))

        return signals


class RankingStrategy(TradingStrategy):
    """
    Ranking-based strategy for multiple assets.

    Selects top N assets by predicted return.
    """

    def __init__(
        self,
        config: StrategyConfig,
        top_n: int = 3,
        bottom_n: int = 0
    ):
        super().__init__(config)
        self.top_n = top_n
        self.bottom_n = bottom_n

    def generate_signals_multi(
        self,
        predictions: Dict[str, np.ndarray],
        timestamps: List[datetime]
    ) -> Dict[str, List[TradeSignal]]:
        """Generate signals for multiple assets"""
        all_signals = {ticker: [] for ticker in predictions}
        n_periods = len(timestamps)

        for t in range(n_periods):
            # Get predictions for this period
            period_preds = {
                ticker: preds[t] for ticker, preds in predictions.items()
            }

            # Rank assets
            sorted_assets = sorted(
                period_preds.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Long top N
            long_assets = sorted_assets[:self.top_n]
            long_weight = 1.0 / self.top_n if self.top_n > 0 else 0

            # Short bottom N (if allowed)
            if self.config.allow_short and self.bottom_n > 0:
                short_assets = sorted_assets[-self.bottom_n:]
                short_weight = 1.0 / self.bottom_n
            else:
                short_assets = []
                short_weight = 0

            # Generate signals
            for ticker in predictions:
                if ticker in [a[0] for a in long_assets]:
                    position = PositionType.LONG
                    weight = long_weight
                elif ticker in [a[0] for a in short_assets]:
                    position = PositionType.SHORT
                    weight = short_weight
                else:
                    position = PositionType.FLAT
                    weight = 0

                all_signals[ticker].append(TradeSignal(
                    timestamp=timestamps[t],
                    ticker=ticker,
                    position=position,
                    confidence=1.0,
                    target_weight=weight,
                    predicted_return=period_preds[ticker]
                ))

        return all_signals

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        """Single asset version - always long if positive prediction"""
        signals = []
        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred > 0:
                position = PositionType.LONG
                weight = 1.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=1.0,
                target_weight=weight,
                predicted_return=pred
            ))
        return signals


class VolatilityScaledStrategy(TradingStrategy):
    """
    Volatility-scaled strategy.

    Scales position size inversely to volatility to target constant risk.
    """

    def __init__(
        self,
        config: StrategyConfig,
        target_volatility: float = 0.15,
        vol_lookback: int = 20,
        vol_floor: float = 0.05
    ):
        super().__init__(config)
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        self.vol_floor = vol_floor

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        # Calculate rolling volatility
        returns = np.diff(np.log(prices))
        returns = np.concatenate([[0], returns])

        vol = pd.Series(returns).rolling(
            window=self.vol_lookback,
            min_periods=5
        ).std() * np.sqrt(252)
        vol = vol.fillna(self.target_volatility).values

        signals = []
        for i, (pred, ts, v) in enumerate(zip(predictions, timestamps, vol)):
            # Scale by volatility
            realized_vol = max(v, self.vol_floor)
            vol_scalar = self.target_volatility / realized_vol

            if pred > 0:
                position = PositionType.LONG
                # Scale position by volatility
                weight = min(
                    self.config.max_position_size * self.config.leverage,
                    vol_scalar
                )
            elif pred < 0 and self.config.allow_short:
                position = PositionType.SHORT
                weight = min(
                    self.config.max_position_size * self.config.leverage,
                    vol_scalar
                )
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=min(1.0, abs(pred) * 100),
                target_weight=weight,
                predicted_return=pred
            ))

        return signals


class ConfidenceWeightedStrategy(TradingStrategy):
    """
    Confidence-weighted strategy.

    Weights positions by model confidence (prediction magnitude).
    """

    def __init__(
        self,
        config: StrategyConfig,
        confidence_scale: float = 100.0,
        min_confidence: float = 0.1
    ):
        super().__init__(config)
        self.confidence_scale = confidence_scale
        self.min_confidence = min_confidence

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        signals = []

        # Normalize predictions to confidence scores
        pred_abs = np.abs(predictions)
        max_pred = np.percentile(pred_abs, 95) if len(pred_abs) > 10 else 1.0

        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            confidence = min(1.0, abs(pred) * self.confidence_scale)

            if confidence < self.min_confidence:
                position = PositionType.FLAT
                weight = 0.0
            elif pred > 0:
                position = PositionType.LONG
                weight = confidence * self.config.max_position_size
            elif self.config.allow_short:
                position = PositionType.SHORT
                weight = confidence * self.config.max_position_size
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=confidence,
                target_weight=weight,
                predicted_return=pred
            ))

        return signals


class StrategyEvaluator:
    """
    Evaluates trading strategies against benchmarks.

    Provides comprehensive performance metrics and comparison.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def evaluate_strategy(
        self,
        strategy: TradingStrategy,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime],
        actual_returns: Optional[np.ndarray] = None
    ) -> StrategyResult:
        """
        Evaluate a strategy's performance.

        Args:
            strategy: Trading strategy to evaluate
            predictions: Model predictions (predicted returns)
            prices: Asset prices
            timestamps: Timestamps
            actual_returns: Actual returns (optional, computed if not provided)

        Returns:
            StrategyResult with all metrics
        """
        # Generate signals
        signals = strategy.generate_signals(predictions, prices, timestamps)

        # Get positions
        positions = strategy.calculate_positions(signals)

        # Calculate actual returns if not provided
        if actual_returns is None:
            price_returns = np.diff(np.log(prices))
            actual_returns = np.concatenate([[0], price_returns])

        # Calculate strategy returns
        strategy_returns = positions[:-1] * actual_returns[1:]
        strategy_returns = np.concatenate([[0], strategy_returns])

        # Calculate turnover and transaction costs
        position_changes = np.abs(np.diff(positions))
        position_changes = np.concatenate([[abs(positions[0])], position_changes])
        turnover = np.sum(position_changes)

        # Apply transaction costs
        costs = position_changes * (self.transaction_cost + self.slippage)
        net_returns = strategy_returns - costs

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + net_returns) - 1

        # Calculate metrics
        total_return = cumulative[-1] if len(cumulative) > 0 else 0
        n_days = len(net_returns)
        ann_factor = 252 / n_days if n_days > 0 else 1

        ann_return = (1 + total_return) ** ann_factor - 1
        volatility = np.std(net_returns) * np.sqrt(252)
        sharpe = FinancialMetrics.sharpe_ratio(net_returns)
        sortino = FinancialMetrics.sortino_ratio(net_returns)
        max_dd = FinancialMetrics.max_drawdown(net_returns)
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.01 else 0

        # Trade statistics
        trades = position_changes > 0.01
        n_trades = np.sum(trades)

        # Win rate
        trade_returns = strategy_returns[trades] if n_trades > 0 else []
        win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0

        # Profit factor
        gains = np.sum(trade_returns[trade_returns > 0]) if len(trade_returns) > 0 else 0
        losses = abs(np.sum(trade_returns[trade_returns < 0])) if len(trade_returns) > 0 else 0
        profit_factor = gains / losses if losses > 0 else gains if gains > 0 else 0

        avg_trade = np.mean(trade_returns) if len(trade_returns) > 0 else 0

        return StrategyResult(
            strategy_name=strategy.config.name,
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade,
            n_trades=int(n_trades),
            turnover=turnover,
            transaction_costs=np.sum(costs),
            net_return=total_return,
            daily_returns=net_returns,
            cumulative_returns=cumulative,
            positions=positions,
            timestamps=timestamps
        )

    def compare_strategies(
        self,
        strategies: List[TradingStrategy],
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime],
        actual_returns: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Returns:
            DataFrame with strategy comparison
        """
        results = []

        for strategy in strategies:
            result = self.evaluate_strategy(
                strategy, predictions, prices, timestamps, actual_returns
            )
            results.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Ann. Return': f"{result.annualized_return:.2%}",
                'Volatility': f"{result.volatility:.2%}",
                'Sharpe': f"{result.sharpe_ratio:.2f}",
                'Sortino': f"{result.sortino_ratio:.2f}",
                'Max DD': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.1%}",
                'Trades': result.n_trades,
                'Turnover': f"{result.turnover:.1f}x"
            })

        return pd.DataFrame(results)


class StrategyFactory:
    """Factory for creating common strategy configurations"""

    @staticmethod
    def create_conservative(name: str = "Conservative") -> TradingStrategy:
        """Conservative threshold strategy"""
        config = StrategyConfig(
            name=name,
            transaction_cost=0.001,
            max_position_size=0.15,
            allow_short=False
        )
        return ThresholdStrategy(config, buy_threshold=0.002, sell_threshold=-0.002)

    @staticmethod
    def create_aggressive(name: str = "Aggressive") -> TradingStrategy:
        """Aggressive threshold strategy"""
        config = StrategyConfig(
            name=name,
            transaction_cost=0.001,
            max_position_size=0.3,
            allow_short=True,
            leverage=1.5
        )
        return ThresholdStrategy(config, buy_threshold=0.001, sell_threshold=-0.001)

    @staticmethod
    def create_vol_targeted(
        name: str = "Vol-Targeted",
        target_vol: float = 0.15
    ) -> TradingStrategy:
        """Volatility-targeted strategy"""
        config = StrategyConfig(
            name=name,
            transaction_cost=0.001,
            max_position_size=1.0,
            allow_short=False
        )
        return VolatilityScaledStrategy(config, target_volatility=target_vol)

    @staticmethod
    def create_confidence_weighted(name: str = "Confidence-Weighted") -> TradingStrategy:
        """Confidence-weighted strategy"""
        config = StrategyConfig(
            name=name,
            transaction_cost=0.001,
            max_position_size=0.25,
            allow_short=False
        )
        return ConfidenceWeightedStrategy(config)


def evaluate_model_trading_value(
    predictions: np.ndarray,
    prices: np.ndarray,
    timestamps: List[datetime],
    actual_returns: Optional[np.ndarray] = None
) -> Dict[str, StrategyResult]:
    """
    Convenience function to evaluate trading value of model predictions.

    Args:
        predictions: Model predictions (returns)
        prices: Asset prices
        timestamps: Timestamps
        actual_returns: Actual returns

    Returns:
        Dictionary of strategy name -> results
    """
    evaluator = StrategyEvaluator()

    strategies = [
        StrategyFactory.create_conservative(),
        StrategyFactory.create_aggressive(),
        StrategyFactory.create_vol_targeted(),
        StrategyFactory.create_confidence_weighted()
    ]

    results = {}
    for strategy in strategies:
        result = evaluator.evaluate_strategy(
            strategy, predictions, prices, timestamps, actual_returns
        )
        results[strategy.config.name] = result

    return results
