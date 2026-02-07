"""
Comprehensive Backtesting Platform for PINN Financial Forecasting

Features:
- Walk-forward optimization
- Multiple strategy support
- Benchmark comparisons (Buy & Hold, SMA, Random Walk)
- Multi-asset portfolio simulation
- Transaction cost modeling
- Position sizing (Kelly, Fixed, Volatility-based)
- Risk management (Stop-loss, Take-profit, Max position)
- Detailed performance analytics
- Monte Carlo simulation for robustness

Author: Claude Code
Date: 2026-02-04
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger
from .financial_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    compute_all_metrics
)

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"
    EQUAL_WEIGHT = "equal_weight"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class TradeRecord:
    """Record of a single trade"""
    timestamp: pd.Timestamp
    ticker: str
    signal: SignalType
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    position_value: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period: int = 0
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_reason: str = ""
    confidence: float = 1.0
    model_name: str = ""


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, float]  # ticker -> quantity
    position_values: Dict[str, float]  # ticker -> market value
    entry_prices: Dict[str, float]  # ticker -> average entry price
    entry_times: Dict[str, pd.Timestamp]  # ticker -> entry timestamp
    total_value: float = 0.0

    def update_value(self, current_prices: Dict[str, float]):
        """Update portfolio value with current prices"""
        self.position_values = {
            ticker: qty * current_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
            if ticker in current_prices
        }
        self.total_value = self.cash + sum(self.position_values.values())


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_position_size: float = 0.20  # 20% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    stop_loss: float = 0.02  # 2%
    take_profit: float = 0.05  # 5%
    position_sizing: PositionSizingMethod = PositionSizingMethod.FIXED
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    risk_free_rate: float = 0.02  # 2% annual
    kelly_fraction: float = 0.5  # Half-Kelly
    target_volatility: float = 0.15  # 15% annual
    use_stop_loss: bool = True
    use_take_profit: bool = True
    allow_short: bool = False
    max_leverage: float = 1.0  # No leverage by default


@dataclass
class StrategyResult:
    """Results from a single strategy backtest"""
    strategy_name: str
    model_name: str
    trades: List[TradeRecord] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_holding_period: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    volatility: float = 0.0

    def to_dict(self) -> Dict:
        """Convert results to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'model_name': self.model_name,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_holding_period': self.avg_holding_period,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'volatility': self.volatility
        }


class Strategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name

    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        """
        Generate trading signal

        Returns:
            Tuple of (signal_type, confidence)
        """
        raise NotImplementedError


class ModelBasedStrategy(Strategy):
    """Strategy based on model predictions"""

    def __init__(
        self,
        model: torch.nn.Module,
        name: str = "Model Strategy",
        threshold: float = 0.02,
        confidence_threshold: float = 0.6,
        device: torch.device = None
    ):
        super().__init__(name)
        self.model = model
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        current_price: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        """Generate signal from model prediction"""
        if predictions is None or len(predictions) == 0:
            return SignalType.HOLD, 0.0

        # Get latest prediction
        pred_price = predictions[-1] if len(predictions.shape) == 1 else predictions[-1, 0]

        # Calculate expected return
        if current_price > 0:
            expected_return = (pred_price - current_price) / current_price
        else:
            expected_return = 0.0

        # Simple confidence based on prediction magnitude
        confidence = min(abs(expected_return) / self.threshold, 1.0)

        # Generate signal
        if expected_return > self.threshold and confidence >= self.confidence_threshold:
            return SignalType.BUY, confidence
        elif expected_return < -self.threshold and confidence >= self.confidence_threshold:
            return SignalType.SELL, confidence
        else:
            return SignalType.HOLD, confidence


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold benchmark"""

    def __init__(self):
        super().__init__("Buy and Hold")
        self.entered = False

    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        if not self.entered and current_position == 0:
            self.entered = True
            return SignalType.BUY, 1.0
        return SignalType.HOLD, 1.0

    def reset(self):
        self.entered = False


class SMACrossoverStrategy(Strategy):
    """Simple Moving Average crossover strategy"""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"SMA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        if len(data) < self.long_window:
            return SignalType.HOLD, 0.0

        # Calculate SMAs
        prices = data['close'].values if 'close' in data.columns else data['price'].values
        short_sma = np.mean(prices[-self.short_window:])
        long_sma = np.mean(prices[-self.long_window:])

        # Previous SMAs for crossover detection
        if len(prices) > self.long_window:
            prev_short_sma = np.mean(prices[-(self.short_window+1):-1])
            prev_long_sma = np.mean(prices[-(self.long_window+1):-1])
        else:
            prev_short_sma = short_sma
            prev_long_sma = long_sma

        # Crossover detection
        if short_sma > long_sma and prev_short_sma <= prev_long_sma:
            return SignalType.BUY, 0.8
        elif short_sma < long_sma and prev_short_sma >= prev_long_sma:
            return SignalType.SELL, 0.8

        return SignalType.HOLD, 0.5


class MomentumStrategy(Strategy):
    """Momentum-based trading strategy"""

    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        super().__init__(f"Momentum ({lookback}d)")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        if len(data) < self.lookback:
            return SignalType.HOLD, 0.0

        prices = data['close'].values if 'close' in data.columns else data['price'].values
        momentum = (prices[-1] - prices[-self.lookback]) / prices[-self.lookback]

        confidence = min(abs(momentum) / self.threshold, 1.0)

        if momentum > self.threshold:
            return SignalType.BUY, confidence
        elif momentum < -self.threshold:
            return SignalType.SELL, confidence

        return SignalType.HOLD, confidence


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands"""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"Mean Reversion (BB {window})")
        self.window = window
        self.num_std = num_std

    def generate_signal(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        current_position: float = 0,
        **kwargs
    ) -> Tuple[SignalType, float]:
        if len(data) < self.window:
            return SignalType.HOLD, 0.0

        prices = data['close'].values if 'close' in data.columns else data['price'].values

        # Bollinger Bands
        sma = np.mean(prices[-self.window:])
        std = np.std(prices[-self.window:])
        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std

        current_price = prices[-1]

        # Calculate z-score for confidence
        z_score = (current_price - sma) / (std + 1e-8)
        confidence = min(abs(z_score) / self.num_std, 1.0)

        if current_price < lower_band:
            return SignalType.BUY, confidence
        elif current_price > upper_band:
            return SignalType.SELL, confidence

        return SignalType.HOLD, confidence


class BacktestingPlatform:
    """
    Comprehensive backtesting platform for financial forecasting models
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results: Dict[str, StrategyResult] = {}

    def reset(self):
        """Reset platform state"""
        self.results = {}

    def _initialize_portfolio(self) -> PortfolioState:
        """Initialize portfolio state"""
        return PortfolioState(
            cash=self.config.initial_capital,
            positions={},
            position_values={},
            entry_prices={},
            entry_times={},
            total_value=self.config.initial_capital
        )

    def _calculate_position_size(
        self,
        portfolio: PortfolioState,
        price: float,
        confidence: float = 1.0,
        volatility: float = None,
        win_rate: float = 0.5,
        avg_win: float = 0.03,
        avg_loss: float = 0.02
    ) -> float:
        """Calculate position size based on configured method"""
        available_cash = portfolio.cash
        portfolio_value = portfolio.total_value

        if self.config.position_sizing == PositionSizingMethod.FIXED:
            # Fixed percentage of portfolio
            position_value = portfolio_value * self.config.max_position_size * confidence

        elif self.config.position_sizing == PositionSizingMethod.KELLY:
            # Kelly Criterion
            if win_rate > 0 and win_rate < 1 and avg_loss > 0:
                b = avg_win / avg_loss
                q = 1 - win_rate
                kelly_fraction = (win_rate * b - q) / b
                kelly_fraction = max(0, kelly_fraction) * self.config.kelly_fraction
                kelly_fraction = min(kelly_fraction, self.config.max_position_size)
                position_value = portfolio_value * kelly_fraction * confidence
            else:
                position_value = portfolio_value * self.config.min_position_size

        elif self.config.position_sizing == PositionSizingMethod.VOLATILITY:
            # Volatility-based sizing
            if volatility and volatility > 0:
                vol_adjusted_size = self.config.target_volatility / volatility
                vol_adjusted_size = min(vol_adjusted_size, self.config.max_position_size)
                position_value = portfolio_value * vol_adjusted_size * confidence
            else:
                position_value = portfolio_value * self.config.max_position_size * confidence

        elif self.config.position_sizing == PositionSizingMethod.CONFIDENCE:
            # Confidence-based sizing
            size_fraction = self.config.min_position_size + \
                           (self.config.max_position_size - self.config.min_position_size) * confidence
            position_value = portfolio_value * size_fraction

        else:  # EQUAL_WEIGHT
            position_value = portfolio_value * self.config.max_position_size

        # Clip to available cash and position limits
        position_value = min(position_value, available_cash)
        position_value = max(position_value, 0)

        # Calculate quantity
        quantity = position_value / (price * (1 + self.config.commission_rate + self.config.slippage_rate))
        return max(0, int(quantity))

    def _apply_transaction_costs(self, value: float, is_buy: bool = True) -> Tuple[float, float, float]:
        """Apply transaction costs"""
        commission = value * self.config.commission_rate
        slippage = value * self.config.slippage_rate

        if is_buy:
            total_cost = value + commission + slippage
        else:
            total_cost = value - commission - slippage

        return total_cost, commission, slippage

    def _check_risk_limits(
        self,
        portfolio: PortfolioState,
        ticker: str,
        current_price: float
    ) -> Optional[str]:
        """Check if risk limits are breached"""
        if ticker not in portfolio.positions or portfolio.positions[ticker] == 0:
            return None

        entry_price = portfolio.entry_prices.get(ticker, current_price)
        pnl_pct = (current_price - entry_price) / entry_price

        if self.config.use_stop_loss and pnl_pct < -self.config.stop_loss:
            return f"Stop-loss triggered ({pnl_pct:.2%})"

        if self.config.use_take_profit and pnl_pct > self.config.take_profit:
            return f"Take-profit triggered ({pnl_pct:.2%})"

        return None

    def run_backtest(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        tickers: List[str] = None,
        model_name: str = "Unknown"
    ) -> StrategyResult:
        """
        Run backtest for a single strategy

        Args:
            strategy: Trading strategy to backtest
            prices: DataFrame with columns [timestamp, ticker, price/close]
            predictions: Optional array of model predictions
            tickers: List of tickers (if multi-asset)
            model_name: Name of the model generating predictions

        Returns:
            StrategyResult with performance metrics
        """
        logger.info(f"Running backtest for strategy: {strategy.name}")

        # Initialize
        portfolio = self._initialize_portfolio()
        result = StrategyResult(
            strategy_name=strategy.name,
            model_name=model_name
        )

        # Reset strategy if applicable
        if hasattr(strategy, 'reset'):
            strategy.reset()

        # Get unique timestamps
        if 'timestamp' in prices.columns:
            timestamps = sorted(prices['timestamp'].unique())
        else:
            timestamps = list(range(len(prices)))

        # Get ticker list
        if tickers is None:
            if 'ticker' in prices.columns:
                tickers = prices['ticker'].unique().tolist()
            else:
                tickers = ['ASSET']

        # Track performance
        portfolio_values = [self.config.initial_capital]
        result_timestamps = [timestamps[0] if len(timestamps) > 0 else pd.Timestamp.now()]
        open_trades: Dict[str, TradeRecord] = {}
        all_trades: List[TradeRecord] = []

        # Run through each timestamp
        for i, ts in enumerate(timestamps):
            # Get current prices
            if 'timestamp' in prices.columns:
                current_data = prices[prices['timestamp'] == ts]
            else:
                current_data = prices.iloc[[i]]

            current_prices = {}
            for ticker in tickers:
                if 'ticker' in current_data.columns:
                    ticker_data = current_data[current_data['ticker'] == ticker]
                    if len(ticker_data) > 0:
                        price_col = 'close' if 'close' in ticker_data.columns else 'price'
                        current_prices[ticker] = ticker_data[price_col].values[0]
                else:
                    price_col = 'close' if 'close' in current_data.columns else 'price'
                    current_prices[ticker] = current_data[price_col].values[0]

            # Update portfolio value
            portfolio.update_value(current_prices)

            # Check risk limits and close positions if needed
            for ticker in list(portfolio.positions.keys()):
                if ticker not in current_prices:
                    continue

                exit_reason = self._check_risk_limits(
                    portfolio, ticker, current_prices[ticker]
                )

                if exit_reason:
                    # Close position
                    qty = portfolio.positions[ticker]
                    exit_price = current_prices[ticker]
                    exit_value, commission, slippage = self._apply_transaction_costs(
                        qty * exit_price, is_buy=False
                    )

                    portfolio.cash += exit_value

                    # Record trade
                    if ticker in open_trades:
                        trade = open_trades[ticker]
                        trade.exit_price = exit_price
                        trade.exit_timestamp = ts
                        trade.pnl = exit_value - (trade.entry_price * trade.quantity)
                        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                        trade.holding_period = i - timestamps.index(trade.timestamp) if trade.timestamp in timestamps else 0
                        trade.exit_reason = exit_reason
                        trade.commission += commission
                        trade.slippage += slippage
                        all_trades.append(trade)
                        del open_trades[ticker]

                    del portfolio.positions[ticker]
                    del portfolio.entry_prices[ticker]
                    if ticker in portfolio.entry_times:
                        del portfolio.entry_times[ticker]

            # Generate signals for each ticker
            for ticker in tickers:
                if ticker not in current_prices:
                    continue

                current_price = current_prices[ticker]
                current_position = portfolio.positions.get(ticker, 0)

                # Get historical data for this ticker
                if 'ticker' in prices.columns:
                    hist_data = prices[
                        (prices['ticker'] == ticker) &
                        (prices['timestamp'] <= ts)
                    ].copy()
                else:
                    hist_data = prices.iloc[:i+1].copy()

                # Get predictions if available
                pred_slice = None
                if predictions is not None and len(predictions) > i:
                    pred_slice = predictions[:i+1]

                # Generate signal
                signal, confidence = strategy.generate_signal(
                    data=hist_data,
                    predictions=pred_slice,
                    current_position=current_position,
                    current_price=current_price
                )

                # Execute signal
                if signal == SignalType.BUY and current_position == 0:
                    # Calculate position size
                    volatility = hist_data['close'].pct_change().std() * np.sqrt(252) if 'close' in hist_data.columns and len(hist_data) > 1 else None
                    quantity = self._calculate_position_size(
                        portfolio, current_price, confidence, volatility
                    )

                    if quantity > 0:
                        cost, commission, slippage = self._apply_transaction_costs(
                            quantity * current_price, is_buy=True
                        )

                        if cost <= portfolio.cash:
                            portfolio.cash -= cost
                            portfolio.positions[ticker] = quantity
                            portfolio.entry_prices[ticker] = current_price
                            portfolio.entry_times[ticker] = ts

                            # Create trade record
                            trade = TradeRecord(
                                timestamp=ts,
                                ticker=ticker,
                                signal=signal,
                                entry_price=current_price,
                                quantity=quantity,
                                position_value=quantity * current_price,
                                commission=commission,
                                slippage=slippage,
                                confidence=confidence,
                                model_name=model_name
                            )
                            open_trades[ticker] = trade

                elif signal == SignalType.SELL and current_position > 0:
                    # Close position
                    qty = portfolio.positions[ticker]
                    exit_value, commission, slippage = self._apply_transaction_costs(
                        qty * current_price, is_buy=False
                    )

                    portfolio.cash += exit_value

                    # Record trade
                    if ticker in open_trades:
                        trade = open_trades[ticker]
                        trade.exit_price = current_price
                        trade.exit_timestamp = ts
                        trade.pnl = exit_value - (trade.entry_price * trade.quantity)
                        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                        trade.holding_period = i - timestamps.index(trade.timestamp) if isinstance(trade.timestamp, type(timestamps[0])) and trade.timestamp in timestamps else 1
                        trade.exit_reason = "Signal"
                        trade.commission += commission
                        trade.slippage += slippage
                        all_trades.append(trade)
                        del open_trades[ticker]

                    del portfolio.positions[ticker]
                    del portfolio.entry_prices[ticker]
                    if ticker in portfolio.entry_times:
                        del portfolio.entry_times[ticker]

            # Record portfolio value
            portfolio.update_value(current_prices)
            portfolio_values.append(portfolio.total_value)
            result_timestamps.append(ts)

        # Close any remaining positions at end
        for ticker, qty in portfolio.positions.items():
            if ticker in current_prices:
                exit_value, _, _ = self._apply_transaction_costs(
                    qty * current_prices[ticker], is_buy=False
                )
                portfolio.cash += exit_value

                if ticker in open_trades:
                    trade = open_trades[ticker]
                    trade.exit_price = current_prices[ticker]
                    trade.exit_timestamp = timestamps[-1] if timestamps else pd.Timestamp.now()
                    trade.pnl = exit_value - (trade.entry_price * trade.quantity)
                    trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) if trade.entry_price * trade.quantity > 0 else 0
                    trade.exit_reason = "End of backtest"
                    all_trades.append(trade)

        # Calculate performance metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        # Populate result
        result.trades = all_trades
        result.portfolio_values = portfolio_values.tolist()
        result.timestamps = result_timestamps
        result.returns = returns

        result.total_return = (portfolio_values[-1] / self.config.initial_capital - 1) * 100

        # Annualized metrics (assuming 252 trading days)
        n_periods = len(returns)
        if n_periods > 0:
            result.annualized_return = ((1 + result.total_return/100) ** (252 / n_periods) - 1) * 100
            result.volatility = np.std(returns) * np.sqrt(252) * 100
            result.sharpe_ratio = calculate_sharpe_ratio(returns, self.config.risk_free_rate)
            result.sortino_ratio = calculate_sortino_ratio(returns, self.config.risk_free_rate)
            result.max_drawdown = calculate_max_drawdown(returns) * 100
            result.calmar_ratio = calculate_calmar_ratio(returns)

        # Trade statistics
        result.total_trades = len(all_trades)
        if result.total_trades > 0:
            winning_trades = [t for t in all_trades if t.pnl > 0]
            losing_trades = [t for t in all_trades if t.pnl < 0]

            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            result.win_rate = result.winning_trades / result.total_trades * 100

            total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
            result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            result.avg_trade_pnl = sum(t.pnl for t in all_trades) / result.total_trades
            result.avg_holding_period = sum(t.holding_period for t in all_trades) / result.total_trades

        # Store result
        self.results[strategy.name] = result

        logger.info(f"Backtest complete: {strategy.name}")
        logger.info(f"  Total Return: {result.total_return:.2f}%")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        logger.info(f"  Total Trades: {result.total_trades}")

        return result

    def run_walk_forward(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        predictions: np.ndarray,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        model_name: str = "Unknown"
    ) -> List[StrategyResult]:
        """
        Run walk-forward optimization/validation

        Args:
            strategy: Trading strategy
            prices: Price data
            predictions: Model predictions
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of data for training window
            model_name: Name of the model

        Returns:
            List of StrategyResult for each walk-forward period
        """
        logger.info(f"Running walk-forward validation with {n_splits} splits")

        n_samples = len(prices)
        results = []

        # Calculate split sizes
        test_size = n_samples // n_splits

        for i in range(n_splits):
            # Define test period
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)

            # Get test data
            test_prices = prices.iloc[test_start:test_end].copy()
            test_predictions = predictions[test_start:test_end] if predictions is not None else None

            # Reset strategy
            if hasattr(strategy, 'reset'):
                strategy.reset()

            # Run backtest on test period
            result = self.run_backtest(
                strategy=strategy,
                prices=test_prices,
                predictions=test_predictions,
                model_name=f"{model_name}_fold{i+1}"
            )

            result.strategy_name = f"{strategy.name} (Fold {i+1})"
            results.append(result)

        # Calculate aggregate statistics
        logger.info("Walk-forward validation complete:")
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        std_sharpe = np.std([r.sharpe_ratio for r in results])
        avg_return = np.mean([r.total_return for r in results])

        logger.info(f"  Avg Sharpe: {avg_sharpe:.2f} ± {std_sharpe:.2f}")
        logger.info(f"  Avg Return: {avg_return:.2f}%")

        return results

    def compare_strategies(
        self,
        strategies: List[Strategy],
        prices: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        model_name: str = "Unknown"
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data

        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(strategies)} strategies")

        comparison_data = []

        for strategy in strategies:
            result = self.run_backtest(
                strategy=strategy,
                prices=prices,
                predictions=predictions,
                model_name=model_name
            )
            comparison_data.append(result.to_dict())

        comparison_df = pd.DataFrame(comparison_data)

        # Rank strategies
        comparison_df['sharpe_rank'] = comparison_df['sharpe_ratio'].rank(ascending=False)
        comparison_df['return_rank'] = comparison_df['total_return'].rank(ascending=False)
        comparison_df['drawdown_rank'] = comparison_df['max_drawdown'].rank(ascending=True)  # Less negative is better

        return comparison_df

    def monte_carlo_simulation(
        self,
        result: StrategyResult,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on strategy returns

        Returns:
            Dictionary with confidence intervals for key metrics
        """
        if len(result.returns) == 0:
            return {}

        returns = result.returns
        n_periods = len(returns)

        # Bootstrap simulation
        simulated_sharpes = []
        simulated_returns = []
        simulated_drawdowns = []

        for _ in range(n_simulations):
            # Resample returns with replacement
            resampled = np.random.choice(returns, size=n_periods, replace=True)

            # Calculate metrics
            simulated_sharpes.append(calculate_sharpe_ratio(resampled))
            simulated_returns.append(np.sum(resampled) * 100)
            simulated_drawdowns.append(calculate_max_drawdown(resampled) * 100)

        # Calculate confidence intervals
        alpha = 1 - confidence_level

        return {
            'sharpe_mean': np.mean(simulated_sharpes),
            'sharpe_lower': np.percentile(simulated_sharpes, alpha/2 * 100),
            'sharpe_upper': np.percentile(simulated_sharpes, (1 - alpha/2) * 100),
            'return_mean': np.mean(simulated_returns),
            'return_lower': np.percentile(simulated_returns, alpha/2 * 100),
            'return_upper': np.percentile(simulated_returns, (1 - alpha/2) * 100),
            'drawdown_mean': np.mean(simulated_drawdowns),
            'drawdown_lower': np.percentile(simulated_drawdowns, alpha/2 * 100),
            'drawdown_upper': np.percentile(simulated_drawdowns, (1 - alpha/2) * 100),
        }

    def generate_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive backtest report

        Returns:
            Dictionary with all results and analysis
        """
        report = {
            'config': {
                'initial_capital': self.config.initial_capital,
                'commission_rate': self.config.commission_rate,
                'slippage_rate': self.config.slippage_rate,
                'max_position_size': self.config.max_position_size,
                'stop_loss': self.config.stop_loss,
                'take_profit': self.config.take_profit,
                'position_sizing': self.config.position_sizing.value,
                'risk_free_rate': self.config.risk_free_rate
            },
            'strategies': {},
            'comparison': None,
            'generated_at': datetime.now().isoformat()
        }

        # Add strategy results
        for name, result in self.results.items():
            report['strategies'][name] = result.to_dict()

            # Add Monte Carlo analysis
            mc_results = self.monte_carlo_simulation(result)
            if mc_results:
                report['strategies'][name]['monte_carlo'] = mc_results

        # Create comparison table if multiple strategies
        if len(self.results) > 1:
            comparison_data = [r.to_dict() for r in self.results.values()]
            report['comparison'] = comparison_data

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")

        return report

    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all backtested strategies"""
        if not self.results:
            return pd.DataFrame()

        data = []
        for name, result in self.results.items():
            data.append({
                'Strategy': name,
                'Model': result.model_name,
                'Total Return (%)': f"{result.total_return:.2f}",
                'Annual Return (%)': f"{result.annualized_return:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{result.sortino_ratio:.2f}",
                'Max Drawdown (%)': f"{result.max_drawdown:.2f}",
                'Calmar Ratio': f"{result.calmar_ratio:.2f}",
                'Win Rate (%)': f"{result.win_rate:.2f}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Total Trades': result.total_trades,
                'Volatility (%)': f"{result.volatility:.2f}"
            })

        return pd.DataFrame(data)


def run_comprehensive_backtest(
    model: torch.nn.Module,
    prices: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str = "PINN",
    output_dir: str = "results/backtests"
) -> Dict:
    """
    Convenience function to run comprehensive backtest with multiple strategies

    Args:
        model: Trained model
        prices: Price data DataFrame
        predictions: Model predictions
        model_name: Name of the model
        output_dir: Directory to save results

    Returns:
        Dictionary with all backtest results
    """
    # Create platform with default config
    platform = BacktestingPlatform()

    # Define strategies to compare
    strategies = [
        ModelBasedStrategy(model, name=f"{model_name} Strategy"),
        BuyAndHoldStrategy(),
        SMACrossoverStrategy(20, 50),
        SMACrossoverStrategy(10, 30),
        MomentumStrategy(20, 0.02),
        MeanReversionStrategy(20, 2.0)
    ]

    # Run backtests
    for strategy in strategies:
        platform.run_backtest(
            strategy=strategy,
            prices=prices,
            predictions=predictions,
            model_name=model_name
        )

    # Generate report
    output_path = f"{output_dir}/{model_name}_backtest_report.json"
    report = platform.generate_report(output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(platform.get_summary_table().to_string(index=False))
    print("=" * 80)

    return report


if __name__ == "__main__":
    # Demo with synthetic data
    print("Backtesting Platform Demo")
    print("=" * 50)

    # Generate synthetic price data
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    prices_data = {
        'timestamp': dates,
        'ticker': ['DEMO'] * n_days,
        'close': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n_days)))
    }
    prices_df = pd.DataFrame(prices_data)

    # Generate synthetic predictions (with some accuracy)
    actual_returns = np.diff(prices_df['close'].values) / prices_df['close'].values[:-1]
    noise = np.random.normal(0, 0.01, len(actual_returns))
    predicted_prices = prices_df['close'].values[:-1] * (1 + actual_returns * 0.5 + noise)
    predicted_prices = np.append(prices_df['close'].values[0], predicted_prices)

    # Create platform
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        stop_loss=0.03,
        take_profit=0.05
    )
    platform = BacktestingPlatform(config)

    # Run strategies
    strategies = [
        BuyAndHoldStrategy(),
        SMACrossoverStrategy(20, 50),
        MomentumStrategy(20, 0.02),
        MeanReversionStrategy(20, 2.0)
    ]

    for strategy in strategies:
        platform.run_backtest(
            strategy=strategy,
            prices=prices_df,
            predictions=predicted_prices,
            model_name="Synthetic"
        )

    # Print summary
    print("\n" + platform.get_summary_table().to_string(index=False))

    # Generate report
    report = platform.generate_report("results/demo_backtest.json")
    print(f"\nReport saved with {len(report['strategies'])} strategies")
