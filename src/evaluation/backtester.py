"""
Backtesting framework for trading strategies

Supports multiple position sizing strategies:
- Fixed percentage
- Kelly Criterion (full/half/quarter)
- Volatility-based
- Confidence-based

Integration with uncertainty estimation from trading agent.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger
from .metrics import MetricsCalculator, calculate_financial_metrics

logger = get_logger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods"""
    FIXED = "fixed"
    KELLY_FULL = "kelly_full"
    KELLY_HALF = "kelly_half"
    KELLY_QUARTER = "kelly_quarter"
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: pd.Timestamp
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    value: float
    commission: float = 0.0
    reason: str = ""
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    position_before: float = 0.0
    position_after: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Represents a position in a security"""
    ticker: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_price(self, price: float):
        """Update position with new price"""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class BacktestResults:
    """Results from a backtest"""
    trades: List[Trade] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: Dict[str, Union[float, str]] = field(default_factory=dict)
    positions_history: List[Dict[str, Position]] = field(default_factory=list)
    weights_history: List[Dict[str, float]] = field(default_factory=list)
    turnover: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'portfolio_value': self.portfolio_values,
            'returns': np.concatenate([[0], self.returns])
        })

    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS SUMMARY",
            "=" * 60,
            f"Total Trades: {len(self.trades)}",
            f"Final Portfolio Value: ${self.portfolio_values[-1]:,.2f}" if self.portfolio_values else "N/A",
            "",
            "Performance Metrics:",
            "-" * 60,
        ]

        for key, value in sorted(self.metrics.items()):
            lines.append(f"{key:30s}: {value:12.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Backtester:
    """
    Backtesting engine for trading strategies

    Supports multiple position sizing methods including Kelly Criterion.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        max_position_size: float = 0.2,  # 20% of portfolio
        stop_loss: float = 0.02,         # 2%
        take_profit: float = 0.05,       # 5%
        position_sizing_method: Union[str, PositionSizingMethod] = PositionSizingMethod.FIXED,
        risk_per_trade: float = 0.02,    # For fixed sizing
        turnover_cost: float = 0.001,    # Cost per unit turnover (e.g., 10 bps)
        enforce_signal_lag: bool = True,
    ):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss: Stop loss as fraction (e.g., 0.02 = 2% loss)
            take_profit: Take profit as fraction
            position_sizing_method: Method for calculating position sizes
            risk_per_trade: Base risk for fixed sizing (fraction of capital)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_per_trade = risk_per_trade
        self.turnover_cost = turnover_cost
        self.enforce_signal_lag = enforce_signal_lag

        # Position sizing
        if isinstance(position_sizing_method, str):
            position_sizing_method = PositionSizingMethod(position_sizing_method)
        self.position_sizing_method = position_sizing_method

        # Initialize position sizers
        self._init_position_sizers()

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []

        # Trading statistics for Kelly Criterion
        self.trade_stats = {
            'wins': 0,
            'losses': 0,
            'total_win_pct': 0.0,
            'total_loss_pct': 0.0
        }

        logger.info(f"Backtester initialized with {position_sizing_method.value} position sizing")

    def _init_position_sizers(self):
        """Initialize position sizing strategies"""
        try:
            from ..trading.position_sizing import (
                FixedRiskSizer,
                KellyCriterionSizer,
                VolatilityBasedSizer,
                ConfidenceBasedSizer
            )

            self.sizers = {
                PositionSizingMethod.FIXED: FixedRiskSizer(
                    risk_per_trade=self.risk_per_trade,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                ),
                PositionSizingMethod.KELLY_FULL: KellyCriterionSizer(
                    fractional_kelly=1.0,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                ),
                PositionSizingMethod.KELLY_HALF: KellyCriterionSizer(
                    fractional_kelly=0.5,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                ),
                PositionSizingMethod.KELLY_QUARTER: KellyCriterionSizer(
                    fractional_kelly=0.25,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                ),
                PositionSizingMethod.VOLATILITY: VolatilityBasedSizer(
                    target_volatility=0.15,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                ),
                PositionSizingMethod.CONFIDENCE: ConfidenceBasedSizer(
                    base_risk=self.risk_per_trade,
                    initial_capital=self.initial_capital,
                    max_position_pct=self.max_position_size
                )
            }
        except ImportError:
            logger.warning("Position sizing module not found, using basic sizing")
            self.sizers = {}

    def reset(self):
        """Reset backtester to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.positions_history = []
        self.weights_history = []
        self.trade_stats = {
            'wins': 0,
            'losses': 0,
            'total_win_pct': 0.0,
            'total_loss_pct': 0.0
        }

    def update_trade_stats(self, pnl_pct: float):
        """
        Update trade statistics for Kelly Criterion calculation

        Args:
            pnl_pct: Profit/loss percentage of completed trade
        """
        if pnl_pct > 0:
            self.trade_stats['wins'] += 1
            self.trade_stats['total_win_pct'] += pnl_pct
        elif pnl_pct < 0:
            self.trade_stats['losses'] += 1
            self.trade_stats['total_loss_pct'] += abs(pnl_pct)

    def get_kelly_params(self) -> Dict[str, float]:
        """
        Calculate parameters needed for Kelly Criterion

        Returns:
            Dict with win_rate, avg_win, avg_loss
        """
        total_trades = self.trade_stats['wins'] + self.trade_stats['losses']

        if total_trades < 10:  # Need minimum trades for reliable estimate
            # Use conservative defaults
            return {
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': 0.02
            }

        win_rate = self.trade_stats['wins'] / total_trades
        avg_win = self.trade_stats['total_win_pct'] / max(1, self.trade_stats['wins'])
        avg_loss = self.trade_stats['total_loss_pct'] / max(1, self.trade_stats['losses'])

        # Ensure reasonable bounds
        win_rate = np.clip(win_rate, 0.1, 0.9)
        avg_win = max(0.001, avg_win)
        avg_loss = max(0.001, avg_loss)

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value

        Args:
            prices: Dictionary of ticker -> current price

        Returns:
            Total portfolio value
        """
        # Update positions with current prices
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.update_price(prices[ticker])

        # Sum up cash and positions
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        confidence: float = 1.0,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on selected sizing method

        Args:
            ticker: Ticker symbol
            price: Current price
            confidence: Prediction confidence (0-1)
            volatility: Stock volatility (for volatility-based sizing)

        Returns:
            Number of shares to buy
        """
        current_capital = self.get_portfolio_value({ticker: price})

        # Use position sizer if available
        if self.sizers and self.position_sizing_method in self.sizers:
            sizer = self.sizers[self.position_sizing_method]

            try:
                if self.position_sizing_method in [
                    PositionSizingMethod.KELLY_FULL,
                    PositionSizingMethod.KELLY_HALF,
                    PositionSizingMethod.KELLY_QUARTER
                ]:
                    # Kelly Criterion needs trade stats
                    kelly_params = self.get_kelly_params()
                    result = sizer.calculate(
                        current_capital=current_capital,
                        current_price=price,
                        win_rate=kelly_params['win_rate'],
                        avg_win=kelly_params['avg_win'],
                        avg_loss=kelly_params['avg_loss'],
                        confidence=confidence
                    )

                elif self.position_sizing_method == PositionSizingMethod.VOLATILITY:
                    # Volatility-based sizing
                    vol = volatility or 0.25  # Default 25% annualized
                    result = sizer.calculate(
                        current_capital=current_capital,
                        current_price=price,
                        stock_volatility=vol
                    )

                elif self.position_sizing_method == PositionSizingMethod.CONFIDENCE:
                    # Confidence-based sizing
                    result = sizer.calculate(
                        current_capital=current_capital,
                        current_price=price,
                        confidence=confidence
                    )

                else:
                    # Fixed risk sizing
                    result = sizer.calculate(
                        current_capital=current_capital,
                        current_price=price
                    )

                # Ensure we have enough cash
                max_shares = int(self.cash / (price * (1 + self.commission_rate + self.slippage_rate)))
                return min(result.position_size, max_shares)

            except Exception as e:
                logger.warning(f"Position sizer error: {e}, using fallback")

        # Fallback: basic position sizing
        max_value = current_capital * self.max_position_size * confidence
        available_cash = min(self.cash, max_value)
        shares = int(available_cash / (price * (1 + self.commission_rate + self.slippage_rate)))

        return max(0, shares)

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        action: str,
        price: float,
        quantity: Optional[float] = None,
        confidence: float = 1.0,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Execute a trade

        Args:
            timestamp: Trade timestamp
            ticker: Ticker symbol
            action: 'BUY', 'SELL', or 'HOLD'
            price: Current price
            quantity: Optional quantity (calculated if None)
            confidence: Prediction confidence
            reason: Reason for trade

        Returns:
            Trade object or None if trade not executed
        """
        if action == 'HOLD':
            return None

        # Apply slippage
        actual_price = price * (1 + self.slippage_rate) if action == 'BUY' else price * (1 - self.slippage_rate)

        if action == 'BUY':
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self.calculate_position_size(ticker, actual_price, confidence)

            if quantity == 0:
                logger.debug(f"Insufficient funds to buy {ticker}")
                return None

            # Calculate cost
            value = quantity * actual_price
            commission = value * self.commission_rate
            total_cost = value + commission

            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {ticker}: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return None

            position_before = self.positions[ticker].quantity if ticker in self.positions else 0.0
            position_after = position_before + quantity
            slippage_amount = price * self.slippage_rate * quantity

            # Execute buy
            self.cash -= total_cost

            # Add or update position
            if ticker in self.positions:
                # Average down/up
                pos = self.positions[ticker]
                total_quantity = pos.quantity + quantity
                avg_price = (pos.entry_price * pos.quantity + actual_price * quantity) / total_quantity

                pos.quantity = total_quantity
                pos.entry_price = avg_price
                pos.current_price = actual_price
                pos.update_price(actual_price)
            else:
                # New position
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    entry_price=actual_price,
                    entry_time=timestamp,
                    current_price=actual_price,
                    stop_loss=actual_price * (1 - self.stop_loss),
                    take_profit=actual_price * (1 + self.take_profit)
                )
                self.positions[ticker].update_price(actual_price)

            # Record trade
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                action=action,
                price=actual_price,
                quantity=quantity,
                value=value,
                commission=commission,
                reason=reason,
                position_before=position_before,
                position_after=position_after,
                slippage=slippage_amount
            )
            self.trades.append(trade)

            logger.debug(f"BUY {quantity:.2f} {ticker} @ ${actual_price:.2f}")
            return trade

        elif action == 'SELL':
            # Check if we have this position
            if ticker not in self.positions:
                logger.debug(f"No position to sell for {ticker}")
                return None

            pos = self.positions[ticker]

            # Quantity to sell
            if quantity is None or quantity > pos.quantity:
                quantity = pos.quantity

            # Calculate proceeds
            value = quantity * actual_price
            commission = value * self.commission_rate
            proceeds = value - commission

            # Calculate PnL percentage for Kelly Criterion tracking
            entry_value = pos.entry_price * quantity
            pnl = (actual_price - pos.entry_price) * quantity - commission
            pnl_pct = pnl / entry_value if entry_value > 0 else 0

            # Update trade statistics for Kelly Criterion
            self.update_trade_stats(pnl_pct)

            position_before = pos.quantity
            position_after = pos.quantity - quantity
            slippage_amount = price * self.slippage_rate * quantity

            # Execute sell
            self.cash += proceeds

            # Update or remove position
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self.positions[ticker]
            else:
                pos.update_price(actual_price)

            # Record trade
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                action=action,
                price=actual_price,
                quantity=quantity,
                value=value,
                commission=commission,
                reason=reason,
                pnl=pnl,
                pnl_percent=pnl_pct * 100,
                position_before=position_before,
                position_after=position_after,
                slippage=slippage_amount
            )
            self.trades.append(trade)

            logger.debug(f"SELL {quantity:.2f} {ticker} @ ${actual_price:.2f} (PnL: {pnl_pct:.2%})")
            return trade

        return None

    def check_stop_loss_take_profit(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float]
    ) -> List[Trade]:
        """
        Check and execute stop-loss and take-profit orders

        Args:
            timestamp: Current timestamp
            prices: Dictionary of current prices

        Returns:
            List of executed trades
        """
        trades = []

        for ticker in list(self.positions.keys()):
            if ticker not in prices:
                continue

            pos = self.positions[ticker]
            current_price = prices[ticker]

            # Check stop-loss
            if pos.stop_loss and current_price <= pos.stop_loss:
                trade = self.execute_trade(
                    timestamp=timestamp,
                    ticker=ticker,
                    action='SELL',
                    price=current_price,
                    quantity=pos.quantity,
                    reason=f"Stop-loss triggered at ${current_price:.2f}"
                )
                if trade:
                    trades.append(trade)

            # Check take-profit
            elif pos.take_profit and current_price >= pos.take_profit:
                trade = self.execute_trade(
                    timestamp=timestamp,
                    ticker=ticker,
                    action='SELL',
                    price=current_price,
                    quantity=pos.quantity,
                    reason=f"Take-profit triggered at ${current_price:.2f}"
                )
                if trade:
                    trades.append(trade)

        return trades

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame
    ) -> BacktestResults:
        """
        Run backtest on historical signals

        Args:
            signals: DataFrame with columns [timestamp, ticker, signal, confidence]
                     signal: 'BUY', 'SELL', or 'HOLD'
            prices: DataFrame with columns [timestamp, ticker, price]

        Returns:
            BacktestResults object
        """
        logger.info("Starting backtest...")
        self.reset()

        # Merge signals and prices
        data = pd.merge(
            signals,
            prices,
            on=['timestamp', 'ticker'],
            how='inner'
        )

        # Sort by timestamp
        data = data.sort_values('timestamp')

        # Track portfolio over time
        portfolio_values = []
        timestamps = []

        # Get unique timestamps
        unique_timestamps = data['timestamp'].unique()

        pending_rows = None  # signals to execute on current timestamp (lagged)

        for timestamp in unique_timestamps:
            # Get data for this timestamp
            timestamp_data = data[data['timestamp'] == timestamp]

            # Build current prices dictionary
            current_prices = dict(zip(timestamp_data['ticker'], timestamp_data['price']))

            # Check stop-loss and take-profit before new signals
            self.check_stop_loss_take_profit(timestamp, current_prices)

            # Execute lagged signals (previous timestamp predictions)
            rows_to_execute = pending_rows if self.enforce_signal_lag else timestamp_data
            if rows_to_execute is not None:
                for _, row in rows_to_execute.iterrows():
                    ticker = cast(str, row['ticker'])
                    signal = cast(str, row['signal'])
                    price = float(cast(Any, row['price']))
                    conf_val = row.get('confidence', 1.0)
                    if isinstance(conf_val, (pd.Series, pd.DataFrame)):
                        conf_val = conf_val.iloc[0] if hasattr(conf_val, "iloc") else 1.0
                    if bool(pd.isna(conf_val)):
                        conf_val = 1.0
                    confidence = float(cast(Any, conf_val))

                    self.execute_trade(
                        timestamp=timestamp,
                        ticker=ticker,
                        action=signal,
                        price=price,
                        confidence=confidence
                    )

            # Queue current signals for next timestamp if lag enforced
            pending_rows = timestamp_data if self.enforce_signal_lag else None

            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            timestamps.append(timestamp)

            # Track weights for turnover/exposure
            if portfolio_value > 0:
                weights = {
                    t: pos.market_value / portfolio_value for t, pos in self.positions.items()
                }
            else:
                weights = {}

            self.positions_history.append(self.positions.copy())
            self.weights_history.append(weights)

        # Calculate returns
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Compute turnover from weights and apply turnover-based cost
        if self.weights_history:
            weights_df = pd.DataFrame(self.weights_history).fillna(0.0)
            turnover = weights_df.diff().abs().sum(axis=1)
            turnover.iloc[0] = 0.0
            turnover_array = turnover.to_numpy()
        else:
            turnover_array = np.zeros_like(portfolio_values)

        if len(turnover_array) > 1 and self.turnover_cost > 0:
            returns -= self.turnover_cost * turnover_array[1:]

        # Calculate metrics
        metrics: Dict[str, Union[float, str]] = dict(
            cast(Dict[str, float], calculate_financial_metrics(returns))
        )
        metrics['num_trades'] = len(self.trades)
        metrics['final_value'] = portfolio_values[-1]
        metrics['total_return_pct'] = ((portfolio_values[-1] / self.initial_capital) - 1) * 100
        metrics['average_turnover'] = float(turnover_array.mean()) if len(turnover_array) else 0.0
        metrics['trading_days_pct'] = float(np.mean(turnover_array > 0)) if len(turnover_array) else 0.0
        metrics['turnover_cost'] = self.turnover_cost

        # Add position sizing info
        metrics['position_sizing_method'] = self.position_sizing_method.value
        metrics['trade_win_rate'] = (
            self.trade_stats['wins'] / max(1, self.trade_stats['wins'] + self.trade_stats['losses'])
        )
        metrics['avg_win_pct'] = (
            self.trade_stats['total_win_pct'] / max(1, self.trade_stats['wins']) * 100
        )
        metrics['avg_loss_pct'] = (
            self.trade_stats['total_loss_pct'] / max(1, self.trade_stats['losses']) * 100
        )

        # Create results
        results = BacktestResults(
            trades=self.trades,
            portfolio_values=portfolio_values.tolist(),
            timestamps=timestamps,
            returns=returns,
            metrics=metrics,
            positions_history=self.positions_history,
            weights_history=self.weights_history,
            turnover=turnover_array
        )

        logger.info(f"Backtest completed ({self.position_sizing_method.value}): {len(self.trades)} trades, "
                   f"Final value: ${portfolio_values[-1]:,.2f}, "
                   f"Return: {metrics['total_return_pct']:.2f}%")

        return results


def compare_position_sizing_methods(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 100000.0,
    methods: Optional[List[PositionSizingMethod]] = None
) -> Dict[str, BacktestResults]:
    """
    Compare different position sizing methods on the same signals

    Args:
        signals: Trading signals DataFrame
        prices: Price history DataFrame
        initial_capital: Starting capital
        methods: List of methods to compare (default: all)

    Returns:
        Dictionary mapping method name to BacktestResults
    """
    if methods is None:
        methods = [
            PositionSizingMethod.FIXED,
            PositionSizingMethod.KELLY_HALF,
            PositionSizingMethod.KELLY_QUARTER,
            PositionSizingMethod.CONFIDENCE
        ]

    results = {}

    for method in methods:
        logger.info(f"Running backtest with {method.value} sizing...")

        backtester = Backtester(
            initial_capital=initial_capital,
            position_sizing_method=method
        )

        result = backtester.run_backtest(signals, prices)
        results[method.value] = result

    # Print comparison
    print("\n" + "=" * 80)
    print("Position Sizing Method Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Final Value':<15} {'Return %':<12} {'Sharpe':<10} {'Trades':<8}")
    print("-" * 80)

    for method_name, result in results.items():
        m = result.metrics
        print(f"{method_name:<20} "
              f"${m.get('final_value', 0):>12,.2f} "
              f"{m.get('total_return_pct', 0):>10.2f}% "
              f"{m.get('sharpe_ratio', 0):>8.2f} "
              f"{m.get('num_trades', 0):>6}")

    print("=" * 80)

    return results
