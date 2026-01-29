"""
Backtesting framework for trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logger import get_logger
from .metrics import MetricsCalculator, calculate_financial_metrics

logger = get_logger(__name__)


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
    metrics: Dict[str, float] = field(default_factory=dict)
    positions_history: List[Dict[str, Position]] = field(default_factory=list)

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
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        max_position_size: float = 0.2,  # 20% of portfolio
        stop_loss: float = 0.02,         # 2%
        take_profit: float = 0.05,       # 5%
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
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []

    def reset(self):
        """Reset backtester to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

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
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on portfolio constraints

        Args:
            ticker: Ticker symbol
            price: Current price
            confidence: Prediction confidence (0-1)

        Returns:
            Number of shares to buy
        """
        # Maximum dollar amount for this position
        max_value = self.initial_capital * self.max_position_size * confidence

        # Available cash
        available_cash = min(self.cash, max_value)

        # Calculate shares (accounting for commission)
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
                reason=reason
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
                reason=reason
            )
            self.trades.append(trade)

            logger.debug(f"SELL {quantity:.2f} {ticker} @ ${actual_price:.2f}")
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

        for timestamp in unique_timestamps:
            # Get data for this timestamp
            timestamp_data = data[data['timestamp'] == timestamp]

            # Build current prices dictionary
            current_prices = dict(zip(timestamp_data['ticker'], timestamp_data['price']))

            # Check stop-loss and take-profit
            self.check_stop_loss_take_profit(timestamp, current_prices)

            # Execute new signals
            for _, row in timestamp_data.iterrows():
                ticker = row['ticker']
                signal = row['signal']
                price = row['price']
                confidence = row.get('confidence', 1.0)

                self.execute_trade(
                    timestamp=timestamp,
                    ticker=ticker,
                    action=signal,
                    price=price,
                    confidence=confidence
                )

            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            timestamps.append(timestamp)

        # Calculate returns
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate metrics
        metrics = calculate_financial_metrics(returns)
        metrics['num_trades'] = len(self.trades)
        metrics['final_value'] = portfolio_values[-1]
        metrics['total_return_pct'] = ((portfolio_values[-1] / self.initial_capital) - 1) * 100

        # Create results
        results = BacktestResults(
            trades=self.trades,
            portfolio_values=portfolio_values.tolist(),
            timestamps=timestamps,
            returns=returns,
            metrics=metrics
        )

        logger.info(f"Backtest completed: {len(self.trades)} trades, "
                   f"Final value: ${portfolio_values[-1]:,.2f}, "
                   f"Return: {metrics['total_return_pct']:.2f}%")

        return results
