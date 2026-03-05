"""
Real-Time Trading Agent

A comprehensive trading agent that:
- Fetches real-time or simulated market data
- Uses trained ML models to generate trading signals
- Executes trades (paper trading or simulation)
- Manages portfolio and risk
- Provides real-time performance tracking

DISCLAIMER: This is for educational/research purposes only.
Not financial advice. Use at your own risk.
"""

import time
import threading
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from queue import Queue
from collections import deque
import pandas as pd
import numpy as np
import torch

# Maximum history sizes to prevent unbounded memory growth
MAX_ORDERS_HISTORY = 10000
MAX_TRADES_HISTORY = 10000
MAX_PORTFOLIO_HISTORY = 50000  # ~200 per day * 250 trading days = 1 year
MAX_ALERTS_HISTORY = 1000

from .realtime_data import RealTimeDataService, SimulatedDataService, MarketQuote
from .agent import SignalGenerator, UncertaintyEstimator, Signal
from .position_sizing import (
    PositionSizer,
    FixedRiskSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
    ConfidenceBasedSizer,
    PositionSizeResult
)
from ..models.model_registry import ModelRegistry, get_model_registry
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class TradingMode(Enum):
    """Trading mode options"""
    PAPER = "paper"  # Paper trading with real-time data
    SIMULATION = "simulation"  # Historical simulation
    LIVE = "live"  # Live trading (disabled by default for safety)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    timestamp: datetime
    ticker: str
    side: str  # 'BUY' or 'SELL'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    reason: str = ""
    model_confidence: float = 0.0
    signal_expected_return: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'side': self.side,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'reason': self.reason,
            'model_confidence': self.model_confidence
        }


@dataclass
class Position:
    """Portfolio position"""
    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    entry_time: datetime
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        self.unrealized_pnl_pct = ((current_price / self.avg_entry_price) - 1) * 100 if self.avg_entry_price > 0 else 0

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'avg_entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


@dataclass
class Trade:
    """Executed trade record"""
    trade_id: str
    order_id: str
    timestamp: datetime
    ticker: str
    side: str
    quantity: float
    price: float
    value: float
    commission: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    model_used: str = ""
    signal_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'value': self.value,
            'commission': self.commission,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'model_used': self.model_used,
            'signal_confidence': self.signal_confidence
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    daily_pnl: float
    total_pnl: float
    total_pnl_pct: float
    num_positions: int
    num_trades: int

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash': self.cash,
            'positions_value': self.positions_value,
            'total_value': self.total_value,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'num_positions': self.num_positions,
            'num_trades': self.num_trades
        }


@dataclass
class Alert:
    """Trading alert"""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'signal', 'risk', 'price', 'execution', 'error'
    severity: str  # 'info', 'warning', 'critical'
    ticker: Optional[str]
    message: str
    data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'ticker': self.ticker,
            'message': self.message,
            'data': self.data
        }


class RealTimeTradingAgent:
    """
    Real-time trading agent with ML model integration

    Features:
    - Multiple model support with model selection
    - Real-time and simulated data modes
    - Paper trading with realistic execution
    - Risk management (stop-loss, take-profit, position limits)
    - Performance tracking and analytics
    - Alert system for important events
    - Trade journal with full history
    """

    def __init__(
        self,
        project_root: Path,
        initial_capital: float = 100000.0,
        trading_mode: TradingMode = TradingMode.PAPER,
        model_key: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,  # 0.05%
        max_position_pct: float = 0.20,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        confidence_threshold: float = 0.6,
        expected_return_threshold: float = 0.02,
        position_sizing_method: str = 'confidence',
        data_update_interval: float = 5.0,
        sequence_length: int = 60
    ):
        """
        Initialize the real-time trading agent

        Args:
            project_root: Project root path
            initial_capital: Starting capital
            trading_mode: Paper, simulation, or live
            model_key: Model to use for predictions
            tickers: List of tickers to trade
            commission_rate: Commission as fraction of trade
            slippage_rate: Slippage as fraction of price
            max_position_pct: Max position size as fraction of portfolio
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            confidence_threshold: Min confidence to trade
            expected_return_threshold: Min expected return to trade
            position_sizing_method: 'fixed', 'kelly', 'volatility', 'confidence'
            data_update_interval: Seconds between data updates
            sequence_length: Input sequence length for models
        """
        self.project_root = Path(project_root)
        self.config = get_config()

        # Trading parameters
        self.initial_capital = initial_capital
        self.trading_mode = trading_mode
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.confidence_threshold = confidence_threshold
        self.expected_return_threshold = expected_return_threshold
        self.sequence_length = sequence_length

        # Default tickers if not provided
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

        # Initialize model registry and load model
        self.model_registry = get_model_registry(self.project_root)
        self.current_model_key = model_key
        self.model = None
        self.signal_generator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_key:
            self.load_model(model_key)

        # Position sizing
        self._init_position_sizer(position_sizing_method)

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        # Use deques with maxlen to prevent unbounded memory growth
        self.orders: deque[Order] = deque(maxlen=MAX_ORDERS_HISTORY)
        self.trades: deque[Trade] = deque(maxlen=MAX_TRADES_HISTORY)
        self.portfolio_history: deque[PortfolioSnapshot] = deque(maxlen=MAX_PORTFOLIO_HISTORY)
        self.alerts: deque[Alert] = deque(maxlen=MAX_ALERTS_HISTORY)

        # Trading statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'max_drawdown': 0.0,
            'peak_value': initial_capital
        }

        # Data service
        self.data_service: Optional[RealTimeDataService] = None
        self._init_data_service(data_update_interval)

        # Prepared sequences for prediction
        self._sequences: Dict[str, torch.Tensor] = {}

        # Threading
        self._running = False
        self._trading_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._trade_callbacks: List[Callable[[Trade], None]] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._portfolio_callbacks: List[Callable[[PortfolioSnapshot], None]] = []

        # Counters for IDs
        self._order_counter = 0
        self._trade_counter = 0
        self._alert_counter = 0

        logger.info(f"RealTimeTradingAgent initialized in {trading_mode.value} mode")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Trading tickers: {self.tickers}")

    def _init_data_service(self, update_interval: float):
        """Initialize the data service based on trading mode"""
        if self.trading_mode == TradingMode.SIMULATION:
            self.data_service = SimulatedDataService(
                tickers=self.tickers,
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                speed_multiplier=10.0
            )
        else:
            self.data_service = RealTimeDataService(
                tickers=self.tickers,
                update_interval=update_interval
            )

    def _init_position_sizer(self, method: str):
        """Initialize position sizing strategy"""
        sizers = {
            'fixed': FixedRiskSizer(
                risk_per_trade=0.02,
                initial_capital=self.initial_capital,
                max_position_pct=self.max_position_pct
            ),
            'kelly': KellyCriterionSizer(
                fractional_kelly=0.5,
                initial_capital=self.initial_capital,
                max_position_pct=self.max_position_pct
            ),
            'volatility': VolatilityBasedSizer(
                target_volatility=0.15,
                initial_capital=self.initial_capital,
                max_position_pct=self.max_position_pct
            ),
            'confidence': ConfidenceBasedSizer(
                base_risk=0.05,
                initial_capital=self.initial_capital,
                max_position_pct=self.max_position_pct
            )
        }

        self.position_sizer = sizers.get(method, sizers['confidence'])
        self.position_sizing_method = method
        logger.info(f"Position sizing: {method}")

    def load_model(self, model_key: str) -> bool:
        """
        Load a model for predictions

        Args:
            model_key: Model key from registry

        Returns:
            True if successful
        """
        model = self.model_registry.load_model(model_key, device=self.device)

        if model is None:
            logger.error(f"Failed to load model: {model_key}")
            return False

        self.model = model
        self.current_model_key = model_key
        self.signal_generator = SignalGenerator(
            model=self.model,
            config=self.config,
            device=self.device,
            n_mc_samples=30  # Reduced for real-time performance
        )

        logger.info(f"Loaded model: {model_key}")
        self._create_alert('info', 'signal', None, f"Model loaded: {model_key}")

        return True

    def get_available_models(self) -> List[Dict]:
        """Get list of available models for selection"""
        models = []
        for key, info in self.model_registry.get_trained_models().items():
            # Skip aliases
            if key.startswith('pinn_') and key[5:] in self.model_registry.models:
                continue

            models.append({
                'key': key,
                'name': info.model_name,
                'type': info.model_type,
                'architecture': info.architecture,
                'description': info.description,
                'trained': info.trained,
                'training_date': info.training_date
            })

        return models

    def prepare_sequences(self, ticker: str) -> Optional[torch.Tensor]:
        """
        Prepare input sequences for a ticker

        Args:
            ticker: Ticker symbol

        Returns:
            Tensor of shape (1, seq_len, features) or None
        """
        # Get historical data
        data = self.data_service.fetch_historical_data(
            ticker,
            period='90d',
            interval='1d'
        )

        if data is None or len(data) < self.sequence_length:
            logger.warning(f"Insufficient data for {ticker}")
            return None

        # Calculate technical indicators
        indicators = self.data_service.calculate_indicators(ticker, data)

        # Prepare features
        try:
            features = pd.DataFrame({
                'close': data['close'],
                'volume': data['volume'],
                'volatility': indicators.get('volatility', data['close'].pct_change().rolling(20).std()),
                'rsi': indicators.get('rsi', pd.Series([50] * len(data))),
                'macd': indicators.get('macd', pd.Series([0] * len(data)))
            }).dropna()

            if len(features) < self.sequence_length:
                return None

            # Normalize features
            features_normalized = (features - features.mean()) / (features.std() + 1e-8)

            # Create sequence
            sequence = features_normalized.iloc[-self.sequence_length:].values

            # Convert to tensor
            tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq_len, features)

            self._sequences[ticker] = tensor
            return tensor

        except Exception as e:
            logger.error(f"Error preparing sequences for {ticker}: {e}")
            return None

    def generate_signal(self, ticker: str, quote: MarketQuote) -> Optional[Signal]:
        """
        Generate a trading signal for a ticker

        Args:
            ticker: Ticker symbol
            quote: Current market quote

        Returns:
            Signal object or None
        """
        if self.signal_generator is None:
            logger.warning("No model loaded")
            return None

        # Prepare sequences if needed
        if ticker not in self._sequences:
            self.prepare_sequences(ticker)

        sequence = self._sequences.get(ticker)
        if sequence is None:
            return None

        try:
            # Generate signals
            signals, uncertainty_details = self.signal_generator.generate_signals(
                sequences=sequence,
                current_prices=np.array([quote.price]),
                tickers=[ticker],
                timestamps=[pd.Timestamp(quote.timestamp)],
                threshold=self.expected_return_threshold,
                confidence_threshold=self.confidence_threshold,
                estimate_uncertainty=True,
                uncertainty_method='mc_dropout'
            )

            if signals:
                return signals[0]

        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")

        return None

    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        confidence: float
    ) -> int:
        """
        Calculate position size for a trade

        Args:
            ticker: Ticker symbol
            price: Current price
            confidence: Signal confidence

        Returns:
            Number of shares
        """
        portfolio_value = self.get_portfolio_value()

        try:
            if self.position_sizing_method == 'kelly':
                # Calculate Kelly parameters from trade history
                win_rate, avg_win, avg_loss = self._calculate_kelly_params()
                result = self.position_sizer.calculate(
                    current_capital=portfolio_value,
                    current_price=price,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    confidence=confidence
                )
            elif self.position_sizing_method == 'volatility':
                # Get volatility from data service
                indicators = self.data_service.calculate_indicators(ticker)
                volatility = indicators.get('volatility', pd.Series([0.25])).iloc[-1]
                result = self.position_sizer.calculate(
                    current_capital=portfolio_value,
                    current_price=price,
                    stock_volatility=float(volatility)
                )
            elif self.position_sizing_method == 'confidence':
                result = self.position_sizer.calculate(
                    current_capital=portfolio_value,
                    current_price=price,
                    confidence=confidence
                )
            else:
                result = self.position_sizer.calculate(
                    current_capital=portfolio_value,
                    current_price=price
                )

            # Ensure we have enough cash
            max_shares = int(self.cash / (price * (1 + self.commission_rate)))
            return min(result.position_size, max_shares)

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            # Fallback
            return int((portfolio_value * 0.02) / price)

    def _calculate_kelly_params(self) -> Tuple[float, float, float]:
        """Calculate Kelly Criterion parameters from trade history"""
        if len(self.trades) < 10:
            return 0.5, 0.02, 0.02

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0.5
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.02
        avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0.02

        return max(0.3, min(0.7, win_rate)), max(0.01, avg_win), max(0.01, avg_loss)

    def create_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reason: str = "",
        confidence: float = 0.0,
        expected_return: float = 0.0
    ) -> Order:
        """
        Create a new order

        Args:
            ticker: Ticker symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: Market, limit, stop, or stop-limit
            price: Limit price
            stop_price: Stop price
            reason: Reason for order
            confidence: Model confidence
            expected_return: Expected return from signal

        Returns:
            Order object
        """
        self._order_counter += 1
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:06d}"

        order = Order(
            order_id=order_id,
            timestamp=datetime.now(),
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reason=reason,
            model_confidence=confidence,
            signal_expected_return=expected_return
        )

        self.orders.append(order)
        logger.info(f"Created order: {order_id} - {side} {quantity} {ticker}")

        return order

    def execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """
        Execute an order

        Args:
            order: Order to execute
            current_price: Current market price

        Returns:
            Trade object if executed
        """
        # Apply slippage
        if order.side == 'BUY':
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)

        # Calculate trade value and commission
        value = order.quantity * execution_price
        commission = value * self.commission_rate

        with self._lock:
            if order.side == 'BUY':
                # Check if we have enough cash
                total_cost = value + commission
                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    order.reason = f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash:.2f}"
                    logger.warning(order.reason)
                    return None

                # Execute buy
                self.cash -= total_cost

                # Update or create position
                if order.ticker in self.positions:
                    pos = self.positions[order.ticker]
                    total_qty = pos.quantity + order.quantity
                    pos.avg_entry_price = (
                        pos.avg_entry_price * pos.quantity + execution_price * order.quantity
                    ) / total_qty
                    pos.quantity = total_qty
                else:
                    self.positions[order.ticker] = Position(
                        ticker=order.ticker,
                        quantity=order.quantity,
                        avg_entry_price=execution_price,
                        current_price=execution_price,
                        entry_time=datetime.now(),
                        stop_loss=execution_price * (1 - self.stop_loss_pct),
                        take_profit=execution_price * (1 + self.take_profit_pct)
                    )

                self.positions[order.ticker].update(execution_price)
                pnl = 0.0
                pnl_pct = 0.0

            else:  # SELL
                if order.ticker not in self.positions:
                    order.status = OrderStatus.REJECTED
                    order.reason = f"No position to sell for {order.ticker}"
                    return None

                pos = self.positions[order.ticker]
                sell_qty = min(order.quantity, pos.quantity)

                # Calculate PnL
                pnl = (execution_price - pos.avg_entry_price) * sell_qty - commission
                pnl_pct = ((execution_price / pos.avg_entry_price) - 1) * 100 if pos.avg_entry_price > 0 else 0

                # Execute sell
                proceeds = sell_qty * execution_price - commission
                self.cash += proceeds

                # Update or remove position
                pos.quantity -= sell_qty
                pos.realized_pnl += pnl

                if pos.quantity <= 0:
                    del self.positions[order.ticker]
                else:
                    pos.update(execution_price)

                # Update stats
                self.stats['total_pnl'] += pnl
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                    self.stats['largest_win'] = max(self.stats['largest_win'], pnl)
                elif pnl < 0:
                    self.stats['losing_trades'] += 1
                    self.stats['largest_loss'] = min(self.stats['largest_loss'], pnl)

        # Create trade record
        self._trade_counter += 1
        trade_id = f"TRD-{datetime.now().strftime('%Y%m%d')}-{self._trade_counter:06d}"

        trade = Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            timestamp=datetime.now(),
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            value=value,
            commission=commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            model_used=self.current_model_key or "",
            signal_confidence=order.model_confidence
        )

        self.trades.append(trade)
        self.stats['total_trades'] += 1

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission

        logger.info(
            f"Executed: {trade_id} - {order.side} {order.quantity} {order.ticker} "
            f"@ ${execution_price:.2f} (PnL: ${pnl:.2f})"
        )

        # Notify callbacks
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

        return trade

    def check_risk_limits(self, quote: MarketQuote) -> List[Order]:
        """
        Check stop-loss and take-profit levels

        Args:
            quote: Current market quote

        Returns:
            List of triggered orders
        """
        orders = []

        with self._lock:
            if quote.ticker not in self.positions:
                return orders

            pos = self.positions[quote.ticker]
            pos.update(quote.price)

            # Check stop-loss
            if pos.stop_loss and quote.price <= pos.stop_loss:
                order = self.create_order(
                    ticker=quote.ticker,
                    side='SELL',
                    quantity=pos.quantity,
                    reason=f"Stop-loss triggered at ${quote.price:.2f}"
                )
                orders.append(order)
                self._create_alert(
                    'warning', 'risk', quote.ticker,
                    f"Stop-loss triggered: {quote.ticker} @ ${quote.price:.2f}"
                )

            # Check take-profit
            elif pos.take_profit and quote.price >= pos.take_profit:
                order = self.create_order(
                    ticker=quote.ticker,
                    side='SELL',
                    quantity=pos.quantity,
                    reason=f"Take-profit triggered at ${quote.price:.2f}"
                )
                orders.append(order)
                self._create_alert(
                    'info', 'risk', quote.ticker,
                    f"Take-profit triggered: {quote.ticker} @ ${quote.price:.2f}"
                )

        return orders

    def process_quote(self, quote: MarketQuote):
        """
        Process a new market quote

        Args:
            quote: Market quote
        """
        # Update existing positions
        with self._lock:
            if quote.ticker in self.positions:
                self.positions[quote.ticker].update(quote.price)

        # Check risk limits
        risk_orders = self.check_risk_limits(quote)
        for order in risk_orders:
            self.execute_order(order, quote.price)

        # Generate trading signal
        signal = self.generate_signal(quote.ticker, quote)

        if signal and signal.action != 'HOLD':
            # Check if we should act on this signal
            if signal.confidence >= self.confidence_threshold:
                if signal.action == 'BUY' and quote.ticker not in self.positions:
                    # Calculate position size
                    quantity = self.calculate_position_size(
                        quote.ticker,
                        quote.price,
                        signal.confidence
                    )

                    if quantity > 0:
                        order = self.create_order(
                            ticker=quote.ticker,
                            side='BUY',
                            quantity=quantity,
                            reason=f"Model signal: expected return {signal.expected_return:.2%}",
                            confidence=signal.confidence,
                            expected_return=signal.expected_return
                        )
                        self.execute_order(order, quote.price)

                        self._create_alert(
                            'info', 'signal', quote.ticker,
                            f"BUY signal executed: {quote.ticker} @ ${quote.price:.2f} "
                            f"(confidence: {signal.confidence:.2%})"
                        )

                elif signal.action == 'SELL' and quote.ticker in self.positions:
                    pos = self.positions[quote.ticker]
                    order = self.create_order(
                        ticker=quote.ticker,
                        side='SELL',
                        quantity=pos.quantity,
                        reason=f"Model signal: expected return {signal.expected_return:.2%}",
                        confidence=signal.confidence,
                        expected_return=signal.expected_return
                    )
                    self.execute_order(order, quote.price)

                    self._create_alert(
                        'info', 'signal', quote.ticker,
                        f"SELL signal executed: {quote.ticker} @ ${quote.price:.2f} "
                        f"(confidence: {signal.confidence:.2%})"
                    )

        # Record portfolio snapshot
        self._record_portfolio_snapshot()

    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': portfolio_value,
            'total_pnl': portfolio_value - self.initial_capital,
            'total_pnl_pct': ((portfolio_value / self.initial_capital) - 1) * 100,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': self.stats['total_pnl'],
            'num_positions': len(self.positions),
            'positions': [pos.to_dict() for pos in self.positions.values()]
        }

    def get_performance_metrics(self) -> Dict:
        """Get trading performance metrics"""
        if not self.trades:
            return {'message': 'No trades yet'}

        returns = []
        for i, snapshot in enumerate(self.portfolio_history[1:], 1):
            prev_value = self.portfolio_history[i-1].total_value
            if prev_value > 0:
                returns.append((snapshot.total_value - prev_value) / prev_value)

        returns = np.array(returns) if returns else np.array([0])

        # Calculate metrics
        win_rate = (
            self.stats['winning_trades'] / self.stats['total_trades']
            if self.stats['total_trades'] > 0 else 0
        )

        # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0

        # Maximum Drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for snapshot in self.portfolio_history:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            drawdown = (peak - snapshot.total_value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': win_rate * 100,
            'total_pnl': self.stats['total_pnl'],
            'largest_win': self.stats['largest_win'],
            'largest_loss': self.stats['largest_loss'],
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'avg_trade_pnl': self.stats['total_pnl'] / max(1, self.stats['total_trades']),
            'model_used': self.current_model_key
        }

    def _record_portfolio_snapshot(self):
        """Record current portfolio state"""
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())

        # Calculate daily PnL
        if self.portfolio_history:
            daily_pnl = portfolio_value - self.portfolio_history[-1].total_value
        else:
            daily_pnl = 0

        # Update peak and drawdown
        if portfolio_value > self.stats['peak_value']:
            self.stats['peak_value'] = portfolio_value

        current_drawdown = (self.stats['peak_value'] - portfolio_value) / self.stats['peak_value']
        self.stats['max_drawdown'] = max(self.stats['max_drawdown'], current_drawdown)

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions_value=positions_value,
            total_value=portfolio_value,
            daily_pnl=daily_pnl,
            total_pnl=portfolio_value - self.initial_capital,
            total_pnl_pct=((portfolio_value / self.initial_capital) - 1) * 100,
            num_positions=len(self.positions),
            num_trades=len(self.trades)
        )

        self.portfolio_history.append(snapshot)

        # Notify callbacks
        for callback in self._portfolio_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Portfolio callback error: {e}")

    def _create_alert(
        self,
        severity: str,
        alert_type: str,
        ticker: Optional[str],
        message: str,
        data: Optional[Dict] = None
    ):
        """Create and store an alert"""
        self._alert_counter += 1
        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:06d}"

        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            ticker=ticker,
            message=message,
            data=data
        )

        self.alerts.append(alert)
        # Note: deque maxlen handles automatic eviction of oldest alerts

        logger.info(f"[{severity.upper()}] {message}")

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_trade_callback(self, callback: Callable[[Trade], None]):
        """Add callback for trade events"""
        self._trade_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alerts"""
        self._alert_callbacks.append(callback)

    def add_portfolio_callback(self, callback: Callable[[PortfolioSnapshot], None]):
        """Add callback for portfolio updates"""
        self._portfolio_callbacks.append(callback)

    def start(self):
        """Start the trading agent"""
        if self._running:
            logger.warning("Agent already running")
            return

        self._running = True

        # Start data service
        if self.trading_mode == TradingMode.SIMULATION:
            self.data_service.load_simulation_data()
            self._trading_thread = threading.Thread(
                target=self._simulation_loop,
                daemon=True
            )
        else:
            self.data_service.add_callback(self.process_quote)
            self.data_service.start_streaming()
            self._trading_thread = threading.Thread(
                target=self._paper_trading_loop,
                daemon=True
            )

        self._trading_thread.start()

        self._create_alert(
            'info', 'execution', None,
            f"Trading agent started in {self.trading_mode.value} mode"
        )

    def stop(self):
        """Stop the trading agent"""
        self._running = False

        if self.data_service:
            if self.trading_mode == TradingMode.SIMULATION:
                self.data_service.stop_simulation()
            else:
                self.data_service.stop_streaming()

        if self._trading_thread:
            self._trading_thread.join(timeout=5.0)

        self._create_alert(
            'info', 'execution', None,
            "Trading agent stopped"
        )

    def _paper_trading_loop(self):
        """Main loop for paper trading"""
        logger.info("Paper trading loop started")

        while self._running:
            # Periodic portfolio snapshot
            self._record_portfolio_snapshot()
            time.sleep(1.0)

    def _simulation_loop(self):
        """Main loop for simulation"""
        logger.info("Simulation loop started")

        def on_quote(quote: MarketQuote):
            if self._running:
                self.process_quote(quote)

        self.data_service.run_simulation(callback=on_quote)
        logger.info("Simulation loop completed")

    def get_trade_history(self) -> List[Dict]:
        """Get trade history as list of dicts"""
        return [trade.to_dict() for trade in self.trades]

    def get_order_history(self) -> List[Dict]:
        """Get order history as list of dicts"""
        return [order.to_dict() for order in self.orders]

    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        return [alert.to_dict() for alert in self.alerts[-limit:]]

    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()

        return pd.DataFrame([s.to_dict() for s in self.portfolio_history])

    def export_state(self, filepath: Path):
        """Export agent state to JSON"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'trading_mode': self.trading_mode.value,
            'model': self.current_model_key,
            'initial_capital': self.initial_capital,
            'portfolio': self.get_portfolio_summary(),
            'performance': self.get_performance_metrics(),
            'trades': self.get_trade_history(),
            'orders': self.get_order_history(),
            'alerts': self.get_alerts(100)
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Agent state exported to {filepath}")

    def reset(self):
        """Reset the agent to initial state"""
        self.stop()

        with self._lock:
            self.cash = self.initial_capital
            self.positions.clear()
            self.orders.clear()  # deque.clear() works the same as list.clear()
            self.trades.clear()
            self.portfolio_history.clear()
            self.alerts.clear()
            self._sequences.clear()

            self.stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'max_drawdown': 0.0,
                'peak_value': self.initial_capital
            }

            self._order_counter = 0
            self._trade_counter = 0
            self._alert_counter = 0

        logger.info("Agent reset to initial state")
