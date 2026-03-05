"""Service for real-time trading agent management."""

import sys
import uuid
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.services.data_service import DataService
from backend.app.core.exceptions import DataFetchError, PredictionError
from backend.app.schemas.trading import (
    AgentStatus,
    AgentConfigRequest,
    Signal,
    Position,
    Trade,
    Order,
    Alert,
    PortfolioAllocation,
    ConfidenceHistoryPoint,
    PerformanceMetrics,
    MarketData,
    SignalAction,
    TradingMode,
    OrderType,
    OrderStatus,
    AlertSeverity,
    AlertType,
)

# Try to import ML components
try:
    import torch
    import torch.nn as nn
    import yfinance as yf
    from src.models.model_registry import ModelRegistry
    from src.trading.agent import SignalGenerator
    HAS_ML = True
except ImportError:
    HAS_ML = False
    torch = None
    yf = None


class TradingAgent:
    """
    Real-time trading agent with ML model integration.

    Features:
    - Real-time and simulated data modes
    - ML model predictions with uncertainty
    - Paper trading with realistic execution
    - Risk management (stop-loss, take-profit)
    - Performance tracking and alerts
    """

    def __init__(self, config: AgentConfigRequest):
        """Initialize trading agent with configuration."""
        if not HAS_ML and not settings.demo_mode:
            raise RuntimeError(
                "ML dependencies (torch, yfinance, src) are unavailable. "
                "Install them or set DEMO_MODE=true to run with mock data."
            )
        self.agent_id = str(uuid.uuid4())[:8]
        self.config = config

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Portfolio
        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Dict] = {}

        # History
        self.trades: List[Dict] = []
        self.orders: List[Dict] = []
        self.alerts: List[Dict] = []
        self.signals: deque = deque(maxlen=100)
        self.confidence_history: deque = deque(maxlen=500)
        self.portfolio_history: List[Dict] = []
        self.price_history: deque = deque(maxlen=500)

        # Counters
        self._order_counter = 0
        self._trade_counter = 0
        self._alert_counter = 0

        # Statistics
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'peak_value': config.initial_capital,
            'max_drawdown': 0.0,
        }

        # ML components
        self.model = None
        self.signal_generator = None
        self.device = None

        # Market data
        self.current_price: Optional[float] = None
        self.market_data: Optional[Dict] = None

        # Timestamps
        self.started_at: Optional[datetime] = None
        self.last_update: Optional[datetime] = None

        # Initialize model if available
        if HAS_ML:
            self._init_ml_components()

    def _init_ml_components(self):
        """Initialize ML model and signal generator."""
        try:
            self.device = torch.device('cpu')
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')

            registry = ModelRegistry(settings.project_root)
            self.model = registry.load_model(
                self.config.model_key,
                device=self.device,
                input_dim=14
            )

            if self.model:
                self.signal_generator = SignalGenerator(
                    model=self.model,
                    device=self.device,
                    n_mc_samples=30
                )
                self._add_alert(
                    AlertType.SIGNAL, AlertSeverity.INFO,
                    f"Model loaded: {self.config.model_key}"
                )
        except Exception as e:
            self._add_alert(
                AlertType.ERROR, AlertSeverity.WARNING,
                f"Failed to load model: {str(e)}"
            )

    def _add_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        ticker: Optional[str] = None,
        data: Optional[Dict] = None
    ):
        """Add an alert."""
        self._alert_counter += 1
        alert = {
            'alert_id': f"ALT-{self.agent_id}-{self._alert_counter:04d}",
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type.value,
            'severity': severity.value,
            'ticker': ticker,
            'message': message,
            'data': data
        }
        self.alerts.append(alert)

        # Keep only last 500 alerts
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

    def fetch_market_data(self) -> Optional[Dict]:
        """Fetch current market data."""
        if settings.demo_mode:
            return self._get_mock_market_data()
            
        if not HAS_ML or not yf:
            raise DataFetchError(
                "Market data unavailable because ML/yfinance dependencies are missing. "
                "Install requirements or enable DEMO_MODE for mock data."
            )

        try:
            ticker = yf.Ticker(self.config.ticker)

            # Get intraday data
            hist = ticker.history(period='1d', interval='1m')
            if hist.empty:
                hist = ticker.history(period='5d')

            if hist.empty:
                if settings.demo_mode:
                    return self._get_mock_market_data()
                raise DataFetchError("No intraday market data returned for ticker.")

            latest = hist.iloc[-1]
            info = ticker.info
            prev_close = info.get('previousClose', latest['Close'])

            self.current_price = float(latest['Close'])

            self.market_data = {
                'ticker': self.config.ticker,
                'price': float(latest['Close']),
                'change': float(latest['Close'] - prev_close) if prev_close else 0.0,
                'change_pct': float((latest['Close'] - prev_close) / prev_close * 100) if prev_close else 0.0,
                'volume': int(latest['Volume']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'timestamp': datetime.now().isoformat(),
                'is_market_open': self._is_market_open()
            }

            # Record price history point
            self.price_history.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
            })

            return self.market_data

        except Exception as e:
            self._add_alert(
                AlertType.ERROR, AlertSeverity.WARNING,
                f"Error fetching market data: {str(e)}",
                self.config.ticker
            )
            if settings.demo_mode:
                return self._get_mock_market_data()
            raise DataFetchError(f"Error fetching market data: {e}")

    def _get_mock_market_data(self) -> Dict:
        """Generate mock market data for testing."""
        if not settings.demo_mode:
            raise RuntimeError("Mock market data is disabled when DEMO_MODE is false.")
        # Use a base price with some randomness
        if self.current_price is None:
            try:
                import yfinance as yf
                ticker = yf.Ticker(self.config.ticker)
                hist = ticker.history(period='5d')
                if not hist.empty:
                    self.current_price = float(hist.iloc[-1]['Close'])
                else:
                    self.current_price = 450.0
            except Exception:
                self.current_price = 450.0
        else:
            # Random walk
            self.current_price *= (1 + np.random.normal(0, 0.001))

        change = np.random.normal(0, 2)

        self.market_data = {
            'ticker': self.config.ticker,
            'price': self.current_price,
            'change': change,
            'change_pct': change / self.current_price * 100,
            'volume': int(np.random.uniform(1e6, 5e6)),
            'open': self.current_price - np.random.uniform(-2, 2),
            'high': self.current_price + np.random.uniform(0, 3),
            'low': self.current_price - np.random.uniform(0, 3),
            'timestamp': datetime.now().isoformat(),
            'is_market_open': self._is_market_open()
        }

        # Record price history point
        self.price_history.append({
            'timestamp': self.market_data['timestamp'],
            'price': self.market_data['price'],
            'volume': self.market_data['volume'],
            'open': self.market_data['open'],
            'high': self.market_data['high'],
            'low': self.market_data['low'],
        })

        return self.market_data

    def _is_market_open(self) -> bool:
        """Check if US stock market is open."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def generate_signal(self) -> Optional[Dict]:
        """Generate a trading signal from the model.

        Uses DataService.prepare_sequences() to build the same 14-feature,
        StandardScaler-normalised input that the model was trained on.
        """
        if self.signal_generator is None or self.current_price is None:
            if settings.demo_mode:
                return self._generate_mock_signal()
            raise PredictionError("Signal generator unavailable or price missing.")

        try:
            if not HAS_ML:
                if settings.demo_mode:
                    return self._generate_mock_signal()
                raise PredictionError("ML dependencies missing for signal generation.")

            # ── Use DataService for consistent 14-feature prep ──
            data_service = DataService()
            sequences, targets, df = data_service.prepare_sequences(
                ticker=self.config.ticker,
                sequence_length=60,
            )

            if sequences.size == 0:
                if settings.demo_mode:
                    return self._generate_mock_signal()
                raise PredictionError("Insufficient data for signal generation.")

            # Take the most recent sequence
            latest = sequences[-1:]  # shape (1, 60, 14)
            sequence = torch.FloatTensor(latest).to(self.device)

            # Generate signal via SignalGenerator
            signals, _ = self.signal_generator.generate_signals(
                sequences=sequence,
                current_prices=np.array([self.current_price]),
                tickers=[self.config.ticker],
                timestamps=[pd.Timestamp.now()],
                threshold=self.config.signal_threshold,
                confidence_threshold=self.config.min_confidence,
                estimate_uncertainty=True,
            )

            if signals:
                signal = signals[0]

                # The model output is in normalised space.  De-normalise
                # the expected_return using the scaler stats so the signal
                # expresses a realistic percentage return.
                close_mean = float(getattr(df, '_scaler_mean', {}).get('close', 0))
                close_std  = float(getattr(df, '_scaler_std',  {}).get('close', 1))

                raw_pred_price = float(signal.predicted_price)
                pred_price = raw_pred_price * close_std + close_mean
                expected_ret = (pred_price - self.current_price) / self.current_price if self.current_price else 0.0

                # Override the action based on de-normalised return
                if signal.confidence >= self.config.min_confidence:
                    if expected_ret > self.config.signal_threshold:
                        action = 'BUY'
                    elif expected_ret < -self.config.signal_threshold:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
                else:
                    action = 'HOLD'

                signal_dict = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'ticker': self.config.ticker,
                    'action': action,
                    'confidence': float(signal.confidence),
                    'price': self.current_price,
                    'expected_return': expected_ret,
                    'model_key': self.config.model_key,
                    'uncertainty_std': float(signal.prediction_std) if signal.prediction_std else None,
                }

                self._record_signal(signal_dict)
                return signal_dict

        except Exception as e:
            self._add_alert(
                AlertType.ERROR, AlertSeverity.WARNING,
                f"Error generating signal: {str(e)}",
            )
            if settings.demo_mode:
                return self._generate_mock_signal()
            raise PredictionError(f"Error generating signal: {e}")

    def _generate_mock_signal(self) -> Dict:
        """Generate a mock trading signal for testing."""
        if not settings.demo_mode:
            raise RuntimeError("Mock signals are disabled when DEMO_MODE is false.")
        if self.current_price is None:
            self.current_price = 450.0

        # Random signal with some logic
        confidence = np.random.uniform(0.4, 0.95)
        expected_return = np.random.normal(0, 0.03)

        if confidence >= self.config.min_confidence:
            if expected_return > self.config.signal_threshold:
                action = 'BUY'
            elif expected_return < -self.config.signal_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'
        else:
            action = 'HOLD'

        signal = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': self.config.ticker,
            'action': action,
            'confidence': confidence,
            'price': self.current_price,
            'expected_return': expected_return,
            'model_key': self.config.model_key,
            'uncertainty_std': np.random.uniform(0.01, 0.05)
        }

        self._record_signal(signal)
        return signal

    def _record_signal(self, signal: Dict):
        """Record a signal in history."""
        self.signals.appendleft(signal)
        self.stats['total_signals'] += 1

        if signal['action'] == 'BUY':
            self.stats['buy_signals'] += 1
        elif signal['action'] == 'SELL':
            self.stats['sell_signals'] += 1

        # Record confidence history
        signal_value = 1 if signal['action'] == 'BUY' else (-1 if signal['action'] == 'SELL' else 0)
        self.confidence_history.append({
            'time': signal['time'],
            'confidence': signal['confidence'],
            'signal': signal_value
        })

    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """Execute a trading signal."""
        if signal['action'] == 'HOLD':
            return None

        with self._lock:
            if signal['action'] == 'BUY':
                return self._execute_buy(signal)
            elif signal['action'] == 'SELL':
                return self._execute_sell(signal)

        return None

    def _execute_buy(self, signal: Dict) -> Optional[Dict]:
        """Execute a buy order (supports scaling into existing positions)."""
        ticker = signal['ticker']

        # ── Check total exposure limit (80 % of portfolio) ──────
        portfolio_value = self.get_portfolio_value()
        existing_value = 0.0
        if ticker in self.positions:
            existing_value = self.positions[ticker]['market_value']

        max_allowed = portfolio_value * 0.80          # up to 80 % in one ticker
        if existing_value >= max_allowed:
            return None

        # ── Per-trade order size: 5 % of portfolio × confidence ─
        order_value = portfolio_value * 0.05 * signal['confidence']
        remaining  = max_allowed - existing_value
        order_value = min(order_value, remaining, self.cash * 0.95)

        price = signal['price'] * 1.0005              # slippage

        # Use fractional shares (paper trading)
        quantity = round(order_value / price, 4)
        if quantity < 0.001 or order_value < 10:
            return None

        value = quantity * price
        commission = value * 0.001                     # 0.1 %
        total_cost = value + commission

        if total_cost > self.cash:
            return None

        # ── Execute ─────────────────────────────────────────────
        self.cash -= total_cost

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_qty = round(pos['quantity'] + quantity, 4)
            pos['avg_entry_price'] = (
                (pos['avg_entry_price'] * pos['quantity']) + (price * quantity)
            ) / total_qty
            pos['quantity'] = total_qty
            pos['market_value'] = total_qty * price
            pos['unrealized_pnl'] = (price - pos['avg_entry_price']) * total_qty
            pos['stop_loss'] = pos['avg_entry_price'] * (1 - self.config.stop_loss_pct)
            pos['take_profit'] = pos['avg_entry_price'] * (1 + self.config.take_profit_pct)
        else:
            self.positions[ticker] = {
                'ticker': ticker,
                'quantity': quantity,
                'avg_entry_price': price,
                'current_price': price,
                'market_value': value,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'realized_pnl': 0.0,
                'entry_time': datetime.now().isoformat(),
                'stop_loss': price * (1 - self.config.stop_loss_pct),
                'take_profit': price * (1 + self.config.take_profit_pct),
            }

        # ── Record trade ────────────────────────────────────────
        trade = self._record_trade(
            ticker, 'BUY', quantity, price, commission, 0.0, signal
        )

        self._add_alert(
            AlertType.EXECUTION, AlertSeverity.INFO,
            f"BUY {quantity:.4f} {ticker} @ ${price:.2f}",
            ticker
        )

        return trade

    def _execute_sell(self, signal: Dict) -> Optional[Dict]:
        """Execute a sell order."""
        if signal['ticker'] not in self.positions:
            return None

        pos = self.positions[signal['ticker']]
        quantity = pos['quantity']
        price = signal['price'] * 0.9995  # Slippage
        value = quantity * price
        commission = value * 0.001

        # Calculate PnL
        pnl = (price - pos['avg_entry_price']) * quantity - commission
        pnl_pct = ((price / pos['avg_entry_price']) - 1) * 100 if pos['avg_entry_price'] > 0 else 0

        # Execute
        self.cash += value - commission
        del self.positions[signal['ticker']]

        # Update stats
        self.stats['total_pnl'] += pnl
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['largest_win'] = max(self.stats['largest_win'], pnl)
        else:
            self.stats['losing_trades'] += 1
            self.stats['largest_loss'] = min(self.stats['largest_loss'], pnl)

        # Record trade
        trade = self._record_trade(
            signal['ticker'], 'SELL', quantity, price, commission, pnl, signal
        )

        self._add_alert(
            AlertType.EXECUTION, AlertSeverity.INFO,
            f"SELL {quantity} {signal['ticker']} @ ${price:.2f} (PnL: ${pnl:.2f})",
            signal['ticker']
        )

        return trade

    def _record_trade(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        commission: float,
        pnl: float,
        signal: Dict
    ) -> Dict:
        """Record a trade."""
        self._trade_counter += 1
        self.stats['total_trades'] += 1

        trade = {
            'trade_id': f"TRD-{self.agent_id}-{self._trade_counter:04d}",
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'commission': commission,
            'pnl': pnl,
            'pnl_pct': (pnl / (quantity * price) * 100) if side == 'SELL' and (quantity * price) > 0 else 0.0,
            'model_used': self.config.model_key,
            'signal_confidence': signal['confidence']
        }

        self.trades.append(trade)
        return trade

    def check_risk_limits(self):
        """Check stop-loss and take-profit for positions."""
        if self.current_price is None:
            return

        with self._lock:
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                pos['current_price'] = self.current_price
                pos['market_value'] = pos['quantity'] * self.current_price
                pos['unrealized_pnl'] = (self.current_price - pos['avg_entry_price']) * pos['quantity']
                pos['unrealized_pnl_pct'] = ((self.current_price / pos['avg_entry_price']) - 1) * 100

                # Check stop-loss
                if pos['stop_loss'] and self.current_price <= pos['stop_loss']:
                    signal = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'ticker': ticker,
                        'action': 'SELL',
                        'confidence': 1.0,
                        'price': self.current_price,
                        'expected_return': pos['unrealized_pnl_pct'] / 100,
                        'model_key': 'stop_loss',
                    }
                    self._execute_sell(signal)
                    self._add_alert(
                        AlertType.RISK, AlertSeverity.WARNING,
                        f"Stop-loss triggered for {ticker} @ ${self.current_price:.2f}",
                        ticker
                    )

                # Check take-profit
                elif pos['take_profit'] and self.current_price >= pos['take_profit']:
                    signal = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'ticker': ticker,
                        'action': 'SELL',
                        'confidence': 1.0,
                        'price': self.current_price,
                        'expected_return': pos['unrealized_pnl_pct'] / 100,
                        'model_key': 'take_profit',
                    }
                    self._execute_sell(signal)
                    self._add_alert(
                        AlertType.RISK, AlertSeverity.INFO,
                        f"Take-profit triggered for {ticker} @ ${self.current_price:.2f}",
                        ticker
                    )

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(
            pos['quantity'] * (self.current_price or pos['current_price'])
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def record_portfolio_snapshot(self):
        """Record current portfolio state."""
        total_value = self.get_portfolio_value()
        positions_value = total_value - self.cash

        # Update peak and drawdown
        if total_value > self.stats['peak_value']:
            self.stats['peak_value'] = total_value

        current_drawdown = (self.stats['peak_value'] - total_value) / self.stats['peak_value']
        self.stats['max_drawdown'] = max(self.stats['max_drawdown'], current_drawdown)

        self.portfolio_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'daily_return': 0.0,
            'cumulative_return': ((total_value / self.initial_capital) - 1) * 100
        })

        # Keep only last 1000 snapshots
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]

        self.last_update = datetime.now()

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        total_value = self.get_portfolio_value()
        positions_value = total_value - self.cash

        # Calculate average confidence
        avg_confidence = 0.0
        if self.signals:
            avg_confidence = np.mean([s['confidence'] for s in self.signals])

        # Calculate today's PnL (from trades today)
        today = datetime.now().date()
        pnl_today = sum(
            t['pnl'] for t in self.trades
            if datetime.fromisoformat(t['timestamp']).date() == today
        )

        # Portfolio allocation
        allocation = []
        if positions_value > 0:
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
            for i, (ticker, pos) in enumerate(self.positions.items()):
                allocation.append(PortfolioAllocation(
                    name=ticker,
                    value=round(pos['market_value'] / total_value * 100, 1),
                    color=colors[i % len(colors)]
                ))

        if self.cash > 0:
            allocation.append(PortfolioAllocation(
                name='Cash',
                value=round(self.cash / total_value * 100, 1),
                color='#6b7280'
            ))

        # Performance metrics
        win_rate = 0.0
        if self.stats['total_trades'] > 0:
            win_rate = self.stats['winning_trades'] / self.stats['total_trades'] * 100

        performance = PerformanceMetrics(
            total_trades=self.stats['total_trades'],
            winning_trades=self.stats['winning_trades'],
            losing_trades=self.stats['losing_trades'],
            win_rate=win_rate,
            total_pnl=self.stats['total_pnl'],
            realized_pnl=self.stats['total_pnl'],
            unrealized_pnl=sum(p['unrealized_pnl'] for p in self.positions.values()),
            largest_win=self.stats['largest_win'],
            largest_loss=self.stats['largest_loss'],
            max_drawdown=self.stats['max_drawdown'] * 100,
            avg_trade_pnl=self.stats['total_pnl'] / max(1, self.stats['total_trades'])
        )

        return AgentStatus(
            is_running=self._running,
            trading_mode=self.config.trading_mode.value,
            model_key=self.config.model_key,
            ticker=self.config.ticker,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            total_pnl=total_value - self.initial_capital,
            total_pnl_pct=((total_value / self.initial_capital) - 1) * 100,
            total_signals=self.stats['total_signals'],
            buy_signals=self.stats['buy_signals'],
            sell_signals=self.stats['sell_signals'],
            avg_confidence=avg_confidence,
            pnl_today=pnl_today,
            config={
                'model_key': self.config.model_key,
                'signal_threshold': self.config.signal_threshold,
                'max_position_size': self.config.max_position_size,
                'min_confidence': self.config.min_confidence,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
            },
            signals=[Signal(**s) for s in list(self.signals)[:20]],
            positions=[Position(**p) for p in self.positions.values()],
            portfolio_allocation=allocation,
            confidence_history=[
                ConfidenceHistoryPoint(**c) for c in list(self.confidence_history)[-100:]
            ],
            performance=performance,
            recent_alerts=[Alert(**a) for a in self.alerts[-10:]],
            market_data=MarketData(**self.market_data) if self.market_data else None,
            started_at=self.started_at.isoformat() if self.started_at else None,
            last_update=self.last_update.isoformat() if self.last_update else None,
        )

    def start(self):
        """Start the trading agent."""
        if self._running:
            return

        self._running = True
        self.started_at = datetime.now()

        self._add_alert(
            AlertType.SIGNAL, AlertSeverity.INFO,
            f"Trading agent started in {self.config.trading_mode.value} mode"
        )

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the trading agent."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._add_alert(
            AlertType.SIGNAL, AlertSeverity.INFO,
            "Trading agent stopped"
        )

    def _run_loop(self):
        """Main trading loop."""
        while self._running:
            try:
                # Fetch market data
                self.fetch_market_data()

                # Check risk limits
                self.check_risk_limits()

                # Generate and execute signals
                signal = self.generate_signal()
                if signal and signal['action'] != 'HOLD':
                    if signal['confidence'] >= self.config.min_confidence:
                        self.execute_signal(signal)

                # Partial take-profit: every 6th cycle, trim large positions
                self._cycle_count = getattr(self, '_cycle_count', 0) + 1
                if self._cycle_count % 6 == 0:
                    self._partial_take_profit()

                # Record portfolio snapshot
                self.record_portfolio_snapshot()

                # Sleep
                time.sleep(5.0)

            except Exception as e:
                self._add_alert(
                    AlertType.ERROR, AlertSeverity.CRITICAL,
                    f"Error in trading loop: {str(e)}"
                )
                time.sleep(10.0)

    # ------------------------------------------------------------------ #
    #  Partial take-profit helper                                         #
    # ------------------------------------------------------------------ #
    def _partial_take_profit(self):
        """Sell half of a position when it exceeds 40 % of portfolio value."""
        if self.current_price is None:
            return

        portfolio_value = self.get_portfolio_value()

        with self._lock:
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                pos_value = pos['quantity'] * self.current_price

                if pos_value > portfolio_value * 0.40:
                    sell_qty = round(pos['quantity'] * 0.5, 4)
                    if sell_qty < 0.001:
                        continue

                    price = self.current_price * 0.9995   # slippage
                    value = sell_qty * price
                    commission = value * 0.001
                    pnl = (price - pos['avg_entry_price']) * sell_qty - commission

                    self.cash += value - commission

                    # Reduce position
                    pos['quantity'] = round(pos['quantity'] - sell_qty, 4)
                    if pos['quantity'] < 0.001:
                        del self.positions[ticker]
                    else:
                        pos['market_value'] = pos['quantity'] * price
                        pos['unrealized_pnl'] = (price - pos['avg_entry_price']) * pos['quantity']

                    # Stats
                    self.stats['total_pnl'] += pnl
                    if pnl > 0:
                        self.stats['winning_trades'] += 1
                        self.stats['largest_win'] = max(self.stats['largest_win'], pnl)
                    else:
                        self.stats['losing_trades'] += 1
                        self.stats['largest_loss'] = min(self.stats['largest_loss'], pnl)

                    trade = self._record_trade(
                        ticker, 'SELL', sell_qty, price, commission, pnl,
                        {'confidence': 1.0, 'model_key': 'take_profit_trim'}
                    )
                    self._add_alert(
                        AlertType.EXECUTION, AlertSeverity.INFO,
                        f"SELL {sell_qty:.4f} {ticker} @ ${price:.2f} (trim, PnL: ${pnl:.2f})",
                        ticker
                    )


class TradingService:
    """Service for managing trading agents."""

    def __init__(self):
        """Initialize trading service."""
        self._agents: Dict[str, TradingAgent] = {}
        self._active_agent: Optional[TradingAgent] = None
        self._lock = threading.Lock()

    def start_agent(self, config: AgentConfigRequest) -> Tuple[str, TradingAgent]:
        """Start a new trading agent."""
        # Stop existing agent if any
        if self._active_agent:
            self.stop_agent()

        with self._lock:
            agent = TradingAgent(config)
            agent.start()

            self._agents[agent.agent_id] = agent
            self._active_agent = agent

            return agent.agent_id, agent

    def stop_agent(self) -> Optional[Dict]:
        """Stop the active trading agent."""
        with self._lock:
            if self._active_agent is None:
                return None

            agent = self._active_agent
            agent.stop()

            result = {
                'success': True,
                'message': 'Agent stopped',
                'final_portfolio_value': agent.get_portfolio_value(),
                'total_pnl': agent.get_portfolio_value() - agent.initial_capital,
                'total_trades': agent.stats['total_trades']
            }

            self._active_agent = None
            return result

    def get_status(self) -> Optional[AgentStatus]:
        """Get status of active agent."""
        if self._active_agent is None:
            return AgentStatus(
                is_running=False,
                trading_mode='paper',
                ticker='^GSPC',
            )
        return self._active_agent.get_status()

    def get_agent(self, agent_id: str) -> Optional[TradingAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._active_agent is not None and self._active_agent._running

    def get_trades(self, page: int = 1, page_size: int = 50) -> Dict:
        """Get trade history."""
        if self._active_agent is None:
            return {'trades': [], 'total': 0, 'page': page, 'page_size': page_size}

        trades = self._active_agent.trades
        total = len(trades)

        start = (page - 1) * page_size
        end = start + page_size

        return {
            'trades': [Trade(**t) for t in trades[start:end]],
            'total': total,
            'page': page,
            'page_size': page_size
        }

    def get_orders(self, page: int = 1, page_size: int = 50) -> Dict:
        """Get order history."""
        if self._active_agent is None:
            return {'orders': [], 'total': 0, 'page': page, 'page_size': page_size}

        orders = self._active_agent.orders
        total = len(orders)

        start = (page - 1) * page_size
        end = start + page_size

        return {
            'orders': [Order(**o) for o in orders[start:end]],
            'total': total,
            'page': page,
            'page_size': page_size
        }

    def get_alerts(self, limit: int = 50) -> Dict:
        """Get recent alerts."""
        if self._active_agent is None:
            return {'alerts': [], 'total': 0, 'unread_count': 0}

        alerts = self._active_agent.alerts[-limit:]

        return {
            'alerts': [Alert(**a) for a in alerts],
            'total': len(self._active_agent.alerts),
            'unread_count': len(alerts)
        }

    def place_manual_order(
        self,
        ticker: str,
        side: str,
        quantity: Optional[int] = None
    ) -> Optional[Dict]:
        """Place a manual order."""
        if self._active_agent is None:
            return None

        agent = self._active_agent

        signal = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': ticker,
            'action': side,
            'confidence': 1.0,
            'price': agent.current_price or 450.0,
            'expected_return': 0.0,
            'model_key': 'manual',
        }

        return agent.execute_signal(signal)

    def close_position(self, ticker: str, quantity: Optional[int] = None) -> Optional[Dict]:
        """Close a position."""
        if self._active_agent is None:
            return None

        agent = self._active_agent

        if ticker not in agent.positions:
            return None

        signal = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': ticker,
            'action': 'SELL',
            'confidence': 1.0,
            'price': agent.current_price or 450.0,
            'expected_return': 0.0,
            'model_key': 'manual_close',
        }

        return agent.execute_signal(signal)


# Singleton instance
_trading_service: Optional[TradingService] = None


def get_trading_service() -> TradingService:
    """Get or create trading service singleton."""
    global _trading_service
    if _trading_service is None:
        _trading_service = TradingService()
    return _trading_service
