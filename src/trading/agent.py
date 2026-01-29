"""
Trading agent for generating signals from model predictions
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """Trading signal"""
    timestamp: pd.Timestamp
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_price: float
    current_price: float
    expected_return: float


class SignalGenerator:
    """
    Generate trading signals from model predictions
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[any] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize signal generator

        Args:
            model: Trained prediction model
            config: Configuration object
            device: Device for model inference
        """
        self.model = model
        self.config = config or get_config()
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        sequences: torch.Tensor
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the model

        Args:
            sequences: Input sequences (batch_size, seq_len, features)

        Returns:
            Tuple of (predictions, uncertainties)
        """
        sequences = sequences.to(self.device)

        # Forward pass
        if hasattr(self.model, 'base_model_type') and self.model.base_model_type in ['lstm', 'gru']:
            predictions, _ = self.model(sequences)
        else:
            predictions = self.model(sequences)

        predictions = predictions.cpu().numpy().flatten()

        # TODO: Implement uncertainty estimation (e.g., using dropout, ensemble, etc.)
        uncertainties = None

        return predictions, uncertainties

    def generate_signals(
        self,
        sequences: torch.Tensor,
        current_prices: np.ndarray,
        tickers: List[str],
        timestamps: List[pd.Timestamp],
        threshold: float = 0.02,  # 2% minimum expected return
        confidence_threshold: float = 0.6
    ) -> List[Signal]:
        """
        Generate trading signals from predictions

        Args:
            sequences: Input sequences for prediction
            current_prices: Current prices for each ticker
            tickers: List of ticker symbols
            timestamps: List of timestamps
            threshold: Minimum expected return to generate signal
            confidence_threshold: Minimum confidence to trade

        Returns:
            List of Signal objects
        """
        # Make predictions
        predicted_prices, uncertainties = self.predict(sequences)

        signals = []

        for i, (pred_price, curr_price, ticker, timestamp) in enumerate(
            zip(predicted_prices, current_prices, tickers, timestamps)
        ):
            # Calculate expected return
            expected_return = (pred_price - curr_price) / curr_price

            # Calculate confidence (simplified - could use uncertainty estimates)
            confidence = 1.0 if uncertainties is None else (1.0 - uncertainties[i])

            # Determine action
            if expected_return > threshold and confidence >= confidence_threshold:
                action = 'BUY'
            elif expected_return < -threshold and confidence >= confidence_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'

            signal = Signal(
                timestamp=timestamp,
                ticker=ticker,
                action=action,
                confidence=confidence,
                predicted_price=pred_price,
                current_price=curr_price,
                expected_return=expected_return
            )

            signals.append(signal)

        return signals

    def signals_to_dataframe(self, signals: List[Signal]) -> pd.DataFrame:
        """Convert signals to DataFrame"""
        return pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'ticker': s.ticker,
                'signal': s.action,
                'confidence': s.confidence,
                'predicted_price': s.predicted_price,
                'current_price': s.current_price,
                'expected_return': s.expected_return
            }
            for s in signals
        ])


class TradingAgent:
    """
    Complete trading agent with risk management
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[any] = None,
        device: Optional[torch.device] = None,
        initial_capital: Optional[float] = None
    ):
        """
        Initialize trading agent

        Args:
            model: Trained prediction model
            config: Configuration object
            device: Device for inference
            initial_capital: Initial capital (uses config if None)
        """
        self.config = config or get_config()
        self.device = device or torch.device('cpu')

        self.signal_generator = SignalGenerator(model, config, device)

        self.initial_capital = initial_capital or self.config.trading.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []

        logger.info(f"Trading agent initialized with ${self.initial_capital:,.2f}")

    def reset(self):
        """Reset agent to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []

    def calculate_position_size(
        self,
        price: float,
        confidence: float,
        risk_per_trade: float = 0.02
    ) -> int:
        """
        Calculate position size using risk management rules

        Args:
            price: Current price
            confidence: Signal confidence
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Number of shares
        """
        # Risk amount
        risk_amount = self.initial_capital * risk_per_trade * confidence

        # Position size based on stop loss
        stop_loss_distance = price * self.config.trading.stop_loss
        shares = int(risk_amount / stop_loss_distance)

        # Limit maximum position value
        max_position_value = self.initial_capital * self.config.trading.max_position_size
        max_shares = int(max_position_value / price)

        return min(shares, max_shares)

    def execute_strategy(
        self,
        data: pd.DataFrame,
        sequences: torch.Tensor,
        return_signals: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Execute trading strategy on data

        Args:
            data: DataFrame with columns [timestamp, ticker, price]
            sequences: Prepared sequences for prediction
            return_signals: Whether to return signals DataFrame

        Returns:
            Tuple of (portfolio_history, signals) or just portfolio_history
        """
        logger.info("Executing trading strategy...")

        # Extract information
        timestamps = data['timestamp'].tolist()
        tickers = data['ticker'].tolist()
        prices = data['price'].values

        # Generate signals
        signals = self.signal_generator.generate_signals(
            sequences=sequences,
            current_prices=prices,
            tickers=tickers,
            timestamps=timestamps
        )

        # Convert to DataFrame
        signals_df = self.signal_generator.signals_to_dataframe(signals)

        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"Buy signals: {(signals_df['signal'] == 'BUY').sum()}")
        logger.info(f"Sell signals: {(signals_df['signal'] == 'SELL').sum()}")
        logger.info(f"Hold signals: {(signals_df['signal'] == 'HOLD').sum()}")

        if return_signals:
            return signals_df, signals_df
        else:
            return signals_df, None

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run backtest with signals

        Args:
            signals_df: DataFrame with signals
            prices_df: DataFrame with prices

        Returns:
            DataFrame with portfolio performance
        """
        from ..evaluation.backtester import Backtester

        # Create backtester
        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_rate=self.config.trading.transaction_cost,
            max_position_size=self.config.trading.max_position_size,
            stop_loss=self.config.trading.stop_loss,
            take_profit=self.config.trading.take_profit
        )

        # Run backtest
        results = backtester.run_backtest(signals_df, prices_df)

        # Print summary
        logger.info("\n" + results.summary())

        return results

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get current portfolio summary

        Args:
            current_prices: Dictionary of ticker -> current price

        Returns:
            Portfolio summary dictionary
        """
        positions_value = sum(
            qty * current_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
        )

        total_value = self.cash + positions_value

        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'return': ((total_value / self.initial_capital) - 1) * 100,
            'positions': dict(self.positions)
        }


class BenchmarkStrategy:
    """Benchmark strategies for comparison"""

    @staticmethod
    def buy_and_hold(
        prices: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        Simple buy-and-hold strategy

        Args:
            prices: DataFrame with columns [timestamp, ticker, price]
            initial_capital: Initial capital

        Returns:
            DataFrame with portfolio values
        """
        # Buy on first day, hold until end
        first_prices = prices.groupby('ticker').first()
        last_prices = prices.groupby('ticker').last()

        # Calculate shares to buy with equal weighting
        n_tickers = len(first_prices)
        capital_per_ticker = initial_capital / n_tickers

        total_return = 0
        for ticker in first_prices.index:
            shares = capital_per_ticker / first_prices.loc[ticker, 'price']
            final_value = shares * last_prices.loc[ticker, 'price']
            total_return += (final_value - capital_per_ticker)

        final_value = initial_capital + total_return

        return pd.DataFrame({
            'strategy': ['buy_and_hold'],
            'initial_capital': [initial_capital],
            'final_value': [final_value],
            'return_pct': [((final_value / initial_capital) - 1) * 100]
        })

    @staticmethod
    def sma_crossover(
        prices: pd.DataFrame,
        short_window: int = 50,
        long_window: int = 200
    ) -> pd.DataFrame:
        """
        Simple Moving Average crossover strategy

        Args:
            prices: DataFrame with columns [timestamp, ticker, price]
            short_window: Short SMA window
            long_window: Long SMA window

        Returns:
            DataFrame with signals
        """
        signals = []

        for ticker in prices['ticker'].unique():
            ticker_df = prices[prices['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('timestamp')

            # Calculate SMAs
            ticker_df['sma_short'] = ticker_df['price'].rolling(window=short_window).mean()
            ticker_df['sma_long'] = ticker_df['price'].rolling(window=long_window).mean()

            # Generate signals
            ticker_df['signal'] = 'HOLD'
            ticker_df.loc[ticker_df['sma_short'] > ticker_df['sma_long'], 'signal'] = 'BUY'
            ticker_df.loc[ticker_df['sma_short'] < ticker_df['sma_long'], 'signal'] = 'SELL'

            ticker_df['confidence'] = 1.0

            signals.append(ticker_df[['timestamp', 'ticker', 'signal', 'confidence', 'price']])

        return pd.concat(signals, ignore_index=True)
