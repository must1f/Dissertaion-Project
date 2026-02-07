"""
Trading agent for generating signals from model predictions

Includes uncertainty estimation via:
- MC Dropout: Multiple forward passes with dropout enabled
- Ensemble predictions: If multiple models available
- Prediction intervals: Based on uncertainty estimates
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """Trading signal with optional uncertainty information"""
    timestamp: pd.Timestamp
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # Derived from uncertainty estimation (higher = more certain)
    predicted_price: float
    current_price: float
    expected_return: float
    # Optional uncertainty fields
    prediction_std: Optional[float] = None  # Standard deviation of prediction
    prediction_interval_lower: Optional[float] = None  # Lower bound of prediction interval
    prediction_interval_upper: Optional[float] = None  # Upper bound of prediction interval


class UncertaintyEstimator:
    """
    Uncertainty estimation methods for neural network predictions

    Supports:
    - MC Dropout: Multiple forward passes with dropout enabled
    - Ensemble: Predictions from multiple models
    - Prediction intervals: Based on uncertainty estimates
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_mc_samples: int = 50,
        ensemble_models: Optional[List[nn.Module]] = None
    ):
        """
        Initialize uncertainty estimator

        Args:
            model: Primary model for MC Dropout
            device: Device for inference
            n_mc_samples: Number of MC Dropout samples
            ensemble_models: Optional list of ensemble models
        """
        self.model = model
        self.device = device
        self.n_mc_samples = n_mc_samples
        self.ensemble_models = ensemble_models or []

    def _enable_dropout(self, model: nn.Module):
        """Enable dropout layers during inference for MC Dropout"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self, model: nn.Module):
        """Disable dropout layers (restore eval mode)"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def _forward_pass(self, model: nn.Module, sequences: torch.Tensor) -> torch.Tensor:
        """Single forward pass handling different model types"""
        model_class_name = model.__class__.__name__.lower()

        if hasattr(model, 'base_model_type') and model.base_model_type in ['lstm', 'gru']:
            predictions, _ = model(sequences)
        elif 'lstm' in model_class_name or 'gru' in model_class_name:
            predictions, _ = model(sequences)
        else:
            predictions = model(sequences)

        return predictions

    def mc_dropout_estimate(
        self,
        sequences: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty estimation

        Performs multiple forward passes with dropout enabled to estimate
        epistemic uncertainty (model uncertainty).

        Args:
            sequences: Input sequences (batch_size, seq_len, features)

        Returns:
            Tuple of (mean_predictions, std_predictions, all_samples)
        """
        sequences = sequences.to(self.device)

        # Enable dropout for MC sampling
        self._enable_dropout(self.model)

        all_predictions = []

        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                preds = self._forward_pass(self.model, sequences)
                all_predictions.append(preds.cpu().numpy())

        # Restore eval mode
        self._disable_dropout(self.model)

        # Stack predictions: (n_samples, batch_size, output_dim)
        all_predictions = np.stack(all_predictions, axis=0)

        # Compute statistics
        mean_preds = np.mean(all_predictions, axis=0).flatten()
        std_preds = np.std(all_predictions, axis=0).flatten()

        return mean_preds, std_preds, all_predictions

    def ensemble_estimate(
        self,
        sequences: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble uncertainty estimation

        Uses predictions from multiple models to estimate uncertainty.

        Args:
            sequences: Input sequences

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if not self.ensemble_models:
            raise ValueError("No ensemble models provided")

        sequences = sequences.to(self.device)
        all_predictions = []

        with torch.no_grad():
            for model in self.ensemble_models:
                model.eval()
                preds = self._forward_pass(model, sequences)
                all_predictions.append(preds.cpu().numpy())

        all_predictions = np.stack(all_predictions, axis=0)

        mean_preds = np.mean(all_predictions, axis=0).flatten()
        std_preds = np.std(all_predictions, axis=0).flatten()

        return mean_preds, std_preds

    def prediction_intervals(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals based on uncertainty estimates

        Assumes approximately Gaussian distribution of predictions.

        Args:
            mean: Mean predictions
            std: Standard deviation of predictions
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = mean - z * std
        upper = mean + z * std

        return lower, upper

    def uncertainty_to_confidence(
        self,
        std: np.ndarray,
        scale: str = 'normalized'
    ) -> np.ndarray:
        """
        Convert uncertainty (std) to confidence score

        Args:
            std: Standard deviation of predictions
            scale: 'normalized' (0-1) or 'raw'

        Returns:
            Confidence scores (higher = more confident)
        """
        if scale == 'normalized':
            # Normalize std to [0, 1] using tanh transformation
            # Higher std -> lower confidence
            # Use tanh to bound the output
            normalized_std = np.tanh(std)  # Maps to [0, 1)
            confidence = 1.0 - normalized_std
        else:
            # Raw inverse relationship
            confidence = 1.0 / (1.0 + std)

        return np.clip(confidence, 0.0, 1.0)


class SignalGenerator:
    """
    Generate trading signals from model predictions with uncertainty estimation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[any] = None,
        device: Optional[torch.device] = None,
        n_mc_samples: int = 50,
        ensemble_models: Optional[List[torch.nn.Module]] = None
    ):
        """
        Initialize signal generator

        Args:
            model: Trained prediction model
            config: Configuration object
            device: Device for model inference
            n_mc_samples: Number of MC Dropout samples for uncertainty
            ensemble_models: Optional list of models for ensemble uncertainty
        """
        self.model = model
        self.config = config or get_config()
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            model=self.model,
            device=self.device,
            n_mc_samples=n_mc_samples,
            ensemble_models=ensemble_models
        )

    def predict(
        self,
        sequences: torch.Tensor,
        estimate_uncertainty: bool = True,
        method: str = 'mc_dropout'
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """
        Make predictions with the model, optionally with uncertainty estimation

        Args:
            sequences: Input sequences (batch_size, seq_len, features)
            estimate_uncertainty: Whether to compute uncertainty estimates
            method: Uncertainty method - 'mc_dropout', 'ensemble', or 'both'

        Returns:
            Tuple of (predictions, uncertainties, uncertainty_details)
            - predictions: Point predictions (mean if uncertainty enabled)
            - uncertainties: Standard deviation of predictions (or None)
            - uncertainty_details: Dict with prediction intervals, confidence, etc.
        """
        sequences = sequences.to(self.device)
        uncertainty_details = None

        if estimate_uncertainty:
            if method == 'mc_dropout':
                # MC Dropout uncertainty estimation
                predictions, uncertainties, all_samples = self.uncertainty_estimator.mc_dropout_estimate(sequences)

                # Compute prediction intervals
                lower, upper = self.uncertainty_estimator.prediction_intervals(
                    predictions, uncertainties, confidence=0.95
                )

                # Convert uncertainty to confidence scores
                confidence_scores = self.uncertainty_estimator.uncertainty_to_confidence(uncertainties)

                uncertainty_details = {
                    'method': 'mc_dropout',
                    'n_samples': self.uncertainty_estimator.n_mc_samples,
                    'prediction_interval_lower': lower,
                    'prediction_interval_upper': upper,
                    'confidence_scores': confidence_scores,
                    'raw_std': uncertainties
                }

            elif method == 'ensemble':
                # Ensemble uncertainty estimation
                if not self.uncertainty_estimator.ensemble_models:
                    logger.warning("No ensemble models available, falling back to point prediction")
                    return self._point_predict(sequences)

                predictions, uncertainties = self.uncertainty_estimator.ensemble_estimate(sequences)

                lower, upper = self.uncertainty_estimator.prediction_intervals(
                    predictions, uncertainties, confidence=0.95
                )

                confidence_scores = self.uncertainty_estimator.uncertainty_to_confidence(uncertainties)

                uncertainty_details = {
                    'method': 'ensemble',
                    'n_models': len(self.uncertainty_estimator.ensemble_models),
                    'prediction_interval_lower': lower,
                    'prediction_interval_upper': upper,
                    'confidence_scores': confidence_scores,
                    'raw_std': uncertainties
                }

            elif method == 'both':
                # Combine MC Dropout and Ensemble for total uncertainty
                mc_preds, mc_std, _ = self.uncertainty_estimator.mc_dropout_estimate(sequences)

                if self.uncertainty_estimator.ensemble_models:
                    ens_preds, ens_std = self.uncertainty_estimator.ensemble_estimate(sequences)

                    # Average predictions
                    predictions = (mc_preds + ens_preds) / 2

                    # Combine uncertainties (root mean square)
                    uncertainties = np.sqrt((mc_std ** 2 + ens_std ** 2) / 2)
                else:
                    predictions = mc_preds
                    uncertainties = mc_std

                lower, upper = self.uncertainty_estimator.prediction_intervals(
                    predictions, uncertainties, confidence=0.95
                )

                confidence_scores = self.uncertainty_estimator.uncertainty_to_confidence(uncertainties)

                uncertainty_details = {
                    'method': 'combined',
                    'prediction_interval_lower': lower,
                    'prediction_interval_upper': upper,
                    'confidence_scores': confidence_scores,
                    'raw_std': uncertainties
                }

            else:
                raise ValueError(f"Unknown uncertainty method: {method}")

        else:
            # Simple point prediction without uncertainty
            predictions, uncertainties, uncertainty_details = self._point_predict(sequences)

        return predictions, uncertainties, uncertainty_details

    @torch.no_grad()
    def _point_predict(
        self,
        sequences: torch.Tensor
    ) -> Tuple[np.ndarray, None, None]:
        """
        Simple point prediction without uncertainty estimation

        Args:
            sequences: Input sequences

        Returns:
            Tuple of (predictions, None, None)
        """
        sequences = sequences.to(self.device)

        # Forward pass
        if hasattr(self.model, 'base_model_type') and self.model.base_model_type in ['lstm', 'gru']:
            predictions, _ = self.model(sequences)
        else:
            predictions = self.model(sequences)

        predictions = predictions.cpu().numpy().flatten()

        return predictions, None, None

    def generate_signals(
        self,
        sequences: torch.Tensor,
        current_prices: np.ndarray,
        tickers: List[str],
        timestamps: List[pd.Timestamp],
        threshold: float = 0.02,  # 2% minimum expected return
        confidence_threshold: float = 0.6,
        estimate_uncertainty: bool = True,
        uncertainty_method: str = 'mc_dropout',
        risk_adjusted: bool = True
    ) -> Tuple[List[Signal], Optional[Dict]]:
        """
        Generate trading signals from predictions with uncertainty-aware decision making

        Args:
            sequences: Input sequences for prediction
            current_prices: Current prices for each ticker
            tickers: List of ticker symbols
            timestamps: List of timestamps
            threshold: Minimum expected return to generate signal
            confidence_threshold: Minimum confidence to trade
            estimate_uncertainty: Whether to use uncertainty estimation
            uncertainty_method: 'mc_dropout', 'ensemble', or 'both'
            risk_adjusted: If True, adjust threshold based on uncertainty

        Returns:
            Tuple of (List of Signal objects, uncertainty_details dict)
        """
        # Make predictions with uncertainty
        predicted_prices, uncertainties, uncertainty_details = self.predict(
            sequences,
            estimate_uncertainty=estimate_uncertainty,
            method=uncertainty_method
        )

        signals = []

        # Get confidence scores from uncertainty estimation
        if uncertainty_details is not None:
            confidence_scores = uncertainty_details.get('confidence_scores', None)
            pred_intervals_lower = uncertainty_details.get('prediction_interval_lower', None)
            pred_intervals_upper = uncertainty_details.get('prediction_interval_upper', None)
        else:
            confidence_scores = None
            pred_intervals_lower = None
            pred_intervals_upper = None

        for i, (pred_price, curr_price, ticker, timestamp) in enumerate(
            zip(predicted_prices, current_prices, tickers, timestamps)
        ):
            # Calculate expected return
            expected_return = (pred_price - curr_price) / curr_price

            # Get confidence from uncertainty estimation or use default
            if confidence_scores is not None:
                confidence = float(confidence_scores[i])
            else:
                confidence = 1.0

            # Risk-adjusted threshold: increase threshold for high uncertainty
            if risk_adjusted and uncertainties is not None:
                # Scale threshold by uncertainty - higher uncertainty means higher threshold
                uncertainty_factor = 1.0 + float(uncertainties[i])
                adjusted_threshold = threshold * uncertainty_factor
            else:
                adjusted_threshold = threshold

            # Determine action with uncertainty-aware logic
            action = 'HOLD'

            if confidence >= confidence_threshold:
                if expected_return > adjusted_threshold:
                    # Additional check: ensure prediction interval is above current price
                    if pred_intervals_lower is not None:
                        lower_return = (pred_intervals_lower[i] - curr_price) / curr_price
                        if lower_return > 0:  # Even lower bound suggests upside
                            action = 'BUY'
                        elif expected_return > adjusted_threshold * 1.5:
                            # Strong signal despite interval crossing zero
                            action = 'BUY'
                    else:
                        action = 'BUY'

                elif expected_return < -adjusted_threshold:
                    # Additional check: ensure prediction interval is below current price
                    if pred_intervals_upper is not None:
                        upper_return = (pred_intervals_upper[i] - curr_price) / curr_price
                        if upper_return < 0:  # Even upper bound suggests downside
                            action = 'SELL'
                        elif expected_return < -adjusted_threshold * 1.5:
                            # Strong signal despite interval crossing zero
                            action = 'SELL'
                    else:
                        action = 'SELL'

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

        # Log uncertainty statistics
        if uncertainties is not None:
            logger.info(f"Uncertainty stats - Mean: {np.mean(uncertainties):.4f}, "
                       f"Std: {np.std(uncertainties):.4f}, "
                       f"Max: {np.max(uncertainties):.4f}")
            logger.info(f"Mean confidence: {np.mean(confidence_scores):.4f}")

        return signals, uncertainty_details

    def signals_to_dataframe(self, signals: List[Signal], include_uncertainty: bool = True) -> pd.DataFrame:
        """
        Convert signals to DataFrame

        Args:
            signals: List of Signal objects
            include_uncertainty: Whether to include uncertainty columns

        Returns:
            DataFrame with signal information
        """
        records = []
        for s in signals:
            record = {
                'timestamp': s.timestamp,
                'ticker': s.ticker,
                'signal': s.action,
                'confidence': s.confidence,
                'predicted_price': s.predicted_price,
                'current_price': s.current_price,
                'expected_return': s.expected_return
            }

            if include_uncertainty:
                record['prediction_std'] = s.prediction_std
                record['prediction_interval_lower'] = s.prediction_interval_lower
                record['prediction_interval_upper'] = s.prediction_interval_upper

            records.append(record)

        return pd.DataFrame(records)


class TradingAgent:
    """
    Complete trading agent with risk management and uncertainty-aware trading
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[any] = None,
        device: Optional[torch.device] = None,
        initial_capital: Optional[float] = None,
        n_mc_samples: int = 50,
        ensemble_models: Optional[List[torch.nn.Module]] = None
    ):
        """
        Initialize trading agent

        Args:
            model: Trained prediction model
            config: Configuration object
            device: Device for inference
            initial_capital: Initial capital (uses config if None)
            n_mc_samples: Number of MC Dropout samples for uncertainty
            ensemble_models: Optional list of models for ensemble uncertainty
        """
        self.config = config or get_config()
        self.device = device or torch.device('cpu')

        self.signal_generator = SignalGenerator(
            model=model,
            config=config,
            device=device,
            n_mc_samples=n_mc_samples,
            ensemble_models=ensemble_models
        )

        self.initial_capital = initial_capital or self.config.trading.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []

        logger.info(f"Trading agent initialized with ${self.initial_capital:,.2f}")
        logger.info(f"Uncertainty estimation: MC Dropout ({n_mc_samples} samples)")
        if ensemble_models:
            logger.info(f"Ensemble models: {len(ensemble_models)}")

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
        return_signals: bool = True,
        estimate_uncertainty: bool = True,
        uncertainty_method: str = 'mc_dropout'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict]]:
        """
        Execute trading strategy on data with uncertainty-aware decision making

        Args:
            data: DataFrame with columns [timestamp, ticker, price]
            sequences: Prepared sequences for prediction
            return_signals: Whether to return signals DataFrame
            estimate_uncertainty: Whether to use uncertainty estimation
            uncertainty_method: 'mc_dropout', 'ensemble', or 'both'

        Returns:
            Tuple of (portfolio_history, signals_df, uncertainty_details)
        """
        logger.info("Executing trading strategy...")
        logger.info(f"Uncertainty estimation: {estimate_uncertainty} (method: {uncertainty_method})")

        # Extract information
        timestamps = data['timestamp'].tolist()
        tickers = data['ticker'].tolist()
        prices = data['price'].values

        # Generate signals with uncertainty
        signals, uncertainty_details = self.signal_generator.generate_signals(
            sequences=sequences,
            current_prices=prices,
            tickers=tickers,
            timestamps=timestamps,
            estimate_uncertainty=estimate_uncertainty,
            uncertainty_method=uncertainty_method
        )

        # Convert to DataFrame
        signals_df = self.signal_generator.signals_to_dataframe(signals)

        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"Buy signals: {(signals_df['signal'] == 'BUY').sum()}")
        logger.info(f"Sell signals: {(signals_df['signal'] == 'SELL').sum()}")
        logger.info(f"Hold signals: {(signals_df['signal'] == 'HOLD').sum()}")

        if uncertainty_details:
            mean_confidence = signals_df['confidence'].mean()
            logger.info(f"Mean confidence: {mean_confidence:.4f}")

        if return_signals:
            return signals_df, signals_df, uncertainty_details
        else:
            return signals_df, None, uncertainty_details

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
