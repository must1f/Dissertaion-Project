"""Service for backtesting wrapping src/evaluation/."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid
import json

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import BacktestError
from backend.app.schemas.backtesting import (
    BacktestRequest,
    BacktestResults,
    Trade,
    TradeAction,
    PortfolioSnapshot,
    DrawdownInfo,
    BacktestSummary,
    PositionSizingMethod,
)
from backend.app.services.model_service import ModelService
from backend.app.services.data_service import DataService
from backend.app.services.prediction_service import PredictionService

# Import from existing src/
try:
    from src.evaluation.backtester import Backtester as SrcBacktester
    from src.evaluation.backtester import PositionSizingMethod as SrcPositionSizing
    HAS_SRC = True
except ImportError:
    HAS_SRC = False
    SrcBacktester = None


class BacktestService:
    """Service for running backtests."""

    def __init__(self):
        """Initialize backtest service."""
        self._model_service = ModelService()
        self._data_service = DataService()
        self._prediction_service = PredictionService()
        self._results: Dict[str, BacktestResults] = {}

    def run_backtest(self, request: BacktestRequest) -> BacktestResults:
        """Run a backtest."""
        result_id = str(uuid.uuid4())[:8]

        try:
            # Get stock data
            stock_data = self._data_service.get_stock_data(
                ticker=request.ticker,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            if len(stock_data.data) < 100:
                raise BacktestError("Insufficient data for backtesting")

            # Convert to DataFrame
            df = pd.DataFrame([d.model_dump() for d in stock_data.data])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            if HAS_SRC:
                results = self._run_with_src(request, df, result_id)
            else:
                results = self._run_simulated(request, df, result_id)

            # Store results
            self._results[result_id] = results

            return results

        except Exception as e:
            raise BacktestError(f"Backtest failed: {str(e)}")

    def _run_with_src(
        self,
        request: BacktestRequest,
        df: pd.DataFrame,
        result_id: str,
    ) -> BacktestResults:
        """Run backtest using src/ modules."""
        import torch
        from src.trading.agent import SignalGenerator

        # Prepare sequence data
        sequence_length = 60
        sequences, targets, df_norm = self._data_service.prepare_sequences(
            ticker=request.ticker,
            sequence_length=sequence_length
        )

        if len(df_norm) <= sequence_length:
            raise BacktestError("Not enough sequence data to run neural network evaluation.")

        # Sequences match rows from index sequence_length onwards
        eval_df = df_norm.iloc[sequence_length:].copy()
        
        # Filter eval_df and sequences to only include dates requested in df
        common_indices = eval_df.index.intersection(df.index)
        if len(common_indices) == 0:
            raise BacktestError("No overlapping dates between sequence data and requested backtest period.")
            
        # Create a boolean mask to filter both the dataframe and the numpy arrays
        mask = eval_df.index.isin(common_indices)
        eval_df = eval_df.loc[mask]
        sequences = sequences[mask]
        
        # Pull exact un-normalized price data matching evaluation windows
        prices = df.loc[common_indices]
        
        data_input = pd.DataFrame({
            'timestamp': prices.index,
            'ticker': request.ticker,
            'price': prices['close'].values
        })

        # Spin up Generator
        model = self._model_service.load_model(request.model_key)
        device = self._prediction_service._device
        
        seq_tensor = torch.FloatTensor(sequences).to(device)
        generator = SignalGenerator(model=model, device=device, n_mc_samples=25)

        # Extract Scaler Mean/Std for Log Return Denormalization
        ret_mean = float(getattr(df_norm, '_scaler_mean', {}).get('log_return', 0.0))
        ret_std = float(getattr(df_norm, '_scaler_std', {}).get('log_return', 1.0))

        # Batch create signals via Model Tensors
        signals_list, _ = generator.generate_signals(
            sequences=seq_tensor,
            current_prices=data_input['price'].values,
            tickers=data_input['ticker'].values,
            timestamps=pd.to_datetime(data_input['timestamp']).tolist(),
            threshold=request.signal_threshold,
            estimate_uncertainty=True,
            price_mean=ret_mean,
            price_std=ret_std,
            prediction_target='log_return'
        )
        
        signals_df = generator.signals_to_dataframe(signals_list)

        # Map position sizing method
        sizing_map = {
            PositionSizingMethod.FIXED: SrcPositionSizing.FIXED,
            PositionSizingMethod.KELLY_FULL: SrcPositionSizing.KELLY_FULL,
            PositionSizingMethod.KELLY_HALF: SrcPositionSizing.KELLY_HALF,
            PositionSizingMethod.KELLY_QUARTER: SrcPositionSizing.KELLY_QUARTER,
            PositionSizingMethod.VOLATILITY: SrcPositionSizing.VOLATILITY,
            PositionSizingMethod.CONFIDENCE: SrcPositionSizing.CONFIDENCE,
        }

        backtester = SrcBacktester(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            max_position_size=request.max_position_size,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            position_sizing_method=sizing_map.get(
                request.position_sizing_method,
                SrcPositionSizing.FIXED,
            ),
        )

        try:
            # Execute actual Core Engine loop
            src_results = backtester.run_backtest(signals_df, data_input)
        except Exception as e:
            raise BacktestError(f"Backtesting engine failed: {e}")

        # Map Results back to API Schema
        portfolio_history = []
        for i, ts in enumerate(src_results.timestamps):
            val = src_results.portfolio_values[i]
            daily_rtn = float(src_results.returns[i-1]) if i > 0 else 0.0
            
            portfolio_history.append(PortfolioSnapshot(
                timestamp=ts,
                portfolio_value=val,
                cash=0.0,
                positions_value=0.0,
                daily_return=daily_rtn,
                cumulative_return=(val / request.initial_capital) - 1,
            ))

        trades = []
        for i, t in enumerate(src_results.trades):
            trades.append(Trade(
                id=f"T{i+1}",
                timestamp=t.timestamp,
                ticker=t.ticker,
                action=TradeAction(t.action),
                price=t.price,
                quantity=t.quantity,
                value=t.value,
                commission=t.commission,
                slippage=t.slippage,
                position_before=t.position_before,
                position_after=t.position_after,
                pnl=t.pnl,
                pnl_percent=t.pnl_percent
            ))
            
        winning = len([t for t in trades if t.pnl and t.pnl > 0])
        losing = len([t for t in trades if t.pnl and t.pnl < 0])

        equity = np.array(src_results.portfolio_values)
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            drawdown_arr = (equity - peak) / peak
        else:
            drawdown_arr = np.array([])
            
        drawdowns = []
        in_drawdown = False
        dd_start = None
        for i, dd in enumerate(drawdown_arr):
            if dd < -0.01 and not in_drawdown:
                in_drawdown = True
                dd_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if dd_start is not None:
                    drawdowns.append(DrawdownInfo(
                        start_date=portfolio_history[dd_start].timestamp if dd_start < len(portfolio_history) else datetime.now(),
                        end_date=portfolio_history[i].timestamp if i < len(portfolio_history) else datetime.now(),
                        drawdown_percent=float(np.min(drawdown_arr[dd_start:i+1])),
                        duration_days=i - dd_start,
                    ))
                    
        total_rtn = src_results.metrics.get('total_return', 0.0)
        # Using the `metrics` map safely and assigning default fallbacks
        rtn_ann = src_results.metrics.get('annualized_return', 0.0)
        if 'annual_return' in src_results.metrics:
            rtn_ann = src_results.metrics['annual_return']

        return BacktestResults(
            model_key=request.model_key,
            ticker=request.ticker,
            start_date=data_input.iloc[0]['timestamp'] if not data_input.empty else datetime.now(),
            end_date=data_input.iloc[-1]['timestamp'] if not data_input.empty else datetime.now(),
            initial_capital=request.initial_capital,
            final_value=src_results.portfolio_values[-1] if src_results.portfolio_values else request.initial_capital,
            total_return=total_rtn,
            annual_return=rtn_ann,
            sharpe_ratio=src_results.metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=src_results.metrics.get('sortino_ratio', 0.0),
            max_drawdown=src_results.metrics.get('max_drawdown', 0.0),
            win_rate=src_results.metrics.get('trade_win_rate', 0.0),
            profit_factor=src_results.metrics.get('profit_factor', 0.0),
            total_trades=src_results.metrics.get('num_trades', 0),
            metrics=src_results.metrics,
            portfolio_history=portfolio_history,
            equity_curve=src_results.portfolio_values,
            returns=src_results.returns.tolist(),
            trades=trades,
            winning_trades=winning,
            losing_trades=losing,
            drawdowns=drawdowns
        )

    def _run_simulated(
        self,
        request: BacktestRequest,
        df: pd.DataFrame,
        result_id: str,
    ) -> BacktestResults:
        """Run simulated backtest."""
        import random

        # Initialize
        capital = request.initial_capital
        position = 0.0
        trades = []
        portfolio_history = []
        equity_curve = [capital]
        returns = []

        # Calculate returns
        df["return"] = df["close"].pct_change()

        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i == 0:
                continue

            current_price = row["close"]

            # Simple signal based on momentum
            if i >= 20:
                momentum = df["close"].iloc[i-20:i].pct_change().mean()

                if momentum > request.signal_threshold and position == 0:
                    # Buy
                    quantity = (capital * request.max_position_size) / current_price
                    cost = quantity * current_price * (1 + request.commission_rate)

                    trades.append(Trade(
                        id=f"T{len(trades)+1}",
                        timestamp=timestamp,
                        ticker=request.ticker,
                        action=TradeAction.BUY,
                        price=current_price,
                        quantity=quantity,
                        value=cost,
                        commission=quantity * current_price * request.commission_rate,
                        slippage=quantity * current_price * request.slippage_rate,
                        position_before=0,
                        position_after=quantity,
                    ))

                    capital -= cost
                    position = quantity

                elif momentum < -request.signal_threshold and position > 0:
                    # Sell
                    proceeds = position * current_price * (1 - request.commission_rate)
                    pnl = proceeds - (position * trades[-1].price if trades else current_price)

                    trades.append(Trade(
                        id=f"T{len(trades)+1}",
                        timestamp=timestamp,
                        ticker=request.ticker,
                        action=TradeAction.SELL,
                        price=current_price,
                        quantity=position,
                        value=proceeds,
                        commission=position * current_price * request.commission_rate,
                        slippage=position * current_price * request.slippage_rate,
                        position_before=position,
                        position_after=0,
                        pnl=pnl,
                        pnl_percent=(pnl / (position * trades[-1].price)) * 100 if trades else 0,
                    ))

                    capital += proceeds
                    position = 0

            # Track portfolio value
            portfolio_value = capital + (position * current_price)
            daily_return = (portfolio_value / equity_curve[-1]) - 1 if equity_curve else 0

            equity_curve.append(portfolio_value)
            returns.append(daily_return)

            portfolio_history.append(PortfolioSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash=capital,
                positions_value=position * current_price,
                daily_return=daily_return,
                cumulative_return=(portfolio_value / request.initial_capital) - 1,
            ))

        # Calculate metrics
        returns_arr = np.array(returns)
        final_value = equity_curve[-1]
        total_return = (final_value / request.initial_capital) - 1
        n_days = len(returns)

        # Annualized return
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe
        excess_returns = returns_arr - (0.02 / 252)
        sharpe = np.mean(excess_returns) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)

        # Sortino
        downside = returns_arr[returns_arr < 0]
        sortino = np.mean(excess_returns) / (np.std(downside) + 1e-8) * np.sqrt(252) if len(downside) > 0 else sharpe

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_dd = float(np.min(drawdown))

        # Win rate
        winning = [t for t in trades if t.pnl and t.pnl > 0]
        losing = [t for t in trades if t.pnl and t.pnl < 0]
        win_rate = len(winning) / max(len(trades), 1) * 100

        # Profit factor
        gross_profit = sum(t.pnl for t in winning if t.pnl)
        gross_loss = abs(sum(t.pnl for t in losing if t.pnl))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        # Drawdown analysis
        drawdowns = []
        in_drawdown = False
        dd_start = None

        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:
                in_drawdown = True
                dd_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if dd_start is not None:
                    drawdowns.append(DrawdownInfo(
                        start_date=portfolio_history[dd_start].timestamp if dd_start < len(portfolio_history) else datetime.now(),
                        end_date=portfolio_history[i].timestamp if i < len(portfolio_history) else datetime.now(),
                        drawdown_percent=float(np.min(drawdown[dd_start:i+1])),
                        duration_days=i - dd_start,
                    ))

        return BacktestResults(
            model_key=request.model_key,
            ticker=request.ticker,
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=request.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            metrics={
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino),
                "max_drawdown": max_dd,
                "win_rate": win_rate,
            },
            portfolio_history=portfolio_history,
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            drawdowns=drawdowns,
        )

    def get_results(self, result_id: str) -> Optional[BacktestResults]:
        """Get backtest results by ID."""
        return self._results.get(result_id)

    def list_results(
        self,
        ticker: Optional[str] = None,
        model_key: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List backtest results."""
        results = list(self._results.values())

        if ticker:
            results = [r for r in results if r.ticker == ticker]
        if model_key:
            results = [r for r in results if r.model_key == model_key]

        # Sort by end date
        results.sort(key=lambda r: r.end_date, reverse=True)

        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        results = results[start:end]

        summaries = [
            BacktestSummary(
                result_id=str(uuid.uuid4())[:8],  # Would use actual ID in real implementation
                model_key=r.model_key,
                ticker=r.ticker,
                run_date=r.end_date,
                total_return=r.total_return,
                sharpe_ratio=r.sharpe_ratio,
                max_drawdown=r.max_drawdown,
                total_trades=r.total_trades,
            )
            for r in results
        ]

        return {
            "results": summaries,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_trades(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get trade history for a backtest."""
        result = self._results.get(result_id)
        if not result:
            return None

        return {
            "result_id": result_id,
            "trades": result.trades,
            "total": len(result.trades),
            "summary": {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
            },
        }

    def save_results(self, result_id: str):
        """Save backtest results to disk."""
        if result_id not in self._results:
            return

        result = self._results[result_id]
        results_dir = settings.results_path / "backtests"
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / f"{result_id}.json"
        with open(filepath, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    def load_results(self, result_id: str) -> Optional[BacktestResults]:
        """Load backtest results from disk."""
        results_dir = settings.results_path / "backtests"
        filepath = results_dir / f"{result_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        return BacktestResults(**data)
