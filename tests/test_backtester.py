"""
Comprehensive tests for backtesting framework
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.evaluation.backtester import Trade, Position, BacktestResults, Backtester


class TestTrade:
    """Test Trade dataclass"""

    def test_trade_creation(self):
        """Test creating a trade"""
        trade = Trade(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='AAPL',
            action='BUY',
            price=150.0,
            quantity=10.0,
            value=1500.0,
            commission=1.5,
            reason="Signal triggered"
        )
        
        assert trade.timestamp == pd.Timestamp('2020-01-01')
        assert trade.ticker == 'AAPL'
        assert trade.action == 'BUY'
        assert trade.price == 150.0
        assert trade.quantity == 10.0
        assert trade.value == 1500.0
        assert trade.commission == 1.5
        assert trade.reason == "Signal triggered"

    def test_trade_default_values(self):
        """Test trade with default values"""
        trade = Trade(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='AAPL',
            action='BUY',
            price=150.0,
            quantity=10.0,
            value=1500.0
        )
        
        assert trade.commission == 0.0
        assert trade.reason == ""


class TestPosition:
    """Test Position dataclass"""

    def test_position_creation(self):
        """Test creating a position"""
        position = Position(
            ticker='AAPL',
            quantity=10.0,
            entry_price=150.0,
            entry_time=pd.Timestamp('2020-01-01'),
            current_price=150.0
        )
        
        assert position.ticker == 'AAPL'
        assert position.quantity == 10.0
        assert position.entry_price == 150.0
        assert position.current_price == 150.0

    def test_position_update_price(self):
        """Test updating position price"""
        position = Position(
            ticker='AAPL',
            quantity=10.0,
            entry_price=150.0,
            entry_time=pd.Timestamp('2020-01-01'),
            current_price=150.0
        )
        
        # Update to higher price
        position.update_price(160.0)
        
        assert position.current_price == 160.0
        assert position.market_value == 1600.0  # 10 * 160
        assert position.unrealized_pnl == 100.0  # (160 - 150) * 10

    def test_position_update_price_loss(self):
        """Test position update with price loss"""
        position = Position(
            ticker='AAPL',
            quantity=10.0,
            entry_price=150.0,
            entry_time=pd.Timestamp('2020-01-01'),
            current_price=150.0
        )
        
        # Update to lower price
        position.update_price(140.0)
        
        assert position.current_price == 140.0
        assert position.market_value == 1400.0
        assert position.unrealized_pnl == -100.0  # (140 - 150) * 10

    def test_position_stop_loss_take_profit(self):
        """Test position with stop loss and take profit"""
        position = Position(
            ticker='AAPL',
            quantity=10.0,
            entry_price=150.0,
            entry_time=pd.Timestamp('2020-01-01'),
            current_price=150.0,
            stop_loss=140.0,
            take_profit=165.0
        )
        
        assert position.stop_loss == 140.0
        assert position.take_profit == 165.0


class TestBacktestResults:
    """Test BacktestResults dataclass"""

    def test_backtest_results_creation(self):
        """Test creating backtest results"""
        results = BacktestResults()
        
        assert results.trades == []
        assert results.portfolio_values == []
        assert results.timestamps == []
        assert len(results.returns) == 0
        assert results.metrics == {}

    def test_backtest_results_to_dataframe(self):
        """Test converting results to DataFrame"""
        results = BacktestResults(
            timestamps=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')],
            portfolio_values=[100000.0, 101000.0],
            returns=np.array([0.01])
        )
        
        df = results.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'portfolio_value' in df.columns
        assert 'returns' in df.columns
        assert len(df) == 2

    def test_backtest_results_summary(self):
        """Test generating results summary"""
        results = BacktestResults(
            trades=[
                Trade(pd.Timestamp('2020-01-01'), 'AAPL', 'BUY', 150.0, 10.0, 1500.0)
            ],
            portfolio_values=[100000.0, 101000.0],
            metrics={'sharpe_ratio': 1.5, 'max_drawdown': -0.05}
        )
        
        summary = results.summary()
        
        assert isinstance(summary, str)
        assert 'Total Trades' in summary
        assert 'Portfolio Value' in summary
        assert 'sharpe_ratio' in summary


class TestBacktester:
    """Test Backtester class"""

    def test_backtester_initialization(self):
        """Test backtester initialization"""
        backtester = Backtester(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        assert backtester.initial_capital == 100000.0
        assert backtester.commission_rate == 0.001
        assert backtester.slippage_rate == 0.0005
        assert backtester.cash == 100000.0
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0

    def test_backtester_reset(self):
        """Test resetting backtester"""
        backtester = Backtester(initial_capital=100000.0)
        
        # Modify state
        backtester.cash = 50000.0
        backtester.positions['AAPL'] = Position('AAPL', 10, 150.0, pd.Timestamp.now(), 150.0)
        backtester.trades.append(Trade(pd.Timestamp.now(), 'AAPL', 'BUY', 150.0, 10, 1500.0))
        
        # Reset
        backtester.reset()
        
        assert backtester.cash == 100000.0
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0

    def test_get_portfolio_value_cash_only(self):
        """Test portfolio value with only cash"""
        backtester = Backtester(initial_capital=100000.0)
        
        portfolio_value = backtester.get_portfolio_value({})
        
        assert portfolio_value == 100000.0

    def test_get_portfolio_value_with_positions(self):
        """Test portfolio value with positions"""
        backtester = Backtester(initial_capital=100000.0)
        
        # Add some positions
        backtester.cash = 50000.0
        backtester.positions['AAPL'] = Position(
            'AAPL', 100, 150.0, pd.Timestamp.now(), 150.0
        )
        backtester.positions['GOOGL'] = Position(
            'GOOGL', 50, 200.0, pd.Timestamp.now(), 200.0
        )
        
        # Get portfolio value with current prices
        prices = {'AAPL': 160.0, 'GOOGL': 210.0}
        portfolio_value = backtester.get_portfolio_value(prices)
        
        # Cash: 50000
        # AAPL: 100 * 160 = 16000
        # GOOGL: 50 * 210 = 10500
        # Total: 76500
        assert portfolio_value == 76500.0

    def test_position_price_update(self):
        """Test that positions are updated when getting portfolio value"""
        backtester = Backtester(initial_capital=100000.0)
        
        position = Position('AAPL', 100, 150.0, pd.Timestamp.now(), 150.0)
        backtester.positions['AAPL'] = position
        
        # Get portfolio value with new price
        prices = {'AAPL': 160.0}
        backtester.get_portfolio_value(prices)
        
        # Position should be updated
        assert position.current_price == 160.0
        assert position.market_value == 16000.0

    def test_commission_calculation(self):
        """Test commission calculation"""
        backtester = Backtester(
            initial_capital=100000.0,
            commission_rate=0.001  # 0.1%
        )
        
        # Buy trade value: 1500
        # Commission: 1500 * 0.001 = 1.5
        trade_value = 1500.0
        commission = trade_value * backtester.commission_rate
        
        assert commission == 1.5

    def test_max_position_size(self):
        """Test maximum position size constraint"""
        backtester = Backtester(
            initial_capital=100000.0,
            max_position_size=0.2  # 20%
        )
        
        # Maximum position value should be 20% of 100000 = 20000
        max_value = backtester.initial_capital * backtester.max_position_size
        
        assert max_value == 20000.0

    def test_stop_loss_level(self):
        """Test stop loss level"""
        backtester = Backtester(
            initial_capital=100000.0,
            stop_loss=0.02  # 2%
        )
        
        # If entry price is 150, stop loss should trigger at 150 * (1 - 0.02) = 147
        entry_price = 150.0
        stop_loss_price = entry_price * (1 - backtester.stop_loss)
        
        assert stop_loss_price == 147.0

    def test_take_profit_level(self):
        """Test take profit level"""
        backtester = Backtester(
            initial_capital=100000.0,
            take_profit=0.05  # 5%
        )
        
        # If entry price is 150, take profit should trigger at 150 * (1 + 0.05) = 157.5
        entry_price = 150.0
        take_profit_price = entry_price * (1 + backtester.take_profit)
        
        assert take_profit_price == 157.5


class TestBacktesterEdgeCases:
    """Test edge cases for backtester"""

    def test_zero_initial_capital(self):
        """Test with zero initial capital"""
        backtester = Backtester(initial_capital=0.0)
        
        assert backtester.cash == 0.0
        
        portfolio_value = backtester.get_portfolio_value({})
        assert portfolio_value == 0.0

    def test_negative_commission_rate(self):
        """Test with negative commission (rebates)"""
        backtester = Backtester(
            initial_capital=100000.0,
            commission_rate=-0.001  # Rebate
        )
        
        assert backtester.commission_rate == -0.001

    def test_zero_commission(self):
        """Test with zero commission"""
        backtester = Backtester(
            initial_capital=100000.0,
            commission_rate=0.0
        )
        
        trade_value = 1000.0
        commission = trade_value * backtester.commission_rate
        
        assert commission == 0.0

    def test_high_slippage(self):
        """Test with high slippage"""
        backtester = Backtester(
            initial_capital=100000.0,
            slippage_rate=0.01  # 1% slippage
        )
        
        assert backtester.slippage_rate == 0.01

    def test_missing_price_for_position(self):
        """Test getting portfolio value when price is missing for a position"""
        backtester = Backtester(initial_capital=100000.0)
        
        backtester.cash = 50000.0
        backtester.positions['AAPL'] = Position(
            'AAPL', 100, 150.0, pd.Timestamp.now(), 150.0
        )
        
        # Get portfolio value without AAPL price
        prices = {'GOOGL': 200.0}  # Missing AAPL
        portfolio_value = backtester.get_portfolio_value(prices)
        
        # Should use last known price (150)
        # Cash: 50000, AAPL: 100 * 150 = 15000
        assert portfolio_value == 65000.0

    def test_empty_positions(self):
        """Test with no positions"""
        backtester = Backtester(initial_capital=100000.0)
        
        portfolio_value = backtester.get_portfolio_value({'AAPL': 150.0})
        
        assert portfolio_value == 100000.0
        assert len(backtester.positions) == 0


class TestBacktesterPortfolio:
    """Test portfolio management"""

    def test_portfolio_value_increases(self):
        """Test portfolio value increases with price gains"""
        backtester = Backtester(initial_capital=100000.0)
        
        # Start with position
        backtester.cash = 50000.0
        backtester.positions['AAPL'] = Position(
            'AAPL', 100, 150.0, pd.Timestamp.now(), 150.0
        )
        
        initial_value = backtester.get_portfolio_value({'AAPL': 150.0})
        
        # Price increases
        new_value = backtester.get_portfolio_value({'AAPL': 160.0})
        
        assert new_value > initial_value
        assert new_value == 50000.0 + 100 * 160.0

    def test_portfolio_value_decreases(self):
        """Test portfolio value decreases with price losses"""
        backtester = Backtester(initial_capital=100000.0)
        
        backtester.cash = 50000.0
        backtester.positions['AAPL'] = Position(
            'AAPL', 100, 150.0, pd.Timestamp.now(), 150.0
        )
        
        initial_value = backtester.get_portfolio_value({'AAPL': 150.0})
        
        # Price decreases
        new_value = backtester.get_portfolio_value({'AAPL': 140.0})
        
        assert new_value < initial_value
        assert new_value == 50000.0 + 100 * 140.0

    def test_multiple_positions(self):
        """Test portfolio with multiple positions"""
        backtester = Backtester(initial_capital=100000.0)
        
        backtester.cash = 20000.0
        backtester.positions['AAPL'] = Position('AAPL', 100, 150.0, pd.Timestamp.now(), 150.0)
        backtester.positions['GOOGL'] = Position('GOOGL', 50, 200.0, pd.Timestamp.now(), 200.0)
        backtester.positions['MSFT'] = Position('MSFT', 200, 100.0, pd.Timestamp.now(), 100.0)
        
        prices = {'AAPL': 155.0, 'GOOGL': 205.0, 'MSFT': 105.0}
        portfolio_value = backtester.get_portfolio_value(prices)
        
        # Cash: 20000
        # AAPL: 100 * 155 = 15500
        # GOOGL: 50 * 205 = 10250
        # MSFT: 200 * 105 = 21000
        # Total: 66750
        assert portfolio_value == 66750.0


class TestBacktesterConfiguration:
    """Test different backtester configurations"""

    def test_different_initial_capitals(self):
        """Test with different initial capital amounts"""
        capitals = [10000.0, 100000.0, 1000000.0]
        
        for capital in capitals:
            backtester = Backtester(initial_capital=capital)
            assert backtester.cash == capital
            assert backtester.initial_capital == capital

    def test_different_commission_rates(self):
        """Test with different commission rates"""
        rates = [0.0, 0.001, 0.005, 0.01]
        
        for rate in rates:
            backtester = Backtester(initial_capital=100000.0, commission_rate=rate)
            assert backtester.commission_rate == rate

    def test_different_max_position_sizes(self):
        """Test with different max position sizes"""
        sizes = [0.1, 0.2, 0.5, 1.0]
        
        for size in sizes:
            backtester = Backtester(initial_capital=100000.0, max_position_size=size)
            assert backtester.max_position_size == size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
