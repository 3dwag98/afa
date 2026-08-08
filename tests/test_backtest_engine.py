"""Tests for the BacktestEngine."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import pytest

from backtest_engine import BacktestEngine


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch):
    """Create synthetic market data for 3 tickers over 100 days."""
    # Generate 100 business days of data
    start_date = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start=start_date, periods=100)
    
    # Create synthetic data for 3 tickers
    tickers = ["SYNTH1.NS", "SYNTH2.NS", "SYNTH3.NS"]
    data_dict = {}
    
    for i, ticker in enumerate(tickers):
        base_price = 100 + i * 50  # Different base prices
        data = {
            'open': [base_price + j * 0.5 + np.random.uniform(-2, 2) for j in range(len(dates))],
            'high': [base_price + j * 0.5 + np.random.uniform(0, 5) for j in range(len(dates))],
            'low': [base_price + j * 0.5 - np.random.uniform(0, 5) for j in range(len(dates))],
            'close': [base_price + j * 0.5 + np.random.uniform(-1, 3) for j in range(len(dates))],
            'volume': [1000000 + j * 1000 + np.random.randint(-10000, 10000) for j in range(len(dates))]
        }
        df = pd.DataFrame(data, index=dates)
        data_dict[ticker] = df
    
    # Mock load_ticker_data to return our synthetic data
    def mock_load_ticker_data(ticker, start_date=None, end_date=None):
        if ticker in data_dict:
            df = data_dict[ticker].copy()
            
            # Apply date filtering if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            return df
        return None
    
    monkeypatch.setattr("backtest_engine.load_ticker_data", mock_load_ticker_data)
    
    return {
        'tickers': tickers,
        'data': data_dict,
        'dates': dates
    }


class TestBacktestEngineInitialization:
    """Test BacktestEngine initialization."""
    
    def test_init_creates_engine(self, synthetic_data, monkeypatch):
        """Test that engine initializes correctly."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        assert engine.initial_capital == 1000000.0
        assert engine.cash == 1000000.0
        assert engine.holdings == {}
        assert engine.portfolio_value == 1000000.0
        assert len(engine.universe_tickers) == 3
    
    def test_init_with_custom_brain(self, synthetic_data, monkeypatch):
        """Test initialization with custom brain state."""
        custom_brain = {
            'weights': {
                'Trend': 30.0,
                'Breakout': 30.0,
                'Volume': 20.0,
                'MC_Prob': 20.0
            },
            'trade_history': [],
            'learning_log': []
        }
        
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers,
            initial_brain=custom_brain
        )
        
        assert engine.agent_brain.weights['Trend'] == 30.0
        assert engine.agent_brain.weights['Breakout'] == 30.0


class TestBacktestEngineRun:
    """Test BacktestEngine run_backtest method."""
    
    def test_run_backtest_equity_curve_length(self, synthetic_data, monkeypatch):
        """Test that equity curve has same length as master_date_index."""
        tickers = synthetic_data['tickers']
        dates = synthetic_data['dates']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        results = engine.run_backtest()
        
        # Assert equity curve is a Series
        assert isinstance(results['daily_equity_curve'], pd.Series)
        
        # Assert equity curve length matches master_date_index
        assert len(results['daily_equity_curve']) == len(engine.master_date_index)
    
    def test_run_backtest_portfolio_value_consistency(self, synthetic_data, monkeypatch):
        """Test that cash + holdings value equals total portfolio value."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        results = engine.run_backtest()
        
        # Get final equity from curve
        final_equity = results['daily_equity_curve'].iloc[-1]
        
        # Calculate holdings value at end
        # Note: We need to check that the accounting is consistent
        # The portfolio_value should equal cash + sum(holding * price)
        
        # At any point during backtest, this invariant should hold
        # We verify it by checking the final state
        assert engine.portfolio_value == engine.cash + sum(
            qty * engine._get_price_at_date(ticker, engine.master_date_index[-1], 'close')
            for ticker, qty in engine.holdings.items()
            if engine._get_price_at_date(ticker, engine.master_date_index[-1], 'close') is not None
        )
        
        # Final equity should match portfolio_value
        assert abs(final_equity - engine.portfolio_value) < 0.01  # Small floating point tolerance
    
    def test_run_backtest_returns_trade_log(self, synthetic_data, monkeypatch):
        """Test that run_backtest returns a trade_log list."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        results = engine.run_backtest()
        
        assert isinstance(results['trade_log'], list)
    
    def test_run_backtest_returns_brain_evolution(self, synthetic_data, monkeypatch):
        """Test that run_backtest returns brain_evolution list."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        results = engine.run_backtest()
        
        assert isinstance(results['brain_evolution'], list)
        # Should have at least one snapshot (final)
        assert len(results['brain_evolution']) >= 1
        
        # Each snapshot should have required keys
        for snapshot in results['brain_evolution']:
            assert 'weights' in snapshot
            assert 'trading_day' in snapshot


class TestLookAheadBiasPrevention:
    """Test that look-ahead bias is properly prevented."""
    
    def test_signals_use_only_historical_data(self, synthetic_data, monkeypatch):
        """Test that signal generation only uses data up to T-1."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        # Pick a date in the middle of the range
        test_date = engine.master_date_index[len(engine.master_date_index) // 2]
        
        # Generate signals
        signals = engine._generate_signals(test_date)
        
        # Verify signals were generated (or empty if not enough data)
        assert isinstance(signals, dict)
        
        # For each signal, verify the historical data used doesn't include test_date
        for ticker, signal_info in signals.items():
            hist_data = engine._get_historical_data_up_to(ticker, test_date)
            if hist_data is not None:
                # Ensure no data from test_date or later is included
                assert hist_data.index.max() < test_date


class TestCorporateActions:
    """Test handling of corporate actions and delisted tickers."""
    
    def test_untradeable_ticker_handling(self, monkeypatch):
        """Test that tickers with no data are marked untradeable."""
        def mock_load_ticker_data(ticker, start_date=None, end_date=None):
            # Return None for one ticker
            if ticker == "BAD.NS":
                return None
            # Return valid data for others
            dates = pd.bdate_range("2023-01-02", periods=50)
            return pd.DataFrame({
                'open': [100] * len(dates),
                'high': [105] * len(dates),
                'low': [99] * len(dates),
                'close': [103] * len(dates),
                'volume': [1000] * len(dates)
            }, index=dates)
        
        monkeypatch.setattr("backtest_engine.load_ticker_data", mock_load_ticker_data)
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-03-31",
            initial_capital=1000000.0,
            universe_tickers=["GOOD.NS", "BAD.NS"]
        )
        
        # BAD.NS should be in untradeable set
        assert "BAD.NS" in engine.untradeable_tickers
        assert "GOOD.NS" not in engine.untradeable_tickers


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_get_performance_metrics(self, synthetic_data, monkeypatch):
        """Test that performance metrics are calculated correctly."""
        tickers = synthetic_data['tickers']
        
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )
        
        engine.run_backtest()
        metrics = engine.get_performance_metrics()
        
        # Check required metrics exist
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'final_portfolio_value' in metrics
        
        # Check types
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['final_portfolio_value'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
