"""Tests for the unified BacktestEngine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import BacktestEngine
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams
from portfolio_agent.config.schema import StrategyConfig


@pytest.fixture
def synthetic_data(monkeypatch):
    """Create synthetic market data for 3 tickers over 300 days (enough for sma_200)."""
    np.random.seed(7)
    start_date = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start=start_date, periods=300)

    tickers = ["SYNTH1.NS", "SYNTH2.NS", "SYNTH3.NS"]
    data_dict = {}

    for i, ticker in enumerate(tickers):
        base_price = 100 + i * 50
        data = {
            'open': [base_price + j * 0.5 + np.random.uniform(-2, 2) for j in range(len(dates))],
            'high': [base_price + j * 0.5 + np.random.uniform(0, 5) for j in range(len(dates))],
            'low': [base_price + j * 0.5 - np.random.uniform(0, 5) for j in range(len(dates))],
            'close': [base_price + j * 0.5 + np.random.uniform(-1, 3) for j in range(len(dates))],
            'volume': [1000000 + j * 1000 + np.random.randint(-10000, 10000) for j in range(len(dates))]
        }
        df = pd.DataFrame(data, index=dates)
        data_dict[ticker] = df

    def mock_load_ticker_data(ticker, start_date=None, end_date=None):
        if ticker in data_dict:
            df = data_dict[ticker].copy()
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            return df
        return None

    monkeypatch.setattr("src.backtest_engine.load_ticker_data", mock_load_ticker_data)

    return {'tickers': tickers, 'data': data_dict, 'dates': dates}


class TestBacktestEngineInitialization:
    """Test BacktestEngine initialization."""

    def test_init_creates_engine_with_default_strategy(self, synthetic_data):
        """With no strategy specified, the engine defaults to the registered rule_based strategy."""
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
        assert engine.strategy.name == "Trend Breakout Volume MC"

    def test_init_with_explicit_strategy_and_risk_params(self, synthetic_data):
        """The engine should accept an explicit strategy/risk_params instance."""
        tickers = synthetic_data['tickers']
        strategy = load_strategy(StrategyConfig(type="rule_based", params={"yaml_path": "config/strategies/trend_breakout.yaml"}))
        risk_params = RiskParams(
            target_prob_profit=0.5, min_reward_risk=1.0, min_price_inr=10.0,
            portfolio_value_inr=1000000.0, risk_per_trade_pct=0.01, max_single_position_pct=0.10,
        )

        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers,
            strategy=strategy,
            risk_params=risk_params,
        )

        assert engine.strategy is strategy
        assert engine.risk_params is risk_params

    def test_init_with_custom_brain(self, synthetic_data):
        """Test initialization with custom brain state."""
        custom_brain = {
            'weights': {'Trend': 30.0, 'Breakout': 30.0, 'Volume': 20.0, 'MC_Prob': 20.0},
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

    def test_run_backtest_equity_curve_length(self, synthetic_data):
        tickers = synthetic_data['tickers']

        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )

        results = engine.run_backtest()

        assert isinstance(results['daily_equity_curve'], pd.Series)
        assert len(results['daily_equity_curve']) == len(engine.master_date_index)

    def test_run_backtest_portfolio_value_consistency(self, synthetic_data):
        tickers = synthetic_data['tickers']

        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-05-31",
            initial_capital=1000000.0,
            universe_tickers=tickers
        )

        results = engine.run_backtest()
        final_equity = results['daily_equity_curve'].iloc[-1]

        assert engine.portfolio_value == engine.cash + sum(
            qty * engine._get_price_at_date(ticker, engine.master_date_index[-1], 'close')
            for ticker, qty in engine.holdings.items()
            if engine._get_price_at_date(ticker, engine.master_date_index[-1], 'close') is not None
        )
        assert abs(final_equity - engine.portfolio_value) < 0.01

    def test_run_backtest_returns_trade_log(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=synthetic_data['tickers']
        )
        results = engine.run_backtest()
        assert isinstance(results['trade_log'], list)

    def test_run_backtest_returns_brain_evolution(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=synthetic_data['tickers']
        )
        results = engine.run_backtest()

        assert isinstance(results['brain_evolution'], list)
        assert len(results['brain_evolution']) >= 1
        for snapshot in results['brain_evolution']:
            assert 'weights' in snapshot
            assert 'trading_day' in snapshot

    def test_parallel_and_serial_runs_produce_signals_on_same_window(self, synthetic_data):
        """Parallel dispatch is a performance change only — both paths should run without error
        and produce a signal dict of the same shape on a fixed date."""
        tickers = synthetic_data['tickers']

        serial_engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=tickers, parallel=False,
        )
        parallel_engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=tickers, parallel=True, max_workers=2,
        )

        test_date = serial_engine.master_date_index[len(serial_engine.master_date_index) // 2]

        serial_signals = serial_engine._generate_signals(test_date)
        parallel_signals = parallel_engine._generate_signals(test_date)

        assert set(serial_signals.keys()) == set(parallel_signals.keys())
        for ticker in serial_signals:
            assert serial_signals[ticker].signal == parallel_signals[ticker].signal
            assert serial_signals[ticker].score == pytest.approx(parallel_signals[ticker].score)


class TestLookAheadBiasPrevention:
    """Test that look-ahead bias is properly prevented."""

    def test_signals_use_only_historical_data(self, synthetic_data):
        tickers = synthetic_data['tickers']

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=tickers
        )

        test_date = engine.master_date_index[len(engine.master_date_index) // 2]
        signals = engine._generate_signals(test_date)

        assert isinstance(signals, dict)
        for ticker, sig in signals.items():
            hist_data = engine._get_historical_data_up_to(ticker, test_date)
            if hist_data is not None:
                assert hist_data.index.max() < test_date


class TestCorporateActions:
    """Test handling of corporate actions and delisted tickers."""

    def test_untradeable_ticker_handling(self, monkeypatch):
        def mock_load_ticker_data(ticker, start_date=None, end_date=None):
            if ticker == "BAD.NS":
                return None
            dates = pd.bdate_range("2023-01-02", periods=50)
            return pd.DataFrame({
                'open': [100] * len(dates), 'high': [105] * len(dates),
                'low': [99] * len(dates), 'close': [103] * len(dates), 'volume': [1000] * len(dates)
            }, index=dates)

        monkeypatch.setattr("src.backtest_engine.load_ticker_data", mock_load_ticker_data)

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1000000.0, universe_tickers=["GOOD.NS", "BAD.NS"]
        )

        assert "BAD.NS" in engine.untradeable_tickers
        assert "GOOD.NS" not in engine.untradeable_tickers


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_get_performance_metrics(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-05-31",
            initial_capital=1000000.0, universe_tickers=synthetic_data['tickers']
        )
        engine.run_backtest()
        metrics = engine.get_performance_metrics()

        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'final_portfolio_value' in metrics
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['final_portfolio_value'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
