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


def _signal(ticker, signal="BUY", score=90.0, entry_price=100.0, extra=None):
    from portfolio_agent.strategies.types import StrategySignal

    return StrategySignal(
        symbol=ticker, signal=signal, score=score, trigger="Momentum",
        entry_price=entry_price, stop_price=entry_price * 0.95,
        target_price=entry_price * 1.10, reward_risk=2.0, probability_profit=0.6,
        extra=extra or {},
    )


class TestPositionScaling:
    """Signals that measure their own risk environment publish a
    position_scale; sizing must honour it wherever the quantity came from."""

    def test_scale_shrinks_the_sized_quantity(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )
        full = engine._apply_position_scale(1000, _signal("A", extra={"position_scale": 1.0}))
        half = engine._apply_position_scale(1000, _signal("A", extra={"position_scale": 0.5}))
        none = engine._apply_position_scale(1000, _signal("A", extra={"position_scale": 0.0}))

        assert full == 1000
        assert half == 500
        assert none == 0

    def test_missing_or_unparseable_scale_leaves_quantity_untouched(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )

        assert engine._apply_position_scale(1000, _signal("A")) == 1000
        assert engine._apply_position_scale(1000, _signal("A", extra={"position_scale": "x"})) == 1000

    def test_scale_above_one_never_levers_up(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )

        assert engine._apply_position_scale(1000, _signal("A", extra={"position_scale": 3.0})) == 1000

    def test_scaled_signal_queues_a_smaller_order(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )
        date = pd.Timestamp("2023-02-01")
        ticker = synthetic_data['tickers'][0]

        engine._create_pending_orders({ticker: _signal(ticker)}, date)
        unscaled = engine.pending_orders[0]['quantity']

        engine.pending_orders.clear()
        engine._create_pending_orders(
            {ticker: _signal(ticker, extra={"position_scale": 0.25})}, date
        )
        scaled = engine.pending_orders[0]['quantity']

        assert scaled == int(unscaled * 0.25)


class TestSectorConcentrationCap:
    def test_cap_trims_the_second_name_in_a_crowded_sector(self, synthetic_data, tmp_path):
        tickers = synthetic_data['tickers']
        sector_csv = tmp_path / "sectors.csv"
        sector_csv.write_text(
            "ticker,sector\n" + "".join(f"{t},IT\n" for t in tickers), encoding="utf-8"
        )

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers,
            max_sector_pct=0.15, sector_map_csv=str(sector_csv),
        )
        date = pd.Timestamp("2023-02-01")

        signals = {t: _signal(t, score=90.0 - i, entry_price=100.0) for i, t in enumerate(tickers)}
        engine._create_pending_orders(signals, date)

        queued_value = sum(o['quantity'] * 100.0 for o in engine.pending_orders)
        # Three 10%-of-portfolio positions would be 30% of the book in one
        # sector; the 15% cap must hold across the whole round, not per order.
        assert queued_value <= 0.15 * engine.portfolio_value + 100.0

    def test_unmapped_tickers_are_capped_as_one_pooled_sector(self, synthetic_data, tmp_path):
        tickers = synthetic_data['tickers']

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers,
            max_sector_pct=0.15, sector_map_csv=str(tmp_path / "absent.csv"),
        )
        date = pd.Timestamp("2023-02-01")

        signals = {t: _signal(t, score=90.0 - i, entry_price=100.0) for i, t in enumerate(tickers)}
        engine._create_pending_orders(signals, date)

        queued_value = sum(o['quantity'] * 100.0 for o in engine.pending_orders)
        assert queued_value <= 0.15 * engine.portfolio_value + 100.0

    def test_disabled_cap_leaves_orders_alone(self, synthetic_data):
        tickers = synthetic_data['tickers']
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers, max_sector_pct=0.0,
        )
        date = pd.Timestamp("2023-02-01")

        signals = {t: _signal(t, score=90.0 - i, entry_price=100.0) for i, t in enumerate(tickers)}
        engine._create_pending_orders(signals, date)

        assert len(engine.pending_orders) == len(tickers)


class TestDrawdownCircuitBreaker:
    def _engine(self, tickers, **kwargs):
        params = dict(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers,
            max_portfolio_drawdown_pct=0.15, drawdown_reentry_pct=0.10,
        )
        params.update(kwargs)
        return BacktestEngine(**params)

    def test_trips_past_the_threshold_and_blocks_new_buys(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'])
        ticker = synthetic_data['tickers'][0]
        date = pd.Timestamp("2023-02-01")

        engine.portfolio_value = 800_000.0  # 20% below the 1,000,000 peak
        engine._create_pending_orders({ticker: _signal(ticker)}, date)

        assert engine.buying_halted is True
        assert engine.pending_orders == []
        assert engine.circuit_breaker_log[-1]['event'] == 'HALT'

    def test_stays_armed_inside_the_threshold(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'])
        ticker = synthetic_data['tickers'][0]

        engine.portfolio_value = 900_000.0  # 10% drawdown, under the 15% trip
        engine._create_pending_orders({ticker: _signal(ticker)}, pd.Timestamp("2023-02-01"))

        assert engine.buying_halted is False
        assert engine.pending_orders

    def test_does_not_re_arm_between_the_two_thresholds(self, synthetic_data):
        """Separate trip and re-entry levels are what stop the breaker
        flickering on every wobble across a single line."""
        engine = self._engine(synthetic_data['tickers'])

        engine.portfolio_value = 800_000.0
        engine._update_circuit_breaker(pd.Timestamp("2023-02-01"))
        assert engine.buying_halted is True

        engine.portfolio_value = 870_000.0  # 13% down: past the trip, short of re-entry
        engine._update_circuit_breaker(pd.Timestamp("2023-02-02"))
        assert engine.buying_halted is True

    def test_re_arms_once_recovered(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'])
        ticker = synthetic_data['tickers'][0]

        engine.portfolio_value = 800_000.0
        engine._update_circuit_breaker(pd.Timestamp("2023-02-01"))
        engine.portfolio_value = 920_000.0  # 8% down, inside the 10% re-entry
        engine._create_pending_orders({ticker: _signal(ticker)}, pd.Timestamp("2023-02-03"))

        assert engine.buying_halted is False
        assert engine.pending_orders
        assert engine.circuit_breaker_log[-1]['event'] == 'RESUME'

    def test_sells_are_still_queued_while_halted(self, synthetic_data):
        """A halt stops new risk; it must not trap capital in open positions."""
        engine = self._engine(synthetic_data['tickers'])
        ticker = synthetic_data['tickers'][0]
        engine.holdings[ticker] = 100
        engine.portfolio_value = 800_000.0

        engine._create_pending_orders(
            {ticker: _signal(ticker, signal="SELL")}, pd.Timestamp("2023-02-01")
        )

        assert engine.buying_halted is True
        assert [o['action'] for o in engine.pending_orders] == ['SELL']

    def test_disabled_breaker_never_trips(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'], max_portfolio_drawdown_pct=0.0)
        ticker = synthetic_data['tickers'][0]

        engine.portfolio_value = 100_000.0  # 90% drawdown
        engine._create_pending_orders({ticker: _signal(ticker)}, pd.Timestamp("2023-02-01"))

        assert engine.buying_halted is False
        assert engine.pending_orders

    def test_peak_tracks_new_highs(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'])

        engine.portfolio_value = 1_500_000.0
        engine._update_circuit_breaker(pd.Timestamp("2023-02-01"))
        assert engine.equity_peak == 1_500_000.0

        engine.portfolio_value = 1_300_000.0  # ~13% off the NEW peak, not the old one
        engine._update_circuit_breaker(pd.Timestamp("2023-02-02"))
        assert engine.buying_halted is False
