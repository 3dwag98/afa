"""Tests for the unified BacktestEngine."""

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.backtest_engine import BacktestEngine
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

    monkeypatch.setattr("portfolio_agent.src.backtest_engine.load_ticker_data", mock_load_ticker_data)

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

        monkeypatch.setattr("portfolio_agent.src.backtest_engine.load_ticker_data", mock_load_ticker_data)

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


class TestPortfolioRiskCap:
    """The only limit in the engine that is not per-position.

    Risk-per-trade, max_single_position_pct and max_sector_pct are all blind
    to correlation: twenty 3% positions correlated 0.6 carry roughly 3.5x the
    volatility of twenty independent ones, and nothing upstream could see it.
    """

    def _engine(self, tickers, **kwargs):
        params = dict(
            start_date="2023-01-02", end_date="2023-12-29",
            initial_capital=1_000_000.0, universe_tickers=tickers,
        )
        params.update(kwargs)
        return BacktestEngine(**params)

    def test_measurement_runs_even_with_the_constraint_disabled(self, synthetic_data):
        """The gap between true and independence-assumed risk is the finding;
        a report that omits it is the state the platform was already in."""
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=0.0)
        date = pd.Timestamp("2023-12-01")
        engine.holdings = {t: 100 for t in synthetic_data['tickers']}

        summary = engine._book_risk_summary(
            engine._current_position_values(date), date
        )

        assert summary is not None
        assert summary['portfolio_volatility'] > 0
        assert summary['independent_volatility'] > 0
        assert summary['n_positions'] == 3

    def test_covariance_uses_only_returns_before_the_decision_date(self, synthetic_data):
        """A covariance that knows how the names co-moved during the period it
        is sizing for is not a risk model, it is a memory."""
        engine = self._engine(synthetic_data['tickers'])
        cutoff = pd.Timestamp("2023-06-01")

        covariance = engine._book_covariance(synthetic_data['tickers'], cutoff)
        assert covariance is not None

        # Corrupt every observation on and after the cutoff; the estimate must
        # be byte-identical, because none of it should have been read.
        for ticker in synthetic_data['tickers']:
            frame = engine.ticker_data[ticker]
            frame.loc[frame.index >= cutoff, 'close'] *= 100.0
        engine._returns_cache.clear()

        assert np.asarray(engine._book_covariance(synthetic_data['tickers'], cutoff)) == pytest.approx(
            np.asarray(covariance)
        )

    def test_trims_a_buy_that_would_breach_the_volatility_target(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=0.05)
        date = pd.Timestamp("2023-12-01")
        ticker = synthetic_data['tickers'][0]
        covariance = engine._book_covariance(synthetic_data['tickers'], date)
        assert covariance is not None

        uncapped = 5000
        capped = engine._apply_portfolio_risk_cap(
            ticker, uncapped, 100.0, {}, date, covariance
        )

        assert 0 <= capped < uncapped
        # And the trimmed size actually satisfies the constraint it was
        # trimmed to, rather than merely being smaller.
        names = [str(c) for c in covariance.columns]
        weights = np.array([
            (capped * 100.0 / engine.portfolio_value) if n == ticker else 0.0
            for n in names
        ])
        from portfolio_agent.src.portfolio import portfolio_volatility
        assert portfolio_volatility(weights, covariance) <= 0.05 + 1e-9

    def test_leaves_a_buy_alone_when_the_book_stays_inside_the_target(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=5.0)
        date = pd.Timestamp("2023-12-01")
        covariance = engine._book_covariance(synthetic_data['tickers'], date)

        assert engine._apply_portfolio_risk_cap(
            synthetic_data['tickers'][0], 500, 100.0, {}, date, covariance
        ) == 500

    def test_refuses_to_add_when_the_book_is_already_over_target(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=0.001)
        date = pd.Timestamp("2023-12-01")
        ticker = synthetic_data['tickers'][0]
        covariance = engine._book_covariance(synthetic_data['tickers'], date)

        existing = {t: 300_000.0 for t in synthetic_data['tickers']}

        assert engine._apply_portfolio_risk_cap(
            ticker, 500, 100.0, existing, date, covariance
        ) == 0

    def test_a_disabled_target_never_trims(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=0.0)
        date = pd.Timestamp("2023-12-01")
        covariance = engine._book_covariance(synthetic_data['tickers'], date)

        assert engine._apply_portfolio_risk_cap(
            synthetic_data['tickers'][0], 9999, 100.0, {}, date, covariance
        ) == 9999

    def test_a_name_with_no_usable_history_is_left_to_the_per_position_limits(
        self, synthetic_data
    ):
        """Inventing a correlation for an unknown name would be worse than
        deferring to the limits that do not need one."""
        engine = self._engine(synthetic_data['tickers'], portfolio_volatility_target=0.05)
        date = pd.Timestamp("2023-12-01")
        covariance = engine._book_covariance(synthetic_data['tickers'], date)

        assert engine._apply_portfolio_risk_cap(
            "NOT_IN_UNIVERSE.NS", 5000, 100.0, {}, date, covariance
        ) == 5000

    def test_too_little_history_yields_no_covariance_rather_than_a_guess(
        self, synthetic_data
    ):
        engine = self._engine(synthetic_data['tickers'])
        # Only a handful of sessions exist before this date.
        assert engine._book_covariance(
            synthetic_data['tickers'], pd.Timestamp("2023-01-10")
        ) is None

    def test_book_risk_statistics_aggregate_the_snapshots(self, synthetic_data):
        engine = self._engine(synthetic_data['tickers'])
        engine.holdings = {t: 100 for t in synthetic_data['tickers']}
        for day, date in enumerate(pd.bdate_range("2023-11-01", periods=3), start=1):
            engine.trading_day_count = day
            engine._record_book_risk(date)

        statistics = engine.book_risk_statistics()

        assert statistics['observations'] == 3
        assert statistics['mean_portfolio_volatility'] > 0
        assert statistics['mean_independent_volatility'] > 0
        assert statistics['mean_positions'] == 3
        # The multiple is the ratio of the two volatilities, whichever way it
        # falls. It exceeds 1 when names are positively correlated — the case
        # that matters, pinned exactly against a known correlation structure in
        # test_portfolio.py — but these three synthetic series are near
        # independent, so asserting > 1 here would be asserting a property of
        # the fixture rather than of the code.
        per_snapshot = [
            row['correlation_risk_multiple'] / (
                row['portfolio_volatility'] / row['independent_volatility']
            )
            for row in engine.book_risk_log
        ]
        assert per_snapshot == pytest.approx([1.0] * 3)

    def test_statistics_are_empty_rather_than_fabricated_without_snapshots(
        self, synthetic_data
    ):
        assert self._engine(synthetic_data['tickers']).book_risk_statistics() == {}


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

    def test_without_a_sector_map_the_cap_is_inactive(self, synthetic_data, tmp_path):
        """Pooling unmapped tickers into one UNKNOWN bucket and capping it
        would turn a 15% *sector* limit into a 15% limit on the whole book,
        leaving 85% of capital idle. A cap that cannot be computed is reported
        as unenforceable, not quietly reinterpreted."""
        tickers = synthetic_data['tickers']

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers,
            max_sector_pct=0.15, sector_map_csv=str(tmp_path / "absent.csv"),
        )
        date = pd.Timestamp("2023-02-01")

        signals = {t: _signal(t, score=90.0 - i, entry_price=100.0) for i, t in enumerate(tickers)}
        engine._create_pending_orders(signals, date)

        assert engine.sector_map == {}
        assert len(engine.pending_orders) == len(tickers)
        queued_value = sum(o['quantity'] * 100.0 for o in engine.pending_orders)
        assert queued_value > 0.15 * engine.portfolio_value

    def test_mapped_and_unmapped_tickers_do_not_share_an_allowance(
        self, synthetic_data, tmp_path
    ):
        """An unmapped holding must not consume a mapped sector's capacity."""
        tickers = synthetic_data['tickers']
        sector_csv = tmp_path / "sectors.csv"
        # Only the first two names are classified.
        sector_csv.write_text(
            "ticker,sector\n" + "".join(f"{t},IT\n" for t in tickers[:2]), encoding="utf-8"
        )

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=tickers,
            max_sector_pct=0.15, sector_map_csv=str(sector_csv),
        )
        date = pd.Timestamp("2023-02-01")

        signals = {t: _signal(t, score=90.0 - i, entry_price=100.0) for i, t in enumerate(tickers)}
        engine._create_pending_orders(signals, date)

        queued = {o['ticker']: o['quantity'] * 100.0 for o in engine.pending_orders}
        it_value = sum(queued.get(t, 0.0) for t in tickers[:2])
        assert it_value <= 0.15 * engine.portfolio_value + 100.0
        # The unmapped name keeps its full 10%-of-portfolio sizing.
        assert queued.get(tickers[2], 0.0) == pytest.approx(100_000.0, rel=0.01)

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


class TestBenchmarkWiring:
    """The crash filter's benchmark has to survive the trip from cache to
    StrategyContext, look-ahead safe."""

    def _engine_with_benchmark(self, monkeypatch, synthetic_data, series):
        real_loader = __import__("portfolio_agent.src.backtest_engine", fromlist=["x"]).load_ticker_data

        def loader(ticker, start_date=None, end_date=None):
            if ticker == "^NSEI":
                return pd.DataFrame({"close": series.values}, index=series.index)
            return real_loader(ticker, start_date=start_date, end_date=end_date)

        monkeypatch.setattr("portfolio_agent.src.backtest_engine.load_ticker_data", loader)
        return BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
            benchmark_symbol="^NSEI",
        )

    def test_loads_the_benchmark_and_truncates_before_the_decision_date(
        self, monkeypatch, synthetic_data
    ):
        dates = pd.bdate_range("2023-01-02", periods=60)
        series = pd.Series(np.linspace(100, 160, 60), index=dates)

        engine = self._engine_with_benchmark(monkeypatch, synthetic_data, series)

        assert engine.benchmark_close is not None
        cutoff = dates[30]
        truncated = engine._benchmark_up_to(cutoff)
        assert truncated is not None
        assert truncated.index.max() < cutoff

    def test_missing_benchmark_degrades_to_none(self, synthetic_data):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
            benchmark_symbol="^NOSUCH",
        )

        assert engine.benchmark_close is None
        assert engine._benchmark_up_to(pd.Timestamp("2023-02-01")) is None

    def test_batch_strategies_score_without_crashing(self, synthetic_data):
        """Regression: _generate_signals reaches the benchmark helper on the
        batch path, which no test exercised."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.cross_sectional import MomentumStrategy

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
            strategy=MomentumStrategy(StrategyConfig(type="momentum", params={"min_universe": 2})),
        )

        signals = engine._generate_signals(pd.Timestamp("2023-03-01"))

        assert isinstance(signals, dict)


class TestKellyUsesNetReturns:
    """Kelly's payoff ratio b is what f* is most sensitive to after p. The
    trade log's `return_pct` is GROSS, so feeding it to Kelly would pair a net
    win/loss classification with a gross payoff ratio and over-bet."""

    def _engine(self, synthetic_data):
        return BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
            use_kelly_sizing=True, kelly_min_trades=10, kelly_shrinkage_strength=0.0,
        )

    @staticmethod
    def _trade(entry_price, quantity, gross_pnl, costs):
        net = gross_pnl - costs
        return {
            'entry_price': entry_price, 'quantity': quantity,
            'gross_pnl': gross_pnl, 'net_pnl': net,
            'return_pct': gross_pnl / (entry_price * quantity) * 100,
            'exit_date': '2023-02-01',
        }

    def test_net_return_pct_deducts_costs_and_taxes(self, synthetic_data):
        engine = self._engine(synthetic_data)
        trade = self._trade(entry_price=100.0, quantity=100, gross_pnl=1000.0, costs=200.0)

        assert trade['return_pct'] == pytest.approx(10.0)          # gross
        assert engine._net_return_pct(trade) == pytest.approx(8.0)  # net

    def test_zero_cost_basis_is_handled(self, synthetic_data):
        engine = self._engine(synthetic_data)

        assert engine._net_return_pct({'entry_price': 0.0, 'quantity': 0, 'net_pnl': 5.0}) == 0.0

    def test_kelly_sizes_smaller_off_net_than_gross_returns(self, synthetic_data):
        """Costs shave the winners more than they help the losers, so the net
        payoff ratio is lower and Kelly must bet less on it."""
        engine = self._engine(synthetic_data)
        engine.risk_params.max_single_position_pct = 0.9
        engine.portfolio_value = 1_000_000.0

        # 12 wins of +10% gross, 8 losses of -5% gross, 2% of cost basis in
        # friction on every trade.
        engine.trade_log = (
            [self._trade(100.0, 100, 1000.0, 200.0) for _ in range(12)]
            + [self._trade(100.0, 100, -500.0, 200.0) for _ in range(8)]
        )

        net_quantity = engine._kelly_quantity(entry_price=100.0)

        # Same history scored off the gross column, as the code used to.
        from portfolio_agent.src.risk import calculate_kelly_quantity, estimate_kelly_inputs
        gross_inputs = estimate_kelly_inputs(
            [
                {"outcome": "WIN" if t["net_pnl"] > 0 else "LOSS", "return_pct": t["return_pct"]}
                for t in engine.trade_log
            ],
            min_trades=10, shrinkage_strength=0.0,
        )
        gross_quantity = calculate_kelly_quantity(
            entry_price=100.0, portfolio_value_inr=engine.portfolio_value,
            max_single_position_pct=0.9,
            win_probability=gross_inputs.win_probability,
            avg_win_pct=gross_inputs.avg_win_pct,
            avg_loss_pct=gross_inputs.avg_loss_pct,
            kelly_fraction=engine.kelly_fraction,
        )

        assert net_quantity < gross_quantity

    def test_open_positions_are_excluded(self, synthetic_data):
        """An open BUY leg carries net_pnl = -costs and no exit; counting it
        would classify every open position as a realized loss."""
        engine = self._engine(synthetic_data)
        engine.trade_log = [
            {'entry_price': 100.0, 'quantity': 10, 'net_pnl': -20.0, 'exit_date': None}
        ] * 50

        assert engine._kelly_quantity(entry_price=100.0) == 0


class TestExitTriggers:
    """A modelled stop assumes a fill is available near it. Two conditions
    invalidate that assumption rather than merely arguing against the position."""

    @staticmethod
    def _engine(synthetic_data, **kwargs):
        return BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-06-30",
            initial_capital=1_000_000.0,
            universe_tickers=synthetic_data['tickers'],
            **kwargs,
        )

    def test_a_lower_circuit_lock_queues_an_immediate_exit(self, synthetic_data):
        ticker = synthetic_data['tickers'][0]
        engine = self._engine(synthetic_data)
        engine.holdings[ticker] = 100

        # Rewrite the last two bars into a 5% lower-circuit close.
        df = engine.ticker_data[ticker]
        lock_date = df.index[-1]
        prev_close = 100.0
        df.loc[df.index[-2], ['open', 'high', 'low', 'close']] = prev_close
        df.loc[lock_date, ['open', 'high', 'low', 'close']] = [98.0, 98.0, 95.0, 95.0]

        engine._create_pending_orders({}, lock_date)

        sells = [o for o in engine.pending_orders if o['action'] == 'SELL']
        assert [o['ticker'] for o in sells] == [ticker]
        assert sells[0]['trigger'] == 'EXIT_TRIGGER'
        assert 'lower circuit' in engine.exit_trigger_log[0]['reason']

    def test_the_lock_exit_can_be_switched_off(self, synthetic_data):
        ticker = synthetic_data['tickers'][0]
        engine = self._engine(synthetic_data, exit_on_lower_circuit_lock=False)
        engine.holdings[ticker] = 100

        df = engine.ticker_data[ticker]
        lock_date = df.index[-1]
        df.loc[df.index[-2], ['open', 'high', 'low', 'close']] = 100.0
        df.loc[lock_date, ['open', 'high', 'low', 'close']] = [98.0, 98.0, 95.0, 95.0]

        engine._create_pending_orders({}, lock_date)

        assert engine.pending_orders == []

    def test_an_ordinary_fall_is_not_an_exit_trigger(self, synthetic_data):
        """Band matching, not a floor: 3.5% is not a statutory limit."""
        ticker = synthetic_data['tickers'][0]
        engine = self._engine(synthetic_data)
        engine.holdings[ticker] = 100

        df = engine.ticker_data[ticker]
        fall_date = df.index[-1]
        df.loc[df.index[-2], ['open', 'high', 'low', 'close']] = 100.0
        df.loc[fall_date, ['open', 'high', 'low', 'close']] = [99.0, 99.0, 96.5, 96.5]

        engine._create_pending_orders({}, fall_date)

        assert engine.pending_orders == []

    def test_drawdown_liquidation_is_opt_in(self, synthetic_data):
        tickers = synthetic_data['tickers']
        date = pd.Timestamp("2023-06-01")

        holding_engine = self._engine(
            synthetic_data, max_portfolio_drawdown_pct=0.15, drawdown_reentry_pct=0.10,
        )
        holding_engine.holdings = {t: 10 for t in tickers}
        holding_engine.portfolio_value = 700_000.0
        holding_engine._create_pending_orders({}, date)

        assert holding_engine.buying_halted is True
        assert holding_engine.pending_orders == []

    def test_drawdown_liquidation_sells_the_whole_book_when_enabled(self, synthetic_data):
        tickers = synthetic_data['tickers']
        engine = self._engine(
            synthetic_data,
            max_portfolio_drawdown_pct=0.15,
            drawdown_reentry_pct=0.10,
            liquidate_on_drawdown_halt=True,
        )
        engine.holdings = {t: 10 for t in tickers}
        engine.portfolio_value = 700_000.0

        engine._create_pending_orders({}, pd.Timestamp("2023-06-01"))

        sells = [o for o in engine.pending_orders if o['action'] == 'SELL']
        assert sorted(o['ticker'] for o in sells) == sorted(tickers)
        assert all(o['trigger'] == 'EXIT_TRIGGER' for o in sells)

    def test_a_forced_exit_does_not_duplicate_a_signal_sell(self, synthetic_data):
        """Both paths want the same position gone; queueing it twice would sell
        a quantity the book does not hold."""
        from portfolio_agent.strategies.types import StrategySignal

        ticker = synthetic_data['tickers'][0]
        engine = self._engine(synthetic_data)
        engine.holdings[ticker] = 100

        df = engine.ticker_data[ticker]
        lock_date = df.index[-1]
        df.loc[df.index[-2], ['open', 'high', 'low', 'close']] = 100.0
        df.loc[lock_date, ['open', 'high', 'low', 'close']] = [98.0, 98.0, 95.0, 95.0]

        signals = {ticker: StrategySignal(
            symbol=ticker, signal="SELL", score=10.0, trigger="Model",
            entry_price=95.0, stop_price=0.0, target_price=0.0,
            reward_risk=0.0, probability_profit=0.0,
        )}
        engine._create_pending_orders(signals, lock_date)

        sells = [o for o in engine.pending_orders if o['action'] == 'SELL']
        assert len(sells) == 1
        assert sells[0]['trigger'] == 'EXIT_TRIGGER'


class TestTriggerUMAThroughTheEngine:
    """End to end: a regime-gated trigger UMA containing a cross-sectional
    member has to survive the engine's full-batch dispatch, and its size
    multiplier has to reach the sized quantity."""

    @staticmethod
    def _uma_yaml(tmp_path, regimes=None):
        import yaml
        spec = {
            "name": "Engine Trigger UMA",
            "method": "trigger",
            "trigger": {
                "mode": "strong_or_consensus",
                "strong_confidence": 0.6,
                "min_net_ev_pct": -100.0,   # the hurdle is exercised elsewhere
            },
            "members": [
                {"type": "momentum",
                 "params": {"name": "mom", "min_universe": 2, "top_percentile": 0.5,
                            "regime_filter": False}},
                {"type": "rule_based", "name": "rules",
                 "config_path": "config/strategies/trend_breakout.yaml"},
            ],
        }
        if regimes is not None:
            spec["regimes"] = regimes
        path = tmp_path / "engine_uma.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(spec, f)
        return str(path)

    def _engine(self, synthetic_data, tmp_path, **kwargs):
        strategy = load_strategy(StrategyConfig(
            type="ensemble", config_path=self._uma_yaml(tmp_path, **kwargs)
        ))
        return BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-12-29",
            initial_capital=1_000_000.0,
            universe_tickers=synthetic_data['tickers'],
            strategy=strategy,
        )

    def test_a_full_batch_uma_runs_to_completion(self, synthetic_data, tmp_path):
        engine = self._engine(synthetic_data, tmp_path)

        assert engine.strategy.requires_full_batch is True

        results = engine.run_backtest()

        assert len(results['daily_equity_curve']) > 0
        assert 'exit_trigger_log' in results

    def test_the_engine_classifies_a_regime_for_the_round(self, synthetic_data, tmp_path):
        """Without this the regime map is inert and every member always speaks."""
        engine = self._engine(synthetic_data, tmp_path)
        engine.benchmark_close = None

        # Nothing to classify from at all -> None, read as "permit all".
        assert engine._classify_regime(None, None) is None

        prices = pd.Series(
            [100.0 * (1.0003 ** i) for i in range(400)],
            index=pd.bdate_range("2021-01-04", periods=400),
        )
        assert engine._classify_regime(prices, None) == "BULL_RISK_ON"

    def test_it_falls_back_to_a_universe_composite_without_an_index(
        self, synthetic_data, tmp_path
    ):
        """Requiring a cached index would leave the regime map inert on every
        installation that never downloaded one — with nothing in the logs to
        say the gating had stopped working."""
        engine = self._engine(synthetic_data, tmp_path)
        index = pd.bdate_range("2021-01-04", periods=400)
        eligible = {
            f"SYN{i}": pd.DataFrame(
                {"close": [100.0 * (1.0004 ** d) for d in range(400)]}, index=index
            )
            for i in range(3)
        }

        assert engine._classify_regime(None, None, eligible) == "BULL_RISK_ON"

    def test_a_composite_too_short_to_judge_returns_none(self, synthetic_data, tmp_path):
        engine = self._engine(synthetic_data, tmp_path)
        eligible = {
            "SYN0": pd.DataFrame(
                {"close": [100.0] * 20}, index=pd.bdate_range("2023-01-02", periods=20)
            )
        }

        assert engine._classify_regime(None, None, eligible) == "UNKNOWN"

    def test_the_trigger_size_multiplier_reaches_the_sized_quantity(self, synthetic_data, tmp_path):
        """The multiplier travels on extra['position_scale'], the same channel
        volatility targeting uses, so sizing picks it up with no extra wiring."""
        from portfolio_agent.strategies.types import StrategySignal

        engine = self._engine(synthetic_data, tmp_path)
        half = StrategySignal(
            symbol="X", signal="BUY", score=80.0, trigger="Trigger:strong_single",
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            reward_risk=2.0, probability_profit=0.6,
            extra={"position_scale": 0.5},
        )

        assert engine._apply_position_scale(100, half) == 50


class TestDrawdownBreakerCooldown:
    """Recovery-only re-arming deadlocks: the breaker halts buying, the open
    positions exit through their stops, and a book that is entirely cash can
    never appreciate back toward the peak it is measured against."""

    @staticmethod
    def _engine(synthetic_data, **kwargs):
        params = dict(
            start_date="2023-01-02",
            end_date="2023-12-29",
            initial_capital=1_000_000.0,
            universe_tickers=synthetic_data['tickers'],
            max_portfolio_drawdown_pct=0.15,
            drawdown_reentry_pct=0.10,
        )
        params.update(kwargs)
        return BacktestEngine(**params)

    def test_a_cash_book_re_arms_on_the_cooldown(self, synthetic_data):
        engine = self._engine(synthetic_data, drawdown_halt_max_days=30)
        engine.portfolio_value = 800_000.0  # 20% below the initial peak

        engine.trading_day_count = 10
        engine._update_circuit_breaker(pd.Timestamp("2023-03-01"))
        assert engine.buying_halted is True

        # Equity frozen: nothing is invested, so it cannot recover on its own.
        engine.trading_day_count = 39
        engine._update_circuit_breaker(pd.Timestamp("2023-04-10"))
        assert engine.buying_halted is True

        engine.trading_day_count = 40
        engine._update_circuit_breaker(pd.Timestamp("2023-04-11"))
        assert engine.buying_halted is False

    def test_the_cooldown_resets_the_peak(self):
        """Without the reset the very next bar trips the breaker again, because
        the old peak is still 20% above current equity."""
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-12-29",
            initial_capital=1_000_000.0, universe_tickers=[],
            max_portfolio_drawdown_pct=0.15, drawdown_reentry_pct=0.10,
            drawdown_halt_max_days=5,
        )
        engine.portfolio_value = 800_000.0
        engine.trading_day_count = 1
        engine._update_circuit_breaker(pd.Timestamp("2023-03-01"))
        engine.trading_day_count = 6
        engine._update_circuit_breaker(pd.Timestamp("2023-03-08"))

        assert engine.equity_peak == pytest.approx(800_000.0)

        engine.trading_day_count = 7
        engine._update_circuit_breaker(pd.Timestamp("2023-03-09"))
        assert engine.buying_halted is False

    def test_the_resume_entry_reports_the_drawdown_that_justified_it(self):
        """Logging after the peak reset would report a meaningless 0%."""
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-12-29",
            initial_capital=1_000_000.0, universe_tickers=[],
            max_portfolio_drawdown_pct=0.15, drawdown_reentry_pct=0.10,
            drawdown_halt_max_days=5,
        )
        engine.portfolio_value = 800_000.0
        engine.trading_day_count = 1
        engine._update_circuit_breaker(pd.Timestamp("2023-03-01"))
        engine.trading_day_count = 6
        engine._update_circuit_breaker(pd.Timestamp("2023-03-08"))

        resume = [e for e in engine.circuit_breaker_log if e['event'] == 'RESUME'][0]
        assert resume['drawdown_pct'] == pytest.approx(20.0)
        assert 'cooldown' in resume['note']

    def test_recovery_still_re_arms_before_the_cooldown(self, synthetic_data):
        engine = self._engine(synthetic_data, drawdown_halt_max_days=60)
        engine.portfolio_value = 800_000.0
        engine.trading_day_count = 1
        engine._update_circuit_breaker(pd.Timestamp("2023-03-01"))

        engine.portfolio_value = 950_000.0  # 5% below peak
        engine.trading_day_count = 5
        engine._update_circuit_breaker(pd.Timestamp("2023-03-06"))

        assert engine.buying_halted is False
        assert engine.equity_peak == pytest.approx(1_000_000.0)

    def test_the_cooldown_can_be_disabled(self, synthetic_data):
        """0 restores the recovery-only behaviour for callers that want it."""
        engine = self._engine(synthetic_data, drawdown_halt_max_days=0)
        engine.portfolio_value = 800_000.0
        engine.trading_day_count = 1
        engine._update_circuit_breaker(pd.Timestamp("2023-03-01"))

        engine.trading_day_count = 5_000
        engine._update_circuit_breaker(pd.Timestamp("2023-12-01"))

        assert engine.buying_halted is True


class TestExitTriggerQueueing:
    def test_a_breaker_tripping_on_the_first_day_still_honours_the_cooldown(self):
        """`or` instead of `is None` reads day zero as 'never tripped', which
        makes the cooldown permanently unsatisfiable."""
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-12-29",
            initial_capital=1_000_000.0, universe_tickers=[],
            max_portfolio_drawdown_pct=0.15, drawdown_reentry_pct=0.10,
            drawdown_halt_max_days=5,
        )
        engine.trading_day_count = 0
        engine.portfolio_value = 800_000.0
        engine._update_circuit_breaker(pd.Timestamp("2023-01-02"))
        assert engine.buying_halted is True
        assert engine.halted_since_day == 0

        engine.trading_day_count = 5
        engine._update_circuit_breaker(pd.Timestamp("2023-01-09"))
        assert engine.buying_halted is False

    def test_an_unfilled_exit_is_not_queued_twice(self, synthetic_data):
        """A sell that could not fill yesterday is still pending; stacking a
        second one sells a quantity the book does not hold."""
        ticker = synthetic_data['tickers'][0]
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-30",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )
        engine.holdings[ticker] = 100

        df = engine.ticker_data[ticker]
        lock_date = df.index[-1]
        df.loc[df.index[-2], ['open', 'high', 'low', 'close']] = 100.0
        df.loc[lock_date, ['open', 'high', 'low', 'close']] = [98.0, 98.0, 95.0, 95.0]

        engine._create_pending_orders({}, lock_date)
        engine._create_pending_orders({}, lock_date)

        assert len([o for o in engine.pending_orders if o['action'] == 'SELL']) == 1


class TestExitLevelsComeFromTheSignal:
    """The engine used to overwrite every strategy's exit plan with a flat
    5%/10% pair, so `min_reward_risk` screened trades on ATR levels the engine
    then ignored — gating on one exit plan and trading another."""

    @staticmethod
    def _engine(synthetic_data):
        return BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-30",
            initial_capital=1_000_000.0, universe_tickers=synthetic_data['tickers'],
        )

    def test_the_signals_distances_are_preserved(self, synthetic_data):
        engine = self._engine(synthetic_data)
        order = {
            'signal_entry_price': 100.0, 'stop_price': 92.0, 'target_price': 116.0,
        }

        stop, target = engine._exit_levels(order, fill_price=200.0)

        # 8% below and 16% above, re-applied to the price actually paid.
        assert stop == pytest.approx(184.0)
        assert target == pytest.approx(232.0)

    def test_a_wider_atr_stop_produces_a_wider_engine_stop(self, synthetic_data):
        """The regression: widening atr_stop_multiplier changed nothing,
        because the engine never read the resulting stop."""
        engine = self._engine(synthetic_data)

        tight = engine._exit_levels(
            {'signal_entry_price': 100.0, 'stop_price': 97.0, 'target_price': 104.0}, 100.0
        )
        wide = engine._exit_levels(
            {'signal_entry_price': 100.0, 'stop_price': 88.0, 'target_price': 116.0}, 100.0
        )

        assert wide[0] < tight[0]
        assert wide[1] > tight[1]

    def test_a_strategy_with_no_exit_plan_gets_the_documented_fallback(self, synthetic_data):
        engine = self._engine(synthetic_data)

        stop, target = engine._exit_levels({}, fill_price=100.0)

        assert stop == pytest.approx(95.0)
        assert target == pytest.approx(110.0)

    @pytest.mark.parametrize("bad", [
        {'signal_entry_price': 100.0, 'stop_price': 105.0, 'target_price': 120.0},
        {'signal_entry_price': 100.0, 'stop_price': 0.0, 'target_price': 120.0},
        {'signal_entry_price': 0.0, 'stop_price': 90.0, 'target_price': 120.0},
    ])
    def test_an_unusable_stop_falls_back_rather_than_inverting(self, synthetic_data, bad):
        """A stop at or above entry would otherwise become a stop *above* the
        fill price, closing the position on any upward move."""
        engine = self._engine(synthetic_data)

        stop, _ = engine._exit_levels(bad, fill_price=100.0)

        assert stop < 100.0

    def test_the_exit_plan_travels_from_the_signal_onto_the_order(self, synthetic_data):
        from portfolio_agent.strategies.types import StrategySignal

        engine = self._engine(synthetic_data)
        ticker = synthetic_data['tickers'][0]
        signals = {ticker: StrategySignal(
            symbol=ticker, signal="BUY", score=90.0, trigger="Momentum",
            entry_price=100.0, stop_price=91.0, target_price=112.0,
            reward_risk=1.33, probability_profit=0.6,
        )}

        engine._create_pending_orders(signals, pd.Timestamp("2023-03-01"))

        buy = [o for o in engine.pending_orders if o['action'] == 'BUY'][0]
        assert buy['stop_price'] == 91.0
        assert buy['target_price'] == 112.0
        assert buy['signal_entry_price'] == 100.0


class TestGapAwareStopFills:
    """Task 4.2: a stop cannot fill at its own price through an overnight gap.

    The simulator assumed every triggered stop filled exactly at the stop
    price. That is only true when the stop is crossed *during* the session. An
    Indian equity that closes at 100 and opens at 90 on an earnings miss or a
    block deal never trades at the 95 stop — the first available price is 90,
    and that is where a stop-market order fills.

    The error is one-directional, so it does not average out: every gap through
    a stop is recorded as a smaller loss than it was. It also feeds back into
    sizing, because Kelly reads its loss magnitude `l` from realized history
    (see src/risk.py::estimate_kelly_inputs) — an understated `l` inflates f*,
    so the book takes larger positions precisely because it has been
    mis-measuring its worst outcomes.
    """

    @staticmethod
    def _engine_holding(monkeypatch, *, open_price, high, low, close=None):
        """An engine holding 100 shares into a day with the given bar."""
        dates = pd.bdate_range("2024-01-01", periods=2)
        frame = pd.DataFrame(
            {
                "open": [100.0, open_price],
                "high": [101.0, high],
                "low": [99.0, low],
                "close": [100.0, close if close is not None else open_price],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        )

        def _load(ticker, start_date=None, end_date=None):
            return frame.copy() if ticker == "GAP.NS" else None

        monkeypatch.setattr("portfolio_agent.src.backtest_engine.load_ticker_data", _load)

        engine = BacktestEngine(
            start_date="2024-01-01", end_date="2024-01-02",
            initial_capital=1_000_000.0, universe_tickers=["GAP.NS"],
        )
        engine.ticker_data = {"GAP.NS": frame}
        engine.holdings = {"GAP.NS": 100}
        # The cost basis lives in open_positions — _get_entry_price_for_tax
        # reads it there, and P&L is zero without it.
        engine.open_positions = {
            "GAP.NS": {
                'entry_price': 100.0,
                'entry_date': dates[0].strftime('%Y-%m-%d'),
                'quantity': 100,
            }
        }
        engine.stop_loss_levels = {"GAP.NS": 95.0}
        engine.take_profit_levels = {}
        return engine, dates[1]

    def test_a_gap_through_the_stop_fills_at_the_open(self, monkeypatch):
        """The acceptance criterion: Close=100, Stop=95, Next_Open=90 -> 90."""
        engine, day = self._engine_holding(
            monkeypatch, open_price=90.0, high=92.0, low=88.0
        )

        trades = engine._check_stop_loss_take_profit(day)

        assert len(trades) == 1
        assert trades[0]['signal_trigger'] == 'STOP_LOSS'
        assert trades[0]['exit_price'] == pytest.approx(90.0)

    def test_an_intraday_cross_still_fills_at_the_stop(self, monkeypatch):
        """The other half. Opening above the stop and only crossing it later in
        the session is the case the original logic was right about, and it must
        keep filling at 95 rather than being penalized to the open."""
        engine, day = self._engine_holding(
            monkeypatch, open_price=99.0, high=99.5, low=93.0, close=94.0
        )

        trades = engine._check_stop_loss_take_profit(day)

        assert len(trades) == 1
        assert trades[0]['exit_price'] == pytest.approx(95.0)

    def test_the_slippage_is_recorded_against_the_stop(self, monkeypatch):
        """The gap has to be visible as gap, not folded silently into P&L —
        otherwise nothing downstream can tell a gapped exit from a clean one.
        """
        engine, day = self._engine_holding(
            monkeypatch, open_price=90.0, high=92.0, low=88.0
        )

        trades = engine._check_stop_loss_take_profit(day)

        assert trades[0]['gap_fill_delta_pct'] == pytest.approx(
            (90.0 - 95.0) / 95.0 * 100.0
        )

    def test_a_clean_stop_records_no_gap_slippage(self, monkeypatch):
        engine, day = self._engine_holding(
            monkeypatch, open_price=99.0, high=99.5, low=93.0, close=94.0
        )

        trades = engine._check_stop_loss_take_profit(day)

        assert trades[0]['gap_fill_delta_pct'] == pytest.approx(0.0)

    def test_the_loss_is_larger_than_the_stop_implied(self, monkeypatch):
        """What the defect actually cost: the realized loss must exceed the
        5% the stop was set at, because the fill was 10% down."""
        engine, day = self._engine_holding(
            monkeypatch, open_price=90.0, high=92.0, low=88.0
        )

        trades = engine._check_stop_loss_take_profit(day)

        assert trades[0]['return_pct'] < -9.0

    def test_a_gap_above_the_target_fills_at_the_open(self, monkeypatch):
        """The favourable side, which is symmetric with the adverse one.

        A take-profit is a resting limit sell, and a limit fills at the limit
        *or better* — so a gap up through the target fills at the open, above
        it. Booking the target would understate the exit for the same reason
        booking the stop overstates it: both assume a price that was never
        available. Because the fill now depends on where the session opened,
        the exit must be evaluated on the *first* bar that reaches the target;
        a test that skips ahead in a rising market is asking a different
        question (see test_report_data_integrity.py::TestTradeLogAccounting).
        """
        engine, day = self._engine_holding(
            monkeypatch, open_price=112.0, high=115.0, low=111.0
        )
        engine.stop_loss_levels = {}
        engine.take_profit_levels = {"GAP.NS": 110.0}

        trades = engine._check_stop_loss_take_profit(day)

        assert len(trades) == 1
        assert trades[0]['signal_trigger'] == 'TAKE_PROFIT'
        assert trades[0]['exit_price'] == pytest.approx(112.0)
        assert trades[0]['gap_fill_delta_pct'] == pytest.approx(
            (112.0 - 110.0) / 110.0 * 100.0
        )

    def test_a_session_touching_both_levels_is_charged_the_stop(self, monkeypatch):
        """Order of evaluation is a modelling choice, so it is pinned.

        A bar whose range spans both the stop and the target gives no
        intraday sequence to read, and assuming the favourable one is how a
        backtest quietly manufactures returns. The stop is checked first.
        """
        engine, day = self._engine_holding(
            monkeypatch, open_price=99.0, high=115.0, low=93.0, close=112.0
        )
        engine.take_profit_levels = {"GAP.NS": 110.0}

        trades = engine._check_stop_loss_take_profit(day)

        assert trades[0]['signal_trigger'] == 'STOP_LOSS'
        assert trades[0]['exit_price'] == pytest.approx(95.0)

    def test_an_intraday_target_cross_fills_at_the_target(self, monkeypatch):
        engine, day = self._engine_holding(
            monkeypatch, open_price=105.0, high=112.0, low=104.0, close=111.0
        )
        engine.stop_loss_levels = {}
        engine.take_profit_levels = {"GAP.NS": 110.0}

        trades = engine._check_stop_loss_take_profit(day)

        assert trades[0]['exit_price'] == pytest.approx(110.0)
