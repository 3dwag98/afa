"""Parallel execution must not change results, only wall-clock time.

The backtest engine can score tickers across a CPU process pool. That is a
performance choice, so a `--parallel` run has to produce byte-identical
trades, equity curve, activity log and Excel report to the serial run it
replaces. These tests pin that down, because the failure mode is silent: the
report still generates, it just contains different numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import BacktestEngine
from src.backtest_reporting import export_backtest_excel
from src.risk_analytics import RiskAnalyzer


@pytest.fixture
def market_data(monkeypatch):
    """Eight synthetic tickers with enough history for the rule-based strategy."""
    np.random.seed(11)
    dates = pd.bdate_range(start="2023-01-02", periods=260)
    tickers = [f"PAR{i}.NS" for i in range(8)]

    data = {}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 25
        trend = np.linspace(0, 30 + i * 5, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        close = base + trend + noise
        data[ticker] = pd.DataFrame(
            {
                'open': close - np.random.uniform(0, 1.5, len(dates)),
                'high': close + np.random.uniform(0.5, 4, len(dates)),
                'low': close - np.random.uniform(0.5, 4, len(dates)),
                'close': close,
                'volume': np.random.randint(500_000, 5_000_000, len(dates)).astype(float),
            },
            index=dates,
        )

    def fake_load(ticker, start_date=None, end_date=None):
        if ticker not in data:
            return None
        df = data[ticker].copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df

    monkeypatch.setattr("src.backtest_engine.load_ticker_data", fake_load)
    return {'tickers': tickers, 'dates': dates}


def _run(tickers, parallel: bool, capital: float = 400_000.0):
    engine = BacktestEngine(
        start_date="2023-01-02",
        end_date="2023-09-29",
        initial_capital=capital,
        universe_tickers=tickers,
        parallel=parallel,
        max_workers=2 if parallel else None,
        mc_simulations=200,  # keep the suite fast; determinism is unaffected
    )
    engine.run_backtest()
    return engine


class TestParallelMatchesSerial:
    """Serial and parallel runs must agree on every reported number."""

    def test_equity_curve_identical(self, market_data):
        serial = _run(market_data['tickers'], parallel=False)
        parallel = _run(market_data['tickers'], parallel=True)

        pd.testing.assert_series_equal(
            serial.daily_equity_curve, parallel.daily_equity_curve
        )

    def test_trade_log_identical(self, market_data):
        serial = _run(market_data['tickers'], parallel=False)
        parallel = _run(market_data['tickers'], parallel=True)

        assert serial.trade_log == parallel.trade_log

    def test_daily_activity_log_identical(self, market_data):
        serial = _run(market_data['tickers'], parallel=False)
        parallel = _run(market_data['tickers'], parallel=True)

        assert serial.daily_activity_log == parallel.daily_activity_log

    def test_exported_report_data_identical(self, market_data, tmp_path):
        """The numbers that reach Excel match, sheet for sheet."""
        serial = _run(market_data['tickers'], parallel=False)
        parallel = _run(market_data['tickers'], parallel=True)

        paths = {}
        for name, engine in (('serial', serial), ('parallel', parallel)):
            analytics = RiskAnalyzer(
                daily_equity_curve=engine.daily_equity_curve,
                trade_log=engine.trade_log,
                risk_free_rate=0.065,
            ).generate_analytics_report()

            path = tmp_path / f"{name}.xlsx"
            export_backtest_excel(
                analytics={
                    'cagr': analytics['cagr_pct'],
                    'sharpe': analytics['sharpe_ratio'],
                    'sortino': analytics['sortino_ratio'],
                    'max_drawdown': analytics['max_drawdown_pct'],
                    'profit_factor': analytics['profit_factor'],
                    'probability_of_ruin': analytics['mc_probability_of_ruin_pct'],
                    'total_return': analytics['total_return_pct'],
                    'volatility': analytics['annualized_volatility_pct'],
                    'win_rate': analytics['win_rate_pct'],
                    'total_trades': analytics['total_trades'],
                    'final_portfolio_value': analytics['final_capital'],
                    'initial_capital': analytics['initial_capital'],
                },
                equity_curve=engine.daily_equity_curve,
                trade_log=engine.trade_log,
                brain_evolution=engine.brain_evolution,
                daily_activity_log=engine.daily_activity_log,
                filepath=str(path),
            )
            paths[name] = path

        for sheet in ('Executive_Summary', 'Equity_Curve', 'Trade_Log',
                      'Daily_Trade_Log', 'Monthly_Heatmap', 'Brain_Evolution',
                      'Monte_Carlo_Simulations'):
            serial_df = pd.read_excel(paths['serial'], sheet_name=sheet)
            parallel_df = pd.read_excel(paths['parallel'], sheet_name=sheet)
            pd.testing.assert_frame_equal(
                serial_df, parallel_df, check_dtype=False,
                obj=f"{sheet} differs between serial and parallel runs",
            )

    def test_parallel_run_is_repeatable(self, market_data):
        """Two parallel runs of the same inputs agree with each other."""
        first = _run(market_data['tickers'], parallel=True)
        second = _run(market_data['tickers'], parallel=True)

        pd.testing.assert_series_equal(
            first.daily_equity_curve, second.daily_equity_curve
        )
        assert first.trade_log == second.trade_log


class TestParallelRobustness:
    """The pool must be an optimization, never a new failure mode."""

    def test_worker_pool_is_reused_across_days(self, market_data):
        """One pool per run, not one per trading day."""
        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-04-28",
            initial_capital=400_000.0,
            universe_tickers=market_data['tickers'],
            parallel=True,
            max_workers=2,
            mc_simulations=200,
        )

        created = []
        original = BacktestEngine._get_scoring_executor

        def counting_executor(self):
            before = self._scoring_executor
            executor = original(self)
            if before is None:
                created.append(executor)
            return executor

        BacktestEngine._get_scoring_executor = counting_executor
        try:
            engine.run_backtest()
        finally:
            BacktestEngine._get_scoring_executor = original

        assert len(created) == 1, f"expected one worker pool, created {len(created)}"
        assert engine._scoring_executor is None, "pool should be shut down after the run"

    def test_failing_worker_skips_only_that_ticker(self, market_data, monkeypatch):
        """One worker exception drops one ticker, it does not abort the run."""
        from concurrent.futures import Future

        from portfolio_agent.strategies.types import StrategySignal

        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=400_000.0, universe_tickers=market_data['tickers'],
            parallel=True, max_workers=2,
        )

        def make_future(ticker):
            future = Future()
            if ticker == "PAR3.NS":
                future.set_exception(RuntimeError("simulated worker failure"))
            else:
                future.set_result(StrategySignal(
                    symbol=ticker, signal="HOLD", score=1.0, trigger="Trend",
                    entry_price=100.0, stop_price=95.0, target_price=110.0,
                    reward_risk=2.0, probability_profit=0.5,
                ))
            return future

        class StubExecutor:
            def submit(self, _fn, ticker, *args, **kwargs):
                return make_future(ticker)

        monkeypatch.setattr(engine, "_get_scoring_executor", lambda: StubExecutor())

        eligible = {t: pd.DataFrame() for t in market_data['tickers']}
        signals = engine._score_tickers_parallel(eligible, weights={})

        assert "PAR3.NS" not in signals
        assert set(signals) == set(market_data['tickers']) - {"PAR3.NS"}

    def test_unstartable_pool_falls_back_to_serial(self, market_data, monkeypatch):
        """If workers cannot start at all, the run continues on one core."""
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-03-31",
            initial_capital=400_000.0, universe_tickers=market_data['tickers'],
            parallel=True, max_workers=2, mc_simulations=100,
        )

        def boom():
            raise OSError("cannot fork")

        monkeypatch.setattr(engine, "_get_scoring_executor", boom)

        signals = engine._generate_signals(engine.master_date_index[-1])

        assert engine.parallel is False, "should stop retrying a pool that cannot start"
        assert isinstance(signals, dict) and signals, "serial fallback should still score"

    def test_orders_are_queued_by_conviction(self, market_data):
        """BUY orders are queued highest-score-first, deterministically."""
        from portfolio_agent.strategies.types import StrategySignal

        engine = BacktestEngine(
            start_date="2023-01-02",
            end_date="2023-03-31",
            initial_capital=400_000.0,
            universe_tickers=market_data['tickers'],
        )

        def signal(symbol, score):
            return StrategySignal(
                symbol=symbol, signal="BUY", score=score, trigger="Trend",
                entry_price=100.0, stop_price=95.0, target_price=110.0,
                reward_risk=2.0, probability_profit=0.6,
            )

        # Deliberately inserted in a different order from their scores.
        signals = {
            "PAR1.NS": signal("PAR1.NS", 55.0),
            "PAR2.NS": signal("PAR2.NS", 90.0),
            "PAR0.NS": signal("PAR0.NS", 72.0),
        }
        engine._create_pending_orders(signals, engine.master_date_index[0])

        assert [o['ticker'] for o in engine.pending_orders] == [
            "PAR2.NS", "PAR0.NS", "PAR1.NS"
        ]
