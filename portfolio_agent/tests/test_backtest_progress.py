"""Tests for backtest progress reporting (src/backtest_engine.py).

The defect these pin down: a backtest printed "Running backtest..." and then
nothing at all — for the several minutes it takes to read a few thousand
tickers off disk and the far longer replay that follows. The engine *did*
report progress, via `logger.info`, but the CLI never configures logging, so
those records went to a handler-less root logger and were discarded. Progress
now goes to a tqdm bar, which does not depend on logging configuration.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.backtest_engine import BacktestEngine


@pytest.fixture
def synthetic_data(monkeypatch):
    """Two tickers with enough history for the rule-based feature set."""
    np.random.seed(11)
    dates = pd.bdate_range(start=pd.Timestamp("2023-01-02"), periods=260)

    data_dict = {}
    for i, ticker in enumerate(["PROG1.NS", "PROG2.NS"]):
        base = 100 + i * 40
        close = base + np.arange(len(dates)) * 0.5 + np.random.uniform(-1, 3, len(dates))
        data_dict[ticker] = pd.DataFrame({
            'open': close - 0.5, 'high': close + 2.0, 'low': close - 2.0,
            'close': close, 'volume': np.random.randint(9e5, 1.1e6, len(dates)),
        }, index=dates)

    def mock_load_ticker_data(ticker, start_date=None, end_date=None):
        return data_dict[ticker].copy() if ticker in data_dict else None

    monkeypatch.setattr("portfolio_agent.src.backtest_engine.load_ticker_data", mock_load_ticker_data)
    return list(data_dict)


def _engine(tickers, **kwargs):
    return BacktestEngine(
        start_date="2023-01-02",
        end_date="2023-06-30",
        initial_capital=1_000_000.0,
        universe_tickers=tickers,
        **kwargs,
    )


class TestProgressBars:
    def test_bars_are_drawn_for_both_slow_phases(self, synthetic_data, monkeypatch):
        """Loading the universe and replaying the days are the two phases that
        take real time, and each needs its own bar — a single bar that only
        appears after loading finishes leaves the longest silence uncovered."""
        created = []

        import portfolio_agent.src.backtest_engine as engine_module
        real_tqdm = engine_module.tqdm

        def spy_tqdm(iterable, **kwargs):
            created.append(kwargs.get("desc"))
            return real_tqdm(iterable, **kwargs)

        monkeypatch.setattr(engine_module, "tqdm", spy_tqdm)

        _engine(synthetic_data, show_progress=True).run_backtest()

        assert "Loading ticker data" in created
        assert "Backtesting" in created

    def test_progress_is_off_by_default_for_library_callers(self, synthetic_data, capsys):
        engine = _engine(synthetic_data)

        assert engine.show_progress is False
        assert capsys.readouterr().out == ""

    def test_disabled_bars_still_iterate_everything(self, synthetic_data):
        """The bar wraps the loop it reports on, so disabling it must not
        change how many days get replayed."""
        quiet = _engine(synthetic_data, show_progress=False)
        loud = _engine(synthetic_data, show_progress=True)

        quiet_result = quiet.run_backtest()
        loud_result = loud.run_backtest()

        assert len(quiet_result['daily_equity_curve']) == len(loud_result['daily_equity_curve'])
        assert len(quiet_result['daily_equity_curve']) == len(quiet.master_date_index)

    def test_run_completes_and_reports_every_trading_day(self, synthetic_data):
        engine = _engine(synthetic_data, show_progress=True)

        result = engine.run_backtest()

        assert engine.trading_day_count == len(engine.master_date_index)
        assert len(result['daily_equity_curve']) == engine.trading_day_count

    def test_bar_is_closed_even_when_the_run_raises(self, synthetic_data, monkeypatch):
        """A bar left open corrupts every line printed after it — including the
        traceback the user needs to read."""
        engine = _engine(synthetic_data, show_progress=True)

        def boom(current_date):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(engine, "_execute_pending_orders", boom)

        closed = []
        import portfolio_agent.src.backtest_engine as engine_module
        real_tqdm = engine_module.tqdm

        def tracking_tqdm(iterable, **kwargs):
            bar = real_tqdm(iterable, **kwargs)
            original_close = bar.close
            bar.close = lambda: (closed.append(kwargs.get("desc")), original_close())[1]
            return bar

        monkeypatch.setattr(engine_module, "tqdm", tracking_tqdm)

        with pytest.raises(RuntimeError, match="simulated failure"):
            engine.run_backtest()

        assert "Backtesting" in closed, "the day bar must be closed on failure"


class TestBacktesterAgentDefaults:
    def test_agent_shows_progress_by_default(self):
        """The agent is the interactive entry point; the engine is not."""
        from portfolio_agent.agents.backtester import BacktesterAgent
        from portfolio_agent.config.schema import AppConfig

        agent = BacktesterAgent(config=AppConfig())

        assert agent.show_progress is True
