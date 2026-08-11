"""The numbers in the generated reports must be the real numbers.

Each test here corresponds to a specific way the exported backtest report used
to disagree with what actually happened in the simulation: P&L measured from
the wrong price, holding periods reported as zero, an equity curve that
ignored the day's fills, analytics computed from a P&L key the engine never
writes, and percentage cells scaled by the wrong power of ten.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import BacktestEngine
from src.backtest_reporting import (
    SUMMARY_METRICS,
    export_backtest_excel,
    _create_monthly_heatmap_df,
    _prepare_equity_curve_df,
)
from src.risk_analytics import RiskAnalyzer


@pytest.fixture
def rising_market(monkeypatch):
    """One ticker that rallies hard enough to hit a take-profit."""
    dates = pd.bdate_range(start="2023-01-02", periods=120)
    close = np.linspace(100.0, 200.0, len(dates))

    df = pd.DataFrame(
        {
            'open': close,
            'high': close * 1.02,
            'low': close * 0.99,
            'close': close,
            'volume': np.full(len(dates), 5_000_000.0),
        },
        index=dates,
    )

    monkeypatch.setattr(
        "src.backtest_engine.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: df.copy() if ticker == "UP.NS" else None,
    )
    return {'dates': dates, 'df': df}


# The first bar of `rising_market` on which the take-profit at 110 is crossed
# *within the session* — it opens at 108.40 and trades up through 110. Later
# bars open well above the target, which is a gap-through and fills at the open
# rather than at the level (see TestGapFills), so a test that wants to assert a
# clean fill at 110 has to exit on a bar where 110 was reachable on the tape.
TARGET_CROSSED_INTRADAY = 10


class TestTradeLogAccounting:
    """Trade records must describe the trade that actually happened."""

    def test_exit_pnl_is_measured_from_the_entry_price(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        # Enter at 100 on day 5, then let the engine's take-profit close it.
        entry_date = engine.master_date_index[5]
        engine.cash -= 100 * 100.0
        engine.holdings["UP.NS"] = 100
        engine._open_position("UP.NS", 100, 100.0, entry_date)
        engine.stop_loss_levels["UP.NS"] = 95.0
        engine.take_profit_levels["UP.NS"] = 110.0

        exit_date = engine.master_date_index[TARGET_CROSSED_INTRADAY]
        trades = engine._check_stop_loss_take_profit(exit_date)

        assert len(trades) == 1
        trade = trades[0]
        assert trade['entry_price'] == pytest.approx(100.0)
        assert trade['exit_price'] == pytest.approx(110.0)
        # 100 shares * ₹10 of gain — not a one-day move measured off yesterday's close.
        assert trade['gross_pnl'] == pytest.approx(1000.0)

    def test_exit_reports_real_holding_period(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        entry_date = engine.master_date_index[5]
        engine.holdings["UP.NS"] = 100
        engine._open_position("UP.NS", 100, 100.0, entry_date)
        engine.take_profit_levels["UP.NS"] = 110.0

        exit_date = engine.master_date_index[TARGET_CROSSED_INTRADAY]
        trade = engine._check_stop_loss_take_profit(exit_date)[0]

        expected_days = (exit_date - entry_date).days
        assert expected_days > 0
        assert trade['holding_days'] == expected_days

    def test_exits_book_costs_and_taxes(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        engine.holdings["UP.NS"] = 100
        engine._open_position("UP.NS", 100, 100.0, engine.master_date_index[5])
        engine.take_profit_levels["UP.NS"] = 110.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[TARGET_CROSSED_INTRADAY])[0]

        assert trade['transaction_costs'] > 0, "sell leg pays brokerage/STT/GST"
        assert trade['taxes'] > 0, "a profitable short-term exit owes STCG"
        assert trade['net_pnl'] == pytest.approx(
            trade['gross_pnl'] - trade['transaction_costs'] - trade['taxes']
        )

    def test_cash_matches_the_booked_net_proceeds(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        engine.holdings["UP.NS"] = 100
        engine._open_position("UP.NS", 100, 100.0, engine.master_date_index[5])
        engine.take_profit_levels["UP.NS"] = 110.0
        cash_before = engine.cash

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[TARGET_CROSSED_INTRADAY])[0]

        expected = cash_before + 100 * 110.0 - trade['transaction_costs'] - trade['taxes']
        assert engine.cash == pytest.approx(expected)

    def test_closing_a_position_clears_its_cost_basis(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        engine.holdings["UP.NS"] = 100
        engine._open_position("UP.NS", 100, 100.0, engine.master_date_index[5])
        engine.take_profit_levels["UP.NS"] = 110.0

        engine._check_stop_loss_take_profit(engine.master_date_index[TARGET_CROSSED_INTRADAY])

        assert "UP.NS" not in engine.open_positions
        assert "UP.NS" not in engine.holdings

    def test_adding_to_a_position_averages_the_cost_basis(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        first = engine.master_date_index[5]
        engine._open_position("UP.NS", 100, 100.0, first)
        engine._open_position("UP.NS", 100, 120.0, engine.master_date_index[20])

        assert engine._get_entry_price_for_tax("UP.NS") == pytest.approx(110.0)
        # Holding period runs from the first entry, which is what decides STCG/LTCG.
        assert engine._get_entry_date_for_ticker("UP.NS") == first.strftime('%Y-%m-%d')


class TestOrderBook:
    """Queued orders must be resolved, never silently stranded."""

    def _engine(self):
        return BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )

    def test_order_dated_on_a_market_holiday_still_fills(self, rising_market):
        """Orders are scheduled for the next weekday, which is sometimes closed.

        Such an order used to carry a date that could never come round again,
        so it sat in the book forever and the signal was silently lost.
        """
        engine = self._engine()
        session = engine.master_date_index[10]
        holiday = session - pd.Timedelta(days=1)
        assert holiday not in engine.master_date_index, "fixture must place a closed day here"

        engine.pending_orders.append({
            'ticker': "UP.NS", 'action': 'BUY', 'quantity': 10,
            'execution_date': holiday, 'trigger': 'SIGNAL',
        })

        executed = engine._execute_pending_orders(session)

        assert len(executed) == 1
        assert executed[0]['ticker'] == "UP.NS"
        assert engine.pending_orders == []

    def test_unaffordable_order_leaves_the_book(self, rising_market):
        """An order that cannot be funded is dropped, not retried forever."""
        engine = self._engine()
        engine.cash = 1.0
        session = engine.master_date_index[10]

        engine.pending_orders.append({
            'ticker': "UP.NS", 'action': 'BUY', 'quantity': 10_000,
            'execution_date': session, 'trigger': 'SIGNAL',
        })

        executed = engine._execute_pending_orders(session)

        assert executed == []
        assert engine.pending_orders == []

    def test_order_book_does_not_grow_across_a_run(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=50_000.0, universe_tickers=["UP.NS"],
        )
        engine.run_backtest()

        # Only orders queued on the final day can still be outstanding.
        assert len(engine.pending_orders) <= 1


class TestEquityCurve:
    """The equity point for day T must include day T's trading."""

    def test_equity_reflects_same_day_fills(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        results = engine.run_backtest()
        curve = results['daily_equity_curve']

        assert len(curve) == len(engine.master_date_index)
        final = engine.cash + sum(
            qty * engine._get_price_at_date(t, engine.master_date_index[-1], 'close')
            for t, qty in engine.holdings.items()
        )
        assert curve.iloc[-1] == pytest.approx(final)

    def test_every_day_logs_exactly_one_mark_to_market(self, rising_market):
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        results = engine.run_backtest()

        marks = [r for r in results['daily_activity_log'] if r['action'] == 'MARK_TO_MARKET']
        assert len(marks) == len(engine.master_date_index)
        assert len({m['date'] for m in marks}) == len(marks)

    def test_mark_to_market_cash_matches_the_ledger(self, rising_market):
        """The EOD row's cash must be the cash after that day's trades."""
        engine = BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["UP.NS"],
        )
        results = engine.run_backtest()

        last_mark = [r for r in results['daily_activity_log']
                     if r['action'] == 'MARK_TO_MARKET'][-1]
        assert last_mark['cash_balance'] == pytest.approx(round(engine.cash, 2))


class TestAnalyticsReadRealPnl:
    """RiskAnalyzer must read the P&L key BacktestEngine actually writes."""

    def _engine_style_log(self):
        return [
            # Open BUY leg: no exit, negative net_pnl (its transaction cost).
            {'ticker': 'A.NS', 'entry_date': '2023-01-05', 'entry_price': 100.0,
             'exit_date': None, 'exit_price': None, 'quantity': 10,
             'gross_pnl': 0.0, 'net_pnl': -20.0, 'return_pct': 0.0},
            {'ticker': 'B.NS', 'entry_date': '2023-01-05', 'entry_price': 100.0,
             'exit_date': '2023-02-05', 'exit_price': 120.0, 'quantity': 10,
             'gross_pnl': 200.0, 'net_pnl': 150.0, 'return_pct': 20.0},
            {'ticker': 'C.NS', 'entry_date': '2023-01-05', 'entry_price': 100.0,
             'exit_date': '2023-02-10', 'exit_price': 90.0, 'quantity': 10,
             'gross_pnl': -100.0, 'net_pnl': -120.0, 'return_pct': -10.0},
            {'ticker': 'D.NS', 'entry_date': '2023-02-01', 'entry_price': 50.0,
             'exit_date': '2023-03-01', 'exit_price': 60.0, 'quantity': 20,
             'gross_pnl': 200.0, 'net_pnl': 180.0, 'return_pct': 20.0},
        ]

    def _curve(self):
        dates = pd.bdate_range("2023-01-02", periods=60)
        return pd.Series(np.linspace(100_000, 110_000, len(dates)), index=dates)

    def test_win_rate_uses_net_pnl(self):
        analyzer = RiskAnalyzer(self._curve(), self._engine_style_log())
        # 2 wins out of 3 closed trades; the open BUY leg is not a trade.
        assert analyzer.calculate_win_rate() == pytest.approx(2 / 3)

    def test_profit_factor_uses_net_pnl(self):
        analyzer = RiskAnalyzer(self._curve(), self._engine_style_log())
        assert analyzer.calculate_profit_factor() == pytest.approx(330.0 / 120.0)

    def test_open_positions_are_not_counted_as_trades(self):
        report = RiskAnalyzer(self._curve(), self._engine_style_log()).generate_analytics_report()
        assert report['total_trades'] == 3
        assert report['total_trade_records'] == 4

    def test_monte_carlo_sees_real_dispersion(self):
        """With real P&L the ruin simulation produces a spread, not a point."""
        report = RiskAnalyzer(self._curve(), self._engine_style_log()).generate_analytics_report()
        assert report['mc_simulations_run'] > 0
        assert report['mc_percentile_95'] > report['mc_percentile_5']

    def test_plain_pnl_logs_still_work(self):
        """Simpler trade logs keyed on 'pnl' keep working."""
        analyzer = RiskAnalyzer(self._curve(), [{'pnl': 100}, {'pnl': -50}, {'pnl': 25}])
        assert analyzer.calculate_win_rate() == pytest.approx(2 / 3)

    def test_monte_carlo_block_is_reproducible(self):
        """Re-running the same backtest must not move the risk-of-ruin cells.

        The bootstrap used to run unseeded, so Probability of Ruin and all
        three terminal-wealth percentiles changed on every export of the very
        same trade log.
        """
        log, curve = self._engine_style_log(), self._curve()

        first = RiskAnalyzer(curve, log).generate_analytics_report()
        second = RiskAnalyzer(curve, log).generate_analytics_report()

        for key in ('mc_probability_of_ruin', 'mc_percentile_5',
                    'mc_median_terminal_wealth', 'mc_percentile_95'):
            assert first[key] == pytest.approx(second[key]), key

    def test_monte_carlo_does_not_disturb_global_rng(self):
        """Drawing from a local generator keeps NumPy's global stream intact."""
        np.random.seed(1234)
        expected = np.random.rand()

        np.random.seed(1234)
        RiskAnalyzer(self._curve(), self._engine_style_log()).generate_analytics_report()
        assert np.random.rand() == pytest.approx(expected)


class TestExcelScaling:
    """Percentage cells must be written at the right magnitude."""

    def _write(self, tmp_path, analytics, equity_curve, trade_log):
        path = tmp_path / "report.xlsx"
        export_backtest_excel(
            analytics=analytics, equity_curve=equity_curve, trade_log=trade_log,
            brain_evolution=[{'trading_day': 1, 'weights': {'Trend': 25.0, 'Breakout': 25.0,
                                                            'Volume': 20.0, 'MC_Prob': 30.0}}],
            daily_activity_log=[], filepath=str(path),
        )
        return path

    def _analytics(self, **overrides):
        base = {
            'cagr': 12.5, 'sharpe': 1.45, 'sortino': 2.1, 'max_drawdown': -15.3,
            'profit_factor': 1.8, 'probability_of_ruin': 3.2, 'total_return': 40.0,
            'volatility': 18.0, 'win_rate': 0.5, 'total_trades': 12,
            'final_portfolio_value': 1_400_000.0, 'initial_capital': 1_000_000.0,
            'monte_carlo_results': {'percentile_5': 900_000.0, 'percentile_50': 1_200_000.0,
                                    'percentile_95': 1_600_000.0},
        }
        base.update(overrides)
        return base

    def _curve(self):
        dates = pd.bdate_range("2022-01-03", periods=400)
        return pd.Series(np.linspace(1_000_000, 1_400_000, len(dates)), index=dates)

    def test_sub_one_percent_values_are_not_inflated(self, tmp_path):
        """A 0.5% win rate must not be written as 50%."""
        path = self._write(tmp_path, self._analytics(), self._curve(), [])
        summary = pd.read_excel(path, sheet_name='Executive_Summary')
        row = summary[summary.iloc[:, 0] == 'Win Rate (%)']

        assert float(row.iloc[0, 1]) == pytest.approx(0.005)

    def test_ratios_are_written_as_plain_numbers(self, tmp_path):
        """Sharpe is dimensionless; it used to carry a percent format."""
        path = self._write(tmp_path, self._analytics(), self._curve(), [])
        summary = pd.read_excel(path, sheet_name='Executive_Summary')

        sharpe = summary[summary.iloc[:, 0] == 'Sharpe Ratio'].iloc[0, 1]
        assert float(sharpe) == pytest.approx(1.45)

    def test_percent_metrics_round_trip(self, tmp_path):
        path = self._write(tmp_path, self._analytics(), self._curve(), [])
        summary = pd.read_excel(path, sheet_name='Executive_Summary')
        values = dict(zip(summary.iloc[:, 0], summary.iloc[:, 1]))

        assert float(values['CAGR (%)']) == pytest.approx(0.125)
        assert float(values['Max Drawdown (%)']) == pytest.approx(-0.153)
        assert float(values['Total Return (%)']) == pytest.approx(0.40)

    def test_currency_metrics_are_not_divided(self, tmp_path):
        path = self._write(tmp_path, self._analytics(), self._curve(), [])
        summary = pd.read_excel(path, sheet_name='Executive_Summary')
        values = dict(zip(summary.iloc[:, 0], summary.iloc[:, 1]))

        assert float(values['Initial Capital (₹)']) == pytest.approx(1_000_000.0)
        assert float(values['Total Trades']) == 12

    def test_small_trade_returns_are_not_inflated(self, tmp_path):
        trade_log = [{
            'trade_id': 'T000001', 'ticker': 'A.NS', 'entry_date': '2022-01-03',
            'entry_price': 100.0, 'exit_date': '2022-01-10', 'exit_price': 100.5,
            'quantity': 10, 'side': 'LONG', 'signal_trigger': 'Trend',
            'gross_pnl': 5.0, 'transaction_costs': 1.0, 'taxes': 0.0,
            'net_pnl': 4.0, 'return_pct': 0.5, 'holding_days': 7,
            'exit_reason': 'target',
        }]
        path = self._write(tmp_path, self._analytics(), self._curve(), trade_log)
        trades = pd.read_excel(path, sheet_name='Trade_Log')

        # 0.5% stays 0.5%, i.e. 0.005 as an Excel fraction.
        assert float(trades['return_pct'].iloc[0]) == pytest.approx(0.005)

    def test_every_summary_metric_declares_a_unit(self):
        for label, key, unit in SUMMARY_METRICS:
            assert unit in ('percent', 'ratio', 'currency', 'count'), label


class TestTradeLogSheetColumns:
    """Values must land in the column their header names."""

    def _report(self, tmp_path, trade_log, daily_log=()):
        dates = pd.bdate_range("2022-01-03", periods=60)
        curve = pd.Series(np.linspace(1_000_000, 1_100_000, len(dates)), index=dates)
        path = tmp_path / "cols.xlsx"
        export_backtest_excel(
            analytics={'cagr': 1.0, 'sharpe': 0.5, 'sortino': 0.5, 'max_drawdown': -1.0,
                       'profit_factor': 1.0, 'probability_of_ruin': 0.0, 'total_return': 10.0,
                       'volatility': 5.0, 'win_rate': 50.0, 'total_trades': 1,
                       'final_portfolio_value': 1_100_000.0, 'initial_capital': 1_000_000.0},
            equity_curve=curve, trade_log=list(trade_log),
            brain_evolution=[], daily_activity_log=list(daily_log), filepath=str(path),
        )
        return path

    def test_exit_date_column_holds_the_exit_date(self, tmp_path):
        """It used to hold the exit *price*, written one column too far left."""
        trade_log = [{
            'trade_id': 'T000001', 'ticker': 'A.NS', 'entry_date': '2022-01-03',
            'entry_price': 100.0, 'exit_date': '2022-02-11', 'exit_price': 123.45,
            'quantity': 10, 'side': 'LONG', 'signal_trigger': 'Trend',
            'gross_pnl': 234.5, 'transaction_costs': 12.0, 'taxes': 3.0,
            'net_pnl': 219.5, 'return_pct': 23.45, 'holding_days': 39,
            'exit_reason': 'target',
        }]
        trades = pd.read_excel(self._report(tmp_path, trade_log), sheet_name='Trade_Log')
        row = trades.iloc[0]

        assert str(row['exit_date']).startswith('2022-02-11')
        assert float(row['exit_price']) == pytest.approx(123.45)
        assert float(row['entry_price']) == pytest.approx(100.0)
        assert float(row['net_pnl']) == pytest.approx(219.5)
        assert float(row['taxes']) == pytest.approx(3.0)
        assert int(row['holding_days']) == 39

    def test_daily_log_values_stay_in_their_columns(self, tmp_path):
        daily_log = [{
            'date': '2022-01-03', 'ticker': 'A.NS', 'action': 'BUY', 'price': 100.0,
            'quantity': 10, 'position_value': 1000.0, 'cash_balance': 990_000.0,
            'total_portfolio_value': 1_000_000.0, 'score': 75.0, 'signal': 'Trend',
            'notes': 'Order executed at 100.0',
        }]
        daily = pd.read_excel(self._report(tmp_path, [], daily_log),
                              sheet_name='Daily_Trade_Log')
        row = daily.iloc[0]

        assert row['action'] == 'BUY'
        assert float(row['price']) == pytest.approx(100.0)
        assert float(row['position_value']) == pytest.approx(1000.0)
        assert float(row['cash_balance']) == pytest.approx(990_000.0)
        assert float(row['total_portfolio_value']) == pytest.approx(1_000_000.0)


class TestEquityCurveSheet:
    """The Equity_Curve sheet layout must match what its charts reference."""

    def test_no_spurious_index_column(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        curve = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)

        df = _prepare_equity_curve_df(curve)

        assert df.index.name == 'Date'
        assert list(df.columns) == ['Portfolio_Value', 'Drawdown_%']

    def test_written_sheet_starts_with_date(self, tmp_path):
        dates = pd.bdate_range("2023-01-02", periods=30)
        curve = pd.Series(np.linspace(100_000.0, 110_000.0, len(dates)), index=dates)

        path = tmp_path / "equity.xlsx"
        export_backtest_excel(
            analytics={'cagr': 1.0, 'sharpe': 0.5, 'sortino': 0.5, 'max_drawdown': -1.0,
                       'profit_factor': 1.0, 'probability_of_ruin': 0.0, 'total_return': 10.0,
                       'volatility': 5.0, 'win_rate': 50.0, 'total_trades': 0,
                       'final_portfolio_value': 110_000.0, 'initial_capital': 100_000.0},
            equity_curve=curve, trade_log=[], brain_evolution=[],
            daily_activity_log=[], filepath=str(path),
        )

        sheet = pd.read_excel(path, sheet_name='Equity_Curve')
        assert list(sheet.columns) == ['Date', 'Portfolio_Value', 'Drawdown_%']
        assert pd.api.types.is_datetime64_any_dtype(sheet['Date'])


class TestMonthlyHeatmap:
    def test_returns_are_in_percent_units(self):
        dates = pd.bdate_range("2023-01-02", periods=90)
        curve = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)

        heatmap = _create_monthly_heatmap_df(curve)
        values = heatmap.values[~pd.isna(heatmap.values)]

        # A ~10% move over three months: percent units, not fractions.
        assert values.size > 0
        assert np.nanmax(np.abs(values)) > 0.1


@pytest.fixture
def gapping_market(monkeypatch):
    """A flat tape interrupted by one gap down and one gap up.

    Bar 40 opens 12% below the previous close and never trades back up; bar 60
    opens 12% above it. Both are ordinary NSE behaviour — the exchange is shut
    for 17.75 hours a day and reopens after the US close and the Asian session.
    """
    dates = pd.bdate_range(start="2023-01-02", periods=120)
    close = np.full(len(dates), 100.0)
    open_ = np.full(len(dates), 100.0)

    open_[40], close[40] = 88.0, 87.0   # gapped down through a 95 stop
    open_[60], close[60] = 112.0, 113.0  # gapped up through a 110 target

    df = pd.DataFrame(
        {
            'open': open_,
            'high': np.maximum(open_, close) + 0.5,
            'low': np.minimum(open_, close) - 0.5,
            'close': close,
            'volume': np.full(len(dates), 5_000_000.0),
        },
        index=dates,
    )
    monkeypatch.setattr(
        "src.backtest_engine.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: df.copy() if ticker == "GAP.NS" else None,
    )
    return {'dates': dates, 'df': df}


class TestGapFills:
    """A level the market gapped through fills at the open, not at the level.

    An ATR stop is an intraday construct: it assumes a continuous tape on which
    a resting order can be worked at the price it names. Overnight there is no
    tape. Booking every stop at the stop credits the book with liquidity that
    was not there, and asymmetrically — the gaps that blow through a long's
    stop are the adverse ones — so the realized loss distribution comes out
    systematically better than it was, which also biases Kelly's payoff ratio.
    """

    def _engine(self):
        return BacktestEngine(
            start_date="2023-01-02", end_date="2023-06-16",
            initial_capital=500_000.0, universe_tickers=["GAP.NS"],
        )

    def test_stop_gapped_through_fills_at_the_open(self, gapping_market):
        engine = self._engine()
        engine.holdings["GAP.NS"] = 100
        engine._open_position("GAP.NS", 100, 100.0, engine.master_date_index[5])
        engine.stop_loss_levels["GAP.NS"] = 95.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[40])[0]

        assert trade['exit_reason'] == 'stop_loss'
        # 88, the price at which the position could actually be sold — not 95,
        # which no one was bidding by the time the market reopened.
        assert trade['exit_price'] == pytest.approx(88.0)
        assert trade['gross_pnl'] == pytest.approx(-1200.0)

    def test_the_unmodelled_slippage_is_a_real_loss_not_a_rounding_error(self, gapping_market):
        """The old model would have booked this exit ₹700 better than it was."""
        engine = self._engine()
        engine.holdings["GAP.NS"] = 100
        engine._open_position("GAP.NS", 100, 100.0, engine.master_date_index[5])
        engine.stop_loss_levels["GAP.NS"] = 95.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[40])[0]

        assumed_at_the_stop = (95.0 - 100.0) * 100
        assert trade['gross_pnl'] < assumed_at_the_stop
        assert assumed_at_the_stop - trade['gross_pnl'] == pytest.approx(700.0)

    def test_target_gapped_through_fills_at_the_open(self, gapping_market):
        """The mirror case runs in the book's favour and must also be modelled.

        A limit sell resting at 110 when the market opens at 112 executes at
        112. Charging the adverse gap but not the favourable one would be a
        different bias, not neutrality.
        """
        engine = self._engine()
        engine.holdings["GAP.NS"] = 100
        engine._open_position("GAP.NS", 100, 100.0, engine.master_date_index[5])
        engine.take_profit_levels["GAP.NS"] = 110.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[60])[0]

        assert trade['exit_reason'] == 'target'
        assert trade['exit_price'] == pytest.approx(112.0)

    def test_a_level_reached_without_a_gap_still_fills_at_the_level(self, gapping_market):
        """The intraday path is unchanged — this only touches gapped bars.

        Bar 60 opens at 112 and trades to a high of 113.5. A target at 113 was
        not gapped through (the open is below it), so it fills at 113.
        """
        engine = self._engine()
        engine.holdings["GAP.NS"] = 100
        engine._open_position("GAP.NS", 100, 100.0, engine.master_date_index[5])
        engine.take_profit_levels["GAP.NS"] = 113.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[60])[0]

        assert trade['exit_price'] == pytest.approx(113.0)

    def test_a_gap_down_beats_a_target_the_session_later_reaches(self, gapping_market):
        """Precedence: the gap is settled at 09:15, before the session runs.

        Bar 60 gaps up to 112 and reaches 113.5, so a position holding both a
        95 stop and a 113 target sees the target intraday — but this bar never
        trades near 95, so the stop must not fire. The converse ordering (a
        gap through the stop outranking an intraday target) is what protects
        the book from booking the good half of a bad day.
        """
        engine = self._engine()
        engine.holdings["GAP.NS"] = 100
        engine._open_position("GAP.NS", 100, 100.0, engine.master_date_index[5])
        engine.stop_loss_levels["GAP.NS"] = 95.0
        engine.take_profit_levels["GAP.NS"] = 113.0

        trade = engine._check_stop_loss_take_profit(engine.master_date_index[60])[0]

        assert trade['exit_reason'] == 'target'
        assert trade['exit_price'] == pytest.approx(113.0)
