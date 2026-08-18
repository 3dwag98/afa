"""The bias that does not look like a bug.

Results for the quarter ending 31 March are published in late May. A backtest
using them from 31 March has read six to eight weeks into the future on every
stock, every quarter, forever — and the output looks like alpha, because the
numbers are real and the dates are real and the strategy simply knew them
early.

So the weight here is on one property: `as_of(D)` returns nothing that was not
publishable on D. Everything else in the module exists to make that property
hard to lose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.data_quality.fundamentals import (
    FUNDAMENTALS_NOTE,
    IMPLAUSIBLY_FAST_DAYS,
    IMPLAUSIBLY_SLOW_DAYS,
    QUARTERLY_DEADLINE_DAYS,
    FundamentalsStore,
    load_fundamentals,
    validate_fundamentals,
)
from portfolio_agent.features.characteristics import (
    CHARACTERISTICS,
    computable_characteristics,
)
from portfolio_agent.features.cross_section import build_cross_section, latest_values

#: Four quarters with realistic Indian filing lags (SEBI LODR gives 45 days).
QUARTERS = [
    ("2023-03-31", "2023-05-12"),
    ("2023-06-30", "2023-08-08"),
    ("2023-09-30", "2023-11-05"),
    ("2023-12-31", "2024-02-09"),
]


def _frame(symbols=("A.NS", "B.NS"), quarters=QUARTERS, **overrides):
    rows = []
    for symbol in symbols:
        for q, (fiscal, report) in enumerate(quarters):
            row = {
                "symbol": symbol, "fiscal_date": fiscal, "report_date": report,
                "total_assets": 1000.0 + 50 * q,
                "total_equity": 400.0 + 20 * q,
                "revenue": 300.0,
                "cost_of_goods_sold": 200.0,
                "net_income": 40.0 + q,
                "cash_flow_operating": 45.0,
                "total_debt": 200.0,
                "shares_outstanding": 10.0,
            }
            row.update(overrides)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The property everything else protects
# --------------------------------------------------------------------------


class TestPointInTime:
    def test_a_quarter_is_invisible_until_it_is_published(self):
        """The whole module in one assertion.

        On 30 June the Q2 numbers *describe* a period that has just ended and
        will not be published until 8 August. A fiscal-date-keyed store would
        hand them over; this one does not.
        """
        store = FundamentalsStore.from_frame(_frame())

        on_quarter_end = store.as_of("2023-06-30")
        assert on_quarter_end.loc["A.NS", "fiscal_date"] == pd.Timestamp("2023-03-31")

        after_publication = store.as_of("2023-08-10")
        assert after_publication.loc["A.NS", "fiscal_date"] == pd.Timestamp("2023-06-30")

    def test_the_day_before_publication_still_shows_the_old_quarter(self):
        store = FundamentalsStore.from_frame(_frame())
        assert store.as_of("2023-08-07").loc["A.NS", "fiscal_date"] == pd.Timestamp(
            "2023-03-31"
        )

    def test_the_publication_day_itself_counts(self):
        """`report_date <= date`, matching T19's inclusive decision-date rule."""
        store = FundamentalsStore.from_frame(_frame())
        assert store.as_of("2023-08-08").loc["A.NS", "fiscal_date"] == pd.Timestamp(
            "2023-06-30"
        )

    def test_nothing_is_known_before_the_first_report(self):
        store = FundamentalsStore.from_frame(_frame())
        assert store.as_of("2023-01-01").empty

    def test_it_takes_the_latest_published_not_the_latest_fiscal(self):
        """A late-filed old quarter must not shadow an already-published new one."""
        store = FundamentalsStore.from_frame(_frame())
        row = store.as_of("2024-03-01").loc["A.NS"]
        assert row["fiscal_date"] == pd.Timestamp("2023-12-31")

    def test_a_restatement_does_not_travel_backwards(self):
        """The earliest report for a period wins.

        A restatement published in November was not knowable in May, and
        keeping the latest would reintroduce the look-ahead by the back door.
        """
        original = _frame(symbols=("A.NS",), quarters=[QUARTERS[0]])
        restated = original.copy()
        restated["report_date"] = "2023-11-20"
        restated["total_equity"] = 999.0

        store = FundamentalsStore.from_frame(pd.concat([original, restated]))
        assert store.as_of("2023-12-01").loc["A.NS", "total_equity"] == 400.0

    def test_symbols_can_be_restricted(self):
        store = FundamentalsStore.from_frame(_frame(symbols=("A.NS", "B.NS", "C.NS")))
        assert list(store.as_of("2023-08-10", ["A.NS"]).index) == ["A.NS"]


class TestThePanel:
    def test_it_is_flat_until_the_first_report_lands(self):
        store = FundamentalsStore.from_frame(_frame())
        dates = pd.bdate_range("2023-04-03", periods=120)
        panel = store.panel(dates, "total_equity")

        assert panel.loc[: pd.Timestamp("2023-05-11")].isna().all().all()
        assert panel.loc[pd.Timestamp("2023-05-12")].tolist() == [400.0, 400.0]

    def test_it_steps_rather_than_interpolating(self):
        """A balance-sheet number is a step function. Smoothing between reports
        would invent quarters that were never published."""
        store = FundamentalsStore.from_frame(_frame())
        dates = pd.bdate_range("2023-05-12", periods=90)
        panel = store.panel(dates, "total_equity")["A.NS"].dropna()

        assert set(panel.unique()) <= {400.0, 420.0}

    def test_a_report_on_a_non_trading_day_still_propagates(self):
        """Union the report dates in before reindexing, or a Saturday filing
        vanishes."""
        saturday = _frame(symbols=("A.NS",), quarters=[("2023-03-31", "2023-05-13")])
        store = FundamentalsStore.from_frame(saturday)
        dates = pd.bdate_range("2023-05-08", periods=10)

        assert store.panel(dates, "total_equity").loc[
            pd.Timestamp("2023-05-15"), "A.NS"
        ] == 400.0

    def test_an_unknown_field_names_what_is_available(self):
        store = FundamentalsStore.from_frame(_frame())
        with pytest.raises(KeyError, match="total_equity"):
            store.panel(pd.bdate_range("2023-05-01", periods=5), "goodwill")


# --------------------------------------------------------------------------
# Validation — in the T02 style
# --------------------------------------------------------------------------


class TestValidation:
    def test_a_clean_file_passes(self):
        result = validate_fundamentals(_frame())
        assert result.ok
        assert not result.errors

    def test_the_median_lag_matches_indian_filing_practice(self):
        result = validate_fundamentals(_frame())
        assert 30 <= result.median_lag_days <= QUARTERLY_DEADLINE_DAYS

    def test_a_missing_report_date_is_an_error_not_a_default(self):
        """Falling back to the fiscal date is the look-ahead itself."""
        frame = _frame().drop(columns=["report_date"])
        result = validate_fundamentals(frame)

        assert not result.ok
        assert any("look-ahead" in error for error in result.errors)

    def test_reporting_before_the_period_ends_is_an_error(self):
        """Not a late filing in the other direction — a column swap."""
        frame = _frame()
        frame["report_date"] = "2023-01-01"
        result = validate_fundamentals(frame)

        assert not result.ok
        assert result.n_negative_lag > 0
        assert any("column swap" in error for error in result.errors)

    def test_an_implausibly_fast_lag_is_flagged(self):
        frame = _frame(symbols=("A.NS",), quarters=[("2023-03-31", "2023-04-02")])
        result = validate_fundamentals(frame)

        assert result.n_implausibly_fast == 1
        assert any("Regulation 33" in w for w in result.warnings)

    def test_an_implausibly_slow_lag_is_flagged(self):
        frame = _frame(symbols=("A.NS",), quarters=[("2023-03-31", "2023-10-31")])
        result = validate_fundamentals(frame)

        assert result.n_implausibly_slow == 1

    def test_identical_lags_are_called_out_as_probably_reconstructed(self):
        """The sharpest check in the module.

        A file whose report dates were made by adding a constant to the fiscal
        date is *worse* than no file: it has the shape of point-in-time data
        and none of the content, so a backtest on it looks rigorous and is not.
        """
        frame = _frame()
        frame["report_date"] = (
            pd.to_datetime(frame["fiscal_date"]) + pd.Timedelta(days=45)
        ).dt.strftime("%Y-%m-%d")

        result = validate_fundamentals(frame)
        assert result.lag_looks_synthetic
        assert any("reconstructed" in w for w in result.warnings)

    def test_a_genuine_file_is_not_called_synthetic(self):
        assert not validate_fundamentals(_frame()).lag_looks_synthetic

    def test_report_equals_fiscal_trips_both_checks(self):
        frame = _frame()
        frame["report_date"] = frame["fiscal_date"]
        result = validate_fundamentals(frame)

        assert result.n_implausibly_fast == len(frame)
        assert result.lag_looks_synthetic

    def test_duplicate_periods_are_reported(self):
        frame = pd.concat([_frame(), _frame()])
        assert validate_fundamentals(frame).n_duplicate_periods > 0

    def test_a_file_with_no_recognized_fact_is_an_error(self):
        frame = _frame()[["symbol", "fiscal_date", "report_date"]]
        result = validate_fundamentals(frame)
        assert not result.ok

    def test_a_store_refuses_to_be_built_from_a_bad_frame(self):
        frame = _frame().drop(columns=["report_date"])
        with pytest.raises(ValueError, match="not usable"):
            FundamentalsStore.from_frame(frame)

    def test_warnings_do_not_block(self):
        """A file with a suspicious lag is still usable — flagged, not refused."""
        frame = _frame(symbols=("A.NS",), quarters=[("2023-03-31", "2023-04-02")])
        store = FundamentalsStore.from_frame(frame)
        assert store.validation.warnings
        assert store.validation.ok

    def test_the_validation_travels_into_a_report(self):
        document = validate_fundamentals(_frame()).to_dict()
        assert "fundamentals_median_lag_days" in document
        assert "fundamentals_lag_looks_synthetic" in document


class TestLoading:
    def test_it_reads_a_csv(self, tmp_path):
        path = tmp_path / "fundamentals.csv"
        _frame().to_csv(path, index=False)

        store = load_fundamentals(path)
        assert store.symbols == ["A.NS", "B.NS"]

    def test_a_missing_file_names_the_resolved_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="OBTAINING_DATA"):
            load_fundamentals(tmp_path / "absent.csv")


# --------------------------------------------------------------------------
# The characteristics
# --------------------------------------------------------------------------


def _panel_frames(n=400, **columns):
    index = pd.bdate_range("2022-01-03", periods=n)
    frames = {}
    for i, symbol in enumerate(["A.NS", "B.NS", "C.NS", "D.NS"]):
        frame = {"close": np.linspace(100.0, 150.0, n) * (1 + 0.1 * i)}
        for name, values in columns.items():
            frame[name] = np.full(n, values[i])
        frames[symbol] = pd.DataFrame(frame, index=index)
    return frames


class TestCharacteristics:
    def test_all_six_are_registered(self):
        from portfolio_agent.features.cross_section import is_cross_sectional_feature

        for name in CHARACTERISTICS:
            assert is_cross_sectional_feature(name), name

    def test_book_to_price_falls_as_price_rises(self):
        frames = _panel_frames(
            total_equity=[400.0] * 4, shares_outstanding=[10.0] * 4
        )
        built = build_cross_section(frames, ["book_to_price"])
        values = latest_values(built["book_to_price"])

        # A.NS is the cheapest name in the fixture, so it has the highest B/P.
        assert values["A.NS"] > values["D.NS"]

    def test_gross_profitability_is_over_assets_not_sales(self):
        """The Novy-Marx choice, checked arithmetically.

        Over assets: (300 - 200) / 1000 = 0.10. Over sales it would be 0.333,
        and the two rank a cross-section differently.
        """
        frames = _panel_frames(
            revenue=[300.0] * 4, cost_of_goods_sold=[200.0] * 4,
            total_assets=[1000.0] * 4,
        )
        built = build_cross_section(frames, ["gross_profitability"])
        assert latest_values(built["gross_profitability"])["A.NS"] == pytest.approx(0.10)

    def test_accruals_are_earnings_minus_cash(self):
        frames = _panel_frames(
            net_income=[40.0] * 4, cash_flow_operating=[45.0] * 4,
            total_assets=[1000.0] * 4,
        )
        built = build_cross_section(frames, ["accruals"])
        # Collected more cash than it reported: negative accrual, the good end.
        assert latest_values(built["accruals"])["A.NS"] == pytest.approx(-0.005)

    def test_non_positive_book_equity_is_dropped_from_the_value_sort(self):
        """Fama and French's own screen, and not a tidying step.

        A firm whose accumulated losses exceed its paid-in capital has a
        *negative* B/P, which sorts it to the extreme growth end — the opposite
        end from where a distressed balance sheet belongs. Keeping them lets
        one end of the value decile be populated by exactly the firms the
        characteristic cannot describe.
        """
        frames = _panel_frames(
            total_equity=[400.0, 0.0, -100.0, 500.0], shares_outstanding=[10.0] * 4
        )
        built = build_cross_section(frames, ["book_to_price"])
        values = latest_values(built["book_to_price"])

        assert "A.NS" in values and "D.NS" in values
        assert "B.NS" not in values and "C.NS" not in values

    def test_earnings_to_price_keeps_loss_makers(self):
        """The deliberate contrast with book-to-price.

        A loss-making firm has a real and interpretable earnings yield and
        sorts to the end a reader would expect, so screening it out would be a
        quality filter wearing a value label.
        """
        frames = _panel_frames(
            net_income=[40.0, -20.0, 10.0, 30.0], shares_outstanding=[10.0] * 4
        )
        built = build_cross_section(frames, ["earnings_to_price"])
        values = latest_values(built["earnings_to_price"])

        assert "B.NS" in values
        assert values["B.NS"] < 0
        assert min(values, key=values.get) == "B.NS"

    def test_leverage_uses_book_equity(self):
        frames = _panel_frames(total_debt=[200.0] * 4, total_equity=[400.0] * 4)
        built = build_cross_section(frames, ["leverage"])
        assert latest_values(built["leverage"])["A.NS"] == pytest.approx(0.5)

    def test_asset_growth_is_year_on_year(self):
        # The step has to land inside the 252-session lookback of the final
        # row, or the comparison is 1200 against 1200 and reports no growth.
        index = pd.bdate_range("2021-01-04", periods=600)
        assets = np.concatenate([np.full(450, 1000.0), np.full(150, 1200.0)])
        frames = {
            s: pd.DataFrame(
                {"close": np.full(600, 100.0), "total_assets": assets}, index=index
            )
            for s in ("A.NS", "B.NS")
        }
        built = build_cross_section(frames, ["asset_growth"])
        assert latest_values(built["asset_growth"])["A.NS"] == pytest.approx(0.2)

    def test_computable_reports_a_partial_dataset_rather_than_failing(self):
        """A fundamentals file covering three fields is the normal case."""
        assert computable_characteristics(["total_equity", "shares_outstanding"]) == [
            "book_to_price"
        ]

    def test_computable_with_everything_returns_everything(self):
        every_fact = [
            "total_assets", "total_equity", "revenue", "cost_of_goods_sold",
            "net_income", "total_debt", "cash_flow_operating", "shares_outstanding",
        ]
        assert computable_characteristics(every_fact) == list(CHARACTERISTICS)

    def test_computable_with_nothing_returns_nothing(self):
        assert computable_characteristics([]) == []


class TestTheRunSaysWhatItLacked:
    def test_no_fundamentals_produces_the_note(self):
        from portfolio_agent.evaluation.harness import fundamentals_notes

        assert fundamentals_notes(None) == [FUNDAMENTALS_NOTE]

    def test_the_note_names_what_is_uncontrolled(self):
        for exposure in ("Size", "value", "profitability", "investment", "quality"):
            assert exposure in FUNDAMENTALS_NOTE

    def test_an_unusable_file_does_not_read_as_though_it_applied(self):
        from portfolio_agent.evaluation.harness import fundamentals_notes

        notes = fundamentals_notes("no_such_file.csv")
        assert any("controls for no accounting characteristic" in n for n in notes)

    def test_a_usable_file_surfaces_its_warnings(self, tmp_path):
        from portfolio_agent.evaluation.harness import fundamentals_notes

        frame = _frame()
        frame["report_date"] = (
            pd.to_datetime(frame["fiscal_date"]) + pd.Timedelta(days=45)
        ).dt.strftime("%Y-%m-%d")
        path = tmp_path / "synthetic.csv"
        frame.to_csv(path, index=False)

        notes = fundamentals_notes(str(path))
        assert any("reconstructed" in note for note in notes)
