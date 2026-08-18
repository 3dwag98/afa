"""Three reference inputs, and the wrong-file failure each one has.

None of these is hard arithmetic. What makes them worth testing is that each
has a specific way of being *supplied wrongly* that produces a plausible
number rather than an error — a total-share count under a free-float heading,
a sector map covering half the universe, a flow series read as though either
leg alone said something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.data_quality.reference import (
    IMPLAUSIBLE_FLOAT_FRACTION_LOW,
    NO_FLOWS_NOTE,
    NO_SECTOR_MAP_NOTE,
    SIZE_PROXY_REPLACED_NOTE,
    FlowSeries,
    FreeFloatStore,
    load_flows,
    load_free_float,
    sector_coverage,
    validate_free_float,
)


def _float_frame(rows=None):
    return pd.DataFrame(rows or [
        # A has a heavy promoter stake; B is widely held.
        {"symbol": "A.NS", "effective_date": "2022-01-01",
         "free_float_shares": 30.0, "total_shares": 100.0},
        {"symbol": "A.NS", "effective_date": "2023-06-01",
         "free_float_shares": 45.0, "total_shares": 100.0},
        {"symbol": "B.NS", "effective_date": "2022-01-01",
         "free_float_shares": 80.0, "total_shares": 100.0},
    ])


def _closes(n=400, start="2022-06-01"):
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"A.NS": np.full(n, 100.0), "B.NS": np.full(n, 100.0)}, index=index
    )


# --------------------------------------------------------------------------
# Free float
# --------------------------------------------------------------------------


class TestFreeFloat:
    def test_a_clean_file_passes(self):
        assert validate_free_float(_float_frame()).ok

    def test_the_market_cap_is_float_times_price(self):
        store = FreeFloatStore.from_frame(_float_frame())
        cap = store.market_cap(_closes())

        assert cap.iloc[0]["A.NS"] == pytest.approx(3000.0)
        assert cap.iloc[0]["B.NS"] == pytest.approx(8000.0)

    def test_two_firms_of_equal_total_size_differ_by_promoter_stake(self):
        """The whole reason free float is not `shares_outstanding`.

        A and B have identical issued shares and identical prices, so a
        total-capitalisation sort calls them the same size. To anyone who has
        to trade them they differ by a factor of 2.7.
        """
        store = FreeFloatStore.from_frame(_float_frame())
        cap = store.market_cap(_closes()).iloc[0]

        assert cap["B.NS"] / cap["A.NS"] == pytest.approx(80.0 / 30.0)

    def test_it_steps_at_the_effective_date(self):
        """A promoter sale in June 2023 must not restate 2022."""
        store = FreeFloatStore.from_frame(_float_frame())
        cap = store.market_cap(_closes())

        assert cap.loc[pd.Timestamp("2023-05-31"), "A.NS"] == pytest.approx(3000.0)
        assert cap.loc[pd.Timestamp("2023-06-01"), "A.NS"] == pytest.approx(4500.0)

    def test_it_never_back_fills(self):
        """Back-filling would apply a post-buyback count to the years before
        it, restating every market cap in one direction."""
        store = FreeFloatStore.from_frame(_float_frame())
        early = _closes(n=60, start="2021-06-01")
        assert store.panel(early.index).isna().all().all()

    def test_a_zero_float_is_an_error_not_a_tiny_company(self):
        frame = _float_frame()
        frame.loc[0, "free_float_shares"] = 0.0
        result = validate_free_float(frame)

        assert not result.ok
        assert any("missing value" in error for error in result.errors)

    def test_float_above_issued_shares_is_an_error(self):
        frame = _float_frame()
        frame.loc[0, "free_float_shares"] = 150.0
        result = validate_free_float(frame)

        assert not result.ok
        assert any("swapped" in error for error in result.errors)

    def test_total_shares_under_a_free_float_heading_is_caught(self):
        """The failure that is invisible from the float column alone.

        A file reporting total capitalisation as free float produces a size
        sort wrong by exactly the promoter stake — largest where promoter
        holdings are largest, which is the opposite of a size correction.
        """
        frame = _float_frame()
        frame["free_float_shares"] = frame["total_shares"]
        frame = pd.concat([frame] * 3, ignore_index=True)
        frame["effective_date"] = pd.date_range(
            "2022-01-01", periods=len(frame), freq="180D"
        ).strftime("%Y-%m-%d")

        result = validate_free_float(frame)
        assert any("total capitalisation" in w for w in result.warnings)

    def test_a_missing_total_shares_column_says_what_cannot_be_checked(self):
        frame = _float_frame().drop(columns=["total_shares"])
        result = validate_free_float(frame)

        assert result.ok
        assert any("undetectable without it" in w for w in result.warnings)

    def test_an_implausibly_small_float_is_flagged_not_rejected(self):
        frame = _float_frame()
        frame.loc[0, "free_float_shares"] = 2.0
        result = validate_free_float(frame)

        assert result.ok
        assert result.n_implausibly_small == 1

    def test_a_missing_column_is_an_error(self):
        frame = _float_frame().drop(columns=["effective_date"])
        assert not validate_free_float(frame).ok

    def test_a_store_refuses_a_bad_frame(self):
        frame = _float_frame()
        frame.loc[0, "free_float_shares"] = -5.0
        with pytest.raises(ValueError, match="not usable"):
            FreeFloatStore.from_frame(frame)

    def test_loading_names_the_resolved_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="OBTAINING_DATA"):
            load_free_float(tmp_path / "absent.csv")

    def test_it_reads_a_csv(self, tmp_path):
        path = tmp_path / "float.csv"
        _float_frame().to_csv(path, index=False)
        assert load_free_float(path).symbols == ["A.NS", "B.NS"]


class TestFreeFloatReplacesTheProxy:
    """The caveat `SIZE_PROXY_NOTE` has been printing since T05."""

    def _panel(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2022-06-01", periods=300)
        rows = []
        for date in dates:
            for symbol in ("A.NS", "B.NS"):
                rows.append({
                    "date": date, "symbol": symbol,
                    "score": float(rng.uniform(0, 100)),
                    "forward_return": float(rng.normal(0, 0.01)),
                    "close": 100.0,
                    "volume": 1e6,
                })
        return pd.DataFrame(rows)

    def test_without_a_store_the_proxy_note_is_printed(self):
        from portfolio_agent.evaluation.neutralize import SIZE_PROXY_NOTE, add_exposures

        _panel, columns, notes = add_exposures(self._panel())
        assert "size" in columns
        assert SIZE_PROXY_NOTE in notes

    def test_with_a_store_the_note_says_it_is_real(self):
        from portfolio_agent.evaluation.neutralize import SIZE_PROXY_NOTE, add_exposures

        store = FreeFloatStore.from_frame(_float_frame())
        _panel, columns, notes = add_exposures(self._panel(), free_float=store)

        assert "size" in columns
        assert SIZE_PROXY_REPLACED_NOTE in notes
        assert SIZE_PROXY_NOTE not in notes

    def test_the_two_produce_different_size_exposures(self):
        """If they agreed there would be nothing to fix.

        Both names trade identical volume at an identical price, so the
        traded-value proxy calls them the same size. Free float does not.
        """
        from portfolio_agent.evaluation.neutralize import add_exposures

        proxy, _c, _n = add_exposures(self._panel())
        real, _c, _n = add_exposures(
            self._panel(), free_float=FreeFloatStore.from_frame(_float_frame())
        )

        proxy_spread = proxy.groupby("symbol")["size"].mean()
        real_spread = real.groupby("symbol")["size"].mean()

        assert proxy_spread["A.NS"] == pytest.approx(proxy_spread["B.NS"])
        assert real_spread["A.NS"] != pytest.approx(real_spread["B.NS"])

    def test_a_store_covering_nothing_falls_back_and_says_so(self):
        """No blending: a size column built from two definitions ranks partly
        on which definition applied."""
        from portfolio_agent.evaluation.neutralize import SIZE_PROXY_NOTE, add_exposures

        elsewhere = FreeFloatStore.from_frame(pd.DataFrame([
            {"symbol": "Z.NS", "effective_date": "2022-01-01",
             "free_float_shares": 10.0, "total_shares": 100.0},
        ]))
        _panel, columns, notes = add_exposures(self._panel(), free_float=elsewhere)

        assert "size" in columns
        assert SIZE_PROXY_NOTE in notes
        assert any("covered none of this universe" in n for n in notes)


# --------------------------------------------------------------------------
# Sector
# --------------------------------------------------------------------------


class TestSectorCoverage:
    def test_full_coverage_reports_full(self):
        coverage = sector_coverage(
            {"A.NS": "Bank", "B.NS": "IT"}, ["A.NS", "B.NS"]
        )
        assert coverage.coverage == 1.0
        assert coverage.n_sectors == 2

    def test_partial_coverage_is_stated_not_implied(self):
        """A map resolving 60% produces a "sector-neutral" result in which the
        rest were neutralized against a pool called UNKNOWN."""
        coverage = sector_coverage(
            {"A.NS": "Bank", "B.NS": "Bank", "C.NS": "IT"},
            ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"],
        )
        assert coverage.coverage == pytest.approx(0.6)
        assert "60%" in coverage.note()
        assert "UNKNOWN" in coverage.note()

    def test_it_lists_what_it_could_not_map(self):
        coverage = sector_coverage({"A.NS": "Bank"}, ["A.NS", "D.NS"])
        assert coverage.unmapped == ["D.NS"]

    def test_it_reports_concentration(self):
        """A map whose largest sector holds most of the universe neutralizes
        almost nothing."""
        coverage = sector_coverage(
            {f"S{i}.NS": ("Bank" if i < 9 else "IT") for i in range(10)},
            [f"S{i}.NS" for i in range(10)],
        )
        assert coverage.largest_sector_share == pytest.approx(0.9)

    def test_an_empty_map_gives_the_no_map_note(self):
        assert sector_coverage({}, ["A.NS", "B.NS"]).note() == NO_SECTOR_MAP_NOTE

    def test_the_note_says_why_it_matters_here(self):
        assert "momentum concentrates" in NO_SECTOR_MAP_NOTE

    def test_full_coverage_does_not_mention_unknown(self):
        note = sector_coverage({"A.NS": "Bank", "B.NS": "IT"}, ["A.NS", "B.NS"]).note()
        assert "UNKNOWN" not in note


# --------------------------------------------------------------------------
# Flows
# --------------------------------------------------------------------------


def _flows(n=200, fii=None, dii=None):
    dates = pd.bdate_range("2023-01-02", periods=n)
    return FlowSeries.from_frame(pd.DataFrame({
        "date": dates,
        "fii_net": np.full(n, -500.0) if fii is None else fii,
        "dii_net": np.full(n, 300.0) if dii is None else dii,
    }))


class TestFlows:
    def test_net_is_fii_minus_dii(self):
        """Domestic flows systematically offset foreign ones, so neither leg
        alone says whether the market was under pressure."""
        flows = _flows()
        assert flows.net.iloc[0] == pytest.approx(-800.0)

    def test_sustained_selling_reads_as_outflow(self):
        flows = _flows()
        states = flows.states(flows.flows.index)
        assert set(states.unique()) == {"outflow"}

    def test_sustained_buying_reads_as_inflow(self):
        flows = _flows(fii=np.full(200, 900.0))
        states = flows.states(flows.flows.index)
        assert set(states.unique()) == {"inflow"}

    def test_the_state_turns_when_the_flow_does(self):
        flows = _flows(
            fii=np.concatenate([np.full(100, -500.0), np.full(100, 900.0)])
        )
        states = flows.states(flows.flows.index)
        assert states.iloc[0] == "outflow"
        assert states.iloc[-1] == "inflow"

    def test_the_trailing_window_is_causal(self):
        """It ends at the row it labels, so this is tradable rather than only
        attribution — the distinction T28's conditional split turns on."""
        base = _flows(fii=np.concatenate([np.full(100, -500.0), np.full(100, 900.0)]))
        dates = base.flows.index[:100]

        tampered = base.flows.copy()
        tampered.iloc[100:] *= -5.0
        after = FlowSeries(flows=tampered)

        pd.testing.assert_series_equal(base.states(dates), after.states(dates))

    def test_dates_without_a_full_window_are_omitted(self):
        """Labelling early dates from a partial window is two conditioners
        under one name."""
        flows = _flows(n=200)
        assert len(flows.states(flows.flows.index, window=63)) == 200 - 62

    def test_a_missing_column_is_refused(self):
        with pytest.raises(ValueError, match="missing required column"):
            FlowSeries.from_frame(pd.DataFrame({"date": ["2023-01-02"], "fii_net": [1.0]}))

    def test_an_unparseable_date_is_refused(self):
        with pytest.raises(ValueError, match="could not be parsed"):
            FlowSeries.from_frame(pd.DataFrame({
                "date": ["not a date"], "fii_net": [1.0], "dii_net": [2.0]
            }))

    def test_it_reads_a_csv(self, tmp_path):
        path = tmp_path / "flows.csv"
        dates = pd.bdate_range("2023-01-02", periods=10)
        pd.DataFrame({
            "date": dates, "fii_net": np.arange(10.0), "dii_net": np.arange(10.0)
        }).to_csv(path, index=False)

        assert len(load_flows(path).flows) == 10

    def test_a_missing_file_names_the_resolved_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="OBTAINING_DATA"):
            load_flows(tmp_path / "absent.csv")

    def test_the_states_drop_into_the_conditional_split(self):
        """The shape T28's `conditional_ic` consumes, so a result can be split
        by flow regime rather than only by realized return."""
        rng = np.random.default_rng(8)
        flows = _flows(
            fii=np.concatenate([np.full(100, -500.0), np.full(100, 900.0)])
        )
        dates = flows.flows.index
        states = flows.states(dates)

        rows = [
            {"date": date, "symbol": f"S{i}", "score": float(rng.uniform(0, 100)),
             "forward_return": float(rng.normal(0, 0.01))}
            for date in dates for i in range(20)
        ]
        panel = pd.DataFrame(rows)

        from portfolio_agent.evaluation.metrics import rank_ic_series, summarize_ic

        for state in ("inflow", "outflow"):
            subset = panel[panel["date"].isin(set(states[states == state].index))]
            assert not subset.empty
            summarize_ic(rank_ic_series(subset), horizon=1)


class TestTheNotesSayWhatWasMissing:
    def test_each_note_points_at_the_acquisition_doc(self):
        for note in (NO_SECTOR_MAP_NOTE, NO_FLOWS_NOTE):
            assert "OBTAINING_DATA" in note

    def test_the_flow_note_says_why_a_pooled_result_is_incomplete(self):
        assert "sustained foreign selling" in NO_FLOWS_NOTE

    def test_the_replaced_size_note_says_free_float_not_total(self):
        assert "Free float rather than total capitalisation" in SIZE_PROXY_REPLACED_NOTE
        assert "50-75%" in SIZE_PROXY_REPLACED_NOTE
