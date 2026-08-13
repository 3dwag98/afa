"""Point-in-time index membership, and the bias it removes.

Every result before this was computed on the names that survived to be
downloaded. A stock in the Nifty 500 in 2012 that was delisted in 2015 has no
parquet file, so it never enters a cross-section, so the 2012 deciles are
formed from a universe that excludes exactly the names that went on to fail.
The published figure for Indian indices is 82.5% membership turnover and
roughly 4.94pp of annual return overstatement — larger than either strategy's
neutralized alpha.

Two properties matter more than the mechanics, and most of the tests here are
about them: a missing file must be an *error* rather than a silently disabled
filter, and a date the file cannot speak for must be reported rather than
quietly dropped.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.data_quality.membership import (
    REQUIRED_COLUMNS,
    SURVIVORSHIP_NOTE,
    IndexMembership,
    MembershipInterval,
    apply_membership,
    load_membership,
)


def rows(*triples):
    """(symbol, start, end) -> membership rows for one index."""
    return [
        {
            "symbol": symbol,
            "index_name": "NIFTY50",
            "start_date": start,
            "end_date": end or "",
        }
        for symbol, start, end in triples
    ]


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(records)
    return path


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------


class TestMembersOn:
    def _membership(self):
        return IndexMembership.from_rows(
            rows(
                ("RELIANCE.NS", "2000-01-01", None),
                ("YESBANK.NS", "2017-03-27", "2020-03-19"),
                ("ADANIENT.NS", "2022-09-30", None),
            )
        )

    def test_an_open_ended_stay_covers_today(self):
        assert "RELIANCE.NS" in self._membership().members_on("2026-08-13")

    def test_a_closed_stay_covers_its_own_end_date(self):
        """Inclusive of both ends — the convention NSE announcements use.

        Off by one here silently drops a name's last day, and on a monthly
        rebalance that is a whole holding period.
        """
        assert "YESBANK.NS" in self._membership().members_on("2020-03-19")

    def test_it_does_not_cover_the_day_after(self):
        assert "YESBANK.NS" not in self._membership().members_on("2020-03-20")

    def test_it_does_not_cover_the_day_before_it_joined(self):
        assert "YESBANK.NS" not in self._membership().members_on("2017-03-26")
        assert "YESBANK.NS" in self._membership().members_on("2017-03-27")

    def test_a_date_before_everything_has_no_members(self):
        """A real answer, not an error: the file usually starts after the
        evaluation window does."""
        assert self._membership().members_on("1990-01-01") == set()

    def test_was_member_agrees_with_members_on(self):
        membership = self._membership()
        for date in ("2018-06-01", "2021-01-01", "2023-01-01"):
            expected = membership.members_on(date)
            for symbol in membership.symbols:
                assert membership.was_member(symbol, date) == (symbol in expected)

    def test_a_time_component_does_not_change_the_answer(self):
        membership = self._membership()
        assert membership.members_on("2020-03-19 15:30:00") == membership.members_on(
            "2020-03-19"
        )

    def test_coverage_is_open_ended_when_any_stay_is(self):
        first, last = self._membership().coverage()
        assert first == pd.Timestamp("2000-01-01")
        assert last is None

    def test_coverage_closes_when_every_stay_has_ended(self):
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2010-01-01", "2015-01-01"), ("B.NS", "2012-01-01", "2018-06-30"))
        )
        first, last = membership.coverage()
        assert (first, last) == (pd.Timestamp("2010-01-01"), pd.Timestamp("2018-06-30"))


class TestReentry:
    """Names leave an index and come back; that is two stays, not one."""

    def _membership(self):
        return IndexMembership.from_rows(
            rows(
                ("BANDHANBNK.NS", "2018-03-01", "2021-09-30"),
                ("BANDHANBNK.NS", "2023-04-01", None),
            )
        )

    def test_both_stays_are_covered(self):
        membership = self._membership()
        assert membership.was_member("BANDHANBNK.NS", "2019-01-01")
        assert membership.was_member("BANDHANBNK.NS", "2024-01-01")

    def test_the_gap_between_them_is_not(self):
        """The whole reason a re-entry is a second interval.

        Merging them would extend membership across a period the name was
        genuinely out of the index — which is the same hindsight error, applied
        at a finer grain.
        """
        assert not self._membership().was_member("BANDHANBNK.NS", "2022-06-01")


# --------------------------------------------------------------------------
# Validation: the file is a constituent set, so a corrupt one is not usable
# --------------------------------------------------------------------------


class TestValidation:
    def test_overlapping_stays_for_one_symbol_are_rejected(self):
        with pytest.raises(ValueError, match="overlaps"):
            IndexMembership.from_rows(
                rows(("A.NS", "2010-01-01", "2015-01-01"), ("A.NS", "2014-01-01", None))
            )

    def test_an_open_ended_stay_followed_by_another_is_rejected(self):
        """An open stay covers everything after it, so anything later overlaps."""
        with pytest.raises(ValueError, match="overlaps"):
            IndexMembership.from_rows(
                rows(("A.NS", "2010-01-01", None), ("A.NS", "2020-01-01", None))
            )

    def test_two_symbols_may_overlap_freely(self):
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2010-01-01", None), ("B.NS", "2011-01-01", None))
        )
        assert len(membership.members_on("2012-01-01")) == 2

    def test_an_end_before_its_start_is_rejected(self):
        with pytest.raises(ValueError, match="precedes"):
            IndexMembership.from_rows(rows(("A.NS", "2015-01-01", "2010-01-01")))

    def test_an_unparseable_date_is_rejected(self):
        with pytest.raises(ValueError, match="is not a date"):
            IndexMembership.from_rows(rows(("A.NS", "not-a-date", None)))

    def test_a_missing_column_names_the_row(self):
        with pytest.raises(ValueError, match="row 2: missing column"):
            IndexMembership.from_rows([{"symbol": "A.NS", "start_date": "2010-01-01"}])

    def test_an_empty_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="empty symbol"):
            IndexMembership.from_rows(rows(("", "2010-01-01", None)))

    def test_a_single_day_stay_is_valid(self):
        membership = IndexMembership.from_rows(rows(("A.NS", "2015-06-01", "2015-06-01")))
        assert membership.was_member("A.NS", "2015-06-01")
        assert not membership.was_member("A.NS", "2015-06-02")


class TestLoading:
    def test_a_missing_file_raises_rather_than_disabling_the_filter(self, tmp_path):
        """The most important test here.

        A typo'd path that quietly produced an empty membership would restore
        the survivorship bias inside a run that claims to have corrected for
        it — a wrong number wearing a correct label.
        """
        with pytest.raises(FileNotFoundError, match="no membership file"):
            IndexMembership.load(tmp_path / "absent.csv")

    def test_none_means_none_was_asked_for(self):
        assert load_membership(None) is None

    def test_a_path_that_was_given_still_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_membership(tmp_path / "absent.csv")

    def test_a_header_with_no_rows_is_rejected(self, tmp_path):
        path = write_csv(tmp_path / "empty.csv", [])
        with pytest.raises(ValueError, match="no rows"):
            IndexMembership.load(path)

    def test_it_round_trips(self, tmp_path):
        original = IndexMembership.from_rows(
            rows(("A.NS", "2010-01-01", None), ("B.NS", "2011-05-06", "2019-12-31"))
        )
        reloaded = IndexMembership.load(original.save(tmp_path / "m.csv"))
        assert reloaded.to_rows() == original.to_rows()

    def test_the_source_is_recorded(self, tmp_path):
        path = write_csv(tmp_path / "m.csv", rows(("A.NS", "2010-01-01", None)))
        assert IndexMembership.load(path).source == str(path)

    def test_a_multi_index_file_can_be_narrowed(self, tmp_path):
        records = [
            {"symbol": "A.NS", "index_name": "NIFTY50",
             "start_date": "2010-01-01", "end_date": ""},
            {"symbol": "B.NS", "index_name": "NIFTY500",
             "start_date": "2010-01-01", "end_date": ""},
        ]
        path = write_csv(tmp_path / "m.csv", records)
        assert IndexMembership.load(path, index_name="NIFTY50").symbols == ["A.NS"]
        assert len(IndexMembership.load(path)) == 2


# --------------------------------------------------------------------------
# Applying it to a panel
# --------------------------------------------------------------------------


def panel_of(symbols, dates):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "score": float(rng.normal()),
                "forward_return": float(rng.normal(0, 0.01)),
            }
            for date in dates
            for symbol in symbols
        ]
    )


class TestApplyMembership:
    def _dates(self):
        return pd.date_range("2015-01-01", periods=10, freq="D")

    def test_a_name_is_kept_only_on_the_dates_it_was_a_member(self):
        dates = self._dates()
        panel = panel_of(["A.NS", "B.NS"], dates)
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2015-01-01", None), ("B.NS", "2015-01-06", None))
        )

        result = apply_membership(panel, membership)
        kept = result.panel
        assert set(kept[kept["symbol"] == "B.NS"]["date"]) == set(dates[5:])
        assert set(kept[kept["symbol"] == "A.NS"]["date"]) == set(dates)

    def test_a_name_that_was_never_a_member_is_dropped_entirely(self):
        panel = panel_of(["A.NS", "GHOST.NS"], self._dates())
        membership = IndexMembership.from_rows(rows(("A.NS", "2015-01-01", None)))

        result = apply_membership(panel, membership)
        assert "GHOST.NS" not in set(result.panel["symbol"])
        assert result.symbols_dropped == 1

    def test_the_removed_share_is_reported(self):
        panel = panel_of(["A.NS", "B.NS"], self._dates())
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2015-01-01", None), ("B.NS", "2015-01-06", None))
        )
        result = apply_membership(panel, membership)

        assert result.n_before == 20
        assert result.n_after == 15
        assert result.removed == 5
        assert result.removed_share == pytest.approx(0.25)

    def test_dates_outside_coverage_are_left_unfiltered_and_counted(self):
        """A shorter window that looks clean is worse than a longer one that
        says which part of it is uncorrected."""
        dates = pd.date_range("2010-01-01", periods=6, freq="365D")
        panel = panel_of(["A.NS", "B.NS"], dates)
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2013-01-01", "2014-06-30"))
        )

        result = apply_membership(panel, membership)
        assert result.uncovered_dates > 0
        assert "outside the membership file's coverage" in result.note()
        # The uncovered dates kept both names, so the panel is not empty.
        assert not result.panel.empty

    def test_the_note_reports_what_was_removed(self):
        panel = panel_of(["A.NS", "B.NS"], self._dates())
        membership = IndexMembership.from_rows(
            rows(("A.NS", "2015-01-01", None), ("B.NS", "2015-01-06", None))
        )
        note = apply_membership(panel, membership).note()
        assert "removed 5 of 20" in note
        assert "25.0%" in note

    def test_an_empty_panel_is_handled(self):
        empty = pd.DataFrame(columns=["date", "symbol", "score", "forward_return"])
        result = apply_membership(
            empty, IndexMembership.from_rows(rows(("A.NS", "2015-01-01", None)))
        )
        assert result.n_before == 0
        assert result.removed_share == 0.0

    def test_to_dict_is_flat(self):
        panel = panel_of(["A.NS", "B.NS"], self._dates())
        membership = IndexMembership.from_rows(rows(("A.NS", "2015-01-01", None)))
        document = apply_membership(panel, membership).to_dict()
        assert document["membership_rows_removed"] == 10
        assert all(not isinstance(v, (list, dict)) for v in document.values())


# --------------------------------------------------------------------------
# The bias, demonstrated
# --------------------------------------------------------------------------


def test_the_survivor_set_overstates_a_signals_returns():
    """The mechanism, on data where the answer is known by construction.

    Two cohorts: survivors that drift up, and names that were index members
    early and were delisted after falling. The survivor-only panel — the one
    every result so far was computed on — never sees the failures, so its mean
    forward return is higher than the point-in-time panel's. That gap is the
    bias, and it exists regardless of the signal.
    """
    dates = pd.date_range("2015-01-01", periods=40, freq="D")
    records = []
    for date in dates:
        for i in range(10):
            records.append(
                {"date": date, "symbol": f"LIVE{i}.NS", "score": float(i),
                 "forward_return": 0.002}
            )
        for i in range(5):
            records.append(
                {"date": date, "symbol": f"DEAD{i}.NS", "score": float(i),
                 "forward_return": -0.010}
            )
    panel = pd.DataFrame(records)

    membership = IndexMembership.from_rows(
        rows(*[(f"LIVE{i}.NS", "2015-01-01", None) for i in range(10)],
             *[(f"DEAD{i}.NS", "2015-01-01", None) for i in range(5)])
    )

    point_in_time = apply_membership(panel, membership).panel
    survivors_only = panel[panel["symbol"].str.startswith("LIVE")]

    assert survivors_only["forward_return"].mean() > point_in_time[
        "forward_return"
    ].mean()
    # Everything was a member, so the filter itself removed nothing — the bias
    # lives in which files exist, not in the filter.
    assert len(point_in_time) == len(panel)


# --------------------------------------------------------------------------
# The harness says so when it has no membership data
# --------------------------------------------------------------------------


class TestTheHarnessIsHonestAboutIt:
    def test_the_survivorship_note_names_the_magnitude(self):
        """A caveat without a number is a caveat people learn to skip."""
        assert "4.94pp" in SURVIVORSHIP_NOTE
        assert "82.5%" in SURVIVORSHIP_NOTE
        assert "OBTAINING_DATA" in SURVIVORSHIP_NOTE

    def test_obtaining_data_documents_how_to_get_one(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent.parent
        text = (repo / "docs" / "OBTAINING_DATA.md").read_text()
        assert "--membership" in text
        assert "index_name,start_date,end_date" in text
        # The honest part: no free feed ships this.
        assert "niftyhistory.in" in text
        assert "Prowess" in text

    def test_the_cli_accepts_a_membership_path(self):
        from portfolio_agent.cli import create_parser

        args = create_parser().parse_args(
            ["evaluate", "--strategy", "momentum",
             "--membership", "universe/m.csv", "--index-name", "NIFTY500"]
        )
        assert args.membership == "universe/m.csv"
        assert args.index_name == "NIFTY500"

    def test_no_membership_is_the_default(self):
        from portfolio_agent.cli import create_parser

        args = create_parser().parse_args(["evaluate", "--strategy", "momentum"])
        assert args.membership is None


def test_an_interval_covers_what_it_should():
    """The primitive, checked directly so the query tests rest on something."""
    stay = MembershipInterval(
        "A.NS", "NIFTY50", pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-31")
    )
    assert not stay.covers(pd.Timestamp("2014-12-31"))
    assert stay.covers(pd.Timestamp("2015-01-01"))
    assert stay.covers(pd.Timestamp("2015-12-31"))
    assert not stay.covers(pd.Timestamp("2016-01-01"))

    open_ended = MembershipInterval(
        "B.NS", "NIFTY50", pd.Timestamp("2015-01-01"), None
    )
    assert open_ended.covers(pd.Timestamp("2099-01-01"))
