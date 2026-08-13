"""Data invariants: one seeded violation per check, plus the gate semantics.

The shape of this file follows the shape of the problem. Every structural check
gets a frame built to break it in exactly one way, so a failure names the check
rather than "something in the data". Every advisory check gets the same, plus a
test that the offending rows *survive* — that is the behavioural correction this
task makes, and it is the one thing a count-based test would not catch.

The severity split is tested as hard as the checks themselves. A gate that
fails a build on a genuine 30% circuit day is a gate someone switches off
within a week, and then nothing is checked at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.data_quality import (
    ADVISORY,
    STRUCTURAL,
    IngestRejected,
    assert_writable,
    check_adjustment_factor,
    check_calendar_coverage,
    check_duplicate_dates,
    check_extreme_returns,
    check_history_length,
    check_monotonic_index,
    check_ohlc_ordering,
    check_price_positivity,
    collect_status,
    infer_trading_calendar,
    validate_frame,
    validate_store,
)


def bars(n: int = 300, start: str = "2023-01-02", level: float = 100.0) -> pd.DataFrame:
    """A clean, boring, entirely valid price series."""
    index = pd.date_range(start, periods=n, freq="B")
    close = level + np.arange(n) * 0.05
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 100_000.0),
        },
        index=index,
    )


def checks_found(violations) -> set:
    return {violation.check for violation in violations}


# --------------------------------------------------------------------------
# A clean frame trips nothing
# --------------------------------------------------------------------------


def test_a_clean_series_produces_no_violations():
    assert validate_frame(bars(), "CLEAN.NS") == []


def test_an_empty_frame_is_structural():
    violations = validate_frame(pd.DataFrame(), "EMPTY.NS")
    assert len(violations) == 1
    assert violations[0].severity == STRUCTURAL
    assert violations[0].check == "empty"


# --------------------------------------------------------------------------
# Structural invariants, one seeded violation each
# --------------------------------------------------------------------------


def test_a_high_below_the_body_is_caught():
    # Below the close but still above the low, so exactly one check fires and a
    # failure names the check rather than "something about this bar".
    frame = bars()
    frame.loc[frame.index[10], "high"] = frame.loc[frame.index[10], "close"] - 0.1
    violations = check_ohlc_ordering(frame, "X.NS")
    assert checks_found(violations) == {"high_below_body"}
    assert violations[0].severity == STRUCTURAL
    assert violations[0].examples == [frame.index[10].strftime("%Y-%m-%d")]


def test_a_low_above_the_body_is_caught():
    # Above the open but still below the high, for the same reason.
    frame = bars()
    frame.loc[frame.index[7], "low"] = frame.loc[frame.index[7], "close"] - 0.1
    violations = check_ohlc_ordering(frame, "X.NS")
    assert checks_found(violations) == {"low_above_body"}


def test_a_high_below_the_low_is_caught():
    frame = bars()
    date = frame.index[3]
    # Break only the high/low relation, leaving the body inside both.
    frame.loc[date, ["open", "close"]] = 100.0
    frame.loc[date, "high"] = 99.0
    frame.loc[date, "low"] = 101.0
    assert "high_below_low" in checks_found(check_ohlc_ordering(frame, "X.NS"))


@pytest.mark.parametrize("bad_close", [0.0, -5.0, np.nan])
def test_a_non_positive_or_missing_close_is_caught(bad_close):
    """A zero close poisons every ratio feature at once."""
    frame = bars()
    frame.loc[frame.index[5], "close"] = bad_close
    violations = check_price_positivity(frame, "X.NS")
    assert "non_positive_price" in checks_found(violations)
    assert all(v.severity == STRUCTURAL for v in violations)


def test_negative_volume_is_caught():
    frame = bars()
    frame.loc[frame.index[2], "volume"] = -1.0
    assert checks_found(check_price_positivity(frame, "X.NS")) == {"negative_volume"}


def test_zero_volume_is_allowed():
    """A genuinely untraded session is not a corrupt one."""
    frame = bars()
    frame.loc[frame.index[2], "volume"] = 0.0
    assert check_price_positivity(frame, "X.NS") == []


def test_a_duplicated_date_is_caught():
    frame = bars(50)
    frame = pd.concat([frame, frame.iloc[[10]]]).sort_index()
    violations = check_duplicate_dates(frame, "X.NS")
    assert checks_found(violations) == {"duplicate_dates"}
    assert violations[0].severity == STRUCTURAL


def test_an_unsorted_index_is_caught():
    """Every rolling feature assumes date order; out of order, it reads ahead."""
    frame = bars(50).iloc[::-1]
    violations = check_monotonic_index(frame, "X.NS")
    assert checks_found(violations) == {"unsorted_index"}
    assert violations[0].severity == STRUCTURAL


def test_back_adjustment_rounding_is_not_a_structural_failure():
    """Multiplying every leg by a float must not manufacture thousands of failures.

    Back-adjustment scales open/high/low/close by the same factor. Without a
    relative tolerance the resulting 15th-digit differences read as a high
    below the body on a large fraction of bars.
    """
    frame = bars(500)
    factor = 0.37419283746
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] * factor / factor * factor
    assert check_ohlc_ordering(frame, "X.NS") == []


# --------------------------------------------------------------------------
# Advisory checks — flagged, never dropped
# --------------------------------------------------------------------------


def test_an_extreme_move_is_flagged_and_the_bar_survives():
    """The behavioural correction: `max_abs_target` *dropped* these.

    Dropping removes genuine corporate-action days along with the errors and
    leaves no record that it happened. Here the row is still in the frame after
    validation, and the finding is advisory.
    """
    frame = bars()
    frame.loc[frame.index[20]:, "close"] *= 1.35
    frame.loc[frame.index[20]:, ["open", "high", "low"]] *= 1.35

    violations = check_extreme_returns(frame, "X.NS")
    assert checks_found(violations) == {"extreme_return"}
    assert violations[0].severity == ADVISORY

    kept = validate_frame(frame, "X.NS")
    assert all(v.severity == ADVISORY for v in kept)
    assert len(frame) == 300  # nothing was removed


def test_a_split_scale_move_gets_its_own_check_name():
    """A 20% move is a circuit limit. An 80% move is a split that escaped."""
    frame = bars()
    frame.loc[frame.index[50]:, ["open", "high", "low", "close"]] *= 0.2

    violations = check_extreme_returns(frame, "X.NS")
    assert checks_found(violations) == {"unadjusted_split_suspected"}
    assert violations[0].severity == ADVISORY
    assert "split" in violations[0].detail


def test_the_two_return_checks_do_not_double_count():
    """One bar produces one finding, under whichever name fits."""
    frame = bars()
    frame.loc[frame.index[50]:, ["open", "high", "low", "close"]] *= 0.2
    violations = check_extreme_returns(frame, "X.NS")
    assert sum(v.count for v in violations) == 1


def test_an_ordinary_circuit_day_is_not_flagged():
    """NSE bands routinely permit 20%; flagging those would bury the real ones."""
    frame = bars()
    frame.loc[frame.index[30]:, ["open", "high", "low", "close"]] *= 1.19
    assert check_extreme_returns(frame, "X.NS") == []


def test_a_short_history_is_advisory():
    violations = check_history_length(bars(100), "X.NS", min_sessions=252)
    assert checks_found(violations) == {"short_history"}
    assert violations[0].severity == ADVISORY


# --------------------------------------------------------------------------
# Adjustment provenance (needs T01's wider schema)
# --------------------------------------------------------------------------


def with_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """The full T01 schema — all four columns, or `has_adjustment_columns` is False."""
    frame = frame.copy()
    frame["adj_close"] = frame["close"]
    frame["adj_factor"] = 1.0
    frame["dividends"] = 0.0
    frame["stock_splits"] = 0.0
    return frame


def test_an_adjustment_factor_step_with_no_recorded_action_is_flagged():
    """Invisible in the prices, and it makes two pulls of one series disagree."""
    frame = with_adjustment(bars())
    frame.loc[frame.index[100]:, "adj_factor"] = 0.5

    violations = check_adjustment_factor(frame, "X.NS")
    assert checks_found(violations) == {"unexplained_adjustment"}
    assert violations[0].severity == ADVISORY
    assert violations[0].examples == [frame.index[100].strftime("%Y-%m-%d")]


def test_an_adjustment_factor_step_on_a_recorded_split_is_fine():
    frame = with_adjustment(bars())
    frame.loc[frame.index[100]:, "adj_factor"] = 0.5
    frame.loc[frame.index[100], "stock_splits"] = 2.0
    assert check_adjustment_factor(frame, "X.NS") == []


def test_an_adjustment_factor_step_on_a_recorded_dividend_is_fine():
    frame = with_adjustment(bars())
    frame.loc[frame.index[100]:, "adj_factor"] = 0.98
    frame.loc[frame.index[100], "dividends"] = 1.5
    assert check_adjustment_factor(frame, "X.NS") == []


def test_a_split_recorded_as_one_is_not_a_corporate_action():
    """The upstream encoding writes 1.0 for "no split"; counting it flags every bar."""
    frame = with_adjustment(bars())
    frame["stock_splits"] = 1.0
    frame.loc[frame.index[100]:, "adj_factor"] = 0.5
    assert checks_found(check_adjustment_factor(frame, "X.NS")) == {"unexplained_adjustment"}


def test_a_cache_without_the_wider_schema_skips_the_check_rather_than_passing_it():
    """Silence here would read as "checked and fine" on 2,397 unchecked files."""
    assert check_adjustment_factor(bars(), "X.NS") == []
    status = collect_status({"X.NS": bars()})
    assert status.without_adjustment_columns
    assert "no adjustment columns" in status.render()


# --------------------------------------------------------------------------
# The inferred trading calendar
# --------------------------------------------------------------------------


def test_a_holiday_is_not_a_missing_session():
    """The check the spec asks for: distinguish a holiday from missing data.

    Every symbol skips the same date, so it never reaches quorum and is not a
    session. Nothing is reported missing.
    """
    frames = {}
    for i in range(10):
        frame = bars(60)
        frames[f"S{i}.NS"] = frame.drop(frame.index[30])

    calendar = infer_trading_calendar(frames)
    assert bars(60).index[30] not in calendar
    for symbol, frame in frames.items():
        assert check_calendar_coverage(frame, symbol, calendar) == []


def test_one_symbol_missing_a_session_the_market_had_is_reported():
    frames = {f"S{i}.NS": bars(60) for i in range(10)}
    gap_date = bars(60).index[30]
    frames["S0.NS"] = frames["S0.NS"].drop(gap_date)

    calendar = infer_trading_calendar(frames)
    assert gap_date in calendar

    violations = check_calendar_coverage(frames["S0.NS"], "S0.NS", calendar)
    assert checks_found(violations) == {"missing_sessions"}
    assert violations[0].examples == [gap_date.strftime("%Y-%m-%d")]
    for symbol in list(frames)[1:]:
        assert check_calendar_coverage(frames[symbol], symbol, calendar) == []


def test_a_late_listing_is_not_blamed_for_dates_before_it_listed():
    """Otherwise every young name reports thousands of missing sessions.

    The symbol here starts halfway through the sample. Its coverage is measured
    against its own span, so it is clean; the check is bounded, not global.
    """
    frames = {f"S{i}.NS": bars(120) for i in range(8)}
    frames["NEW.NS"] = bars(120).iloc[60:]

    calendar = infer_trading_calendar(frames)
    assert check_calendar_coverage(frames["NEW.NS"], "NEW.NS", calendar) == []


def test_coverage_counts_a_run_of_missing_sessions_as_a_run():
    """A hundred scattered gaps is a download bug; a hundred consecutive is a
    suspension, and the two need different responses."""
    frames = {f"S{i}.NS": bars(120) for i in range(8)}
    index = bars(120).index
    frames["S0.NS"] = frames["S0.NS"].drop(index[40:50])

    calendar = infer_trading_calendar(frames)
    violations = check_calendar_coverage(frames["S0.NS"], "S0.NS", calendar)
    assert violations[0].count == 10
    assert "longest run 10" in violations[0].detail


def test_a_calendar_cannot_be_inferred_from_one_symbol():
    """With a single series every date it has is trivially a session."""
    assert len(infer_trading_calendar({"ONLY.NS": bars()})) == 0


def test_the_calendar_quorum_excludes_symbols_outside_their_own_span():
    """Without that, every date before the newest listing falls below quorum.

    Nine symbols cover the whole window and one covers only its tail. The early
    dates must still be sessions.
    """
    frames = {f"S{i}.NS": bars(200) for i in range(9)}
    frames["LATE.NS"] = bars(200).iloc[150:]
    calendar = infer_trading_calendar(frames)
    assert len(calendar) == 200


# --------------------------------------------------------------------------
# The gate: which findings may fail a build
# --------------------------------------------------------------------------


def test_a_structural_violation_fails_the_gate():
    frame = bars()
    frame.loc[frame.index[5], "close"] = 0.0
    report = validate_store({"BAD.NS": frame})
    assert not report.ok
    assert report.exit_code == 1


def test_an_advisory_finding_does_not_fail_the_gate():
    """A store with real circuit days in it is a correct store.

    Failing on those is how a gate becomes something people disable, after
    which nothing is checked at all.
    """
    frame = bars(100)          # short history: advisory
    frame.loc[frame.index[50]:, ["open", "high", "low", "close"]] *= 1.4
    report = validate_store({"NOISY.NS": frame})
    assert report.advisories
    assert report.ok
    assert report.exit_code == 0


def test_ingest_refuses_a_structurally_invalid_frame():
    frame = bars()
    frame.loc[frame.index[5], "high"] = 1.0
    with pytest.raises(IngestRejected) as excinfo:
        assert_writable(frame, "BROKEN.NS")
    assert "BROKEN.NS" in str(excinfo.value)
    assert excinfo.value.violations


def test_ingest_accepts_a_frame_with_only_advisory_findings():
    """A symbol with one genuine 30% day is a symbol worth having."""
    frame = bars(60)
    frame.loc[frame.index[30]:, ["open", "high", "low", "close"]] *= 1.4
    assert assert_writable(frame, "SPIKY.NS") is frame


def test_the_ingest_gate_runs_only_structural_checks():
    """Ingest has one symbol and no cross-section, so advisories are noise there."""
    frame = bars(30)   # short history — advisory only
    assert assert_writable(frame, "SHORT.NS") is frame
    assert checks_found(validate_frame(frame, "SHORT.NS")) == {"short_history"}
    assert validate_frame(frame, "SHORT.NS", structural_only=True) == []


def test_the_hf_ingest_path_calls_the_gate():
    """Wiring, asserted rather than assumed: a gate nobody calls is decoration."""
    import inspect

    from portfolio_agent.src import hf_dataset

    source = inspect.getsource(hf_dataset.sync_hf_to_cache)
    assert "assert_writable" in source
    assert "IngestRejected" in source


# --------------------------------------------------------------------------
# The report object
# --------------------------------------------------------------------------


def test_the_report_groups_by_check_rather_than_listing_flat():
    """A flat list of 2,400 findings hides that 2,300 of them are one cause."""
    frames = {}
    for i in range(12):
        frame = bars(60)
        frame.loc[frame.index[5], "close"] = 0.0
        frames[f"S{i}.NS"] = frame

    text = validate_store(frames).render()
    assert "non_positive_price   12 symbol(s)" in text
    assert "and 8 more symbol(s)" in text
    assert "FAIL" in text


def test_the_report_serializes():
    frame = bars()
    frame.loc[frame.index[5], "close"] = 0.0
    document = validate_store({"BAD.NS": frame}).to_dict()
    assert document["ok"] is False
    assert document["structural"] >= 1
    assert "non_positive_price" in document["by_check"]
    json.dumps(document)  # must not raise


def test_a_clean_store_says_so():
    report = validate_store({f"S{i}.NS": bars() for i in range(5)})
    assert report.ok
    assert "No violations." in report.render()
    assert report.symbols_checked == 5
    assert report.rows_checked == 1500


# --------------------------------------------------------------------------
# data status
# --------------------------------------------------------------------------


def test_status_reports_the_span_that_went_unnoticed():
    """The five-year window, printed. This is the whole motivation."""
    frames = {f"S{i}.NS": bars(1250, start="2021-08-09") for i in range(5)}
    status = collect_status(frames, min_sessions=252)

    start, end = status.span()
    assert start.strftime("%Y-%m-%d") == "2021-08-09"
    assert status.n_symbols == 5
    assert status.total_bars == 6250
    text = status.render()
    assert "span" in text
    assert "4.8 years" in text or "years" in text


def test_status_counts_symbols_below_the_usable_threshold():
    frames = {"LONG.NS": bars(400), "SHORT.NS": bars(80), "TINY.NS": bars(10)}
    status = collect_status(frames, min_sessions=252)
    assert {s.symbol for s in status.below_threshold} == {"SHORT.NS", "TINY.NS"}
    assert "2 symbol(s)" in status.render()


def test_status_reports_coverage_and_the_longest_gap():
    frames = {f"S{i}.NS": bars(200) for i in range(8)}
    index = bars(200).index
    frames["S0.NS"] = frames["S0.NS"].drop(index[40:55])

    status = collect_status(frames)
    by_symbol = {s.symbol: s for s in status.symbols}
    assert by_symbol["S0.NS"].longest_gap == 15
    assert by_symbol["S0.NS"].coverage < 1.0
    assert by_symbol["S1.NS"].coverage == 1.0
    assert "15 consecutive sessions missing" in status.render()


def test_status_counts_corporate_actions_per_year():
    frame = with_adjustment(bars(600, start="2022-01-03"))
    frame.loc[frame.index[100], "dividends"] = 2.0
    frame.loc[frame.index[400], "stock_splits"] = 2.0
    frame.loc[frame.index[401], "stock_splits"] = 1.0   # "no split", must not count

    status = collect_status({"X.NS": frame, "Y.NS": with_adjustment(bars(600, start="2022-01-03"))})
    assert sum(status.actions_per_year().values()) == 2
    assert "Corporate actions per year" in status.render()


def test_status_says_when_no_corporate_actions_exist_at_all():
    """Zero across an Indian universe over years is not plausible, and was true."""
    frames = {f"S{i}.NS": with_adjustment(bars(600)) for i in range(3)}
    text = collect_status(frames).render()
    assert "No corporate actions recorded anywhere" in text


def test_status_frame_is_one_row_per_symbol():
    frames = {f"S{i}.NS": bars(300) for i in range(4)}
    table = collect_status(frames).to_frame()
    assert len(table) == 4
    assert {"symbol", "sessions", "coverage", "longest_gap"} <= set(table.columns)


# --------------------------------------------------------------------------
# The CLI, against a real parquet cache
# --------------------------------------------------------------------------


@pytest.fixture
def seeded_cache(tmp_path):
    """A small on-disk cache: nine clean symbols and one broken one."""
    from portfolio_agent.src.data_store import DataStore

    store = DataStore(cache_dir=tmp_path)
    for i in range(9):
        store.save_ticker_data(f"S{i}.NS", bars(300).copy())

    broken = bars(300)
    broken.loc[broken.index[10], "low"] = broken.loc[broken.index[10], "close"] + 5.0
    store.save_ticker_data("BROKEN.NS", broken)
    return tmp_path


def test_cli_validate_exits_non_zero_on_a_seeded_structural_violation(
    seeded_cache, capsys
):
    from portfolio_agent.cli import main

    code = main(["data", "validate", "--cache-dir", str(seeded_cache)])
    assert code == 1
    output = capsys.readouterr().out
    assert "low_above_body" in output
    assert "BROKEN.NS" in output
    assert "FAIL" in output


def test_cli_validate_exits_zero_on_a_clean_cache(tmp_path, capsys):
    from portfolio_agent.cli import main
    from portfolio_agent.src.data_store import DataStore

    store = DataStore(cache_dir=tmp_path)
    for i in range(5):
        store.save_ticker_data(f"S{i}.NS", bars(300).copy())

    assert main(["data", "validate", "--cache-dir", str(tmp_path)]) == 0
    assert "No violations." in capsys.readouterr().out


def test_cli_validate_strict_fails_on_advisories(tmp_path, capsys):
    from portfolio_agent.cli import main
    from portfolio_agent.src.data_store import DataStore

    store = DataStore(cache_dir=tmp_path)
    for i in range(5):
        store.save_ticker_data(f"S{i}.NS", bars(100).copy())   # short history

    assert main(["data", "validate", "--cache-dir", str(tmp_path)]) == 0
    assert main(["data", "validate", "--cache-dir", str(tmp_path), "--strict"]) == 1
    assert "--strict" in capsys.readouterr().out


def test_cli_status_reports_the_span(seeded_cache, capsys):
    from portfolio_agent.cli import main

    assert main(["data", "status", "--cache-dir", str(seeded_cache)]) == 0
    output = capsys.readouterr().out
    assert "Data store status" in output
    assert "symbols          10" in output
    assert "span" in output


def test_cli_status_emits_json(seeded_cache, capsys):
    from portfolio_agent.cli import main

    assert main([
        "data", "status", "--cache-dir", str(seeded_cache), "--json", "--per-symbol"
    ]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["symbols"] == 10
    assert len(document["per_symbol"]) == 10


def test_cli_validate_emits_json(seeded_cache, capsys):
    from portfolio_agent.cli import main

    assert main(["data", "validate", "--cache-dir", str(seeded_cache), "--json"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    assert document["symbols_checked"] == 10


def test_cli_reports_an_empty_cache_rather_than_passing_it(tmp_path, capsys):
    """Exit 0 on an empty store would make the gate vacuously green."""
    from portfolio_agent.cli import main

    assert main(["data", "validate", "--cache-dir", str(tmp_path)]) == 1
    assert "No readable data" in capsys.readouterr().out


def test_cli_limits_and_symbol_selection(seeded_cache, capsys):
    from portfolio_agent.cli import main

    assert main([
        "data", "status", "--cache-dir", str(seeded_cache), "--symbols", "S0.NS,S1.NS"
    ]) == 0
    assert "symbols          2" in capsys.readouterr().out

    assert main(["data", "status", "--cache-dir", str(seeded_cache), "--limit", "3"]) == 0
    assert "symbols          3" in capsys.readouterr().out


def test_read_cached_bars_does_not_forward_fill(tmp_path):
    """A gap detector reading through a gap filler can never report a gap."""
    from portfolio_agent.src.data_store import DataStore, read_cached_bars

    frame = bars(60)
    gap_date = frame.index[30]
    frame = frame.drop(gap_date)
    DataStore(cache_dir=tmp_path).save_ticker_data("GAPPY.NS", frame.copy())

    raw = read_cached_bars("GAPPY.NS", tmp_path)
    assert gap_date not in raw.index
    assert len(raw) == 59
