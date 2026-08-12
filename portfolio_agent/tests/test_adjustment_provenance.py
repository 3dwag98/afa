"""The adjustment columns the ingest used to throw away.

`normalize_frame` previously applied the back-adjustment and then dropped
`adj_close`, and never read `dividends` or `stock_splits` at all — so the
corporate-action data was present in the source and discarded on the way in.
Two consequences these tests pin:

- Raw prices could not be recovered, which makes any price-*level* check (a
  circuit band, a tick-size rule) compare against a number that never traded.
- There was no corporate-action record, so a demerger or rights issue arrived
  as an unexplained large return and was dropped by the label filter rather
  than recognised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hf_dataset import (
    ADJUSTMENT_COLUMNS,
    OHLCV_COLUMNS,
    RAW_PRICE_COLUMNS,
    corporate_actions_from_frame,
    normalize_frame,
)


def raw_hub_frame(n=10, split_at=None, split_ratio=2.0, dividend_at=None):
    """A frame shaped like one Hub parquet, optionally carrying an action.

    The split is modelled the way the source presents it: the traded price
    halves on the ex-date, and `adj_close` carries the back-adjusted series, so
    `adj_close / close` is the factor that undoes the discontinuity.
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0, dtype=float)

    if split_at is not None:
        close[split_at:] = 100.0 / split_ratio

    adj_close = close.copy()
    if split_at is not None:
        # Everything before the ex-date is scaled down to meet the new level.
        adj_close[:split_at] = close[:split_at] / split_ratio

    frame = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "adj_close": adj_close,
        "volume": np.full(n, 1_000.0),
        "dividends": np.zeros(n),
        "stock_splits": np.zeros(n),
    })
    if split_at is not None:
        frame.loc[split_at, "stock_splits"] = split_ratio
    if dividend_at is not None:
        frame.loc[dividend_at, "dividends"] = 1.5
    return frame


# --------------------------------------------------------------------------
# What survives the ingest
# --------------------------------------------------------------------------


def test_ohlcv_columns_come_first_and_unchanged():
    """The historical contract still holds, so nothing downstream moves."""
    out = normalize_frame(raw_hub_frame())
    assert list(out.columns[:5]) == list(OHLCV_COLUMNS)


def test_raw_prices_are_preserved():
    out = normalize_frame(raw_hub_frame(split_at=5, split_ratio=2.0))
    for column in RAW_PRICE_COLUMNS:
        assert column in out.columns

    # Raw is what traded: 100 before the split, 50 after.
    assert out["close_raw"].iloc[0] == pytest.approx(100.0)
    assert out["close_raw"].iloc[-1] == pytest.approx(50.0)


def test_adjusted_close_is_continuous_across_a_split():
    """The whole point of adjusting: no 50% gap where a split happened."""
    out = normalize_frame(raw_hub_frame(split_at=5, split_ratio=2.0))
    returns = out["close"].pct_change().dropna()
    assert returns.abs().max() < 1e-9

    # And the raw series does show the gap, which is what makes it useful.
    raw_returns = out["close_raw"].pct_change().dropna()
    assert raw_returns.min() == pytest.approx(-0.5)


def test_adjustment_columns_are_kept():
    out = normalize_frame(raw_hub_frame(split_at=5, dividend_at=7))
    for column in ADJUSTMENT_COLUMNS:
        assert column in out.columns
    assert out["stock_splits"].sum() == pytest.approx(2.0)
    assert out["dividends"].sum() == pytest.approx(1.5)


def test_adj_factor_is_recorded_even_with_no_adjustment():
    """1.0 distinguishes 'nothing to adjust' from 'never attempted'."""
    frame = raw_hub_frame().drop(columns=["adj_close"])
    out = normalize_frame(frame)
    assert "adj_factor" in out.columns
    assert (out["adj_factor"] == 1.0).all()


def test_raw_legs_survive_adjust_prices_false():
    """With adjustment off the main legs are raw, and the raw legs agree."""
    out = normalize_frame(raw_hub_frame(split_at=5), adjust_prices=False)
    pd.testing.assert_series_equal(
        out["close"], out["close_raw"], check_names=False
    )


def test_missing_legs_are_filled_before_raw_capture():
    """A close-only file still yields a uniform column set, raw included."""
    frame = raw_hub_frame().drop(columns=["open", "high", "low"])
    out = normalize_frame(frame)
    for column in RAW_PRICE_COLUMNS:
        assert column in out.columns
    assert out["open_raw"].notna().all()


# --------------------------------------------------------------------------
# Corporate actions
# --------------------------------------------------------------------------


def test_split_is_recovered_and_agrees_with_the_factor():
    out = normalize_frame(raw_hub_frame(n=12, split_at=6, split_ratio=2.0))
    actions = corporate_actions_from_frame(out)

    splits = actions[actions["kind"] == "split"]
    assert len(splits) == 1
    assert splits["stated_value"].iloc[0] == pytest.approx(2.0)
    # The source stated a split and the factor moved for it — the good case.
    assert bool(splits["agrees"].iloc[0])


def test_dividend_is_recovered():
    out = normalize_frame(raw_hub_frame(n=12, dividend_at=4))
    actions = corporate_actions_from_frame(out)
    assert (actions["kind"] == "dividend").sum() == 1


def test_adjustment_with_no_stated_cause_is_flagged():
    """A demerger or rights issue moves the factor without stating a split.

    Reported as `unexplained_adjustment` rather than silently absorbed, because
    a naive split factor handles neither correctly and the difference matters.
    """
    frame = raw_hub_frame(n=12, split_at=6, split_ratio=2.0)
    frame["stock_splits"] = 0.0  # the adjustment happened; nothing declared it

    actions = corporate_actions_from_frame(normalize_frame(frame))
    assert (actions["kind"] == "unexplained_adjustment").sum() >= 1


def test_stated_split_the_factor_never_applied_does_not_agree():
    """The expensive case: source says a split, prices were never adjusted.

    Left in the returns this is a genuine 50% single-day loss that no model
    should be asked to explain.
    """
    frame = raw_hub_frame(n=12)
    frame["stock_splits"] = 0.0
    frame.loc[6, "stock_splits"] = 2.0   # declared, but adj_close == close

    actions = corporate_actions_from_frame(normalize_frame(frame))
    splits = actions[actions["kind"] == "split"]
    assert len(splits) == 1
    assert not bool(splits["agrees"].iloc[0])


def test_quiet_history_yields_no_actions():
    actions = corporate_actions_from_frame(normalize_frame(raw_hub_frame(n=20)))
    assert actions.empty


def test_old_narrow_frames_are_tolerated():
    """A cache written before this change has no provenance columns.

    It must return an empty table rather than raising, so an existing install
    keeps working until it refetches.
    """
    narrow = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    assert corporate_actions_from_frame(narrow).empty


# --------------------------------------------------------------------------
# Round trip through the cache
# --------------------------------------------------------------------------


def test_columns_survive_a_parquet_round_trip(tmp_path):
    """The store must persist the wider schema, not just accept it."""
    from src.data_store import DataStore

    out = normalize_frame(raw_hub_frame(n=12, split_at=6, dividend_at=3))
    store = DataStore(cache_dir=tmp_path)
    store.save_ticker_data("TEST.NS", out.copy())

    reloaded = pd.read_parquet(tmp_path / "TEST.NS.parquet")
    for column in list(RAW_PRICE_COLUMNS) + list(ADJUSTMENT_COLUMNS):
        assert column in reloaded.columns
