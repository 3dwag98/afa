"""Purging and embargoing.

The load-bearing test is `test_no_training_label_reaches_into_the_test_fold`:
it asserts the property directly on index sets, without training anything. A
validation scheme is only as trustworthy as the assertion that it did what it
claims, and a score is not that assertion — a leak and a genuinely good model
both produce a high number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.validation import (
    Fold,
    PurgedWalkForward,
    assert_no_leakage,
    label_window_overlaps,
    purged_train_positions,
)

DATES = pd.bdate_range("2020-01-01", periods=500)


# --------------------------------------------------------------------------
# The property
# --------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 5, 10, 21])
@pytest.mark.parametrize("embargo", [0, 5])
def test_no_training_label_reaches_into_the_test_fold(horizon, embargo):
    """The guarantee, checked on every fold at several horizons."""
    splitter = PurgedWalkForward(n_splits=5, horizon=horizon, embargo=embargo)

    folds = list(splitter.split(DATES))
    assert folds, "expected at least one usable fold"

    for fold in folds:
        assert_no_leakage(fold, DATES, horizon)


def test_assert_no_leakage_actually_catches_a_leak():
    """The assertion must fail on a fold that leaks, or it proves nothing."""
    position = {d: i for i, d in enumerate(DATES)}
    test = DATES[200:250]

    # Training right up to the test boundary: the last five samples carry
    # labels computed from prices inside the test period.
    leaky = Fold(train=DATES[:200], test=test)

    with pytest.raises(AssertionError, match="label window reaching into"):
        assert_no_leakage(leaky, DATES, horizon=5)

    _ = position


def test_purge_removes_exactly_the_horizon_before_the_test_fold():
    """With an expanding window the purge is the last `horizon` sessions."""
    kept, purged, embargoed = purged_train_positions(
        n_dates=500,
        train_positions=range(0, 200),
        test_start_pos=200,
        test_end_pos=249,
        horizon=5,
        embargo=0,
    )
    # Positions 195..199 have label windows ending at or after 200.
    assert sorted(purged) == [195, 196, 197, 198, 199]
    assert kept.max() == 194
    assert embargoed.size == 0


def test_embargo_removes_sessions_after_the_test_fold():
    """Purging handles label overlap; the embargo handles what follows.

    Only relevant when training data exists on the far side of a fold, which is
    why it is exercised with an explicit candidate range rather than through
    the expanding-window splitter.
    """
    kept, purged, embargoed = purged_train_positions(
        n_dates=500,
        train_positions=range(0, 400),
        test_start_pos=200,
        test_end_pos=249,
        horizon=5,
        embargo=10,
    )
    assert sorted(embargoed) == list(range(250, 260))
    assert 260 in kept
    assert 259 not in kept


def test_purge_and_embargo_are_disjoint():
    """They cannot overlap, and the split is on the right side of the fold.

    Purging catches samples at or before the test end whose label window
    reaches forward into it; the embargo catches samples strictly after the
    test end. So a position is never both, and the boundary between them is
    the last test session.
    """
    kept, purged, embargoed = purged_train_positions(
        n_dates=100,
        train_positions=range(0, 60),
        test_start_pos=40,
        test_end_pos=45,
        horizon=10,
        embargo=10,
    )
    assert set(purged) & set(embargoed) == set()
    assert max(purged) <= 45          # purging stops at the test end
    assert min(embargoed) == 46       # the embargo begins immediately after
    assert set(kept) & (set(purged) | set(embargoed)) == set()


def test_horizon_zero_purges_nothing():
    """A label known at `t` cannot overlap anything."""
    mask = label_window_overlaps(np.arange(0, 100), 50, 60, horizon=0)
    assert not mask.any()


def test_embargo_is_bounded_by_the_index():
    """An embargo running past the end must not invent positions."""
    _, _, embargoed = purged_train_positions(
        n_dates=100,
        train_positions=range(0, 100),
        test_start_pos=90,
        test_end_pos=95,
        horizon=1,
        embargo=50,
    )
    assert embargoed.max() < 100


# --------------------------------------------------------------------------
# The demonstration
# --------------------------------------------------------------------------


def test_purging_removes_a_measurable_advantage_from_an_overlapping_label():
    """Without purging, the boundary sample's label is partly the test answer.

    A deliberately memorising 'model': predict the label of the nearest
    training date. Unpurged, the nearest date to the test start is the session
    immediately before it, whose 5-day label is built from four days *inside*
    the test fold — so it agrees with the test label far more than it should.
    Purged, the nearest usable date is `horizon` sessions earlier, and the
    agreement drops to what the signal actually supports.

    This is the mechanism made visible. The index assertion above is the proof;
    this shows what the assertion is protecting against.
    """
    horizon = 5
    rng = np.random.default_rng(0)

    # A label that is a rolling forward sum, so adjacent labels genuinely
    # overlap — which is the whole condition being tested.
    steps = pd.Series(rng.normal(size=len(DATES)), index=DATES)
    labels = steps.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))

    test = DATES[300:340]
    test_labels = labels.reindex(test).dropna()

    def nearest_label_prediction(train_dates):
        """Predict each test date with the label of the closest training date."""
        usable = [d for d in train_dates if pd.notna(labels.get(d, np.nan))]
        if not usable:
            return np.array([])
        last = max(usable)
        return np.full(len(test_labels), labels[last])

    unpurged = nearest_label_prediction(DATES[:300])

    splitter = PurgedWalkForward(n_splits=5, horizon=horizon)
    fold = next(f for f in splitter.split(DATES) if f.test.min() <= test.min() <= f.test.max())
    purged = nearest_label_prediction(fold.train)

    # Agreement measured as absolute error against the first test label, which
    # is the one the boundary sample's window actually overlaps.
    first_actual = test_labels.iloc[0]
    unpurged_error = abs(unpurged[0] - first_actual)
    purged_error = abs(purged[0] - first_actual)

    assert unpurged_error < purged_error, (
        "the unpurged boundary sample should agree with the test label more "
        "closely than a properly purged one — if it does not, the overlap "
        "being tested is not present"
    )


# --------------------------------------------------------------------------
# Splitter behaviour
# --------------------------------------------------------------------------


def test_test_folds_are_chronological_and_disjoint():
    folds = list(PurgedWalkForward(n_splits=5, horizon=5).split(DATES))

    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test.max() < later.test.min()


def test_training_window_expands():
    folds = list(PurgedWalkForward(n_splits=5, horizon=5).split(DATES))
    sizes = [f.n_train for f in folds]

    assert sizes == sorted(sizes)


def test_training_always_precedes_its_test_fold():
    for fold in PurgedWalkForward(n_splits=5, horizon=5).split(DATES):
        assert fold.train.max() < fold.test.min()


def test_removed_dates_are_reported():
    """Exclusions are visible, so the mechanism can be audited."""
    fold = next(iter(PurgedWalkForward(n_splits=4, horizon=7, embargo=0).split(DATES)))

    assert len(fold.purged) == 7
    assert set(fold.purged).isdisjoint(set(fold.train))


def test_summary_describes_the_fold():
    fold = next(iter(PurgedWalkForward(n_splits=4, horizon=5).split(DATES)))
    summary = fold.summary()

    assert summary["train"] > 0 and summary["test"] > 0
    assert summary["train_end"] < summary["test_start"]


def test_unsorted_and_duplicated_dates_are_handled():
    shuffled = DATES.to_list()[::-1] + DATES.to_list()[:20]
    folds = list(PurgedWalkForward(n_splits=3, horizon=5).split(pd.DatetimeIndex(shuffled)))

    assert folds
    for fold in folds:
        assert fold.train.is_monotonic_increasing
        assert fold.test.is_monotonic_increasing


def test_an_index_too_short_yields_no_folds_rather_than_raising():
    tiny = pd.bdate_range("2020-01-01", periods=3)
    splitter = PurgedWalkForward(n_splits=5, horizon=10)

    assert splitter.n_usable_splits(tiny) == 0


def test_empty_index_yields_nothing():
    assert list(PurgedWalkForward().split(pd.DatetimeIndex([]))) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 0},
        {"horizon": -1},
        {"embargo": -1},
        {"min_train_fraction": 0.0},
        {"min_train_fraction": 1.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PurgedWalkForward(**kwargs)


def test_splits_are_deterministic():
    a = [f.summary() for f in PurgedWalkForward(n_splits=5, horizon=5).split(DATES)]
    b = [f.summary() for f in PurgedWalkForward(n_splits=5, horizon=5).split(DATES)]

    assert a == b
