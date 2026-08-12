"""Purging and embargoing for cross-validation with overlapping labels.

A 5-day forward return observed daily means consecutive samples share four days
of outcome. Two distinct problems follow, and they need two distinct fixes:

**Label overlap — fixed by purging.** A training sample dated `t` carries a
label computed from prices in `(t, t+h]`. If that window reaches into the test
fold, the sample's label is partly *made of* the test period, and training on
it is training on the answer. Dropping those samples is purging.

**Serial correlation — fixed by embargoing.** Purging removes samples whose
label window overlaps. It does not remove samples immediately after the test
fold, whose features are nearly identical to the test fold's features because
financial series are persistent. A model that memorizes those is still
recovering test-period information, through the inputs rather than the labels.
An embargo drops a further `e` samples on the far side of the fold.

`agents/trainer.py::run_walk_forward_validation` already purged — it dropped
the final `horizon_days` rows of each fold's training history, under a comment
calling it an embargo. That was the right operation with the wrong name, and it
was inline, so nothing asserted the property it was supposed to guarantee. This
module makes it a unit that can be tested directly, and adds the embargo that
was genuinely missing.

Positions, not calendar days
----------------------------
A horizon of 5 means five *trading sessions*, not five calendar days. A weekend
or an exchange holiday would make a calendar-day arithmetic silently wrong by a
variable amount, so everything here is computed on positions within a sorted
date index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/test split, with what was removed kept visible.

    `purged` and `embargoed` are returned rather than silently discarded so a
    caller can assert on them and a reader can see the mechanism working. A
    validation scheme whose exclusions are invisible is one nobody checks.
    """

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purged: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    embargoed: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def summary(self) -> dict:
        return {
            "train": len(self.train),
            "test": len(self.test),
            "purged": len(self.purged),
            "embargoed": len(self.embargoed),
            "train_end": self.train.max() if len(self.train) else None,
            "test_start": self.test.min() if len(self.test) else None,
            "test_end": self.test.max() if len(self.test) else None,
        }


def label_window_overlaps(
    positions: np.ndarray,
    test_start_pos: int,
    test_end_pos: int,
    horizon: int,
) -> np.ndarray:
    """Which candidate positions have a label window touching the test fold.

    A sample at position `p` is labelled over `(p, p + horizon]`. It overlaps
    the closed test span `[test_start_pos, test_end_pos]` when its window ends
    at or after the start and the sample itself begins at or before the end.

    Args:
        positions: Candidate training positions in the sorted date index.
        test_start_pos: First position of the test fold.
        test_end_pos: Last position of the test fold, inclusive.
        horizon: Label horizon in sessions. Zero means the label is known at
            `p` itself, so nothing overlaps and nothing is purged.

    Returns:
        Boolean mask, True where the sample must be purged.
    """
    if horizon <= 0:
        return np.zeros(len(positions), dtype=bool)
    return (positions + horizon >= test_start_pos) & (positions <= test_end_pos)


def purged_train_positions(
    n_dates: int,
    train_positions: Sequence[int],
    test_start_pos: int,
    test_end_pos: int,
    horizon: int,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split candidate training positions into kept, purged and embargoed.

    Args:
        n_dates: Length of the full date index, used to bound the embargo.
        train_positions: Candidate training positions.
        test_start_pos: First test position.
        test_end_pos: Last test position, inclusive.
        horizon: Label horizon in sessions.
        embargo: Extra sessions excluded *after* the test fold, on top of the
            purge. Guards against serial correlation in the features rather
            than overlap in the labels, so it is a separate knob — an embargo
            of zero is a deliberate choice, not a default that happens to work.

    Returns:
        `(kept, purged, embargoed)` position arrays.
    """
    candidates = np.asarray(sorted(set(int(p) for p in train_positions)), dtype=int)
    if candidates.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    purge_mask = label_window_overlaps(candidates, test_start_pos, test_end_pos, horizon)

    embargo_mask = np.zeros(len(candidates), dtype=bool)
    if embargo > 0:
        limit = min(n_dates - 1, test_end_pos + embargo)
        embargo_mask = (candidates > test_end_pos) & (candidates <= limit)

    # The two are disjoint by construction and no reconciliation is needed:
    # purging requires `p <= test_end_pos` and the embargo requires
    # `p > test_end_pos`. Stated because the alternative — an intersection
    # quietly resolved one way or the other — is what a reader will assume is
    # happening, and the disjointness is asserted in the tests.
    keep_mask = ~(purge_mask | embargo_mask)

    return candidates[keep_mask], candidates[purge_mask], candidates[embargo_mask]


class PurgedWalkForward:
    """Expanding-window walk-forward with purging and an optional embargo.

    Fold `i` trains on everything before its test block, minus the samples
    whose labels reach into it, minus the embargo. The test blocks are
    contiguous, non-overlapping and chronological, so pooling their predictions
    yields a genuinely out-of-sample series.

    Args:
        n_splits: Number of test folds.
        horizon: Label horizon in sessions.
        embargo: Sessions excluded after each test fold.
        min_train_fraction: Share of the index reserved for the first fold's
            training block, so the earliest fold is not fitted on almost
            nothing.
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 5,
        embargo: int = 0,
        min_train_fraction: float = 0.4,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if horizon < 0:
            raise ValueError(f"horizon cannot be negative, got {horizon}")
        if embargo < 0:
            raise ValueError(f"embargo cannot be negative, got {embargo}")
        if not 0.0 < min_train_fraction < 1.0:
            raise ValueError(
                f"min_train_fraction must be in (0, 1), got {min_train_fraction}"
            )
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.min_train_fraction = min_train_fraction

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Fold]:
        """Yield folds over a sorted, unique date index.

        Folds with an empty training or test block are skipped rather than
        yielded, since a fold that trains on nothing is not a fold — but a
        caller that gets fewer folds than it asked for should know why, which
        is what `n_usable_splits` is for.
        """
        dates = pd.DatetimeIndex(dates).unique().sort_values()
        n = len(dates)
        if n == 0:
            return

        boundaries = np.unique(
            np.linspace(self.min_train_fraction, 1.0, self.n_splits + 1) * n
        ).astype(int)
        boundaries = np.clip(boundaries, 0, n)

        for i in range(len(boundaries) - 1):
            test_start_pos = int(boundaries[i])
            test_stop_pos = int(boundaries[i + 1])   # exclusive
            if test_stop_pos <= test_start_pos:
                continue
            test_end_pos = test_stop_pos - 1

            # Expanding window: everything before this fold's test block.
            candidates = np.arange(0, test_start_pos, dtype=int)
            kept, purged, embargoed = purged_train_positions(
                n_dates=n,
                train_positions=candidates,
                test_start_pos=test_start_pos,
                test_end_pos=test_end_pos,
                horizon=self.horizon,
                embargo=self.embargo,
            )
            if kept.size == 0:
                continue

            yield Fold(
                train=dates[kept],
                test=dates[test_start_pos:test_stop_pos],
                purged=dates[purged],
                embargoed=dates[embargoed],
            )

    def n_usable_splits(self, dates: pd.DatetimeIndex) -> int:
        """How many folds this index can actually support."""
        return sum(1 for _ in self.split(dates))


def assert_no_leakage(fold: Fold, dates: pd.DatetimeIndex, horizon: int) -> None:
    """Raise if any training sample's label window reaches into the test fold.

    The property purging exists to guarantee, checked directly rather than
    inferred from a score. Cheap enough to call in a test for every fold, which
    is the point: a validation scheme is only as trustworthy as the assertion
    that it did what it claims.

    Raises:
        AssertionError: naming the offending dates, since knowing *which* rows
            leaked is what makes the failure diagnosable.
    """
    if len(fold.train) == 0 or len(fold.test) == 0 or horizon <= 0:
        return

    dates = pd.DatetimeIndex(dates).unique().sort_values()
    position = {date: i for i, date in enumerate(dates)}

    test_start = position[fold.test.min()]
    test_end = position[fold.test.max()]

    offenders = [
        date for date in fold.train
        if position[date] + horizon >= test_start and position[date] <= test_end
    ]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} training date(s) have a label window reaching into "
            f"the test fold [{fold.test.min().date()}, {fold.test.max().date()}] "
            f"at horizon {horizon}: first offenders "
            f"{[str(d.date()) for d in offenders[:5]]}"
        )
