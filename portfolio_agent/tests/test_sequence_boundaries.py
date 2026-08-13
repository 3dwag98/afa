"""A training sequence must come from one instrument.

`_stack_blocks` concatenates per-ticker frames into one matrix and
`TimeSeriesDataset` slid a window across the joins, so a window starting near
the end of ticker A's block and ending inside ticker B's fed the model one
stock's price history and asked it to predict a different stock's move.

`load_data`'s docstring used to acknowledge this and dismiss it:

    Sequence windows that straddle two concatenated tickers' boundaries mix
    data from different instruments; this is a bounded, documented limitation
    ... it affects at most sequence_length * (n_tickers - 1) windows out of the
    full panel.

The bound is arithmetically right and the conclusion drawn from it is wrong,
because the panel it is a fraction *of* is not the panel that matters. Applied
to each 15% split of a ticker with the shipped `min_history_days`, the same
formula gives **every** window — 37 rows per ticker against a 60-row window
means no window fits inside one ticker at all. See
`test_the_shipped_default_could_not_build_one_clean_validation_sample`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.data.dataset import (
    TimeSeriesDataset,
    chronological_split_bounds,
    group_lengths_in_slice,
    sequence_target_positions,
)


def blocks(n_tickers=3, n_rows=12):
    """Feature column counts 0..n-1 within each ticker.

    That is what makes a straddling window detectable: the sequence steps
    backwards exactly where it crosses a join.
    """
    return [
        pd.DataFrame(
            {
                "feature": np.arange(n_rows, dtype=float),
                "target": np.arange(n_rows, dtype=float) + 100 * t,
            },
            index=pd.date_range("2024-01-01", periods=n_rows) + pd.Timedelta(days=t),
        )
        for t in range(n_tickers)
    ]


def stacked(n_tickers=3, n_rows=12):
    frames = blocks(n_tickers, n_rows)
    panel = pd.concat(frames)
    return (
        panel.iloc[:, :-1].values,
        panel.iloc[:, -1].values,
        [len(f) for f in frames],
    )


# --------------------------------------------------------------------------
# The positions arithmetic, which three callers share
# --------------------------------------------------------------------------


class TestSequenceTargetPositions:
    def test_one_group_matches_the_old_contiguous_slide(self):
        """A single instrument was never the broken case, and must not move."""
        np.testing.assert_array_equal(
            sequence_target_positions([20], 5), np.arange(5, 20)
        )

    def test_each_group_pays_its_own_history(self):
        positions = sequence_target_positions([10, 10], 3)
        np.testing.assert_array_equal(
            positions, np.concatenate([np.arange(3, 10), np.arange(13, 20)])
        )

    def test_a_group_no_longer_than_the_window_contributes_nothing(self):
        np.testing.assert_array_equal(sequence_target_positions([5], 5), np.empty(0))
        np.testing.assert_array_equal(sequence_target_positions([4], 5), np.empty(0))

    def test_short_groups_are_skipped_without_shifting_the_rest(self):
        """The offset must keep advancing past a skipped group.

        If it did not, every later group's positions would point at the wrong
        rows — which is the contamination this replaces, not a fix for it.
        """
        positions = sequence_target_positions([10, 2, 10], 3)
        np.testing.assert_array_equal(
            positions, np.concatenate([np.arange(3, 10), np.arange(15, 22)])
        )

    def test_the_count_is_the_sum_of_what_each_group_can_give(self):
        lengths = [30, 12, 45, 8]
        assert len(sequence_target_positions(lengths, 10)) == sum(
            max(0, n - 10) for n in lengths
        )

    def test_positions_are_ascending(self):
        """Sample i's row must be the i-th predicted row, or the dates that
        `_stacked_dates` derives from the same function name other rows."""
        positions = sequence_target_positions([10, 20, 15], 4)
        assert (np.diff(positions) > 0).all()


class TestGroupLengthsInSlice:
    def test_a_cut_on_a_boundary_keeps_whole_groups(self):
        assert group_lengths_in_slice([10, 10, 10], 10, 20) == [10]

    def test_a_cut_inside_a_group_splits_it(self):
        """The per-ticker 70/15/15 cuts floor independently of the panel-level
        one, so their sum drifts and the cut lands mid-block."""
        assert group_lengths_in_slice([10, 10], 5, 15) == [5, 5]

    def test_empty_overlaps_are_dropped_not_recorded_as_zero(self):
        assert group_lengths_in_slice([10, 10, 10], 0, 10) == [10]

    def test_the_pieces_sum_to_the_slice(self):
        lengths = [7, 13, 5, 21]
        for start, stop in ((0, 20), (5, 40), (12, 46)):
            assert sum(group_lengths_in_slice(lengths, start, stop)) == stop - start


# --------------------------------------------------------------------------
# The guarantee
# --------------------------------------------------------------------------


class TestNoWindowSpansTwoTickers:
    def test_every_sequence_is_internally_contiguous(self):
        features, targets, lengths = stacked(n_tickers=4, n_rows=10)
        dataset = TimeSeriesDataset(features, targets, 4, lengths)

        for i in range(len(dataset)):
            sequence, _ = dataset[i]
            steps = np.diff(sequence.numpy().ravel())
            assert (steps == 1).all(), f"sample {i} spans a boundary"

    def test_without_group_lengths_the_old_contamination_is_reproducible(self):
        """The defect, kept measurable rather than described.

        `group_lengths=None` still means "one series", which is correct for a
        single instrument and is exactly what the stacked panel was passing.
        """
        features, targets, _ = stacked(n_tickers=4, n_rows=10)
        dataset = TimeSeriesDataset(features, targets, 4)

        straddling = 0
        for i in range(len(dataset)):
            sequence, _ = dataset[i]
            if not (np.diff(sequence.numpy().ravel()) == 1).all():
                straddling += 1
        assert straddling > 0

    def test_mismatched_lengths_are_refused(self):
        """Lengths that do not add up would shift every boundary — the same
        contamination, now harder to see."""
        features, targets, _ = stacked(n_tickers=3, n_rows=10)
        with pytest.raises(ValueError, match="does not|do not add up|sums to"):
            TimeSeriesDataset(features, targets, 3, [10, 10])

    def test_a_panel_with_no_window_that_fits_says_what_to_change(self):
        """The shipped default lands here, so the message has to be actionable."""
        features, targets, _ = stacked(n_tickers=5, n_rows=8)
        with pytest.raises(ValueError) as error:
            TimeSeriesDataset(features, targets, 10, [8] * 5)

        message = str(error.value)
        assert "span two instruments" in message
        assert "sequence_length" in message
        assert "min_history_days" in message

    def test_the_single_series_message_is_unchanged(self):
        """One instrument was never the broken case; its error should not
        acquire advice about a panel it is not part of."""
        features = np.arange(20, dtype=float).reshape(10, 2)
        with pytest.raises(ValueError, match="must be less than"):
            TimeSeriesDataset(features, np.arange(10, dtype=float), 10)


# --------------------------------------------------------------------------
# How much was contaminated
# --------------------------------------------------------------------------


class TestTheMeasuredRate:
    @staticmethod
    def _rates(n_tickers, rows, window):
        old = n_tickers * rows - window
        clean = len(sequence_target_positions([rows] * n_tickers, window))
        return old, clean, (old - clean) / old

    def test_roughly_a_fifth_of_a_typical_training_panel(self):
        """50 tickers x 300 rows at the shipped 60-row window."""
        _, _, share = self._rates(50, 300, 60)
        assert 0.19 < share < 0.21

    def test_it_grows_with_the_ticker_count_not_the_row_count(self):
        """`sequence_length x (n_tickers - 1)` windows are lost regardless of
        how long each history is, so a wide shallow panel is the bad case."""
        _, _, wide = self._rates(200, 300, 60)
        _, _, deep = self._rates(50, 1000, 60)
        assert wide > 3 * deep

    def test_the_shipped_default_could_not_build_one_clean_validation_sample(self):
        """The finding that makes this a defect rather than a rounding error.

        `create_dataloaders` splits 70/15/15, and each ticker's 15% share of
        `data.min_history_days=250` is ~37 rows against a
        `training.sequence_length=60` window. Nothing fits. Every validation
        and every test sample the old code produced spanned at least one join —
        so the held-out metrics those loaders reported were computed on
        sequences that no single stock ever experienced.
        """
        from portfolio_agent.config.loader import load_config

        config = load_config()
        window = config.training.sequence_length
        rows = config.data.min_history_days
        _, val_end = chronological_split_bounds(rows)

        per_ticker_val = val_end - chronological_split_bounds(rows)[0]
        per_ticker_test = rows - val_end

        assert per_ticker_val < window
        assert per_ticker_test < window
        assert len(sequence_target_positions([per_ticker_test] * 50, window)) == 0


# --------------------------------------------------------------------------
# The stacked panel refuses to be windowed blind
# --------------------------------------------------------------------------


def test_create_dataloaders_refuses_a_stacked_panel_without_boundaries():
    """A backwards date step is proof the frame is several tickers stacked.

    Caught because the alternative failure is silent: the run trains,
    converges, reports metrics, and nothing says that some fraction of its
    samples were one stock's history labelled with another's forward return.
    """
    from portfolio_agent.config.schema import AppConfig
    from portfolio_agent.data.dataset import create_dataloaders

    frames = blocks(n_tickers=3, n_rows=200)
    panel = pd.concat(frames)

    config = AppConfig().training
    config.sequence_length = 5
    with pytest.raises(ValueError, match="several tickers"):
        create_dataloaders(panel, config)


def test_a_single_ticker_frame_is_still_accepted_without_boundaries():
    """Monotonic dates, so nothing to detect and nothing to refuse."""
    from portfolio_agent.config.schema import AppConfig
    from portfolio_agent.data.dataset import create_dataloaders

    frame = blocks(n_tickers=1, n_rows=300)[0]
    config = AppConfig().training
    config.sequence_length = 5
    config.batch_size = 8

    train, val, test = create_dataloaders(frame, config)
    assert len(train.dataset) > 0
    assert len(val.dataset) > 0
    assert len(test.dataset) > 0
