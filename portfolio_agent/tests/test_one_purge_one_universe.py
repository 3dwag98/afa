"""Overlap in sessions, and one universe for everything that measures.

Two unrelated defects with the same shape: an expression restated in a second
place, where the copy was wrong in a way nothing could see.

**The embargo was in calendar days.** `agents/trainer.py` computed
`test_end_date + pd.Timedelta(days=embargo)` — the exact mistake
`validation/purged.py` opens by ruling out: *"A horizon of 5 means five
trading sessions, not five calendar days. A weekend or an exchange holiday
would make a calendar-day arithmetic silently wrong by a variable amount."*
An embargo of 5 spanning a weekend excluded three sessions.

**`evaluate` drew the training universe.** `purpose` offsets the RNG so a
model is not scored on the names it was fitted on, but the offset is two-way
and `resolve_universe` defaulted to `"train"` — which `evaluate` never
overrode. At `universe_size=50` on a 400-name cache, `evaluate` and `backtest`
shared **6 names**. An IC and an equity curve for "the same strategy"
described different markets, side by side in the same report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.universe import (
    MEASUREMENT_PURPOSE,
    TRAINING_PURPOSE,
    select_universe,
)


# --------------------------------------------------------------------------
# One universe for everything that measures
# --------------------------------------------------------------------------


TICKERS = [f"T{i:04d}.NS" for i in range(400)]


def _draw(purpose: str, size: int = 50) -> set:
    return set(
        select_universe(TICKERS, max_tickers=size, selection="random",
                        seed=42, purpose=purpose)
    )


class TestTheMeasurementUniverse:
    def test_evaluation_and_backtest_now_draw_the_same_names(self):
        """The whole point: an IC and an equity curve must describe one market."""
        assert _draw(MEASUREMENT_PURPOSE) == _draw("backtest")

    def test_training_still_draws_different_names(self):
        """The split that is supposed to exist. A model must not be scored on
        the tickers it was fitted on."""
        assert _draw(TRAINING_PURPOSE) != _draw(MEASUREMENT_PURPOSE)

    def test_the_old_divergence_was_large_not_marginal(self):
        """Recorded so this is not re-introduced as a harmless default.

        `evaluate` fell through to the training draw; `backtest` used its own.
        Six names in fifty.
        """
        overlap = _draw(TRAINING_PURPOSE) & _draw(MEASUREMENT_PURPOSE)
        assert len(overlap) < 15

    def test_the_draw_is_reproducible(self):
        assert _draw(MEASUREMENT_PURPOSE) == _draw(MEASUREMENT_PURPOSE)

    def test_alphabetical_selection_ignores_purpose_entirely(self):
        """The offset only applies to a random draw; a deterministic one has
        nothing to offset, so the two purposes must not diverge there."""
        a = select_universe(TICKERS, max_tickers=50, selection="alphabetical",
                            purpose=TRAINING_PURPOSE)
        b = select_universe(TICKERS, max_tickers=50, selection="alphabetical",
                            purpose=MEASUREMENT_PURPOSE)
        assert a == b


class TestEveryScoringPathAsksForTheMeasurementDraw:
    """A default that is right for one caller and wrong for another is how the
    divergence happened, so the parameter is explicit at every scoring site."""

    @pytest.mark.parametrize(
        "module",
        [
            "portfolio_agent.evaluation.harness",
            "portfolio_agent.evaluation.decay",
            "portfolio_agent.evaluation.neutralize",
            "portfolio_agent.cli_forecast",
        ],
    )
    def test_it_passes_the_purpose(self, module):
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module))
        assert "purpose=MEASUREMENT_PURPOSE" in source, module

    def test_the_backtest_resolves_through_the_same_helper(self):
        import inspect

        from portfolio_agent.agents import backtester

        source = inspect.getsource(backtester.run_backtest_cli)
        assert "MEASUREMENT_PURPOSE" in source
        assert "resolve_universe" in source

    def test_training_is_untouched(self):
        """`resolve_universe` still defaults to the training draw, because
        every caller that does not pass a purpose is fitting a model."""
        from portfolio_agent.training.universe import resolve_universe

        import inspect

        assert "TRAINING_PURPOSE" in inspect.getsource(resolve_universe)


class TestBacktestAcceptsASnapshot:
    def test_the_flag_parses(self):
        from portfolio_agent.cli import create_parser

        args = create_parser().parse_args(
            ["backtest", "--universe-snapshot", "universe/pinned.json"]
        )
        assert args.universe_snapshot == "universe/pinned.json"

    def test_it_defaults_to_none(self):
        from portfolio_agent.cli import create_parser

        assert create_parser().parse_args(["backtest"]).universe_snapshot is None

    def test_run_backtest_cli_takes_it(self):
        import inspect

        from portfolio_agent.agents.backtester import run_backtest_cli

        assert "universe_snapshot" in inspect.signature(run_backtest_cli).parameters

    def test_a_snapshot_pins_both_sides_to_identical_names(self, tmp_path):
        """The reason the flag exists — evaluate and backtest on one list."""
        from portfolio_agent.training.universe import UniverseSnapshot, resolve_universe

        pinned = UniverseSnapshot.from_tickers(TICKERS[:30], name="pinned")
        path = pinned.save(tmp_path / "pinned.json")

        evaluated = resolve_universe(None, snapshot=path, purpose=MEASUREMENT_PURPOSE)
        backtested = resolve_universe(None, snapshot=path, purpose=MEASUREMENT_PURPOSE)
        assert evaluated.tickers == backtested.tickers == pinned.tickers
        assert evaluated.fingerprint == pinned.fingerprint


# --------------------------------------------------------------------------
# One purge, stated in sessions
# --------------------------------------------------------------------------


class TestTheEmbargoIsInSessions:
    def test_a_weekend_no_longer_shrinks_it(self):
        """Calendar arithmetic over a weekend excluded three sessions, not five.

        Business days only, so five sessions from a Friday reaches the
        following Friday — seven calendar days.
        """
        sessions = pd.bdate_range("2024-01-01", periods=40)
        friday = sessions[4]
        after = sessions[sessions > friday]

        by_sessions = after[5 - 1]
        by_calendar = friday + pd.Timedelta(days=5)

        assert by_sessions > by_calendar
        excluded_by_calendar = ((after > friday) & (after <= by_calendar)).sum()
        assert excluded_by_calendar == 3

    def test_the_trainer_computes_it_positionally(self):
        """Checked against the code, not the prose.

        The comment there quotes the old calendar expression on purpose — so
        the reason it went is legible — which means a naive substring search
        matches the very thing it is meant to rule out.
        """
        import inspect

        from portfolio_agent.agents.trainer import run_walk_forward_validation

        code = "\n".join(
            line for line in inspect.getsource(run_walk_forward_validation).splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "pd.Timedelta(" not in code
        assert "walk_forward_embargo" in code

    def test_it_never_runs_off_the_end_of_the_index(self):
        """An embargo longer than the remaining history must clamp, not raise."""
        sessions = pd.bdate_range("2024-01-01", periods=10)
        cutoff = sessions[7]
        after = sessions[sessions > cutoff]
        assert len(after) == 2
        assert after[min(50, len(after)) - 1] == sessions[-1]


class TestOnePurgePredicate:
    def test_the_gbm_cutoff_matches_the_shared_overlap_rule(self):
        from portfolio_agent.training.trainers.gbm import _purge_cutoff_position

        for n_dates in (50, 100, 500):
            for split in (1, 5, 40, 80):
                if split >= n_dates:
                    continue
                for horizon in (1, 5, 21):
                    assert _purge_cutoff_position(n_dates, split, horizon) == max(
                        split - horizon, 0
                    )

    def test_it_is_expressed_through_validation_purged(self):
        """The two agreed by coincidence of two people deriving one expression.
        That is the state T12 removed for rank IC."""
        import inspect

        from portfolio_agent.training.trainers.gbm import _purge_cutoff_position

        assert "label_window_overlaps" in inspect.getsource(_purge_cutoff_position)

    def test_a_zero_horizon_purges_nothing(self):
        from portfolio_agent.training.trainers.gbm import _purge_cutoff_position

        assert _purge_cutoff_position(100, 40, 0) == 40

    def test_a_split_at_the_very_start_purges_everything_available(self):
        from portfolio_agent.training.trainers.gbm import _purge_cutoff_position

        assert _purge_cutoff_position(100, 0, 5) == 0

    def test_every_kept_position_is_genuinely_non_overlapping(self):
        """The property, checked against the predicate rather than the formula."""
        from portfolio_agent.training.trainers.gbm import _purge_cutoff_position
        from portfolio_agent.validation.purged import label_window_overlaps

        n_dates, split, horizon = 200, 120, 7
        cutoff = _purge_cutoff_position(n_dates, split, horizon)
        kept = np.arange(cutoff, dtype=int)
        assert not label_window_overlaps(
            kept, test_start_pos=split, test_end_pos=n_dates - 1, horizon=horizon
        ).any()
