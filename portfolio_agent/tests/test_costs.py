"""Costs charged against a forecast, and the one place they must *not* be.

The evaluation layer reported gross spreads while an accurate Indian cost model
sat unused in `src/execution_sim.py`. Everything here is about closing that gap
without inventing precision that is not there — in particular, without
reporting a "net IC", which would be the gross one copied.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.evaluation.costs import (
    TRADING_DAYS_PER_YEAR,
    CostModel,
    bucket_membership,
    cost_notes,
    count_rebalances,
    evaluate_net,
    one_way_turnover,
)
from portfolio_agent.evaluation.metrics import rank_ic_series


def panel_from(scores_by_date, returns_by_date) -> pd.DataFrame:
    rows = []
    for date in scores_by_date:
        for symbol, score in scores_by_date[date].items():
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "score": score,
                    "forward_return": returns_by_date[date][symbol],
                }
            )
    return pd.DataFrame(rows)


def stable_panel(n_dates=60, n_names=40, seed=0, noise=0.5, edge=0.0004):
    """A signal whose ordering barely changes from date to date."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
        for i in range(n_names):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{i:02d}",
                    "score": i + rng.normal(0, noise),
                    "forward_return": edge * i + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The cost model is read, not restated
# --------------------------------------------------------------------------


class TestCostModel:
    def test_the_round_trip_is_about_eighty_basis_points(self):
        """The figure `execution_sim` documents, reached from the eval layer."""
        model = CostModel.from_execution_sim()
        assert 0.007 < model.round_trip < 0.009

    def test_stt_is_charged_on_both_legs(self):
        """0.1% each way on delivery is the single largest statutory component.

        A cost model that charged it once would understate a round trip by
        10 bps, which is a quarter of some signals' entire edge.
        """
        from portfolio_agent.src.execution_sim import ExecutionSimulator

        zero_slippage = CostModel.from_execution_sim(slippage_per_side=0.0)
        assert zero_slippage.buy > ExecutionSimulator.STT_RATE
        assert zero_slippage.sell > ExecutionSimulator.STT_RATE
        assert zero_slippage.round_trip > 2 * ExecutionSimulator.STT_RATE

    def test_the_buy_leg_costs_more_because_of_stamp_duty(self):
        model = CostModel.from_execution_sim(slippage_per_side=0.0)
        from portfolio_agent.src.execution_sim import ExecutionSimulator

        assert model.buy - model.sell == pytest.approx(
            ExecutionSimulator.STAMP_DUTY_RATE
        )

    def test_it_tracks_execution_sim_rather_than_copying_it(self):
        """Change the statutory rate and the evaluation layer must follow.

        Two copies of the STT rate is one copy that stops matching the day the
        Union Budget moves it — the same argument T12 made about rank IC.
        """
        from portfolio_agent.src.execution_sim import ExecutionSimulator

        class DoubledSTT(ExecutionSimulator):
            STT_RATE = ExecutionSimulator.STT_RATE * 2

        base = CostModel.from_execution_sim(slippage_per_side=0.0)
        from portfolio_agent.src.execution_sim import cost_fraction_per_side

        doubled = cost_fraction_per_side("BUY", 0.0, DoubledSTT)
        assert doubled - base.buy == pytest.approx(ExecutionSimulator.STT_RATE)

    def test_slippage_is_configurable_and_additive_per_side(self):
        tight = CostModel.from_execution_sim(slippage_per_side=0.0)
        wide = CostModel.from_execution_sim(slippage_per_side=0.005)
        assert wide.round_trip - tight.round_trip == pytest.approx(0.01)

    def test_negative_slippage_is_refused(self):
        with pytest.raises(ValueError, match="not be negative"):
            CostModel.from_execution_sim(slippage_per_side=-0.001)


# --------------------------------------------------------------------------
# The thing costs cannot do
# --------------------------------------------------------------------------


class TestCostsDoNotMoveTheRanks:
    def test_a_uniform_cost_leaves_rank_ic_bit_identical(self):
        """Why there is no `net_ic` column.

        Spearman is invariant under any monotone transform of either side, and
        subtracting a constant is monotone. A "net IC" would be the gross
        number with a different label — the most expensive kind of metric,
        because it looks like corroboration.
        """
        panel = stable_panel()
        gross = rank_ic_series(panel)

        netted = panel.copy()
        netted["forward_return"] = netted["forward_return"] - 0.008

        pd.testing.assert_series_equal(gross, rank_ic_series(netted))

    def test_a_cost_that_differs_by_name_does_move_them(self):
        """The case that would justify a net IC, kept measurable.

        Slippage scales with illiquidity, so a cost proportional to each name's
        own spread is not a constant and the ranks can move. Nothing reports
        this today; the test records that the distinction is real rather than
        pedantic.
        """
        panel = stable_panel(seed=3)
        per_name = panel["symbol"].map(lambda s: 0.02 if s < "S20" else 0.0)

        netted = panel.copy()
        netted["forward_return"] = netted["forward_return"] - per_name.to_numpy()

        assert rank_ic_series(panel).mean() != pytest.approx(
            rank_ic_series(netted).mean()
        )

    def test_the_notes_say_so_rather_than_leaving_it_implied(self):
        result = evaluate_net(stable_panel(), horizon=5)
        assert any("unchanged by costs" in note for note in cost_notes(result))


# --------------------------------------------------------------------------
# Turnover, measured
# --------------------------------------------------------------------------


class TestTurnover:
    def test_a_frozen_signal_turns_over_nothing(self):
        """Same ordering every date, so the same names in the top decile."""
        rows = []
        for t in range(20):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
            for i in range(30):
                rows.append(
                    {"date": date, "symbol": f"S{i:02d}", "score": float(i),
                     "forward_return": 0.001}
                )
        assert one_way_turnover(pd.DataFrame(rows)) == 0.0

    def test_a_random_signal_turns_over_almost_everything(self):
        rng = np.random.default_rng(11)
        rows = []
        for t in range(40):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
            for i in range(50):
                rows.append(
                    {"date": date, "symbol": f"S{i:02d}",
                     "score": rng.normal(), "forward_return": rng.normal(0, 0.01)}
                )
        assert one_way_turnover(pd.DataFrame(rows)) > 0.8

    def test_a_slow_signal_lands_between_the_two(self):
        turnover = one_way_turnover(stable_panel(noise=2.0, seed=5))
        assert 0.0 < turnover < 0.8

    def test_turnover_is_measured_over_the_top_bucket_by_default(self):
        panel = stable_panel()
        top = bucket_membership(panel, n_buckets=10)
        explicit = bucket_membership(panel, n_buckets=10, bucket=9)
        assert top == explicit

    def test_a_different_bucket_can_be_tracked(self):
        panel = stable_panel()
        bottom = bucket_membership(panel, n_buckets=10, bucket=0)
        top = bucket_membership(panel, n_buckets=10, bucket=9)
        first = sorted(bottom)[0]
        assert bottom[first] != top[first]

    def test_an_out_of_range_bucket_raises(self):
        with pytest.raises(ValueError, match="outside"):
            bucket_membership(stable_panel(), n_buckets=10, bucket=10)

    def test_thin_dates_are_not_bucketed(self):
        """Eight names in ten deciles makes membership an artifact."""
        rows = [
            {"date": pd.Timestamp("2024-01-01"), "symbol": f"S{i}",
             "score": float(i), "forward_return": 0.01}
            for i in range(8)
        ]
        assert bucket_membership(pd.DataFrame(rows), n_buckets=10) == {}

    def test_a_single_rebalance_reports_none_rather_than_zero(self):
        """Zero turnover and unmeasured turnover are different claims."""
        rows = [
            {"date": pd.Timestamp("2024-01-01"), "symbol": f"S{i:02d}",
             "score": float(i), "forward_return": 0.01 - 0.0001 * i}
            for i in range(30)
        ]
        panel = pd.DataFrame(rows)
        assert count_rebalances(panel) == 0
        result = evaluate_net(panel, horizon=5)
        assert result.n_rebalances == 0
        assert any("not measured" in note for note in cost_notes(result))

    def test_rebalance_every_compares_against_a_further_back_book(self):
        panel = stable_panel(noise=2.0, seed=9)
        daily = one_way_turnover(panel, rebalance_every=1)
        monthly = one_way_turnover(panel, rebalance_every=21)
        # Rebalancing less often turns over more per rebalance: the book has
        # had longer to drift.
        assert monthly >= daily

    def test_rebalance_every_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            one_way_turnover(stable_panel(), rebalance_every=0)


# --------------------------------------------------------------------------
# The net spread
# --------------------------------------------------------------------------


class TestNetSpread:
    def test_the_long_short_leg_pays_two_round_trips(self):
        """Both books turn over: the long sells what leaves the top decile,
        the short covers what leaves the bottom."""
        result = evaluate_net(stable_panel(noise=2.0, seed=13), horizon=5)
        assert result.gross - result.net == pytest.approx(
            2 * result.cost_per_rebalance
        )

    def test_the_long_only_leg_pays_one(self):
        result = evaluate_net(stable_panel(noise=2.0, seed=14), horizon=5)
        assert result.long_only_gross - result.long_only_net == pytest.approx(
            result.cost_per_rebalance
        )

    def test_cost_per_rebalance_is_turnover_times_the_round_trip(self):
        result = evaluate_net(stable_panel(noise=2.0, seed=15), horizon=5)
        assert result.cost_per_rebalance == pytest.approx(
            result.turnover * result.costs.round_trip
        )

    def test_at_the_breakeven_cost_the_net_spread_is_zero(self):
        panel = stable_panel(noise=2.0, seed=16)
        result = evaluate_net(panel, horizon=5)
        assert math.isfinite(result.breakeven_cost)

        at_breakeven = evaluate_net(
            panel,
            horizon=5,
            costs=CostModel(
                buy=result.breakeven_cost / 2,
                sell=result.breakeven_cost / 2,
                slippage_per_side=0.0,
            ),
        )
        assert at_breakeven.net == pytest.approx(0.0, abs=1e-12)

    def test_a_signal_with_no_turnover_has_an_infinite_breakeven(self):
        """It never trades, so no cost level can stop it."""
        rows = []
        for t in range(20):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
            for i in range(30):
                rows.append(
                    {"date": date, "symbol": f"S{i:02d}", "score": float(i),
                     "forward_return": 0.001 * i}
                )
        result = evaluate_net(pd.DataFrame(rows), horizon=5)
        assert result.breakeven_cost == math.inf
        assert result.net == pytest.approx(result.gross)

    def test_a_fast_signal_with_a_thin_edge_does_not_survive(self):
        """The case the whole task exists for.

        High turnover against a small gross spread is the shape that looks like
        alpha gross and is a fee generator net.
        """
        rng = np.random.default_rng(21)
        rows = []
        for t in range(60):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
            order = rng.permutation(50)
            for i in range(50):
                rows.append(
                    {"date": date, "symbol": f"S{i:02d}",
                     "score": float(order[i]),
                     "forward_return": 0.00002 * order[i] + rng.normal(0, 0.001)}
                )
        result = evaluate_net(pd.DataFrame(rows), horizon=5)

        assert result.gross > 0
        assert result.turnover > 0.8
        assert not result.survives
        assert result.cost_share > 1.0

    def test_cost_share_is_nan_against_a_non_positive_gross_spread(self):
        """A ratio whose denominator is negative has no readable sign."""
        rng = np.random.default_rng(23)
        rows = []
        for t in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
            for i in range(30):
                rows.append(
                    {"date": date, "symbol": f"S{i:02d}", "score": float(i),
                     "forward_return": -0.0005 * i + rng.normal(0, 0.001)}
                )
        result = evaluate_net(pd.DataFrame(rows), horizon=5)
        assert result.gross < 0
        assert math.isnan(result.cost_share)

    def test_annualization_follows_the_stride_not_the_horizon(self):
        """Evaluating a 21-day label daily does not mean rebalancing daily."""
        panel = stable_panel(noise=2.0, seed=17)
        daily = evaluate_net(panel, horizon=21, stride=1)
        weekly = evaluate_net(panel, horizon=21, stride=5)

        assert daily.periods_per_year == pytest.approx(TRADING_DAYS_PER_YEAR)
        assert weekly.periods_per_year == pytest.approx(TRADING_DAYS_PER_YEAR / 5)

    def test_the_benchmark_is_deducted_from_the_long_only_leg(self):
        panel = stable_panel(noise=2.0, seed=18)
        absolute = evaluate_net(panel, horizon=5)
        relative = evaluate_net(panel, horizon=5, benchmark_return=0.001)
        assert absolute.long_only_gross - relative.long_only_gross == pytest.approx(
            0.001
        )

    def test_to_dict_is_flat_and_carries_the_verdict(self):
        result = evaluate_net(stable_panel(noise=2.0, seed=19), horizon=5)
        document = result.to_dict()
        for key in ("spread_gross", "spread_net", "turnover_one_way",
                    "cost_round_trip_pct", "breakeven_round_trip_cost",
                    "survives_costs", "long_only_survives_costs"):
            assert key in document
        assert all(not isinstance(v, (list, dict)) for v in document.values())


# --------------------------------------------------------------------------
# Wired into the harness
# --------------------------------------------------------------------------


class TestHarnessIntegration:
    def test_costs_are_charged_by_default(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        result = evaluate_panel(stable_panel(noise=2.0, seed=20), horizon=5)
        assert result.costs is not None
        assert result.costs.cost_per_rebalance > 0

    def test_they_can_be_switched_off(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        result = evaluate_panel(
            stable_panel(noise=2.0, seed=20), horizon=5, charge_costs=False
        )
        assert result.costs is None

    def test_none_reads_differently_from_a_zero_cost(self):
        """'Not charged' and 'charged nothing' are different claims."""
        from portfolio_agent.evaluation.harness import evaluate_panel

        panel = stable_panel(noise=2.0, seed=20)
        free = evaluate_panel(panel, horizon=5, slippage_per_side=0.0)
        assert free.costs is not None
        assert free.costs.costs.slippage_per_side == 0.0
        assert free.costs.cost_per_rebalance > 0  # statutory charges remain

    def test_the_report_shows_the_net_section(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        rendered = evaluate_panel(
            stable_panel(noise=2.0, seed=20), horizon=5
        ).render()
        assert "Net of costs" in rendered
        assert "breakeven cost" in rendered
        assert "turnover" in rendered

    def test_the_report_omits_it_when_not_charged(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        rendered = evaluate_panel(
            stable_panel(noise=2.0, seed=20), horizon=5, charge_costs=False
        ).render()
        assert "Net of costs" not in rendered

    def test_the_manifest_dict_carries_the_cost_keys(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        document = evaluate_panel(
            stable_panel(noise=2.0, seed=20), horizon=5
        ).to_dict()
        assert document["cost_round_trip_pct"] > 0
        assert "spread_net" in document

    def test_to_frame_stays_one_flat_row(self):
        from portfolio_agent.evaluation.harness import evaluate_panel

        frame = evaluate_panel(
            stable_panel(noise=2.0, seed=20), horizon=5
        ).to_frame()
        assert len(frame) == 1
        assert "spread_net" in frame.columns

    def test_the_ic_is_the_same_whether_or_not_costs_are_charged(self):
        """The invariance, asserted end to end rather than only on the panel."""
        from portfolio_agent.evaluation.harness import evaluate_panel

        panel = stable_panel(noise=2.0, seed=20)
        charged = evaluate_panel(panel, horizon=5)
        gross = evaluate_panel(panel, horizon=5, charge_costs=False)
        assert charged.ic.mean == pytest.approx(gross.ic.mean)


# --------------------------------------------------------------------------
# The CLI surface
# --------------------------------------------------------------------------


class TestCliFlags:
    def _parser(self):
        from portfolio_agent.cli import create_parser

        return create_parser()

    def test_evaluate_accepts_gross_and_slippage(self):
        args = self._parser().parse_args(
            ["evaluate", "--strategy", "momentum", "--gross", "--slippage-bps", "40"]
        )
        assert args.gross is True
        assert args.slippage_bps == 40.0

    def test_net_is_the_default(self):
        args = self._parser().parse_args(["evaluate", "--strategy", "momentum"])
        assert args.gross is False
        assert args.slippage_bps is None
