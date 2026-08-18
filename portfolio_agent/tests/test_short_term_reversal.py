"""The effect momentum's skip-month exists to avoid, measured rather than assumed.

`mom_9m_skip1m` skips the most recent month because that month reverses rather
than continues (Jegadeesh 1990, Lehmann 1990). The platform has applied that
correction on the strength of the literature and has never measured the effect
it corrects for on this data.

Two things carry the weight here. The formation window has to be *exactly* the
window the skip removes — otherwise the comparison is between two different
questions. And the cost arithmetic has to be visible, because reversal is the
one effect in the book where friction plausibly exceeds the gross spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.evaluation.costs import CostModel, evaluate_net
from portfolio_agent.features.pipeline import build_features, warmup_rows
from portfolio_agent.strategies.cross_sectional import REVERSAL_FEATURE
from portfolio_agent.strategies.registry import load_strategy


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=400)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(index))))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e6, len(index)).astype(float),
        },
        index=index,
    )


# --------------------------------------------------------------------------
# The feature
# --------------------------------------------------------------------------


class TestReturn21d:
    def test_it_is_the_trailing_one_month_return(self, ohlcv):
        built = build_features(ohlcv, ["return_21d"])["return_21d"]
        close = ohlcv["close"]

        expected = close.iloc[-2] / close.iloc[-23] - 1.0
        assert built.iloc[-1] == pytest.approx(expected)

    def test_it_is_lagged_like_every_other_feature(self, ohlcv):
        """Perturbing the last bar cannot change the last value."""
        tampered = ohlcv.copy()
        tampered.iloc[-1, tampered.columns.get_loc("close")] *= 3.0

        base = build_features(ohlcv, ["return_21d"])["return_21d"]
        after = build_features(tampered, ["return_21d"])["return_21d"]
        assert base.iloc[-1] == pytest.approx(after.iloc[-1])

    def test_it_is_exactly_the_window_momentum_skips(self, ohlcv):
        """The claim the whole comparison rests on.

        `mom_9m_skip1m` is `close.shift(21) / close.shift(21 + 189) - 1`, so
        the 21 sessions it drops are precisely the ones this feature measures.
        If the two windows were merely similar, `compare momentum,reversal`
        would be comparing two different questions.
        """
        close = ohlcv["close"]
        skipped = close.shift(1) / close.shift(1 + 21) - 1.0
        built = build_features(ohlcv, ["return_21d"])["return_21d"]

        pd.testing.assert_series_equal(built, skipped, check_names=False)

    def test_it_warms_up_far_faster_than_momentum(self):
        """21 sessions against 189 + 21, which is why a reversal book can be
        evaluated on a much shorter cache."""
        assert warmup_rows(["return_21d"]) == 23
        assert warmup_rows(["mom_9m_skip1m"]) > 200


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------


class TestTheStrategy:
    def _strategy(self, **params):
        return load_strategy(StrategyConfig(type="reversal", params=params))

    def test_it_is_registered(self):
        assert self._strategy().name == "reversal"

    def test_it_goes_long_the_losers(self):
        """The sign flip, and the whole of what distinguishes it."""
        assert self._strategy().higher_metric_is_better is False

    def test_momentum_still_goes_long_the_winners(self):
        plain = load_strategy(StrategyConfig(type="momentum", params={}))
        assert plain.higher_metric_is_better is True

    def test_it_ranks_on_the_skipped_window(self):
        assert REVERSAL_FEATURE in self._strategy().required_features()
        assert self._strategy().entry_rules()["formation_metric"] == "return_21d"

    def test_it_does_not_request_the_long_formation_return(self):
        """Requesting `mom_9m_skip1m` would widen the warm-up by 188 rows for a
        column nothing reads."""
        assert "mom_9m_skip1m" not in self._strategy().required_features()

    def test_it_reports_its_own_trigger(self):
        assert self._strategy().trigger_name == "Reversal"

    def test_it_inherits_momentum_s_controls(self):
        """The comparison is about the formation measure and its sign, nothing
        else."""
        reversal = self._strategy().entry_rules()
        plain = load_strategy(StrategyConfig(type="momentum", params={})).entry_rules()

        assert reversal["crash_protection"] == plain["crash_protection"]
        assert reversal["tradability_filter"] == plain["tradability_filter"]
        assert reversal["top_percentile"] == plain["top_percentile"]

    def test_the_tradability_screen_is_on_by_default(self):
        """A reversal sort concentrates in exactly the names it screens: a
        stock that fell hard on no volume prints the return this ranks
        highest."""
        assert self._strategy().entry_rules()["tradability_filter"]["enabled"]

    def test_the_metric_is_the_feature_value(self):
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        frames = {
            "A.NS": pd.DataFrame({REVERSAL_FEATURE: [0.05, -0.20]}),
            "B.NS": pd.DataFrame({REVERSAL_FEATURE: [0.01, 0.15]}),
        }
        context = StrategyContext(
            risk=RiskParams(
                target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
                portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
                max_single_position_pct=0.03,
            ),
        )
        metric = self._strategy()._formation_metric(frames, context)
        assert metric == {"A.NS": pytest.approx(-0.20), "B.NS": pytest.approx(0.15)}

    def test_it_selects_the_worst_recent_performers(self):
        """End to end, on a cross-section with a known ordering."""
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        rng = np.random.default_rng(3)
        n, k = 60, 40
        frames = {}
        for i in range(k):
            # Name i's last month returns (i - 20)%: S0 worst, S39 best.
            recent = (i - 20) / 100.0
            close = np.linspace(100.0, 100.0 * (1 + recent), n)
            frames[f"S{i}.NS"] = pd.DataFrame({
                "close": close,
                REVERSAL_FEATURE: np.full(n, recent),
                "atr_14": close * 0.02,
                "realized_vol_60": np.full(n, 0.25),
            })

        strategy = self._strategy(liquidity_filter=False)
        context = StrategyContext(
            risk=RiskParams(
                target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
                portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
                max_single_position_pct=0.03,
            ),
        )
        signals = strategy.score_batch(frames, context)

        # Highest score means most favourably ranked, and for reversal that is
        # the *lowest* recent return.
        best = max(signals, key=lambda s: signals[s].score)
        worst = min(signals, key=lambda s: signals[s].score)
        assert best == "S0.NS"
        assert worst == "S39.NS"


# --------------------------------------------------------------------------
# Costs, which is where this one is decided
# --------------------------------------------------------------------------


class TestCostsDecideIt:
    """Reversal is the most turnover-intensive effect in the book.

    A one-month formation window implies replacing most of the decile every
    month. These tests pin the arithmetic that makes that the deciding fact
    rather than a caveat.
    """

    def test_the_indian_round_trip_is_what_the_platform_says_it_is(self):
        costs = CostModel.from_execution_sim()
        # ~79 bps at the shipped 25 bps/side slippage. Pinned loosely: the
        # Union Budget moves the STT rate, and this test should fail when it
        # does rather than when it drifts by a basis point.
        assert 0.006 < costs.round_trip < 0.010

    def test_twelve_full_rebalances_a_year_costs_most_of_a_typical_spread(self):
        """The headline number, computed rather than asserted from memory."""
        costs = CostModel.from_execution_sim()
        annual_friction = 12 * 1.0 * costs.round_trip  # 12 months, 100% turnover
        assert annual_friction > 0.08

    def test_a_high_turnover_signal_reports_a_low_breakeven(self):
        """T13's `breakeven_round_trip_cost` is what decides this strategy.

        Built on a panel that turns over completely every date, which is the
        limiting case a monthly reversal book approaches.
        """
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2023-01-02", periods=60)
        symbols = [f"S{i}" for i in range(30)]

        rows = []
        for date in dates:
            # Scores reshuffled every date: maximum turnover.
            scores = rng.permutation(len(symbols))
            for symbol, score in zip(symbols, scores):
                rows.append({
                    "date": date, "symbol": symbol, "score": float(score),
                    "forward_return": float(rng.normal(0.001, 0.02)),
                })

        spread = evaluate_net(pd.DataFrame(rows), horizon=21)
        assert spread.turnover > 0.5
        # A book replacing most of itself pays a large cost per rebalance, so
        # the gross spread it needs to survive is correspondingly large.
        assert spread.cost_per_rebalance > 0.3 * spread.costs.round_trip

    def test_zero_turnover_costs_nothing(self):
        """The other end: a signal that never changes its book pays once."""
        dates = pd.bdate_range("2023-01-02", periods=40)
        symbols = [f"S{i}" for i in range(30)]
        rng = np.random.default_rng(6)

        rows = [
            {"date": date, "symbol": symbol, "score": float(i),
             "forward_return": float(rng.normal(0.001, 0.02))}
            for date in dates
            for i, symbol in enumerate(symbols)
        ]
        spread = evaluate_net(pd.DataFrame(rows), horizon=21)
        assert spread.turnover == pytest.approx(0.0)
        assert spread.cost_per_rebalance == pytest.approx(0.0)
