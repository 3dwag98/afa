"""Long the low-beta decile, and the split that says when it pays.

Frazzini & Pedersen: investors who want more risk than they can borrow to
obtain bid up high-beta stocks instead, so beta is overpriced and the security
market line is flatter than CAPM says. NSE 2001-2016 finds the effect positive
across capitalizations after controlling for size, value and momentum.

The part worth testing hardest is not the ranking — that is a sort on a feature
T24 already registered and tested. It is the **conditional** measurement: 2025
Asian work finds the effect concentrated in downturns, and a pooled IC made of
a strong down-market number and a flat up-market one describes neither state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.evaluation.conditional import (
    DOWN,
    UP,
    ConditionalIC,
    conditional_ic,
    conditional_notes,
    market_return_by_date,
    realized_states,
    trailing_states,
)
from portfolio_agent.strategies.registry import load_strategy


def _panel(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def beta_world():
    """A low-beta signal: it pays when the market falls and loses when it rises.

    Score is inverted beta, so a *high* score means a *low* beta — the shape
    the strategy emits. The forward return is beta times the market plus noise,
    which makes the conditionality exact rather than approximate: low-beta
    names must outperform on down days and underperform on up days.
    """
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2022-01-03", periods=300)
    symbols = [f"S{i}" for i in range(30)]
    market = pd.Series(rng.normal(0.0002, 0.012, len(dates)), index=dates)

    rows = []
    for date in dates:
        raw = rng.uniform(0, 100, len(symbols))
        betas = 0.5 + 1.5 * (raw / 100.0)
        forward = betas * market[date] + rng.normal(0, 0.010, len(symbols))
        for symbol, value, ret in zip(symbols, raw, forward):
            rows.append({
                "date": date, "symbol": symbol,
                "score": float(100.0 - value),   # high score == low beta
                "forward_return": float(ret),
            })
    return _panel(rows), market


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


class TestMarketStates:
    def test_the_market_proxy_is_the_cross_sectional_mean(self, beta_world):
        panel, _ = beta_world
        proxy = market_return_by_date(panel)
        one = panel[panel["date"] == panel["date"].iloc[0]]

        assert proxy.iloc[0] == pytest.approx(one["forward_return"].mean())

    def test_realized_states_split_on_the_sign(self, beta_world):
        panel, _ = beta_world
        proxy = market_return_by_date(panel)
        states = realized_states(panel)

        assert (states[proxy >= 0] == UP).all()
        assert (states[proxy < 0] == DOWN).all()

    def test_both_states_occur(self, beta_world):
        panel, _ = beta_world
        counts = realized_states(panel).value_counts()
        assert counts[UP] > 20 and counts[DOWN] > 20

    def test_trailing_states_use_only_the_past(self, beta_world):
        """The window ends at the decision date, matching T19."""
        panel, market = beta_world
        dates = sorted(panel["date"].unique())

        tampered = market.copy()
        tampered.loc[tampered.index > dates[200]] *= -5.0

        base = trailing_states(market, dates[:200])
        after = trailing_states(tampered, dates[:200])
        pd.testing.assert_series_equal(base, after)

    def test_trailing_needs_a_full_window(self, beta_world):
        """A partial window would label early dates from less history than the
        later ones, which is two conditioners under one name."""
        panel, market = beta_world
        dates = sorted(panel["date"].unique())
        states = trailing_states(market, dates, window=63)

        assert len(states) < len(dates)


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


class TestConditionalIC:
    def test_it_separates_the_two_states(self, beta_world):
        panel, _ = beta_world
        result = conditional_ic(panel, horizon=1)

        assert set(result.by_state) == {UP, DOWN}
        assert result.n_dates[UP] + result.n_dates[DOWN] == panel["date"].nunique()

    def test_a_low_beta_signal_pays_when_the_market_falls(self, beta_world):
        panel, _ = beta_world
        result = conditional_ic(panel, horizon=1)

        assert result.by_state[DOWN].mean > 0
        assert result.by_state[UP].mean < 0
        assert result.gap > 0

    def test_the_pooled_number_would_describe_neither_state(self, beta_world):
        """The claim the whole module exists to make, measured.

        Pooling a strong positive and a strong negative gives roughly zero, and
        "no skill" is the one description that is wrong about both halves.
        """
        from portfolio_agent.evaluation.metrics import rank_ic_series, summarize_ic

        panel, _ = beta_world
        pooled = summarize_ic(rank_ic_series(panel), horizon=1)
        split = conditional_ic(panel, horizon=1)

        assert abs(pooled.mean) < abs(split.by_state[DOWN].mean)
        assert abs(pooled.mean) < abs(split.by_state[UP].mean)

    def test_opposite_signs_in_both_states_counts_as_conditional(self, beta_world):
        """The case my first `is_conditional` missed.

        It tested "significant in one state and not the other", so a signal
        significant in *both* with opposite signs — the strongest form of state
        dependence, and the easiest to miss because both halves look healthy —
        was reported as unconditional.
        """
        result = conditional_ic(beta_world[0], horizon=1)

        assert result.signs_disagree
        assert result.by_state[UP].significant
        assert result.by_state[DOWN].significant
        assert result.is_conditional

    def test_a_state_independent_signal_is_not_flagged(self):
        """A signal with the same skill in both states must not trip the flag."""
        rng = np.random.default_rng(9)
        dates = pd.bdate_range("2022-01-03", periods=250)
        symbols = [f"S{i}" for i in range(30)]

        rows = []
        for date in dates:
            market = rng.normal(0.0002, 0.012)
            scores = rng.uniform(0, 100, len(symbols))
            # Skill unrelated to the market: the score predicts the *residual*.
            forward = market + (scores / 100.0 - 0.5) * 0.01 + rng.normal(
                0, 0.004, len(symbols)
            )
            for symbol, score, ret in zip(symbols, scores, forward):
                rows.append({
                    "date": date, "symbol": symbol,
                    "score": float(score), "forward_return": float(ret),
                })

        result = conditional_ic(_panel(rows), horizon=1)
        assert not result.signs_disagree
        assert result.by_state[UP].mean > 0 and result.by_state[DOWN].mean > 0

    def test_the_gap_is_signed_toward_down_markets(self, beta_world):
        """Positive means "works better when the market falls" — the direction
        the low-risk anomaly is claimed to have."""
        result = conditional_ic(beta_world[0], horizon=1)
        assert result.gap == pytest.approx(
            result.by_state[DOWN].mean - result.by_state[UP].mean
        )

    def test_it_reports_the_spread_per_state_too(self, beta_world):
        result = conditional_ic(beta_world[0], horizon=1)
        assert set(result.buckets) == {UP, DOWN}

    def test_an_empty_panel_says_so_rather_than_raising(self):
        empty = pd.DataFrame(columns=["date", "symbol", "score", "forward_return"])
        result = conditional_ic(empty, horizon=5)
        assert result.by_state == {}
        assert any("empty" in note for note in result.notes)


class TestTheConditionerIsExplicit:
    def test_realized_says_it_is_not_tradable(self, beta_world):
        result = conditional_ic(beta_world[0], horizon=1)
        assert any("not" in n and "tradable" in n for n in result.notes)

    def test_trailing_needs_a_market_series(self, beta_world):
        """Falling back to `realized` would silently answer a different
        question — attribution instead of timing."""
        panel, _ = beta_world
        with pytest.raises(ValueError, match="needs a `market` return series"):
            conditional_ic(panel, horizon=1, conditioner="trailing")

    def test_trailing_works_when_given_one(self, beta_world):
        panel, market = beta_world
        result = conditional_ic(
            panel, horizon=1, conditioner="trailing", market=market
        )
        assert result.conditioner == "trailing"
        assert set(result.by_state) <= {UP, DOWN}

    def test_an_unknown_conditioner_lists_the_known_ones(self, beta_world):
        with pytest.raises(ValueError, match="Unknown conditioner"):
            conditional_ic(beta_world[0], horizon=1, conditioner="vix")

    def test_the_conditioner_travels_into_the_result(self, beta_world):
        result = conditional_ic(beta_world[0], horizon=1)
        assert result.to_dict()["conditioner"] == "realized"

    def test_a_thin_state_is_flagged(self):
        """A Newey-West t on a dozen dates is not evidence, and the note says so."""
        rng = np.random.default_rng(2)
        dates = pd.bdate_range("2023-01-02", periods=40)
        rows = []
        for i, date in enumerate(dates):
            # Almost every date is an up day.
            drift = 0.02 if i % 20 else -0.02
            for j in range(10):
                rows.append({
                    "date": date, "symbol": f"S{j}",
                    "score": float(rng.uniform(0, 100)),
                    "forward_return": float(drift + rng.normal(0, 0.001)),
                })
        result = conditional_ic(_panel(rows), horizon=1, min_dates_per_state=20)
        assert any("not evidence" in note for note in result.notes)


class TestConditionalNotes:
    def test_it_states_both_states(self, beta_world):
        lines = conditional_notes(conditional_ic(beta_world[0], horizon=1))
        joined = " ".join(lines)
        assert "falling markets" in joined and "rising markets" in joined

    def test_it_names_the_sign_flip(self, beta_world):
        lines = conditional_notes(conditional_ic(beta_world[0], horizon=1))
        assert any("opposite ways" in line for line in lines)

    def test_it_refuses_to_call_the_flag_a_p_value(self, beta_world):
        """Comparing two t-statistics is not a test of their difference."""
        lines = conditional_notes(conditional_ic(beta_world[0], horizon=1))
        assert any("not a p-value" in line for line in lines)


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------


class TestTheStrategy:
    def _strategy(self, **params):
        return load_strategy(StrategyConfig(type="bab", params=params))

    def test_it_is_registered(self):
        assert self._strategy().name == "bab"

    def test_it_sorts_low_beta_first(self):
        """The one thing that distinguishes it from momentum's machinery."""
        assert self._strategy().higher_metric_is_better is False

    def test_momentum_still_sorts_the_other_way(self):
        plain = load_strategy(StrategyConfig(type="momentum", params={}))
        assert plain.higher_metric_is_better is True

    def test_it_ranks_on_a_registered_beta_feature(self):
        assert self._strategy().required_cross_sectional_features() == [
            "market_beta_252"
        ]

    def test_the_window_is_configurable_and_reaches_the_feature(self):
        assert self._strategy(beta_window=60).required_cross_sectional_features() == [
            "market_beta_60"
        ]

    def test_an_unregistered_window_fails_at_construction(self):
        """Not on the first scored date, and not by rounding to a neighbour."""
        with pytest.raises(ValueError, match="No 'market_beta' feature"):
            self._strategy(beta_window=37)

    def test_it_reports_its_own_trigger(self):
        assert self._strategy().trigger_name == "BettingAgainstBeta"

    def test_it_inherits_momentum_s_controls(self):
        bab = self._strategy().entry_rules()
        plain = load_strategy(StrategyConfig(type="momentum", params={})).entry_rules()

        assert bab["crash_protection"] == plain["crash_protection"]
        assert bab["tradability_filter"] == plain["tradability_filter"]

    def test_its_rule_says_bottom_decile(self):
        assert "bottom decile" in self._strategy().entry_rules()["rule"]

    def test_a_universe_of_one_scores_nothing(self):
        """A beta is measured against a cross-section."""
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        rng = np.random.default_rng(1)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        one = {"S0": pd.DataFrame({"close": close})}
        context = StrategyContext(
            risk=RiskParams(
                target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
                portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
                max_single_position_pct=0.03,
            ),
        )
        assert self._strategy()._formation_metric(one, context) == {}

    def test_it_selects_the_lowest_beta_names(self):
        """End to end: build a cross-section with known betas and check which
        names the strategy actually goes long."""
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        rng = np.random.default_rng(6)
        n, k = 500, 40
        market = rng.normal(0.0004, 0.011, n)
        betas = np.linspace(0.3, 2.0, k)

        frames = {}
        for i in range(k):
            returns = betas[i] * market + rng.normal(0, 0.008, n)
            close = 100 * (1 + returns).cumprod()
            frames[f"S{i}.NS"] = pd.DataFrame({
                "close": close,
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
        metric = strategy._formation_metric(frames, context)
        ranked = sorted(metric, key=metric.get)

        # The four lowest estimated betas should come from the low-beta end of
        # the construction. Estimates, so this is about the ordering holding
        # broadly rather than exactly.
        low_indices = [int(name[1:].split(".")[0]) for name in ranked[:4]]
        assert max(low_indices) < k // 2
