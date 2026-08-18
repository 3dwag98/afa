"""Pairs, and the two ways a pairs backtest lies.

`QUANT_RESEARCH.md` §7 scoped this out on an architectural argument. The
architecture turned out to be fine — T24's registry lets a feature see the
whole cross-section — so what is left is the two failure modes that make pairs
backtests famously optimistic, and both are worth testing harder than the
arithmetic.

**Selection look-ahead.** Screening for cointegration on the whole sample and
then trading the pairs it found is not a subtle bias: the pairs are chosen
*because* their spread mean-reverted over the period being evaluated.

**Multiple testing.** A 20-name universe is 190 tests. At p<0.05 about ten
pairs pass on pure noise, so an uncorrected screen reliably "finds"
cointegration in random walks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.cointegration import (
    DEFAULT_FORMATION_WINDOW,
    PAIRS_NOT_NEUTRAL_NOTE,
    Pair,
    engle_granger,
    pair_cheapness_feature,
    pair_scores,
    rolling_pair_scores,
    select_pairs,
)
from portfolio_agent.strategies.registry import load_strategy


def _cointegrated(n: int, rng, beta: float = 0.8, noise: float = 1.0):
    """A price pair sharing a stochastic trend with a stationary AR(1) spread."""
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0, noise)
    return base, base * beta + spread


@pytest.fixture
def mixed_universe():
    """Three genuinely cointegrated pairs among fourteen independent walks."""
    rng = np.random.default_rng(5)
    n = 700
    dates = pd.bdate_range("2020-01-01", periods=n)

    columns = {}
    truth = []
    for i in range(0, 6, 2):
        left, right = _cointegrated(n, rng)
        columns[f"C{i}"], columns[f"C{i + 1}"] = left, right
        truth.append((f"C{i}", f"C{i + 1}"))
    for j in range(14):
        columns[f"R{j}"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))

    return pd.DataFrame(columns, index=dates), truth


@pytest.fixture
def random_walks():
    """Twenty independent random walks. Nothing here is cointegrated."""
    rng = np.random.default_rng(21)
    n = 400
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {f"W{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
         for i in range(20)},
        index=dates,
    )


# --------------------------------------------------------------------------
# The test itself
# --------------------------------------------------------------------------


class TestEngleGranger:
    def test_it_finds_a_constructed_relationship(self, mixed_universe):
        closes, truth = mixed_universe
        left, right = truth[0]
        _beta, p_value = engle_granger(closes[left], closes[right])
        assert p_value < 0.05

    def test_it_recovers_the_hedge_ratio(self):
        rng = np.random.default_rng(2)
        left, right = _cointegrated(600, rng, beta=0.8, noise=0.5)
        index = pd.bdate_range("2021-01-04", periods=600)

        # `right ~ 0.8 * left`, so regressing left on right gives about 1/0.8.
        beta, _p = engle_granger(
            pd.Series(left, index=index), pd.Series(right, index=index)
        )
        assert beta == pytest.approx(1.25, rel=0.15)

    def test_two_independent_walks_are_usually_not_cointegrated(self, random_walks):
        p_values = [
            engle_granger(random_walks["W0"], random_walks[f"W{i}"])[1]
            for i in range(1, 20)
        ]
        assert np.nanmean([p > 0.05 for p in p_values]) > 0.7

    def test_a_degenerate_series_returns_nan_rather_than_raising(self):
        index = pd.bdate_range("2023-01-02", periods=100)
        flat = pd.Series(np.full(100, 50.0), index=index)
        moving = pd.Series(np.arange(100, dtype=float), index=index)

        beta, p_value = engle_granger(moving, flat)
        assert np.isnan(p_value) and np.isnan(beta)

    def test_too_little_history_returns_nan(self):
        index = pd.bdate_range("2023-01-02", periods=10)
        a = pd.Series(np.arange(10, dtype=float), index=index)
        assert np.isnan(engle_granger(a, a * 2 + 1)[1])


# --------------------------------------------------------------------------
# Multiple testing — the reason an uncorrected screen finds so much
# --------------------------------------------------------------------------


class TestMultipleTesting:
    def test_an_uncorrected_screen_finds_pairs_in_pure_noise(self, random_walks):
        """Twenty independent random walks, 190 tests, ~9.5 expected passes.

        This is the number that makes the correction non-optional: without it
        the screen reports cointegration among series that have none, and
        reports a lot of it.
        """
        uncorrected = select_pairs(random_walks, correction="none")

        assert uncorrected.n_tested == 190
        assert uncorrected.expected_false_positives == pytest.approx(9.5)
        assert len(uncorrected.pairs) > 2

    def test_bonferroni_removes_almost_all_of_them(self, random_walks):
        corrected = select_pairs(random_walks, correction="bonferroni")

        assert corrected.expected_false_positives == pytest.approx(0.05)
        assert len(corrected.pairs) <= 1

    def test_the_correction_is_on_by_default(self, random_walks):
        assert select_pairs(random_walks).correction == "bonferroni"

    def test_the_effective_threshold_is_divided_by_the_test_count(self, random_walks):
        selection = select_pairs(random_walks)
        assert selection.effective_threshold == pytest.approx(0.05 / 190)

    def test_turning_it_off_says_what_that_costs(self, random_walks):
        selection = select_pairs(random_walks, correction="none")
        assert any("by chance alone" in note for note in selection.notes)

    def test_the_test_count_travels_into_the_result(self, random_walks):
        document = select_pairs(random_walks).to_dict()
        assert document["n_pairs_tested"] == 190
        assert document["pair_expected_false_positives"] == pytest.approx(0.05)

    def test_an_unknown_correction_is_refused(self, random_walks):
        with pytest.raises(ValueError, match="Unknown correction"):
            select_pairs(random_walks, correction="holm")

    def test_it_still_finds_a_real_pair_through_the_correction(self, mixed_universe):
        """The correction must be conservative, not blind."""
        closes, _truth = mixed_universe
        selection = select_pairs(closes.iloc[:252])

        assert len(selection.pairs) >= 1
        found = {(p.left, p.right) for p in selection.pairs}
        assert all(a.startswith("C") and b.startswith("C") for a, b in found)


# --------------------------------------------------------------------------
# Look-ahead — the reason pairs backtests are famously optimistic
# --------------------------------------------------------------------------


class TestSelectionIsCausal:
    def test_a_score_cannot_depend_on_a_later_price(self, mixed_universe):
        """The assertion the whole module is built around.

        Screening on the full sample and trading what it found is the severe
        version of this mistake. Perturbing every price after a cut date must
        leave every score before it untouched.
        """
        closes, _ = mixed_universe
        cut = closes.index[400]

        tampered = closes.copy()
        tampered.loc[tampered.index > cut] *= 1.5

        base = rolling_pair_scores(closes)
        after = rolling_pair_scores(tampered)

        pd.testing.assert_frame_equal(
            base.loc[base.index <= cut], after.loc[after.index <= cut]
        )

    def test_nothing_is_scored_before_the_first_formation_window_closes(
        self, mixed_universe
    ):
        closes, _ = mixed_universe
        scores = rolling_pair_scores(closes, formation=252)

        assert scores.iloc[:252].isna().all().all()

    def test_a_panel_shorter_than_the_formation_window_scores_nothing(self):
        rng = np.random.default_rng(3)
        index = pd.bdate_range("2023-01-02", periods=100)
        closes = pd.DataFrame(
            {f"S{i}": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
             for i in range(5)},
            index=index,
        )
        assert rolling_pair_scores(closes, formation=252).isna().all().all()

    def test_the_zscore_uses_the_formation_window_s_moments(self):
        """Re-estimating on the trading window would centre it at zero.

        A z-score computed against its own window's mean can never say the
        spread is stretched, which is the one thing it exists to say.
        """
        index = pd.bdate_range("2023-01-02", periods=10)
        closes = pd.DataFrame(
            {"A": np.full(10, 120.0), "B": np.full(10, 100.0)}, index=index
        )
        pair = Pair(
            left="A", right="B", hedge_ratio=1.0, p_value=0.01,
            spread_mean=10.0, spread_std=2.0,
        )
        # Spread is 20 throughout; formation said mean 10, sd 2 -> z = +5.
        assert pair.zscore(closes).iloc[-1] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


class TestPairScores:
    def _pair(self, **overrides):
        base = dict(
            left="A", right="B", hedge_ratio=1.0, p_value=0.01,
            spread_mean=0.0, spread_std=1.0,
        )
        base.update(overrides)
        return Pair(**base)

    def _selection(self, pairs):
        from portfolio_agent.features.cointegration import PairSelection

        return PairSelection(
            pairs=pairs, n_tested=1, p_threshold=0.05, correction="bonferroni"
        )

    def test_the_cheap_leg_scores_positive(self):
        """Spread `A - B` far below its mean means A is cheap against B."""
        index = pd.bdate_range("2023-01-02", periods=3)
        closes = pd.DataFrame({"A": [95.0, 95.0, 95.0], "B": [100.0, 100.0, 100.0]},
                              index=index)

        scores = pair_scores(closes, self._selection([self._pair()]))
        assert scores["A"].iloc[-1] > 0
        assert scores["B"].iloc[-1] < 0

    def test_the_two_legs_are_exact_opposites(self):
        index = pd.bdate_range("2023-01-02", periods=3)
        closes = pd.DataFrame({"A": [103.0] * 3, "B": [100.0] * 3}, index=index)

        scores = pair_scores(closes, self._selection([self._pair()]))
        assert scores["A"].iloc[-1] == pytest.approx(-scores["B"].iloc[-1])

    def test_a_symbol_in_two_pairs_takes_the_mean(self):
        """Not the extreme: one stretched pair out of several is as likely to
        be that pair breaking down as it is an opportunity."""
        index = pd.bdate_range("2023-01-02", periods=3)
        closes = pd.DataFrame(
            {"A": [98.0] * 3, "B": [100.0] * 3, "C": [100.0] * 3}, index=index
        )
        selection = self._selection([
            self._pair(),                                    # A vs B -> +2
            self._pair(right="C", spread_std=1.0),           # A vs C -> +2
        ])
        scores = pair_scores(closes, selection)
        assert scores["A"].iloc[-1] == pytest.approx(2.0)

    def test_a_symbol_in_no_pair_is_absent_not_zero(self):
        """Zero would read as "fairly valued", a claim the screen did not make."""
        index = pd.bdate_range("2023-01-02", periods=3)
        closes = pd.DataFrame(
            {"A": [98.0] * 3, "B": [100.0] * 3, "LONELY": [50.0] * 3}, index=index
        )
        scores = pair_scores(closes, self._selection([self._pair()]))
        assert "LONELY" not in scores.columns

    def test_no_surviving_pairs_scores_nothing(self):
        index = pd.bdate_range("2023-01-02", periods=3)
        closes = pd.DataFrame({"A": [1.0] * 3}, index=index)
        assert pair_scores(closes, self._selection([])).empty


# --------------------------------------------------------------------------
# The registered feature and the strategy
# --------------------------------------------------------------------------


class TestTheRegisteredFeature:
    def test_the_family_is_registered(self):
        from portfolio_agent.features.cross_section import is_cross_sectional_feature

        assert is_cross_sectional_feature("pair_cheapness_252")
        assert is_cross_sectional_feature("pair_cheapness_126")

    def test_an_unregistered_window_is_refused_rather_than_rounded(self):
        with pytest.raises(ValueError, match="No 'pair_cheapness' feature"):
            pair_cheapness_feature(200)

    def test_the_default_window_resolves(self):
        assert pair_cheapness_feature(DEFAULT_FORMATION_WINDOW) == "pair_cheapness_252"


class TestTheStrategy:
    def _strategy(self, **params):
        return load_strategy(StrategyConfig(type="pairs", params=params))

    def test_it_is_registered(self):
        assert self._strategy().name == "pairs"

    def test_it_declares_the_pair_feature(self):
        assert self._strategy().required_cross_sectional_features() == [
            "pair_cheapness_252"
        ]

    def test_the_formation_window_reaches_the_feature(self):
        assert self._strategy(formation_window=126).required_cross_sectional_features() == [
            "pair_cheapness_126"
        ]

    def test_an_unregistered_window_fails_at_construction(self):
        with pytest.raises(ValueError, match="No 'pair_cheapness' feature"):
            self._strategy(formation_window=90)

    def test_it_declares_that_it_is_not_market_neutral(self):
        """The concession section 7 named. A report must not use the words
        "pairs trading" and leave the reader assuming neutrality."""
        rules = self._strategy().entry_rules()
        assert rules["market_neutral"] is False
        assert PAIRS_NOT_NEUTRAL_NOTE in rules["notes"]

    def test_the_note_says_why_it_is_not_neutral(self):
        assert "short" in PAIRS_NOT_NEUTRAL_NOTE
        assert "not comparable" in PAIRS_NOT_NEUTRAL_NOTE

    def test_it_ranks_cheapest_first(self):
        """The metric is stated as undervaluation, not as a return, so higher
        is better even though this is a mean-reversion signal."""
        assert self._strategy().higher_metric_is_better is True

    def test_its_minimum_universe_is_lower_than_a_decile_sort(self):
        """A different constraint, not a looser one: a decile of 20 names is
        2 stocks and not a ranking; a pair screen over 20 names is 190 tests."""
        pairs = self._strategy().entry_rules()["min_universe"]
        momentum = load_strategy(
            StrategyConfig(type="momentum", params={})
        ).entry_rules()["min_universe"]
        assert pairs < momentum

    def test_it_requires_the_full_batch(self):
        assert self._strategy().requires_full_batch is True

    def test_a_universe_of_one_scores_nothing(self):
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

    def test_it_reports_its_own_trigger(self):
        assert self._strategy().trigger_name == "Pairs"

    def test_it_inherits_the_tradability_screen(self):
        assert self._strategy().entry_rules()["tradability_filter"]["enabled"]
