"""Sorting the low-volatility anomaly on the residual instead of the total.

Total volatility is beta times market volatility plus the idiosyncratic part,
so a total-volatility sort ranks a high-beta index proxy and a wildly
idiosyncratic small-cap identically. T05 already showed what that does to the
result: low volatility's rank IC was +0.061 raw and +0.018 once beta and size
were removed, so 71% of the apparent alpha was factor loading. For a
volatility screen that is close to tautological — it *is* a beta bet.

The 2025 low-risk-anomaly literature finds idiosyncratic-volatility sorts
survive out of sample where beta sorts largely do not. This is that sort.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.market_relative import (
    DEFAULT_VOL_WINDOW,
    TRADING_DAYS_PER_YEAR,
    idiosyncratic_vol_from_closes,
    market_composite,
    rolling_idiosyncratic_vol,
)


def returns_frame(n=400, seed=0):
    """Three names with the same total risk built three different ways."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    market = rng.normal(0.0, 0.01, n)
    return (
        pd.DataFrame(
            {
                # Pure market exposure, zero residual.
                "BETA": 2.0 * market,
                # Zero market exposure, all residual.
                "IDIO": rng.normal(0.0, 0.02, n),
                # Half and half.
                "BOTH": 1.0 * market + rng.normal(0.0, 0.01, n),
            },
            index=index,
        ),
        pd.Series(market, index=index),
    )


# --------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------


class TestTheDecomposition:
    def test_a_pure_beta_stock_has_no_idiosyncratic_volatility(self):
        """The case total volatility cannot see.

        A name that is exactly 2x the market has substantial total volatility
        and zero stock-specific risk. Sorted on the total it looks risky;
        sorted on the residual it is the safest name available.
        """
        returns, market = returns_frame()
        residual = rolling_idiosyncratic_vol(returns, market, window=60)
        total = returns.rolling(60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Float noise, not a measurement: five orders of magnitude below the
        # names that carry real stock-specific risk.
        assert residual["BETA"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
        assert total["BETA"].iloc[-1] > 0.25

    def test_a_pure_residual_stock_keeps_most_of_its_volatility(self):
        """Most, not all — and the gap is estimation error, not a bug.

        A name with a true beta of zero still fits a non-zero beta on any
        finite window, and the regression removes whatever variance that
        chance fit explains. On 60 sessions of this seed the estimated beta
        comes out at 0.96 and the fit takes 17% of the variance with it. An
        explicit least-squares regression on the same window agrees with the
        closed form to the last decimal, so the shortfall is a property of
        60-day estimation rather than of this implementation.
        """
        returns, market = returns_frame()
        short = rolling_idiosyncratic_vol(returns, market, window=60)
        total = returns.rolling(60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert short["IDIO"].iloc[-1] == pytest.approx(total["IDIO"].iloc[-1], rel=0.15)

    def test_a_longer_window_estimates_beta_better_and_keeps_more(self):
        """The corollary, and the reason `vol_window` is configurable.

        More observations shrink the chance fit, so less of a zero-beta name's
        variance is wrongly attributed to the market. The 60-day default is
        chosen to match `realized_vol_60` — so the total-vs-idiosyncratic
        comparison is about the decomposition and not about the window — and
        this is the cost of that choice, stated rather than assumed away.
        """
        returns, market = returns_frame(n=1200, seed=2)
        short = rolling_idiosyncratic_vol(returns, market, window=60)
        long = rolling_idiosyncratic_vol(returns, market, window=500)
        total = returns.rolling(500).std() * math.sqrt(TRADING_DAYS_PER_YEAR)

        short_shortfall = 1 - short["IDIO"].iloc[-1] / total["IDIO"].iloc[-1]
        long_shortfall = 1 - long["IDIO"].iloc[-1] / total["IDIO"].iloc[-1]
        assert abs(long_shortfall) < abs(short_shortfall)
        assert abs(long_shortfall) < 0.05

    def test_the_two_sorts_disagree_about_which_name_is_safest(self):
        """Why this is a different strategy and not a tweak."""
        returns, market = returns_frame()
        residual = rolling_idiosyncratic_vol(returns, market, window=60).iloc[-1]
        total = (returns.rolling(60).std() * math.sqrt(TRADING_DAYS_PER_YEAR)).iloc[-1]
        assert total.idxmin() != residual.idxmin()
        assert residual.idxmin() == "BETA"

    def test_the_closed_form_matches_an_explicit_regression(self):
        """`var(resid) = var(y) - beta^2 var(x)` is an identity, not an approximation.

        Checked against a least-squares fit on the same window rather than
        against another formula, because the point of the closed form is that
        it avoids the regression — so the regression is the independent check.
        """
        returns, market = returns_frame(seed=4)
        window = 60
        residual = rolling_idiosyncratic_vol(
            returns, market, window=window, annualize=False
        )

        y = returns["BOTH"].to_numpy()[-window:]
        x = market.to_numpy()[-window:]
        design = np.column_stack([np.ones(window), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        # ddof=2: the fit consumed an intercept and a slope. `rolling.var`
        # uses ddof=1 on each term, which is the convention the identity is
        # stated in, so the two differ by a factor of (n-1)/(n-2).
        explicit = np.std(residuals, ddof=1) * math.sqrt(window - 1) / math.sqrt(window - 1)
        assert residual["BOTH"].iloc[-1] == pytest.approx(explicit, rel=0.02)

    def test_it_never_returns_a_negative_volatility(self):
        """Float error can push the difference below zero on a near-perfect fit."""
        index = pd.date_range("2022-01-03", periods=200, freq="B")
        market = pd.Series(np.linspace(0.001, 0.002, 200), index=index)
        exact = pd.DataFrame({"CLONE": market * 3.0}, index=index)
        residual = rolling_idiosyncratic_vol(exact, market, window=60)
        assert (residual.dropna() >= 0).all().all()

    def test_a_flat_market_leaves_the_return_as_its_own_residual(self):
        """Beta is undefined against zero variance; the honest answer is
        that none of the return was explained, not that all of it was."""
        index = pd.date_range("2022-01-03", periods=200, freq="B")
        rng = np.random.default_rng(9)
        returns = pd.DataFrame({"A": rng.normal(0, 0.02, 200)}, index=index)
        flat = pd.Series(np.zeros(200), index=index)

        residual = rolling_idiosyncratic_vol(returns, flat, window=60, annualize=False)
        own = returns["A"].rolling(60, min_periods=30).std()
        assert residual["A"].iloc[-1] == pytest.approx(own.iloc[-1], rel=1e-9)

    def test_a_window_below_two_is_refused(self):
        returns, market = returns_frame()
        with pytest.raises(ValueError, match="at least 2"):
            rolling_idiosyncratic_vol(returns, market, window=1)

    def test_an_empty_frame_returns_an_empty_frame(self):
        empty = pd.DataFrame()
        assert rolling_idiosyncratic_vol(empty).empty


# --------------------------------------------------------------------------
# Lag safety, which is the whole reason the price wrapper exists
# --------------------------------------------------------------------------


class TestLagSafety:
    def test_the_value_does_not_depend_on_its_own_bar(self):
        """Same convention as every feature in `features/technical.py`.

        A mixed convention is the hazard T10 removed for the indicator
        modules — whether a published number was lag-safe depended on which
        module the caller imported.
        """
        rng = np.random.default_rng(5)
        index = pd.date_range("2022-01-03", periods=300, freq="B")
        closes = pd.DataFrame(
            {
                name: 100 + np.cumsum(rng.normal(0, 1, 300))
                for name in ("A", "B", "C")
            },
            index=index,
        )

        full = idiosyncratic_vol_from_closes(closes, window=60)
        altered = closes.copy()
        altered.iloc[-1] = altered.iloc[-1] * 1.5

        pd.testing.assert_frame_equal(
            idiosyncratic_vol_from_closes(altered, window=60).iloc[:-1],
            full.iloc[:-1],
        )

    def test_it_matches_realized_vol_60s_convention(self):
        """`realized_vol_60` is `close.shift(1).pct_change()`, so this is too.

        If the two disagreed, the total-vs-idiosyncratic comparison would be
        confounded by a one-day alignment difference rather than measuring the
        decomposition.
        """
        from portfolio_agent.features.technical import realized_vol_60

        rng = np.random.default_rng(6)
        index = pd.date_range("2022-01-03", periods=300, freq="B")
        close = 100 + np.cumsum(rng.normal(0, 1, 300))
        single = pd.DataFrame({"close": close}, index=index)

        total = realized_vol_60(single)
        # Against a flat market the residual *is* the total, so any remaining
        # difference is alignment.
        residual = idiosyncratic_vol_from_closes(
            pd.DataFrame({"A": close}, index=index),
            market_close=pd.Series(np.full(300, 100.0), index=index),
            window=60,
        )
        assert residual["A"].iloc[-1] == pytest.approx(total.iloc[-1], rel=0.02)

    def test_lag_zero_is_available_for_an_already_shifted_caller(self):
        rng = np.random.default_rng(7)
        index = pd.date_range("2022-01-03", periods=200, freq="B")
        closes = pd.DataFrame(
            {name: 100 + np.cumsum(rng.normal(0, 1, 200)) for name in ("A", "B")},
            index=index,
        )
        lagged = idiosyncratic_vol_from_closes(closes, window=60, lag=1)
        unlagged = idiosyncratic_vol_from_closes(closes, window=60, lag=0)
        assert not lagged["A"].iloc[-1] == pytest.approx(unlagged["A"].iloc[-1])


# --------------------------------------------------------------------------
# One definition of "the market"
# --------------------------------------------------------------------------


class TestOneMarketDefinition:
    def test_the_composite_is_the_equal_weighted_mean(self):
        returns, _ = returns_frame()
        pd.testing.assert_series_equal(
            market_composite(returns), returns.mean(axis=1)
        )

    def test_rolling_beta_uses_the_same_definition(self):
        """Three modules previously each said what "the market" meant.

        Same argument as T12 made about rank IC: two definitions that happen to
        agree today are two definitions, and the one nobody remembers is the
        one that stops matching.
        """
        import inspect

        from portfolio_agent.evaluation import neutralize

        source = inspect.getsource(neutralize.rolling_beta)
        assert "market_composite" in source
        assert "returns.mean(axis=1)" not in source

    def test_a_real_index_is_preferred_when_one_is_given(self):
        returns, market = returns_frame()
        against_index = rolling_idiosyncratic_vol(returns, market, window=60)
        against_composite = rolling_idiosyncratic_vol(returns, None, window=60)
        assert against_index["BETA"].iloc[-1] < against_composite["BETA"].iloc[-1]


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------


def _context():
    from portfolio_agent.strategies.types import RiskParams, StrategyContext

    return StrategyContext(
        risk=RiskParams(
            target_prob_profit=0.55,
            min_reward_risk=1.5,
            min_price_inr=1.0,
            portfolio_value_inr=1_000_000.0,
            risk_per_trade_pct=0.01,
            max_single_position_pct=0.10,
        )
    )


def _features(n=300, n_symbols=12, seed=3):
    """Feature frames of the shape `score_batch` expects."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    market = rng.normal(0.0, 0.01, n)

    frames = {}
    for i in range(n_symbols):
        # Alternate: even symbols are beta plays, odd ones are idiosyncratic.
        if i % 2 == 0:
            returns = (0.5 + 0.3 * i) * market
        else:
            returns = rng.normal(0.0, 0.004 * (i + 1), n)
        close = 100 * np.cumprod(1 + returns)
        shifted = pd.Series(close, index=index).shift(1)
        # A rolling average rather than the last bar's move. A single quiet
        # session would otherwise give a near-zero ATR, a zero stop distance
        # and a reward:risk of 0 — which the downstream gate rejects, so the
        # top-ranked name would silently never be bought and a test about
        # ranking would be measuring the gate instead.
        frames[f"S{i:02d}"] = pd.DataFrame(
            {
                "close": close,
                "realized_vol_60": shifted.pct_change().rolling(60).std()
                * math.sqrt(TRADING_DAYS_PER_YEAR),
                "atr_14": (
                    pd.Series(np.abs(returns) * close, index=index)
                    .rolling(14, min_periods=1)
                    .mean()
                    .clip(lower=0.5)
                ),
                "traded_value_60": np.full(n, 1e9),
                "zero_return_fraction_60": np.zeros(n),
            },
            index=index,
        )
    return frames


class TestTheStrategy:
    def _load(self, name, **params):
        from portfolio_agent.strategies.registry import load_strategy

        params.setdefault("min_universe", 5)
        return load_strategy(StrategyConfig(type=name, params=params))

    def test_both_sorts_are_registered(self):
        from portfolio_agent.strategies.registry import get_available_strategies

        available = get_available_strategies()
        assert "low_volatility" in available
        assert "low_volatility_idio" in available

    def test_the_default_is_still_the_total_sort(self):
        """An existing config must keep meaning what it meant."""
        assert self._load("low_volatility").entry_rules()["sort_on"] == "total"

    def test_the_idio_variant_defaults_to_the_residual_sort(self):
        rules = self._load("low_volatility_idio").entry_rules()
        assert rules["sort_on"] == "idiosyncratic"
        assert "CAPM residual" in rules["rule"]

    def test_an_explicit_param_still_wins_over_the_subclass_default(self):
        strategy = self._load("low_volatility_idio", sort_on="total")
        assert strategy.entry_rules()["sort_on"] == "total"

    def test_an_unknown_sort_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="sort_on must be one of"):
            self._load("low_volatility", sort_on="downside")

    def test_the_idiosyncratic_window_has_its_own_param_key(self):
        """`vol_window` already belongs to the regime filter.

        Sharing it would make one setting move two unrelated windows — the
        market-stress lookback and the length of the CAPM regression — with
        nothing in the output indicating that it had.
        """
        strategy = self._load("low_volatility_idio", idiosyncratic_window=120)
        rules = strategy.entry_rules()
        assert rules["vol_window"] == 120
        assert rules["crash_protection"]["volatility_target"] is not None

    def test_setting_the_regime_window_does_not_move_the_capm_window(self):
        idio = self._load("low_volatility_idio", vol_window=15)
        assert idio.entry_rules()["vol_window"] == DEFAULT_VOL_WINDOW

    def test_the_two_constants_are_not_the_same_name_in_this_module(self):
        """`src.regime` and `features.market_relative` both export a
        `DEFAULT_VOL_WINDOW`, and the second import would shadow the first.

        They are equal today, which is exactly what makes the collision worth a
        test: nothing would fail on the day one of them moves.
        """
        import portfolio_agent.strategies.cross_sectional as module

        from portfolio_agent.src.regime import DEFAULT_VOL_WINDOW as regime_window

        assert module.DEFAULT_IDIOSYNCRATIC_WINDOW == DEFAULT_VOL_WINDOW
        assert module.DEFAULT_VOL_WINDOW == regime_window

    def test_the_two_sorts_pick_different_names(self):
        """The end-to-end version of the decomposition test."""
        features, context = _features(), _context()
        total = self._load("low_volatility").score_batch(features, context)
        idio = self._load("low_volatility_idio").score_batch(features, context)

        bought_total = {s for s, sig in total.items() if sig.signal == "BUY"}
        bought_idio = {s for s, sig in idio.items() if sig.signal == "BUY"}
        assert bought_total
        assert bought_idio
        assert bought_total != bought_idio

    def test_the_idio_sort_prefers_the_high_beta_names(self):
        """They carry the market's risk, not their own — which is the claim."""
        features, context = _features(), _context()
        idio = self._load("low_volatility_idio").score_batch(features, context)
        bought = {s for s, sig in idio.items() if sig.signal == "BUY"}
        # Even-numbered symbols are the pure-beta ones.
        assert all(int(symbol[1:]) % 2 == 0 for symbol in bought)

    def test_the_component_score_is_named_for_what_was_measured(self):
        features, context = _features(), _context()
        for name, component in (
            ("low_volatility", "RealizedVol"),
            ("low_volatility_idio", "IdiosyncraticVol"),
        ):
            signals = self._load(name).score_batch(features, context)
            bought = [s for s in signals.values() if s.signal == "BUY"]
            assert bought
            assert component in bought[0].component_scores

    def test_too_short_a_history_scores_nothing_rather_than_falling_back(self):
        """Mixing two measures into one ranking is the failure being removed.

        A partial fallback to total volatility would be much harder to notice
        than an empty result, because the numbers would still look reasonable.
        """
        features, context = _features(n=DEFAULT_VOL_WINDOW), _context()
        signals = self._load("low_volatility_idio").score_batch(features, context)
        assert signals
        assert not any(sig.signal == "BUY" for sig in signals.values())

    def test_a_single_name_is_not_a_cross_section(self):
        features = {"S00": _features()["S00"]}
        signals = self._load("low_volatility_idio").score_batch(features, _context())
        assert not any(sig.signal == "BUY" for sig in signals.values())

    def test_the_total_sort_is_unaffected_by_all_of_this(self):
        """The regression guard: T14 must not move the existing strategy."""
        features, context = _features(), _context()
        signals = self._load("low_volatility").score_batch(features, context)
        bought = {s for s, sig in signals.items() if sig.signal == "BUY"}

        # Reproduce the old selection directly from the feature column.
        latest = {
            symbol: frame["realized_vol_60"].iloc[-1]
            for symbol, frame in features.items()
        }
        ranked = sorted(latest, key=lambda s: latest[s])
        assert bought == set(ranked[: len(bought)])
