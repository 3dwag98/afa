"""Momentum measured on the residual, and the trap that makes it zero.

Price momentum's return is substantially a bet on whatever the market has been
rewarding — round two measured this platform's own momentum at 58% factor
loading. Blitz, Huij & Martens (2011) rank on the residual's information ratio
instead and report roughly double the risk-adjusted profit with shallower
drawdowns.

Two things here are worth testing rather than asserting. The residualization
has to actually remove the exposure, and the standardization has to actually be
doing work — a raw cumulated residual still ranks high-residual-volatility
names highest, which puts back the risk that residualizing removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.cross_section import build_cross_section, warmup_rows
from portfolio_agent.features.market_relative import (
    RESIDUAL_BETA_WINDOW,
    RESIDUAL_FORMATION_DAYS,
    RESIDUAL_SKIP_DAYS,
    residual_momentum,
    residual_returns,
)
from portfolio_agent.strategies.registry import load_strategy


@pytest.fixture
def factor_world():
    """Forty names with known betas and five with genuine alpha.

    Betas run 0.4 to 1.8, so a formation measure that has not removed the
    market exposure will rank by beta. Five names carry a positive daily alpha
    and five a negative one, so a measure that *has* removed it has something
    true to find.
    """
    rng = np.random.default_rng(7)
    n, k = 700, 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    market = pd.Series(rng.normal(0.0005, 0.011, n), index=dates)
    betas = pd.Series(np.linspace(0.4, 1.8, k), index=[f"S{i}" for i in range(k)])
    alphas = pd.Series(0.0, index=betas.index)
    alphas.iloc[:5] = 0.0009
    alphas.iloc[-5:] = -0.0009

    returns = pd.DataFrame(
        {
            symbol: alphas[symbol] + betas[symbol] * market.to_numpy()
            + rng.normal(0, 0.010, n)
            for symbol in betas.index
        },
        index=dates,
    )
    closes = 100.0 * (1.0 + returns).cumprod()
    return {
        "returns": returns, "closes": closes, "market": market,
        "betas": betas, "alphas": alphas, "dates": dates,
    }


# --------------------------------------------------------------------------
# The residual
# --------------------------------------------------------------------------


class TestTheResidualRemovesTheExposure:
    def test_it_is_far_less_correlated_with_the_market(self, factor_world):
        raw = factor_world["returns"]
        residual = residual_returns(raw, factor_world["market"])

        raw_correlation = float(raw.corrwith(factor_world["market"]).abs().mean())
        residual_correlation = float(
            residual.corrwith(factor_world["market"]).abs().mean()
        )

        assert raw_correlation > 0.5
        assert residual_correlation < 0.10

    def test_a_pure_market_name_has_almost_no_residual(self, factor_world):
        """A stock that *is* the market has nothing left after the market."""
        market = factor_world["market"]
        returns = factor_world["returns"].copy()
        returns["CLONE"] = market

        residual = residual_returns(returns, market)
        settled = residual["CLONE"].dropna()

        assert settled.abs().mean() < 1e-8

    def test_no_intercept_is_subtracted(self, factor_world):
        """The intercept is the alpha this feature exists to measure.

        Removing it would difference away the signal along with the exposure,
        and a positive-alpha name's residual would centre on zero like every
        other name's.
        """
        residual = residual_returns(factor_world["returns"], factor_world["market"])
        settled = residual.dropna(how="all")

        alpha_name = factor_world["alphas"].idxmax()
        flat_name = factor_world["alphas"][factor_world["alphas"] == 0.0].index[0]

        assert settled[alpha_name].mean() > settled[flat_name].mean()
        assert settled[alpha_name].mean() > 0


class TestTheInterceptTrap:
    def test_fitting_an_intercept_over_the_formation_window_zeroes_it(self):
        """Why the estimation window is longer than the formation window.

        OLS residuals sum to zero over the window they were fitted on — so a
        cumulative residual formed over exactly the estimation window is
        identically zero, and a strategy ranking on it would be ranking pure
        floating-point noise. Demonstrated here so the constant that avoids it
        has a reason attached rather than a number.
        """
        rng = np.random.default_rng(1)
        n = 200
        alpha = 0.0008
        # Noiseless, so the arithmetic is exact and the point is not a
        # question of sample size: the stock *is* alpha plus beta times market.
        market = rng.normal(0.0004, 0.01, n)
        stock = alpha + 1.3 * market

        design = np.column_stack([np.ones(n), market])
        coefficients, *_ = np.linalg.lstsq(design, stock, rcond=None)
        fitted_residual = stock - design @ coefficients

        # Fitted with an intercept, the residual is zero at every point — so
        # cumulating it over the fitting window measures nothing at all.
        assert np.allclose(fitted_residual, 0.0, atol=1e-12)

        # Without the intercept the residual keeps the alpha, which is the
        # whole quantity of interest. It recovers most rather than all of it:
        # a market with a non-zero mean lets beta absorb a little.
        beta_only = np.linalg.lstsq(market.reshape(-1, 1), stock, rcond=None)[0]
        recovered = (stock - market * beta_only).mean()
        assert recovered > 0.5 * alpha
        assert recovered <= alpha + 1e-12

    def test_the_estimation_window_is_longer_than_the_formation_window(self):
        assert RESIDUAL_BETA_WINDOW > RESIDUAL_FORMATION_DAYS - RESIDUAL_SKIP_DAYS


# --------------------------------------------------------------------------
# The formation measure
# --------------------------------------------------------------------------


class TestResidualMomentum:
    def test_it_orders_the_cross_section_by_true_alpha(self, factor_world):
        """The aggregate property, not membership of a particular top five.

        With a daily alpha of 0.0009 against an idiosyncratic 0.010, the mean
        over a 189-day formation window has a standard error of 0.00073 — so
        `alpha / se` is about 1.2 and *which* five names land on top is largely
        the draw. Asserting exact membership would be tuning the fixture until
        the test passed. The rank correlation across the whole cross-section is
        the claim that actually holds.
        """
        resmom = residual_momentum(
            factor_world["returns"], factor_world["market"]
        ).iloc[-1].dropna()

        alphas = factor_world["alphas"].reindex(resmom.index)
        assert float(alphas.rank().corr(resmom.rank())) > 0.5

    def test_the_alpha_names_outscore_the_negative_alpha_names(self, factor_world):
        resmom = residual_momentum(
            factor_world["returns"], factor_world["market"]
        ).iloc[-1].dropna()
        alphas = factor_world["alphas"].reindex(resmom.index)

        winners = resmom[alphas > 0].mean()
        losers = resmom[alphas < 0].mean()
        flat = resmom[alphas == 0].mean()

        assert winners > flat > losers

    def test_it_is_less_beta_driven_than_raw_momentum(self, factor_world):
        """The claim the strategy makes, measured on the same panel."""
        closes = factor_world["closes"]
        raw = (
            closes.shift(RESIDUAL_SKIP_DAYS)
            / closes.shift(RESIDUAL_SKIP_DAYS + RESIDUAL_FORMATION_DAYS)
            - 1.0
        ).iloc[-1].dropna()
        resmom = residual_momentum(
            factor_world["returns"], factor_world["market"]
        ).iloc[-1].dropna()

        beta_rank = factor_world["betas"].rank()
        raw_loading = abs(float(beta_rank.reindex(raw.index).corr(raw.rank())))
        residual_loading = abs(
            float(beta_rank.reindex(resmom.index).corr(resmom.rank()))
        )
        assert residual_loading < raw_loading

    def test_the_standardization_changes_the_ranking(self, factor_world):
        """It is the substance of the effect, not a tidying step.

        A raw cumulated residual still ranks high-residual-volatility names
        highest, reintroducing exactly the risk exposure residualizing removed.
        If dividing by the residual's dispersion produced the same order there
        would be nothing to the argument.
        """
        residual = residual_returns(factor_world["returns"], factor_world["market"])
        window = RESIDUAL_FORMATION_DAYS
        floor = max(2, window // 2)

        raw_sum = (
            residual.rolling(window, min_periods=floor).mean()
            .shift(RESIDUAL_SKIP_DAYS).iloc[-1].dropna()
        )
        standardized = residual_momentum(
            factor_world["returns"], factor_world["market"]
        ).iloc[-1].dropna()

        shared = raw_sum.index.intersection(standardized.index)
        assert list(raw_sum[shared].rank()) != list(standardized[shared].rank())

    def test_a_flat_residual_is_not_ranked_first(self, factor_world):
        """Dividing by a zero dispersion would rank it on a sign bit."""
        returns = factor_world["returns"].copy()
        returns["FLAT"] = factor_world["market"] * 1.0

        resmom = residual_momentum(returns, factor_world["market"]).iloc[-1]
        value = resmom.get("FLAT")
        assert value is None or not np.isinf(value)

    def test_the_skip_keeps_the_recent_month_out(self, factor_world):
        """Perturbing the last three weeks cannot move the formation measure."""
        base = residual_momentum(factor_world["returns"], factor_world["market"])

        tampered = factor_world["returns"].copy()
        tampered.iloc[-RESIDUAL_SKIP_DAYS:] += 0.05
        after = residual_momentum(tampered, factor_world["market"])

        pd.testing.assert_series_equal(base.iloc[-1], after.iloc[-1])


# --------------------------------------------------------------------------
# The registered feature and the strategy
# --------------------------------------------------------------------------


class TestTheRegisteredFeature:
    def test_it_is_in_the_cross_sectional_registry(self):
        from portfolio_agent.features.cross_section import is_cross_sectional_feature

        assert is_cross_sectional_feature("residual_momentum_9m_skip1m")

    def test_its_warm_up_is_the_full_chain(self):
        """Half a beta window, then a whole formation window, then the skip.

        The asymmetry is deliberate. Beta is a *parameter estimate* and
        degrades gracefully as observations are lost, so it keeps this module's
        half-window convention. A formation window is not an estimate — its
        length is part of what the signal is, and allowing half would rank on
        4.5-month momentum under a name that says nine.
        """
        needed = warmup_rows(["residual_momentum_9m_skip1m"])
        expected = (
            RESIDUAL_BETA_WINDOW // 2 + RESIDUAL_FORMATION_DAYS + RESIDUAL_SKIP_DAYS
        )
        assert needed == expected + 1  # +1 converts a position to a count

    def test_a_half_filled_formation_window_produces_nothing(self, factor_world):
        """The mixed-formation-length defect, asserted directly."""
        resmom = residual_momentum(factor_world["returns"], factor_world["market"])
        first = int(np.argmax(resmom.notna().any(axis=1).to_numpy()))
        assert first >= RESIDUAL_BETA_WINDOW // 2 + RESIDUAL_FORMATION_DAYS

    def test_building_it_reproduces_the_direct_call(self, factor_world):
        """The registry path and the function must not diverge.

        The decorator applies the one-bar lag, so the direct call is compared
        against returns built from a shifted close — which is what the panel
        hands the feature body.
        """
        closes = factor_world["closes"]
        frames = {s: pd.DataFrame({"close": closes[s]}) for s in closes.columns}

        built = build_cross_section(frames, ["residual_momentum_9m_skip1m"])[
            "residual_momentum_9m_skip1m"
        ]
        direct = residual_momentum(closes.shift(1).pct_change())

        pd.testing.assert_frame_equal(built, direct)


class TestTheStrategy:
    def _strategy(self, **params):
        return load_strategy(
            StrategyConfig(type="residual_momentum", params=params)
        )

    def test_it_is_registered(self):
        assert self._strategy().name == "residual_momentum"

    def test_it_declares_the_cross_sectional_feature(self):
        assert self._strategy().required_cross_sectional_features() == [
            "residual_momentum_9m_skip1m"
        ]

    def test_it_does_not_request_raw_momentum(self):
        """Requesting it would widen every warm-up by 211 rows for a column
        nothing reads."""
        assert "mom_9m_skip1m" not in self._strategy().required_features()

    def test_it_requires_the_full_batch(self):
        """A residual is defined against a cross-section."""
        assert self._strategy().requires_full_batch is True

    def test_it_reports_its_own_trigger(self, factor_world):
        """Every momentum variant reporting "Momentum" would make a comparison
        between two of them unreadable in the trade log."""
        assert self._strategy().trigger_name == "ResidualMomentum"

    def test_its_entry_rules_name_the_formation_measure(self):
        rules = self._strategy().entry_rules()
        assert rules["formation_metric"] == "residual_momentum_9m_skip1m"
        assert "residual" in rules["rule"].lower()

    def test_it_inherits_momentum_s_controls(self):
        """The comparison is only about the formation measure if everything
        else is identical."""
        residual = self._strategy().entry_rules()
        plain = load_strategy(StrategyConfig(type="momentum", params={})).entry_rules()

        assert residual["crash_protection"] == plain["crash_protection"]
        assert residual["tradability_filter"] == plain["tradability_filter"]
        assert residual["top_percentile"] == plain["top_percentile"]

    def test_a_universe_of_one_scores_nothing(self, factor_world):
        """Below two names there is no cross-section to residualize against."""
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        closes = factor_world["closes"]
        one = {"S0": pd.DataFrame({"close": closes["S0"]})}
        context = StrategyContext(
            risk=RiskParams(
                target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
                portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
                max_single_position_pct=0.03,
            ),
        )
        assert self._strategy()._formation_metric(one, context) == {}


class TestBothRegistriesSetTheWarmUp:
    """The gap this strategy exposed, and the reason it was worth exposing.

    `_required_history_rows` and the harness's `min_history` both consulted the
    per-ticker registry only. `residual_momentum` needs 242 rows for its
    formation window and nothing per-ticker beyond 62, so either would have
    admitted tickers whose ranking key was still NaN — the exact defect T23
    removed, re-created by T24's second registry.
    """

    def test_the_cross_sectional_warm_up_is_the_larger_one_here(self):
        from portfolio_agent.features.pipeline import warmup_rows as per_ticker

        strategy = load_strategy(StrategyConfig(type="residual_momentum", params={}))
        assert warmup_rows(strategy.required_cross_sectional_features()) > per_ticker(
            strategy.required_features()
        )

    def test_the_engine_takes_the_maximum_across_both(self, monkeypatch):
        import numpy as np

        from portfolio_agent.src import backtest_engine as module

        rng = np.random.default_rng(2)
        index = pd.bdate_range("2020-01-01", periods=600)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(index))))
        ohlcv = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(len(index), 1e6)},
            index=index,
        )
        monkeypatch.setattr(
            module, "load_ticker_data",
            lambda ticker, start_date=None, end_date=None: ohlcv.copy(),
        )

        engine = module.BacktestEngine(
            start_date="2022-01-03", end_date="2022-03-31",
            initial_capital=1_000_000.0, universe_tickers=["A.NS"],
            strategy=load_strategy(
                StrategyConfig(type="residual_momentum", params={})
            ),
        )
        assert engine._required_history_rows() > 200
