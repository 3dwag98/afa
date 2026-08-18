"""From a ranking to a book, and whether the complexity buys anything.

`evaluate` reported a decile spread and stopped. A decile spread is the return
of an equal-weighted basket of the top bucket — a book — and nothing said so,
because the choice was implicit in `bucket_analysis` taking a mean.

These tests are about the seam and about the two places it can lie: an
allocation rule that silently degrades to equal weights, and a return reported
without the breadth that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.evaluation.allocation import (
    DEFAULT_MAX_WEIGHT,
    WEIGHTING_SCHEMES,
    build_book,
    cap_is_binding,
    compare_schemes,
    evaluate_book,
    weight_turnover,
)
from portfolio_agent.evaluation.costs import CostModel


@pytest.fixture
def market():
    """Sixty names on a shared factor, with deliberately unequal volatility.

    Name `i` has both a higher market loading and a wider idiosyncratic term,
    so a volatility-aware scheme has something real to respond to. Without that
    spread, inverse-vol and HRP would coincide with equal weights and the tests
    would pass while measuring nothing.
    """
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", periods=200)
    symbols = [f"S{i}" for i in range(60)]
    factor = rng.normal(0.0004, 0.009, len(dates))
    returns = pd.DataFrame(
        {
            s: factor * (0.6 + 0.02 * i) + rng.normal(0, 0.010 + 0.0004 * i, len(dates))
            for i, s in enumerate(symbols)
        },
        index=dates,
    )
    rows = [
        {
            "date": date, "symbol": symbol,
            "score": float(rng.uniform(0, 100)),
            "forward_return": float(returns.loc[date, symbol]),
        }
        for date in dates[60:]
        for symbol in symbols
    ]
    return pd.DataFrame(rows), returns


# --------------------------------------------------------------------------
# The book is well-formed
# --------------------------------------------------------------------------


class TestTheBookIsWellFormed:
    @pytest.mark.parametrize("scheme", ["equal", "inverse_vol", "hrp"])
    def test_weights_sum_to_one_on_every_date(self, market, scheme):
        panel, returns = market
        book = build_book(panel, scheme=scheme, returns=returns)

        sums = book.weights.sum(axis=1)
        assert np.allclose(sums, 1.0)

    @pytest.mark.parametrize("scheme", ["equal", "inverse_vol", "hrp"])
    def test_no_weight_is_negative(self, market, scheme):
        """Long-only is a hard constraint, not a tendency."""
        panel, returns = market
        book = build_book(panel, scheme=scheme, returns=returns)
        assert (book.weights >= -1e-12).all().all()

    def test_it_holds_the_names_the_spread_measured(self, market):
        """The comparison is only meaningful if the selection is the same."""
        from portfolio_agent.evaluation.costs import bucket_membership

        panel, returns = market
        book = build_book(panel, scheme="equal", returns=returns)
        membership = bucket_membership(panel)

        for date, row in book.weights.iterrows():
            held = set(row[row > 0].index)
            assert held == membership[date]

    def test_a_thin_date_produces_no_book(self):
        """Same threshold `bucket_analysis` uses, for the same reason."""
        rows = [
            {"date": pd.Timestamp("2023-01-02"), "symbol": f"S{i}",
             "score": float(i), "forward_return": 0.01}
            for i in range(3)
        ]
        book = build_book(pd.DataFrame(rows), scheme="equal")
        assert book.n_dates == 0
        assert book.n_skipped == 1
        assert any("too thin" in note for note in book.notes)


# --------------------------------------------------------------------------
# The schemes are actually different
# --------------------------------------------------------------------------


class TestTheSchemesDiffer:
    def test_equal_weights_are_equal(self, market):
        panel, returns = market
        book = build_book(panel, scheme="equal", returns=returns)
        row = book.weights.iloc[0]
        held = row[row > 0]
        assert np.allclose(held, held.iloc[0])

    def test_inverse_vol_is_not_equal(self, market):
        panel, returns = market
        book = build_book(panel, scheme="inverse_vol", returns=returns)
        row = book.weights.iloc[-1]
        held = row[row > 0]
        assert held.std() > 1e-6

    def test_inverse_vol_underweights_the_noisier_name(self, market):
        """The property, on the fixture's own volatility ordering.

        Name `S59` has roughly four times `S0`'s idiosyncratic volatility, so
        wherever both are held the quieter one takes more of the book.
        """
        panel, returns = market
        book = build_book(panel, scheme="inverse_vol", returns=returns)

        both = book.weights[(book.weights["S0"] > 0) & (book.weights["S59"] > 0)]
        if both.empty:
            pytest.skip("the random scores never held both names on one date")
        assert (both["S0"] > both["S59"]).mean() > 0.9

    def test_hrp_lowers_realized_volatility_against_equal(self, market):
        """What HRP is for. It uses the covariance structure without inverting
        it, and on a factor-driven cross-section that should show up as a
        quieter book rather than a higher return."""
        panel, returns = market

        equal = evaluate_book(build_book(panel, scheme="equal", returns=returns), panel)
        hrp = evaluate_book(build_book(panel, scheme="hrp", returns=returns), panel)

        assert hrp.annualized_volatility < equal.annualized_volatility

    def test_the_four_schemes_do_not_collapse_to_one(self, market):
        """The regression this module was written with, and nearly shipped.

        With a per-name cap below 1/n every scheme is forced to equal weights,
        and `compare_schemes` prints four indistinguishable rows. The obvious
        reading — "the weighting rule doesn't matter" — would be entirely an
        artifact of the cap.
        """
        panel, returns = market
        table = compare_schemes(
            panel, returns=returns, schemes=["equal", "inverse_vol", "hrp"]
        )
        assert table["book_mean_max_weight"].nunique() == 3


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


class TestTheCap:
    def test_the_default_does_not_bind(self):
        """None, deliberately: a decile book is too small for a low cap."""
        assert DEFAULT_MAX_WEIGHT is None

    def test_a_cap_below_one_over_n_is_unsatisfiable(self):
        assert cap_is_binding(0.10, 6) is False
        assert cap_is_binding(0.10, 20) is True
        assert cap_is_binding(None, 6) is False

    def test_an_unsatisfiable_cap_is_reported_not_swallowed(self, market):
        panel, returns = market
        book = build_book(panel, scheme="hrp", returns=returns, max_weight=0.10)

        assert any("unsatisfiable" in note for note in book.notes)

    def test_a_satisfiable_cap_actually_binds(self, market):
        """One clip-and-rescale does not converge; the loop has to."""
        panel, returns = market
        book = build_book(
            panel, scheme="hrp", returns=returns, n_buckets=3, max_weight=0.10
        )
        assert (book.weights <= 0.10 + 1e-9).all().all()
        assert np.allclose(book.weights.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# What a scheme is allowed to assume
# --------------------------------------------------------------------------


class TestSchemeInputs:
    def test_equal_needs_no_returns(self, market):
        panel, _ = market
        assert build_book(panel, scheme="equal").n_dates > 0

    @pytest.mark.parametrize("scheme", ["inverse_vol", "hrp", "mean_variance"])
    def test_the_others_refuse_rather_than_degrade(self, market, scheme):
        """Silently falling back would report an equal-weighted book under
        another scheme's name."""
        panel, _ = market
        with pytest.raises(ValueError, match="needs a trailing returns panel"):
            build_book(panel, scheme=scheme)

    def test_an_unknown_scheme_lists_the_known_ones(self, market):
        panel, _ = market
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            build_book(panel, scheme="kelly")

    def test_a_thin_covariance_falls_back_and_says_so(self, market):
        panel, returns = market
        book = build_book(
            panel, scheme="hrp", returns=returns, min_observations=10_000
        )
        assert any("fell back to equal weights" in note for note in book.notes)


class TestTheCovarianceIsCausal:
    def test_it_uses_only_returns_observable_on_the_date(self, market):
        """T19's convention, applied to the covariance estimate.

        Perturbing returns strictly after date D must not change the book built
        on D. Without the `<= date` slice the estimator would see the future of
        the very names it is sizing.
        """
        panel, returns = market
        cut = returns.index[120]

        tampered = returns.copy()
        tampered.loc[tampered.index > cut] *= 5.0

        base = build_book(panel, scheme="hrp", returns=returns)
        after = build_book(panel, scheme="hrp", returns=tampered)

        pd.testing.assert_frame_equal(
            base.weights.loc[base.weights.index <= cut],
            after.weights.loc[after.weights.index <= cut],
        )


# --------------------------------------------------------------------------
# Turnover and costs
# --------------------------------------------------------------------------


class TestTurnover:
    def test_an_unchanged_book_turns_over_nothing(self):
        weights = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
            index=pd.bdate_range("2023-01-02", periods=3),
        )
        assert list(weight_turnover(weights)[1:]) == [0.0, 0.0]

    def test_establishing_the_book_is_charged_one_leg(self):
        """Going from cash reports 0.5, and that is the right number.

        100% of the book is bought, so 0.5 looks like an understatement — but
        the caller charges `turnover x round_trip`, and establishing pays only
        the buy leg. Half a round trip *is* one leg. Pinned because it is a
        coincidence of two conventions rather than a derivation.
        """
        weights = pd.DataFrame(
            {"A": [0.5], "B": [0.5]}, index=pd.bdate_range("2023-01-02", periods=1)
        )
        turnover = weight_turnover(weights).iloc[0]
        assert turnover == pytest.approx(0.5)

        costs = CostModel.from_execution_sim()
        assert turnover * costs.round_trip == pytest.approx(costs.round_trip / 2.0)

    def test_a_complete_swap_is_one_hundred_percent(self):
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=pd.bdate_range("2023-01-02", periods=2),
        )
        assert weight_turnover(weights).iloc[1] == pytest.approx(1.0)

    def test_swapping_half_the_book_is_fifty_percent(self):
        """The halving is what makes it one-way, matching `one_way_turnover`."""
        weights = pd.DataFrame(
            {"A": [0.5, 0.5], "B": [0.5, 0.0], "C": [0.0, 0.5]},
            index=pd.bdate_range("2023-01-02", periods=2),
        )
        assert weight_turnover(weights).iloc[1] == pytest.approx(0.5)


class TestPerformance:
    def test_a_costless_book_nets_its_gross(self, market):
        panel, returns = market
        book = build_book(panel, scheme="equal", returns=returns)
        free = evaluate_book(
            book, panel, costs=CostModel(buy=0.0, sell=0.0, slippage_per_side=0.0)
        )
        assert free.mean_net == pytest.approx(free.mean_gross)
        assert free.cost_drag == pytest.approx(0.0, abs=1e-15)

    def test_costs_only_subtract(self, market):
        panel, returns = market
        book = build_book(panel, scheme="equal", returns=returns)
        charged = evaluate_book(book, panel)

        assert charged.mean_net < charged.mean_gross
        assert charged.cost_drag > 0

    def test_the_equity_curve_compounds_the_net_series(self, market):
        panel, returns = market
        performance = evaluate_book(
            build_book(panel, scheme="equal", returns=returns), panel
        )
        assert performance.equity_curve.iloc[-1] == pytest.approx(
            float((1.0 + performance.net).prod())
        )

    def test_total_return_matches_the_curve(self, market):
        panel, returns = market
        performance = evaluate_book(
            build_book(panel, scheme="equal", returns=returns), panel
        )
        assert performance.total_return == pytest.approx(
            performance.equity_curve.iloc[-1] - 1.0
        )

    def test_a_drawdown_is_never_positive(self, market):
        panel, returns = market
        performance = evaluate_book(
            build_book(panel, scheme="equal", returns=returns), panel
        )
        assert performance.max_drawdown <= 0.0

    def test_an_empty_book_evaluates_to_nothing_rather_than_crashing(self):
        from portfolio_agent.evaluation.allocation import BookWeights

        empty = BookWeights(weights=pd.DataFrame(), scheme="equal", n_dates=0)
        performance = evaluate_book(empty, pd.DataFrame(
            columns=["date", "symbol", "score", "forward_return"]
        ))
        assert performance.total_return == 0.0
        assert any("nothing was evaluated" in note for note in performance.notes)

    def test_breadth_travels_with_the_return(self, market):
        """A return without breadth hides the difference between a book spread
        across the decile and one holding a single name."""
        panel, returns = market
        performance = evaluate_book(
            build_book(panel, scheme="equal", returns=returns), panel
        )
        document = performance.to_dict()
        assert document["book_mean_names_held"] > 0
        assert 0 < document["book_mean_max_weight"] <= 1.0


# --------------------------------------------------------------------------
# The harness reports it
# --------------------------------------------------------------------------


class TestTheHarnessReportsTheBook:
    def test_no_weighting_means_no_book(self, market):
        from portfolio_agent.evaluation import evaluate_panel

        panel, _ = market
        assert evaluate_panel(panel, horizon=5).book is None

    def test_asking_for_one_produces_one(self, market):
        from portfolio_agent.evaluation import evaluate_panel

        panel, returns = market
        result = evaluate_panel(panel, horizon=5, weighting="hrp", returns=returns)

        assert result.book is not None
        assert result.book.scheme == "hrp"
        assert "book_sharpe" in result.to_dict()

    def test_equal_weighting_reproduces_the_spread_s_own_allocation(self, market):
        """The decile spread already assumes equal weights, so the book's gross
        return must be the top bucket's mean return — the same number the
        spread is built from."""
        from portfolio_agent.evaluation import evaluate_panel

        panel, returns = market
        result = evaluate_panel(panel, horizon=5, weighting="equal", returns=returns)

        assert result.book is not None
        assert result.book.mean_gross == pytest.approx(
            result.buckets.mean_returns[-1], rel=1e-9
        )
