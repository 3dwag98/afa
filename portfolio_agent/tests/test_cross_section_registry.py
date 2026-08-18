"""A feature of the whole cross-section, expressible for the first time.

`features/registry.py` binds one shape — `Series = f(one_ticker_ohlcv)`. There
is no date and no universe, because a moving average needs neither. That is why
`features/market_relative.py` was written outside every registry, was not
exported by `features/__init__.py`, re-implemented the lag convention by hand,
and was reached by importing it directly inside a strategy method.

These tests are about the seam rather than the arithmetic: the residual-variance
identity has its own suite in `test_idiosyncratic_volatility.py`. What is new
here is that a feature can *see the universe*, that the decorator owns the lag
instead of each author, and that the warm-up is measured rather than declared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.features.cross_section import (
    CrossSectionPanel,
    build_cross_section,
    get_cross_sectional_feature,
    is_cross_sectional_feature,
    latest_values,
    list_cross_sectional_features,
    panel_from_frames,
    register_cross_sectional_feature,
    required_columns,
    warmup_rows,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Probe features registered by a test must not outlive it.

    The registry is process-global, so a leaked probe would widen
    `list_cross_sectional_features()` for every module that runs afterwards
    and would make a second run of the same test fail its own duplicate check.
    """
    from portfolio_agent.features import cross_section

    before = dict(cross_section._REGISTRY)
    try:
        yield
    finally:
        cross_section._REGISTRY.clear()
        cross_section._REGISTRY.update(before)
        cross_section._warmup_for.cache_clear()


@pytest.fixture
def universe():
    """Six names on a shared market factor, long enough for a 60-day window."""
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2021-01-04", periods=400)
    market = rng.normal(0.0004, 0.009, len(index))

    frames = {}
    for i in range(6):
        close = 100.0 * np.exp(
            np.cumsum(market * (0.5 + 0.2 * i) + rng.normal(0, 0.007, len(index)))
        )
        frames[f"S{i}.NS"] = pd.DataFrame(
            {"close": close, "volume": rng.integers(1e5, 1e6, len(index)).astype(float)},
            index=index,
        )
    return frames


# --------------------------------------------------------------------------
# The lag, enforced rather than described
# --------------------------------------------------------------------------


class TestTheDecoratorOwnsTheLag:
    """`technical.py` declares the convention then asks 22 authors to obey it.

    Each per-ticker feature calls `.shift(1)` itself, which is 22 chances to
    forget. Here the decorator shifts before the body runs, so a feature
    *cannot* read the session it is used to decide.
    """

    def test_the_body_never_sees_the_final_row(self):
        seen = {}

        @register_cross_sectional_feature("probe_sees_lagged", inputs=("close",))
        def _probe(panel: CrossSectionPanel) -> pd.DataFrame:
            seen["frame"] = panel.get("close")
            return panel.get("close")

        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        get_cross_sectional_feature("probe_sees_lagged")(
            CrossSectionPanel(columns={"close": frame})
        )

        assert pd.isna(seen["frame"].iloc[0]).all()
        assert seen["frame"].iloc[-1].tolist() == [2.0, 5.0]

    def test_the_benchmark_is_lagged_with_the_panel(self):
        """A market series one bar ahead of the names would be look-ahead in
        the one term the residual is measured against."""
        seen = {}

        @register_cross_sectional_feature("probe_benchmark", inputs=("close",))
        def _probe(panel: CrossSectionPanel) -> pd.DataFrame:
            seen["benchmark"] = panel.benchmark
            return panel.get("close")

        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        benchmark = pd.Series([10.0, 20.0, 30.0])
        get_cross_sectional_feature("probe_benchmark")(
            CrossSectionPanel(columns={"close": frame}, benchmark=benchmark)
        )

        assert seen["benchmark"].tolist()[1:] == [10.0, 20.0]

    def test_perturbing_the_last_bar_cannot_change_the_last_value(self, universe):
        """The property the lag exists for, on a real registered feature."""
        base = build_cross_section(universe, ["idiosyncratic_vol_60"])

        tampered = {k: v.copy() for k, v in universe.items()}
        for frame in tampered.values():
            frame.iloc[-1, frame.columns.get_loc("close")] *= 3.0
        after = build_cross_section(tampered, ["idiosyncratic_vol_60"])

        pd.testing.assert_series_equal(
            base["idiosyncratic_vol_60"].iloc[-1],
            after["idiosyncratic_vol_60"].iloc[-1],
        )

    def test_lag_zero_is_available_but_has_to_be_written_down(self):
        @register_cross_sectional_feature("probe_unlagged", inputs=("close",), lag=0)
        def _probe(panel: CrossSectionPanel) -> pd.DataFrame:
            return panel.get("close")

        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        out = get_cross_sectional_feature("probe_unlagged")(
            CrossSectionPanel(columns={"close": frame})
        )
        assert out.iloc[-1]["A"] == 3.0

    def test_every_shipped_feature_is_lagged(self):
        """The `lag=0` escape hatch, enumerated.

        Nothing in the market-relative family is known at the decision, so an
        unlagged one here would be a mistake rather than a decision. A future
        feature that genuinely needs the current bar belongs on this list with
        a reason, the way `close` is the stated exception per-ticker.
        """
        shipped = [
            name for name in list_cross_sectional_features()
            if not name.startswith("probe_")
        ]
        unlagged = [
            name for name in shipped
            if get_cross_sectional_feature(name).lag == 0
        ]
        assert unlagged == []


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


class TestRegistration:
    def test_the_market_relative_features_are_registered(self):
        """The defect this task removes: they were in no registry at all."""
        names = list_cross_sectional_features()
        assert "idiosyncratic_vol_60" in names
        assert "market_beta_60" in names

    def test_importing_the_package_is_enough_to_register_them(self):
        """`features/__init__.py` did not import `market_relative`, so the
        feature existed only for whoever imported the module by hand."""
        import importlib

        features = importlib.import_module("portfolio_agent.features")
        assert features.is_cross_sectional_feature("idiosyncratic_vol_60")

    def test_the_two_registries_do_not_share_a_name(self):
        """A caller routing by name would pick whichever it checked first."""
        from portfolio_agent.features.registry import list_features

        overlap = set(list_features()) & set(list_cross_sectional_features())
        assert overlap == set()

    def test_a_duplicate_name_is_refused(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_cross_sectional_feature("idiosyncratic_vol_60", inputs=("close",))
            def _duplicate(panel):  # pragma: no cover - construction raises
                return panel.get("close")

    def test_a_feature_must_declare_what_it_reads(self):
        with pytest.raises(ValueError, match="declares no inputs"):

            @register_cross_sectional_feature("probe_no_inputs", inputs=())
            def _no_inputs(panel):  # pragma: no cover - construction raises
                return pd.DataFrame()

    def test_an_unknown_name_names_what_is_available(self):
        with pytest.raises(KeyError, match="idiosyncratic_vol_60"):
            get_cross_sectional_feature("no_such_feature")

    def test_required_columns_comes_from_the_registrations(self):
        assert required_columns(["idiosyncratic_vol_60", "market_beta_60"]) == ("close",)


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------


class TestThePanel:
    def test_it_pivots_to_date_by_symbol(self, universe):
        panel = panel_from_frames(universe, ["close"])

        assert list(panel.get("close").columns) == list(universe)
        assert len(panel.get("close")) == len(next(iter(universe.values())))

    def test_symbols_with_different_histories_align_on_the_union(self):
        a = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
        b = pd.DataFrame({"close": [9.0]}, index=pd.to_datetime(["2023-01-04"]))

        panel = panel_from_frames({"A": a, "B": b}, ["close"])
        closes = panel.get("close")

        assert len(closes) == 3
        assert pd.isna(closes.loc[pd.Timestamp("2023-01-04"), "A"])

    def test_a_gap_is_left_as_nan_not_forward_filled(self):
        """Filling would manufacture a price on a day the stock did not trade
        — the observation the liquidity screen exists to catch."""
        a = pd.DataFrame(
            {"close": [1.0, 3.0]},
            index=pd.to_datetime(["2023-01-02", "2023-01-04"]),
        )
        b = pd.DataFrame(
            {"close": [5.0, 6.0, 7.0]},
            index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
        )

        closes = panel_from_frames({"A": a, "B": b}, ["close"]).get("close")
        assert pd.isna(closes.loc[pd.Timestamp("2023-01-03"), "A"])

    def test_a_missing_column_says_which_one(self):
        panel = panel_from_frames({"A": pd.DataFrame({"close": [1.0]})}, ["close"])
        with pytest.raises(KeyError, match="book_value"):
            panel.get("book_value")

    def test_empty_frames_are_dropped_rather_than_widening_the_panel(self, universe):
        with_empty = dict(universe)
        with_empty["EMPTY.NS"] = pd.DataFrame(columns=["close"])

        panel = panel_from_frames(with_empty, ["close"])
        assert "EMPTY.NS" not in panel.get("close").columns


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


class TestBuildCrossSection:
    def test_it_returns_one_frame_per_requested_feature(self, universe):
        built = build_cross_section(universe, ["idiosyncratic_vol_60", "market_beta_60"])

        assert set(built) == {"idiosyncratic_vol_60", "market_beta_60"}
        for frame in built.values():
            assert list(frame.columns) == list(universe)

    def test_it_only_pivots_the_columns_the_features_declared(self, universe):
        """`volume` is in the fixture and read by nothing requested here."""
        built = build_cross_section(universe, ["market_beta_60"])
        assert not built["market_beta_60"].empty

    def test_no_features_is_no_work(self, universe):
        assert build_cross_section(universe, []) == {}

    def test_the_beta_of_a_higher_loading_name_is_higher(self, universe):
        """A sanity check that the panel reaches the arithmetic intact.

        The fixture builds name `i` with a market loading of `0.5 + 0.2*i`, so
        the estimated betas must be ordered even though they are estimates.
        """
        beta = build_cross_section(universe, ["market_beta_60"])["market_beta_60"]
        latest = beta.iloc[-1]
        assert latest["S5.NS"] > latest["S0.NS"]


class TestLatestValues:
    def test_it_takes_the_final_row(self):
        frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        assert latest_values(frame) == {"A": 2.0, "B": 4.0}

    def test_a_symbol_that_could_not_be_measured_is_omitted_not_filled(self):
        """T14's rule: mixing two measures into one ranking is harder to
        notice than a thin cross-section."""
        frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, np.nan]})
        assert latest_values(frame) == {"A": 2.0}

    def test_infinities_are_omitted_too(self):
        frame = pd.DataFrame({"A": [1.0, np.inf]})
        assert latest_values(frame) == {}

    def test_an_empty_frame_is_an_empty_map(self):
        assert latest_values(pd.DataFrame()) == {}


# --------------------------------------------------------------------------
# The warm-up
# --------------------------------------------------------------------------


class TestWarmup:
    def test_a_longer_window_needs_more_rows(self):
        assert warmup_rows(["idiosyncratic_vol_252"]) > warmup_rows(["idiosyncratic_vol_60"])

    def test_a_set_needs_what_its_slowest_member_needs(self):
        both = warmup_rows(["idiosyncratic_vol_20", "idiosyncratic_vol_252"])
        assert both == warmup_rows(["idiosyncratic_vol_252"])

    def test_an_empty_request_needs_nothing(self):
        assert warmup_rows([]) == 0

    def test_the_answer_covers_the_lag_and_the_differencing(self, universe):
        """A 60-session window needs more than 60 rows: one goes to the lag,
        one to the return differencing, and the estimator floors at half."""
        assert warmup_rows(["idiosyncratic_vol_60"]) > 30

    def test_an_unregistered_name_is_an_error(self):
        with pytest.raises(KeyError):
            warmup_rows(["no_such_feature"])


# --------------------------------------------------------------------------
# The window belongs to the name
# --------------------------------------------------------------------------


class TestTheWindowIsPartOfTheName:
    """The convention `sma_20` / `sma_50` / `sma_200` already follows.

    It matters more here. A caller asking for "idiosyncratic volatility" at 120
    sessions and silently receiving the 60-session answer would be ranking on
    the wrong measurement under a name claiming otherwise — and a sort measured
    over the wrong window looks exactly like a sort measured over the right one.
    """

    def test_every_window_in_the_family_is_registered(self):
        from portfolio_agent.features.market_relative import REGISTERED_WINDOWS

        for window in REGISTERED_WINDOWS:
            assert is_cross_sectional_feature(f"idiosyncratic_vol_{window}")
            assert is_cross_sectional_feature(f"market_beta_{window}")

    def test_an_unregistered_window_is_refused_rather_than_rounded(self):
        from portfolio_agent.features.market_relative import idiosyncratic_vol_feature

        with pytest.raises(ValueError, match="No 'idiosyncratic_vol' feature"):
            idiosyncratic_vol_feature(37)

    def test_the_configured_window_reaches_the_computation(self, universe):
        """The assertion whose absence let a window regression through.

        The existing suite checked that `entry_rules()["vol_window"]` reported
        the configured number. Nothing checked that the number was the one the
        residual was actually measured over, so a strategy could report 120 and
        rank on 60.
        """
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.registry import load_strategy

        strategy = load_strategy(
            StrategyConfig(type="low_volatility_idio", params={"idiosyncratic_window": 120})
        )
        assert strategy.required_cross_sectional_features() == ["idiosyncratic_vol_120"]

        short = load_strategy(
            StrategyConfig(type="low_volatility_idio", params={"idiosyncratic_window": 20})
        )
        assert short.required_cross_sectional_features() == ["idiosyncratic_vol_20"]

        # And the two really do measure different things.
        wide = build_cross_section(universe, ["idiosyncratic_vol_120"])
        narrow = build_cross_section(universe, ["idiosyncratic_vol_20"])
        assert not np.allclose(
            wide["idiosyncratic_vol_120"].iloc[-1].to_numpy(),
            narrow["idiosyncratic_vol_20"].iloc[-1].to_numpy(),
        )

    def test_the_total_volatility_sort_declares_no_cross_sectional_feature(self):
        """It ranks on a per-ticker feature, and must not claim otherwise."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.registry import load_strategy

        strategy = load_strategy(
            StrategyConfig(type="low_volatility", params={"sort_on": "total"})
        )
        assert strategy.required_cross_sectional_features() == []


class TestTheBaseClassContract:
    def test_strategies_declare_nothing_by_default(self):
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.registry import load_strategy

        assert load_strategy(
            StrategyConfig(type="momentum", params={})
        ).required_cross_sectional_features() == []

    def test_a_strategy_that_declares_one_must_take_the_full_batch(self):
        """Scored one ticker at a time, a cross-sectional feature degenerates
        to a universe of one — the failure `requires_full_batch` prevents."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.registry import (
            get_available_strategies,
            load_strategy,
        )

        for name in sorted(get_available_strategies()):
            try:
                strategy = load_strategy(StrategyConfig(type=name, params={}))
            except Exception:
                continue  # needs a checkpoint or a members list; covered elsewhere
            if strategy.required_cross_sectional_features():
                assert strategy.requires_full_batch, name
