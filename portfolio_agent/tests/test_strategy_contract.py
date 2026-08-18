"""What a strategy author has to know, stated in the type rather than learned.

`BaseStrategy` declared three members. Callers relied on five: `load_strategy`
constructs every strategy as `cls(config)`, and two modules probed for `load()`
with `hasattr` — the shape of a contract that exists in practice and not in the
type. A fourth registry idiom, a module-private 175-line selection routine, and
two untyped `extra` keys reaching position sizing completed the surface a new
strategy had to reverse-engineer.

These tests pin the contract. They are deliberately about *the seam*: whether
momentum ranks well is a different question, asked in `test_strategies.py`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.registry import (
    STRATEGY_REGISTRY,
    get_available_strategies,
    get_strategy,
    is_strategy_registered,
    list_strategies,
    load_strategy,
    register_strategy,
    unavailable_strategies,
)
from portfolio_agent.strategies.types import (
    POSITION_SCALE_KEY,
    TRADABILITY_REJECT_KEY,
    StrategySignal,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """A strategy registered by a test must not outlive it.

    The built-ins are forced in *before* the snapshot. Registration is lazy and
    one-shot — `_BUILTINS_LOADED` is set on the first lookup — so snapshotting
    an empty registry and restoring it would leave the process with no
    strategies and no way to get them back.
    """
    from portfolio_agent.strategies import registry as module

    module._ensure_builtins_loaded()
    before = dict(STRATEGY_REGISTRY)
    try:
        yield
    finally:
        STRATEGY_REGISTRY.clear()
        STRATEGY_REGISTRY.update(before)


def _signal(**extra) -> StrategySignal:
    return StrategySignal(
        symbol="A.NS", signal="BUY", score=80.0, trigger="Trend",
        entry_price=100.0, stop_price=95.0, target_price=115.0,
        reward_risk=3.0, probability_profit=0.6, extra=extra,
    )


# --------------------------------------------------------------------------
# The registry is a decorator, like the other three
# --------------------------------------------------------------------------


class TestTheRegistryIsADecorator:
    def test_a_class_registers_itself(self):
        @register_strategy("probe_decorated")
        class Probe(BaseStrategy):
            @property
            def name(self) -> str:
                return "probe"

            def required_features(self):
                return []

            def score(self, symbol, features, context):  # pragma: no cover
                raise NotImplementedError

        assert get_strategy("probe_decorated") is Probe

    def test_the_decorator_returns_the_class_unchanged(self):
        class Probe(BaseStrategy):
            @property
            def name(self) -> str:
                return "probe"

            def required_features(self):
                return []

            def score(self, symbol, features, context):  # pragma: no cover
                raise NotImplementedError

        assert register_strategy("probe_identity")(Probe) is Probe

    def test_a_duplicate_name_is_refused(self):
        """Two classes under one key means whichever imported last wins, which
        is not a thing to discover from a backtest result."""
        with pytest.raises(ValueError, match="already registered"):

            @register_strategy("momentum")
            class Impostor(BaseStrategy):  # pragma: no cover - raises first
                @property
                def name(self) -> str:
                    return "impostor"

                def required_features(self):
                    return []

                def score(self, symbol, features, context):
                    raise NotImplementedError

    def test_re_registering_the_same_class_is_not_an_error(self):
        """Import order can register a module twice; that is not a conflict."""
        cls = get_strategy("momentum")
        register_strategy("momentum")(cls)
        assert get_strategy("momentum") is cls


class TestRegistrationLivesWithTheClass:
    def test_the_built_ins_are_all_registered(self):
        assert set(list_strategies()) >= {
            "rule_based", "momentum", "low_volatility", "low_volatility_idio",
            "ensemble",
        }

    def test_registration_happens_without_importing_the_modules_by_hand(self):
        """Lazy: the registry imports the strategy modules on first lookup."""
        assert is_strategy_registered("momentum")

    def test_rule_based_is_registered_exactly_once(self):
        """It used to be registered from `registry.py` *and* `__init__.py`,
        the second call silently overwriting the first."""
        classes = list(get_available_strategies().values())
        rule_based = get_strategy("rule_based")
        assert classes.count(rule_based) == 1

    def test_an_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match="Available:"):
            load_strategy(StrategyConfig(type="no_such_strategy", params={}))

    def test_an_unavailable_built_in_says_what_to_install(self):
        """Absence and unavailability are different facts.

        A name simply missing from the list looks like a typo; "needs the gpu
        extra" says what to do. Skipped when torch is present, since then
        nothing is unavailable.
        """
        unavailable = unavailable_strategies()
        if not unavailable:
            pytest.skip("torch is installed, so no built-in is unavailable")
        for name, reason in unavailable.items():
            assert "extra" in reason
            with pytest.raises(ValueError, match="built-in"):
                load_strategy(StrategyConfig(type=name, params={}))


# --------------------------------------------------------------------------
# The two contracts callers were probing for
# --------------------------------------------------------------------------


class TestTheConstructorAndLoadAreDeclared:
    def test_every_registered_strategy_constructs_from_a_config(self):
        for name in list_strategies():
            try:
                strategy = load_strategy(StrategyConfig(type=name, params={}))
            except ValueError:
                continue  # needs a members list or a checkpoint path
            assert isinstance(strategy, BaseStrategy), name

    def test_load_is_on_the_base_class(self):
        assert hasattr(BaseStrategy, "load")

    def test_a_strategy_with_nothing_to_load_is_ready_as_constructed(self):
        """The default is True rather than abstract: a rule-based strategy
        should not implement a method to say it needs nothing."""
        strategy = load_strategy(StrategyConfig(type="momentum", params={}))
        assert strategy.load() is True

    def test_callers_no_longer_probe_for_it(self):
        """The `hasattr(strategy, "load")` guard was the symptom."""
        import inspect

        from portfolio_agent.agents import backtester
        from portfolio_agent.strategies import ensemble

        for module in (backtester, ensemble):
            source = inspect.getsource(module)
            assert 'hasattr(strategy, "load")' not in source
            assert 'hasattr(member, "load")' not in source

    def test_the_base_constructor_records_the_config(self):
        class Probe(BaseStrategy):
            @property
            def name(self) -> str:
                return "probe"

            def required_features(self):
                return []

            def score(self, symbol, features, context):  # pragma: no cover
                raise NotImplementedError

        config = StrategyConfig(type="probe", params={"k": 1})
        assert Probe(config)._config is config


# --------------------------------------------------------------------------
# The selection routine a second cross-sectional strategy needs
# --------------------------------------------------------------------------


class TestRankAndSelectIsPublic:
    def test_it_is_importable_without_an_underscore(self):
        from portfolio_agent.strategies.cross_sectional import rank_and_select

        assert callable(rank_and_select)

    def test_the_private_name_is_gone(self):
        from portfolio_agent.strategies import cross_sectional

        assert not hasattr(cross_sectional, "_rank_and_select_decile")

    def test_a_new_strategy_needs_only_one_number_per_symbol(self, monkeypatch):
        """The claim the rename makes, exercised.

        Everything a cross-sectional strategy needs — tradability rejections,
        the minimum-universe abstention, the percentile score, the
        volatility-targeted scale, the reward:risk gate — is behind this one
        call. A new sort supplies a metric and a direction.
        """
        from portfolio_agent.src.regime import neutral_regime
        from portfolio_agent.strategies.cross_sectional import (
            CrashProtection,
            rank_and_select,
        )
        from portfolio_agent.strategies.types import RiskParams, StrategyContext

        symbols = [f"S{i}.NS" for i in range(40)]
        metric = {symbol: float(i) for i, symbol in enumerate(symbols)}
        latest = {
            symbol: pd.Series({
                "close": 100.0, "atr_14": 3.0, "realized_vol_60": 0.25,
            })
            for symbol in symbols
        }

        signals = rank_and_select(
            metric_by_symbol=metric,
            latest_by_symbol=latest,
            context=StrategyContext(
                risk=RiskParams(
                    target_prob_profit=0.55, min_reward_risk=1.5,
                    min_price_inr=20.0, portfolio_value_inr=1_000_000.0,
                    risk_per_trade_pct=0.01, max_single_position_pct=0.03,
                ),
            ),
            top_fraction=0.1,
            higher_is_better=True,
            trigger="Probe",
            component_name="Probe",
            min_universe=30,
            protection=CrashProtection(),
            regime=neutral_regime(),
            rejected={},
        )

        assert set(signals) == set(symbols)
        # The highest metric ranks first, so it scores 100 on the percentile.
        assert signals["S39.NS"].score == pytest.approx(100.0)
        assert signals["S0.NS"].score == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The two load-bearing `extra` keys
# --------------------------------------------------------------------------


class TestTheExtraKeysAreTyped:
    def test_position_scale_reads_through_the_accessor(self):
        assert _signal(**{POSITION_SCALE_KEY: 0.4}).position_scale == 0.4

    def test_absent_is_none_rather_than_one(self):
        """"This strategy does not scale positions" and "scale by one" are
        different statements, and both call sites treated them the same."""
        assert _signal().position_scale is None

    def test_an_unparseable_scale_is_none(self):
        assert _signal(**{POSITION_SCALE_KEY: "not a number"}).position_scale is None

    def test_a_non_finite_scale_is_none(self):
        assert _signal(**{POSITION_SCALE_KEY: float("inf")}).position_scale is None

    def test_scaling_shrinks(self):
        assert _signal(**{POSITION_SCALE_KEY: 0.5}).scaled_quantity(100) == 50

    def test_a_scale_at_or_above_one_cannot_lever_up(self):
        """The multiplier only ever shrinks a position."""
        assert _signal(**{POSITION_SCALE_KEY: 2.0}).scaled_quantity(100) == 100
        assert _signal(**{POSITION_SCALE_KEY: 1.0}).scaled_quantity(100) == 100

    def test_a_negative_scale_floors_at_zero(self):
        assert _signal(**{POSITION_SCALE_KEY: -0.5}).scaled_quantity(100) == 0

    def test_no_scale_leaves_the_quantity_alone(self):
        assert _signal().scaled_quantity(100) == 100

    def test_an_unparseable_scale_sizes_fully(self):
        """Preserved, not chosen. Both call sites have always done this, and
        tightening it is a risk decision rather than an ergonomics one — the
        docstring records the argument for changing it."""
        assert _signal(**{POSITION_SCALE_KEY: None}).scaled_quantity(100) == 100
        assert _signal(**{POSITION_SCALE_KEY: "x"}).scaled_quantity(100) == 100

    def test_the_reject_reason_reads_through_the_accessor(self):
        signal = _signal(**{TRADABILITY_REJECT_KEY: "illiquid"})
        assert signal.tradability_reject_reason == "illiquid"
        assert signal.is_tradable is False

    def test_presence_is_the_signal_not_truthiness(self):
        """A name that passed carries no key at all."""
        assert _signal().tradability_reject_reason is None
        assert _signal().is_tradable is True

    def test_the_verdict_reads_the_screen_marker(self):
        from portfolio_agent.strategies.types import ModelVerdict

        rejected = ModelVerdict.from_signal(_signal(**{TRADABILITY_REJECT_KEY: "zombie"}))
        assert rejected.liquidity_pass is False
        assert ModelVerdict.from_signal(_signal()).liquidity_pass is True


class TestBothSizingPathsShareOneImplementation:
    """They were byte-for-byte identical, kept in step by a comment saying so."""

    def test_the_engine_delegates(self):
        from portfolio_agent.src.backtest_engine import BacktestEngine

        signal = _signal(**{POSITION_SCALE_KEY: 0.25})
        assert BacktestEngine._apply_position_scale(80, signal) == 20

    def test_the_orchestrator_delegates(self):
        from portfolio_agent.execution.orchestrator import _scaled_quantity

        signal = _signal(**{POSITION_SCALE_KEY: 0.25})
        assert _scaled_quantity(80, signal) == 20

    @pytest.mark.parametrize(
        "scale", [None, 0.0, 0.25, 0.5, 0.999, 1.0, 2.0, -1.0, "x", float("nan")]
    )
    def test_they_agree_on_every_case(self, scale):
        from portfolio_agent.execution.orchestrator import _scaled_quantity
        from portfolio_agent.src.backtest_engine import BacktestEngine

        signal = _signal() if scale is None else _signal(**{POSITION_SCALE_KEY: scale})
        assert BacktestEngine._apply_position_scale(137, signal) == _scaled_quantity(
            137, signal
        )
