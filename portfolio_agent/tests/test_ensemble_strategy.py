"""Tests for EnsembleStrategy ("UMA" — combining multiple strategies)."""

import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.ensemble import EnsembleStrategy
from portfolio_agent.strategies.registry import load_strategy, register_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal


class _FixedStrategy(BaseStrategy):
    """A test double that always returns the same signal, for deterministic ensemble tests."""

    def __init__(self, config: StrategyConfig):
        self._name = config.params.get("name", "fixed")
        self._signal = config.params.get("signal", "BUY")
        self._score = float(config.params.get("score", 80.0))

    @property
    def name(self) -> str:
        return self._name

    def required_features(self):
        return []

    def score(self, symbol, features, context) -> StrategySignal:
        return StrategySignal(
            symbol=symbol, signal=self._signal, score=self._score, trigger="Fixed",
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            reward_risk=2.0, probability_profit=0.6,
        )


register_strategy("fixed_test")(_FixedStrategy)


def _risk_params() -> RiskParams:
    return RiskParams(
        target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
        portfolio_value_inr=1000000.0, risk_per_trade_pct=0.01, max_single_position_pct=0.03,
    )


def _write_uma_yaml(tmp_path, method="weighted_blend", vote_mode="majority", members=None):
    if members is None:
        members = [
            {"type": "fixed_test", "weight": 0.5, "params": {"name": "A", "signal": "BUY", "score": 80.0}},
            {"type": "fixed_test", "weight": 0.5, "params": {"name": "B", "signal": "AVOID", "score": 20.0}},
        ]
    import yaml
    spec = {"name": "Test UMA", "method": method, "vote": {"mode": vote_mode}, "members": members}
    path = tmp_path / "uma.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)
    return path


class TestEnsembleStrategyBasics:
    def test_is_a_base_strategy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path)
        strategy = load_strategy(StrategyConfig(type="ensemble", params={"yaml_path": str(yaml_path)}))
        assert isinstance(strategy, BaseStrategy)
        assert strategy.name == "Test UMA"

    def test_missing_members_raises(self, tmp_path):
        import yaml
        path = tmp_path / "empty.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"name": "Empty"}, f)

        with pytest.raises(ValueError, match="members"):
            EnsembleStrategy(StrategyConfig(params={"yaml_path": str(path)}))

    def test_required_features_is_union_of_members(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "rule_based", "weight": 1.0, "config_path": "config/strategies/trend_breakout.yaml"},
            {"type": "fixed_test", "weight": 1.0, "params": {"name": "F"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        features = strategy.required_features()
        assert "sma_50" in features
        assert "atr_14" in features


class TestWeightedBlend:
    def test_equal_weight_opposing_signals_cancel_to_hold(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="weighted_blend")
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        # BUY (strength 1.0) at weight 0.5 + AVOID (strength -0.3) at weight 0.5
        # -> blended strength = 0.35 -> WATCH band, not BUY
        assert sig.trigger == "Ensemble:weighted_blend"
        assert sig.signal in ("WATCH", "HOLD")
        assert "member_signals" in sig.extra

    def test_unanimous_buy_gives_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "fixed_test", "weight": 0.7, "params": {"name": "A", "signal": "BUY", "score": 90.0}},
            {"type": "fixed_test", "weight": 0.3, "params": {"name": "B", "signal": "BUY", "score": 70.0}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"
        assert sig.score == pytest.approx(90.0 * 0.7 + 70.0 * 0.3)

    def test_weights_bias_the_blend(self, tmp_path):
        """A heavily-weighted BUY member should dominate a lightly-weighted AVOID member."""
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "fixed_test", "weight": 0.9, "params": {"name": "A", "signal": "BUY", "score": 90.0}},
            {"type": "fixed_test", "weight": 0.1, "params": {"name": "B", "signal": "AVOID", "score": 10.0}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"


class TestVote:
    def test_majority_vote_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="majority", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "C", "signal": "SELL"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"
        assert sig.trigger == "Ensemble:vote:majority"

    def test_majority_vote_no_consensus_gives_hold(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="majority", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "SELL"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "HOLD"

    def test_unanimous_vote_requires_all_agreement(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="unanimous", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "C", "signal": "AVOID"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "HOLD"

    def test_unanimous_vote_all_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="unanimous", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"


class TestExampleUmaYaml:
    def test_example_uma_yaml_loads(self):
        """The shipped example_uma.yaml (rule_based + lstm) must load without error."""
        strategy = load_strategy(StrategyConfig(
            type="ensemble", params={"yaml_path": "config/strategies/example_uma.yaml"}
        ))
        assert strategy.name == "Trend+ML Blend"
        assert strategy.method == "weighted_blend"
        assert len(strategy._members) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTriggerMethodUMA:
    """`method: trigger` routes members through the arbitration engine instead
    of averaging them, so conflict blocks rather than blends."""

    @staticmethod
    def _members(*specs):
        return [
            {"type": "fixed_test", "weight": 1.0, "params": dict(name=name, signal=signal, score=score)}
            for name, signal, score in specs
        ]

    def _uma(self, tmp_path, members, trigger=None, regimes=None):
        import yaml
        spec = {"name": "Trigger UMA", "method": "trigger", "members": members}
        if trigger is not None:
            spec["trigger"] = trigger
        if regimes is not None:
            spec["regimes"] = regimes
        path = tmp_path / "trigger_uma.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(spec, f)
        return EnsembleStrategy(StrategyConfig(type="ensemble", config_path=str(path)))

    def test_conflicting_strong_members_block_instead_of_averaging(self, tmp_path):
        """The DoD: BUY(0.90) against SELL(0.85) is a BLOCK. A weighted blend
        of the same two members reports a positive strength."""
        members = self._members(("A", "BUY", 90.0), ("B", "SELL", 15.0))
        strategy = self._uma(tmp_path, members)

        signal = strategy.score("ACME", pd.DataFrame(), StrategyContext(risk=_risk_params()))

        assert signal.signal == "AVOID"
        assert signal.trigger == "Trigger:BLOCK"
        assert signal.extra["position_scale"] == 0.0
        assert "conflict" in signal.rationale

    def test_agreeing_members_fire_and_carry_a_size_multiplier(self, tmp_path):
        members = self._members(("A", "BUY", 85.0), ("B", "BUY", 80.0))
        strategy = self._uma(tmp_path, members)

        signal = strategy.score("ACME", pd.DataFrame(), StrategyContext(risk=_risk_params()))

        assert signal.signal == "BUY"
        assert signal.trigger.startswith("Trigger:")
        assert 0.5 <= signal.extra["position_scale"] <= 1.0

    def test_the_levels_come_from_a_contributing_member_not_an_average(self, tmp_path):
        """Averaging a wide stop with a tight one produces a stop that belongs
        to neither thesis."""
        members = self._members(("A", "BUY", 90.0), ("B", "AVOID", 10.0))
        strategy = self._uma(tmp_path, members)

        signal = strategy.score("ACME", pd.DataFrame(), StrategyContext(risk=_risk_params()))

        assert signal.signal == "BUY"
        assert signal.stop_price == 95.0
        assert signal.target_price == 110.0

    def test_thresholds_are_configurable_from_the_yaml(self, tmp_path):
        members = self._members(("A", "BUY", 60.0))
        blocked = self._uma(tmp_path, members, trigger={"mode": "strong_single", "strong_confidence": 0.9})
        context = StrategyContext(risk=_risk_params())

        assert blocked.score("ACME", pd.DataFrame(), context).signal == "AVOID"

        allowed = self._uma(tmp_path, members, trigger={"mode": "strong_single", "strong_confidence": 0.5})
        assert allowed.score("ACME", pd.DataFrame(), context).signal == "BUY"

    def test_the_regime_map_mutes_members_out_of_season(self, tmp_path):
        members = self._members(("momentum", "BUY", 95.0), ("low_vol", "BUY", 20.0))
        strategy = self._uma(
            tmp_path, members,
            trigger={"mode": "strong_single", "strong_confidence": 0.7},
            regimes={"BULL_RISK_ON": ["momentum"], "BEAR_CRASH_RISK": ["low_vol"]},
        )

        bull = strategy.score(
            "ACME", pd.DataFrame(),
            StrategyContext(risk=_risk_params(), regime_label="BULL_RISK_ON"),
        )
        bear = strategy.score(
            "ACME", pd.DataFrame(),
            StrategyContext(risk=_risk_params(), regime_label="BEAR_CRASH_RISK"),
        )

        assert bull.signal == "BUY"
        assert bear.signal == "AVOID"
        assert "momentum" in bear.extra["trigger_muted_models"]

    def test_an_unknown_regime_permits_every_member(self, tmp_path):
        """Not knowing the regime is not evidence that every model is wrong;
        standing the whole book down on a lookup miss is the worse failure."""
        members = self._members(("momentum", "BUY", 95.0))
        strategy = self._uma(
            tmp_path, members,
            trigger={"mode": "strong_single", "strong_confidence": 0.7},
            regimes={"BULL_RISK_ON": ["momentum"]},
        )

        signal = strategy.score(
            "ACME", pd.DataFrame(),
            StrategyContext(risk=_risk_params(), regime_label="SOMETHING_ELSE"),
        )

        assert signal.signal == "BUY"

    def test_weighted_blend_would_have_produced_a_buy_from_the_same_conflict(self, tmp_path):
        """Pins the contrast the trigger method exists to fix."""
        import yaml
        members = self._members(("A", "BUY", 90.0), ("B", "SELL", 15.0))
        path = tmp_path / "blend.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"name": "Blend", "method": "weighted_blend", "members": members}, f)
        blended = EnsembleStrategy(StrategyConfig(type="ensemble", config_path=str(path)))

        signal = blended.score("ACME", pd.DataFrame(), StrategyContext(risk=_risk_params()))

        assert signal.signal != "AVOID"
        assert signal.score > 0


class TestMetaOrchestratorConfig:
    """The shipped multi-regime configuration must actually load and describe
    a coherent set of members — a regime map naming a member that does not
    exist mutes nothing and fails silently."""

    CONFIG = "portfolio_agent/config/strategies/uma_meta_orchestrator.yaml"

    def _spec(self):
        import yaml
        from pathlib import Path
        return yaml.safe_load(Path(self.CONFIG).read_text())

    def test_every_regime_maps_only_to_declared_members(self):
        spec = self._spec()
        declared = {
            m.get("name") or (m.get("params") or {}).get("name")
            for m in spec["members"]
        }

        for regime, members in spec["regimes"].items():
            unknown = set(members) - declared
            assert not unknown, f"{regime} names members that do not exist: {sorted(unknown)}"

    def test_every_member_is_reachable_in_some_regime(self):
        spec = self._spec()
        declared = {
            m.get("name") or (m.get("params") or {}).get("name")
            for m in spec["members"]
        }
        mapped = {name for members in spec["regimes"].values() for name in members}

        assert declared == mapped

    def test_momentum_is_muted_in_the_crash_regime(self):
        """The whole point of the map: momentum's catastrophic drawdowns are
        concentrated in exactly this state."""
        spec = self._spec()

        assert "quality_momentum" not in spec["regimes"]["BEAR_CRASH_RISK"]
        assert "defensive_low_vol" in spec["regimes"]["BEAR_CRASH_RISK"]

    def test_the_defensive_sleeve_is_muted_in_the_bull_regime(self):
        spec = self._spec()

        assert "defensive_low_vol" not in spec["regimes"]["BULL_RISK_ON"]
        assert "quality_momentum" in spec["regimes"]["BULL_RISK_ON"]

    def test_trend_following_is_muted_in_chop(self):
        spec = self._spec()

        assert "quality_momentum" not in spec["regimes"]["SIDEWAYS_CHOP"]

    def test_it_uses_the_trigger_method(self):
        assert self._spec()["method"] == "trigger"


class TestCrossSectionalMembers:
    """Cross-sectional rankers are legal UMA members under `method: trigger`,
    which scores every member across the whole universe before arbitrating."""

    @staticmethod
    def _universe(n=40):
        rows = {}
        for i in range(1, n + 1):
            rows[f"SYM{i}"] = pd.DataFrame([{
                "close": 100.0,
                "atr_14": 2.0,
                "mom_9m_skip1m": i / 100.0,
                # Rises with the momentum metric, so the low-volatility ranker
                # favours exactly the names momentum ranks last.
                "realized_vol_60": 0.10 + i / 200.0,
                "traded_value_60": 50_000_000.0,
                "zero_return_fraction_60": 0.0,
                "circuit_lock_fraction_60": 0.0,
                "circuit_locked_today": 0.0,
                "operator_trap_fraction_60": 0.0,
                "operator_trap_today": 0.0,
            }])
        return rows

    def _uma(self, tmp_path, method="trigger", regimes=None):
        import yaml
        spec = {
            "name": "Cross-sectional UMA",
            "method": method,
            "trigger": {"mode": "strong_or_consensus", "strong_confidence": 0.75},
            "members": [
                {"type": "momentum", "params": {"name": "mom", "min_universe": 10}},
                {"type": "low_volatility", "params": {"name": "lowvol", "min_universe": 10}},
            ],
        }
        if regimes is not None:
            spec["regimes"] = regimes
        path = tmp_path / "cs_uma.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(spec, f)
        return EnsembleStrategy(StrategyConfig(type="ensemble", config_path=str(path)))

    def test_a_cross_sectional_uma_propagates_full_batch(self, tmp_path):
        strategy = self._uma(tmp_path)

        assert strategy.requires_full_batch is True

    def test_it_ranks_across_the_universe_not_per_ticker(self, tmp_path):
        strategy = self._uma(tmp_path)
        universe = self._universe()

        signals = strategy.score_batch(universe, StrategyContext(risk=_risk_params()))

        assert len(signals) == len(universe)
        # A universe-of-one would make every name the top of its own ranking.
        assert sum(1 for s in signals.values() if s.signal == "BUY") < len(universe)

    def test_the_regime_map_selects_which_ranker_speaks(self, tmp_path):
        strategy = self._uma(
            tmp_path, regimes={"BULL_RISK_ON": ["mom"], "BEAR_CRASH_RISK": ["lowvol"]}
        )
        universe = self._universe()

        bull = strategy.score_batch(
            universe, StrategyContext(risk=_risk_params(), regime_label="BULL_RISK_ON")
        )
        bear = strategy.score_batch(
            universe, StrategyContext(risk=_risk_params(), regime_label="BEAR_CRASH_RISK")
        )

        bull_buys = {s for s, sig in bull.items() if sig.signal == "BUY"}
        bear_buys = {s for s, sig in bear.items() if sig.signal == "BUY"}

        # The two rankers order the universe oppositely, so the regimes must
        # not pick the same names.
        assert bull_buys and bear_buys
        assert bull_buys != bear_buys

    def test_averaging_methods_still_reject_cross_sectional_members(self, tmp_path):
        with pytest.raises(ValueError, match="method: trigger"):
            self._uma(tmp_path, method="weighted_blend")

    def test_duplicate_member_names_are_rejected(self, tmp_path):
        import yaml
        spec = {
            "name": "Dupes", "method": "trigger",
            "members": [
                {"type": "fixed_test", "params": {"name": "A", "signal": "BUY", "score": 90.0}},
                {"type": "fixed_test", "params": {"name": "A", "signal": "BUY", "score": 90.0}},
            ],
        }
        path = tmp_path / "dupes.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(spec, f)

        with pytest.raises(ValueError, match="unique"):
            EnsembleStrategy(StrategyConfig(type="ensemble", config_path=str(path)))
