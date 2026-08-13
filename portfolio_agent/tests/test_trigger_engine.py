"""Tests for signal arbitration (src/trigger_engine.py).

The behaviour under test is the one linear blending gets wrong: when two
strong models point in opposite directions, the answer is "stand aside", not
"take a small position in whichever direction won the average".
"""

import pytest

from portfolio_agent.strategies.types import ModelVerdict, StrategySignal
from portfolio_agent.src.trigger_engine import TriggerConfig, TriggerDecision, TriggerEngine


def _verdict(name, action="BUY", confidence=0.8, ev=5.0, **kwargs):
    return ModelVerdict(
        model_name=name,
        action=action,
        confidence=confidence,
        expected_net_ev_pct=ev,
        **kwargs,
    )


class TestConflictHandling:
    def test_two_strong_opposing_models_block_the_trade(self):
        """The headline case. A weighted blend of BUY(0.90) and SELL(0.85)
        reports a mild BUY; maximal disagreement is not mild evidence."""
        engine = TriggerEngine()

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.90),
            _verdict("mean_reversion", "SELL", 0.85),
        ])

        assert decision.action == "BLOCK"
        assert decision.allowed is False
        assert decision.size_multiplier == 0.0
        assert "conflict" in decision.reason
        assert decision.vetoes == ["model_conflict"]

    def test_the_conflict_penalty_is_multiplicative(self):
        """c_eff = c_buy * (1 - max(c_opposing)), so a weak dissent shaves
        conviction rather than being averaged away."""
        engine = TriggerEngine(TriggerConfig(conflict_veto_confidence=1.01))

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 1.0),
            _verdict("quant", "SELL", 0.2),
        ])

        assert decision.effective_confidence == pytest.approx(0.8)

    def test_a_weak_dissent_below_the_veto_does_not_block_outright(self):
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.6))

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.95),
            _verdict("quant", "SELL", 0.2),
        ])

        assert decision.action == "BUY"
        assert decision.opposing_models == ["quant"]

    def test_an_abstaining_model_is_not_an_opposing_one(self):
        """AVOID means 'no opinion'. Counting it against the trade would let a
        model that simply has no view veto one that does."""
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.7))

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.9),
            _verdict("lstm", "AVOID", 0.0),
        ])

        assert decision.action == "BUY"
        assert decision.opposing_models == []


class TestGlobalVetoes:
    def test_a_failed_liquidity_screen_blocks_regardless_of_conviction(self):
        engine = TriggerEngine()

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 1.0, liquidity_pass=False),
            _verdict("lstm", "BUY", 1.0),
        ])

        assert decision.action == "BLOCK"
        assert decision.vetoes == ["liquidity_pass"]

    def test_expected_value_below_the_hurdle_blocks(self):
        engine = TriggerEngine(TriggerConfig(min_net_ev_pct=1.0))

        decision = engine.evaluate([_verdict("momentum", "BUY", 0.95, ev=0.4)])

        assert decision.action == "BLOCK"
        assert decision.vetoes == ["min_net_ev_pct"]
        assert "hurdle" in decision.reason

    def test_an_unestimable_expected_value_does_not_trip_the_hurdle(self):
        """None means 'cannot estimate', not 'zero'. A ranking model with no
        probability estimate must not be vetoed for being honest about it."""
        engine = TriggerEngine(TriggerConfig(min_net_ev_pct=1.0, strong_confidence=0.7))

        decision = engine.evaluate([_verdict("momentum", "BUY", 0.95, ev=None)])

        assert decision.action == "BUY"
        assert decision.expected_net_ev_pct is None

    def test_expected_value_is_confidence_weighted(self):
        engine = TriggerEngine(TriggerConfig(strong_confidence=0.6))

        decision = engine.evaluate([
            _verdict("a", "BUY", 0.9, ev=10.0),
            _verdict("b", "BUY", 0.1, ev=0.0),
        ])

        assert decision.expected_net_ev_pct == pytest.approx(9.0)

    def test_no_verdicts_blocks(self):
        assert TriggerEngine().evaluate([]).action == "BLOCK"

    def test_only_abstentions_blocks(self):
        decision = TriggerEngine().evaluate([_verdict("a", "AVOID", 0.0)])

        assert decision.action == "BLOCK"
        assert "no model voted to buy" in decision.reason


class TestRegimeGating:
    def test_a_muted_model_is_silenced_not_amplified_into_a_veto(self):
        """One sleeve being out of season must not stop the sleeve that is in
        it — otherwise the regime map can only ever reduce the book to nothing."""
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.7))

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.95, regime_compatible=False),
            _verdict("low_volatility", "BUY", 0.9),
        ])

        assert decision.action == "BUY"
        assert decision.muted_models == ["momentum"]
        assert decision.contributing_models == ["low_volatility"]

    def test_muting_every_buyer_blocks(self):
        engine = TriggerEngine()

        decision = engine.evaluate([_verdict("momentum", "BUY", 0.95, regime_compatible=False)])

        assert decision.action == "BLOCK"
        assert decision.vetoes == ["regime_compatible"]
        assert "muted" in decision.reason

    def test_veto_policy_blocks_on_any_incompatible_model(self):
        engine = TriggerEngine(TriggerConfig(regime_policy="veto"))

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.95, regime_compatible=False),
            _verdict("low_volatility", "BUY", 0.9),
        ])

        assert decision.action == "BLOCK"
        assert decision.vetoes == ["regime_compatible"]


class TestFiringModes:
    def test_strong_single_fires_on_one_convinced_model(self):
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.75))

        assert engine.evaluate([_verdict("a", "BUY", 0.8)]).action == "BUY"
        assert engine.evaluate([_verdict("a", "BUY", 0.7)]).action == "BLOCK"

    def test_strong_single_ignores_consensus(self):
        """Three mildly positive models are not a substitute for one convinced
        one when the mode says otherwise."""
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.9))

        decision = engine.evaluate([
            _verdict("a", "BUY", 0.6), _verdict("b", "BUY", 0.6), _verdict("c", "BUY", 0.6),
        ])

        assert decision.action == "BLOCK"

    def test_consensus_needs_enough_agreeing_models(self):
        engine = TriggerEngine(
            TriggerConfig(mode="consensus", consensus_confidence=0.55, min_consensus_models=2)
        )

        assert engine.evaluate([_verdict("a", "BUY", 0.99)]).action == "BLOCK"
        assert engine.evaluate(
            [_verdict("a", "BUY", 0.6), _verdict("b", "BUY", 0.6)]
        ).action == "BUY"

    def test_consensus_ignores_a_lone_strong_model(self):
        engine = TriggerEngine(TriggerConfig(mode="consensus"))

        decision = engine.evaluate([_verdict("a", "BUY", 1.0), _verdict("b", "BUY", 0.1)])

        assert decision.action == "BLOCK"

    def test_strong_or_consensus_accepts_either_path(self):
        engine = TriggerEngine(TriggerConfig(mode="strong_or_consensus"))

        lone_strong = engine.evaluate([_verdict("a", "BUY", 0.9), _verdict("b", "BUY", 0.1)])
        agreeing = engine.evaluate([_verdict("a", "BUY", 0.6), _verdict("b", "BUY", 0.6)])

        assert lone_strong.action == "BUY"
        assert lone_strong.fired_rule == "strong_single"
        assert agreeing.action == "BUY"
        assert agreeing.fired_rule == "consensus"


class TestSizeMultiplier:
    def test_a_trade_that_barely_clears_its_threshold_is_sized_down(self):
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.75))

        decision = engine.evaluate([_verdict("a", "BUY", 0.75)])

        assert decision.size_multiplier == pytest.approx(0.5)

    def test_full_conviction_earns_full_size(self):
        engine = TriggerEngine(TriggerConfig(mode="strong_single", strong_confidence=0.75))

        decision = engine.evaluate([_verdict("a", "BUY", 1.0)])

        assert decision.size_multiplier == pytest.approx(1.0)

    def test_size_never_exceeds_the_configured_ceiling(self):
        engine = TriggerEngine(
            TriggerConfig(mode="strong_single", strong_confidence=0.2, max_size_multiplier=0.8)
        )

        decision = engine.evaluate([_verdict("a", "BUY", 1.0)])

        assert decision.size_multiplier <= 0.8

    def test_a_blocked_trade_is_sized_at_zero(self):
        decision = TriggerEngine().evaluate([_verdict("a", "BUY", 0.1)])

        assert decision.size_multiplier == 0.0


class TestModelVerdictMapping:
    @staticmethod
    def _signal(**overrides):
        base = dict(
            symbol="ACME", signal="BUY", score=80.0, trigger="Momentum",
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            reward_risk=2.0, probability_profit=0.6,
        )
        base.update(overrides)
        return StrategySignal(**base)

    def test_a_buy_maps_score_straight_to_confidence(self):
        verdict = ModelVerdict.from_signal(self._signal(), model_name="momentum")

        assert verdict.action == "BUY"
        assert verdict.confidence == pytest.approx(0.8)
        assert verdict.model_name == "momentum"

    def test_a_sell_takes_the_complement_of_the_score(self):
        """A model emitting SELL at score 15 is 85% convinced, not 15%."""
        verdict = ModelVerdict.from_signal(self._signal(signal="SELL", score=15.0))

        assert verdict.action == "SELL"
        assert verdict.confidence == pytest.approx(0.85)

    @pytest.mark.parametrize("signal", ["HOLD", "WATCH", "AVOID"])
    def test_non_directional_signals_abstain(self, signal):
        verdict = ModelVerdict.from_signal(self._signal(signal=signal))

        assert verdict.action == "AVOID"
        assert verdict.confidence == 0.0

    def test_expected_value_uses_the_already_costed_reward_risk(self):
        """EV_R = p*b - (1-p) = 0.6*2 - 0.4 = 0.8 R; risk is 5% of entry."""
        verdict = ModelVerdict.from_signal(self._signal())

        assert verdict.expected_net_ev_pct == pytest.approx(4.0)

    def test_expected_value_is_none_without_a_probability(self):
        verdict = ModelVerdict.from_signal(self._signal(probability_profit=0.0))

        assert verdict.expected_net_ev_pct is None

    def test_a_screened_name_reports_a_liquidity_failure(self):
        verdict = ModelVerdict.from_signal(
            self._signal(signal="AVOID", extra={"tradability_reject_reason": "illiquid"})
        )

        assert verdict.liquidity_pass is False

    def test_the_verdict_is_immutable(self):
        """Verdicts are evidence, not scratch space: an arbitration pass must
        not be able to edit the opinions it is weighing."""
        verdict = ModelVerdict.from_signal(self._signal())

        with pytest.raises(Exception):
            verdict.confidence = 0.1


class TestTriggerConfigFromParams:
    def test_unknown_keys_are_ignored_and_defaults_preserved(self):
        config = TriggerConfig.from_params({"mode": "consensus", "nonsense": 1})

        assert config.mode == "consensus"
        assert config.strong_confidence == TriggerConfig().strong_confidence

    def test_none_yields_defaults(self):
        assert TriggerConfig.from_params(None) == TriggerConfig()


class TestDecisionShape:
    def test_blocked_decisions_report_who_was_involved(self):
        engine = TriggerEngine()

        decision = engine.evaluate([
            _verdict("momentum", "BUY", 0.9),
            _verdict("reversion", "SELL", 0.9),
            _verdict("lowvol", "BUY", 0.4),
        ])

        assert isinstance(decision, TriggerDecision)
        assert decision.contributing_models == ["lowvol", "momentum"]
        assert decision.opposing_models == ["reversion"]
