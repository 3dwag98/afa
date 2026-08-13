"""A strategy must not need a field only one caller happens to set.

`StrategyContext` has one required field and six optional ones, and which of
the six arrive depends entirely on who is calling. The backtest fills weights,
regime and (on the per-ticker path) a Monte Carlo result; the evaluation
harness fills the benchmark and nothing else.

`rule_based` read its component weights from `context.weights` alone. Under
`evaluate` that mapping is empty, `normalize_weights({})` returns `{}`, and the
weighted sum runs over nothing — so **every score was 0.0**. The resulting
"score dispersion 0.016, one floor value for 98% of the universe, no
cross-section left to rank" was published in `docs/tasks/README.md` as a
finding about the strategy. It was a finding about the harness.

The measurement, on a 40-name synthetic cross-section:

    score_dispersion, weights never supplied : 0.025
    score_dispersion, strategy self-supplies : 1.000

Nothing raised, at any point, in either direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.evaluation.metrics import score_dispersion
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.rule_based import (
    DEFAULT_COMPONENT_WEIGHTS,
    RuleBasedStrategy,
)
from portfolio_agent.strategies.types import RiskParams, StrategyContext

YAML = "portfolio_agent/config/strategies/trend_breakout.yaml"


def _risk() -> RiskParams:
    return RiskParams(
        target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=1.0,
        portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
        max_single_position_pct=0.10,
    )


@pytest.fixture
def strategy() -> RuleBasedStrategy:
    return load_strategy(StrategyConfig(type="rule_based", params={"yaml_path": YAML}))


@pytest.fixture
def features(strategy):
    """Forty names with a deliberate spread of drifts, so a working score ranks."""
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2022-01-03", periods=400)
    panel = {}
    for i in range(40):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005 * (i - 20) / 20, 0.015, 400)))
        frame = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": rng.integers(1e5, 1e6, 400).astype(float)},
            index=index,
        )
        panel[f"S{i:02d}"] = build_features(frame, strategy.required_features())
    return panel


def _panel(signals, date):
    return pd.DataFrame(
        [
            {"date": date, "symbol": s, "score": v.score,
             "forward_return": 0.01 * (int(s[1:]) - 20)}
            for s, v in signals.items()
        ]
    )


# --------------------------------------------------------------------------
# The defect, and that it is gone
# --------------------------------------------------------------------------


class TestScoringWithoutContextWeights:
    def test_the_strategy_scores_without_them(self, strategy, features):
        """The harness case. Every score used to be 0.0."""
        signals = strategy.score_batch(features, StrategyContext(risk=_risk()))
        scores = np.array([s.score for s in signals.values()])

        assert scores.max() > 0.0
        assert scores.std() > 0.0

    def test_dispersion_goes_from_a_floor_to_a_full_cross_section(
        self, strategy, features
    ):
        """The published finding, measured both ways.

        `score_dispersion` is the harness's own metric, so this is the number
        that appeared in the task index — not a proxy for it.
        """
        date = pd.Timestamp("2023-07-14")
        after = _panel(strategy.score_batch(features, StrategyContext(risk=_risk())), date)

        original = strategy._effective_weights
        strategy._effective_weights = lambda context: {}
        try:
            before = _panel(
                strategy.score_batch(features, StrategyContext(risk=_risk())), date
            )
        finally:
            strategy._effective_weights = original

        assert score_dispersion(before) < 0.05
        assert score_dispersion(after) > 0.9

    def test_an_empty_weight_map_really_does_zero_the_score(self):
        """The mechanism, isolated, so the cause stays legible."""
        from portfolio_agent.strategies.weighting import combine_weighted, normalize_weights

        components = {"Trend": 0.8, "Breakout": 0.6, "Volume": 0.7, "MC_Prob": 0.55}
        assert normalize_weights({}) == {}
        assert combine_weighted(components, {})[0] == 0.0
        assert combine_weighted(components, DEFAULT_COMPONENT_WEIGHTS)[0] > 0.0

    def test_both_paths_now_agree(self, strategy, features):
        """A backtest passing the same weights must not change the answer."""
        harness = strategy.score_batch(features, StrategyContext(risk=_risk()))
        backtest = strategy.score_batch(
            features,
            StrategyContext(risk=_risk(), weights=dict(DEFAULT_COMPONENT_WEIGHTS)),
        )
        for symbol in harness:
            assert harness[symbol].score == pytest.approx(backtest[symbol].score)


# --------------------------------------------------------------------------
# The weights are now read from the rules file
# --------------------------------------------------------------------------


class TestConfiguredWeights:
    def test_they_come_from_the_yaml(self, strategy):
        """`scoring.weights` was dead config — read nowhere, overridden always."""
        assert strategy._weights == {
            "Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0,
        }

    def test_editing_the_yaml_changes_the_score(self, tmp_path, features):
        import yaml as yaml_module

        source = yaml_module.safe_load(open(YAML))
        source["scoring"]["weights"] = {
            "trend": 100.0, "breakout": 0.0, "volume": 0.0, "model_probability": 0.0,
        }
        path = tmp_path / "trend_only.yaml"
        path.write_text(yaml_module.safe_dump(source))

        trend_only = load_strategy(
            StrategyConfig(type="rule_based", params={"yaml_path": str(path)})
        )
        assert trend_only._weights["Trend"] == 100.0

        default = load_strategy(
            StrategyConfig(type="rule_based", params={"yaml_path": YAML})
        )
        a = trend_only.score_batch(features, StrategyContext(risk=_risk()))
        b = default.score_batch(features, StrategyContext(risk=_risk()))
        assert any(a[s].score != pytest.approx(b[s].score) for s in a)

    def test_a_misspelled_weight_key_raises(self, tmp_path):
        """Silently weighting a component at zero is the failure being removed."""
        import yaml as yaml_module

        source = yaml_module.safe_load(open(YAML))
        source["scoring"]["weights"] = {"trend": 50.0, "momentum": 50.0}
        path = tmp_path / "typo.yaml"
        path.write_text(yaml_module.safe_dump(source))

        with pytest.raises(ValueError, match="unknown key"):
            load_strategy(StrategyConfig(type="rule_based", params={"yaml_path": str(path)}))

    def test_a_rules_file_without_weights_falls_back_to_the_defaults(self, tmp_path):
        import yaml as yaml_module

        source = yaml_module.safe_load(open(YAML))
        source["scoring"].pop("weights", None)
        path = tmp_path / "no_weights.yaml"
        path.write_text(yaml_module.safe_dump(source))

        strategy = load_strategy(
            StrategyConfig(type="rule_based", params={"yaml_path": str(path)})
        )
        assert strategy._weights == DEFAULT_COMPONENT_WEIGHTS

    def test_the_defaults_match_what_a_backtest_starts_from(self):
        """Otherwise an evaluation and day one of a backtest weigh differently."""
        from portfolio_agent.src.models import AgentBrain

        assert AgentBrain().weights == DEFAULT_COMPONENT_WEIGHTS

    def test_context_weights_still_override(self, strategy, features):
        """The backtest evolves weights across a run; that must keep working."""
        learned = {"Trend": 100.0, "Breakout": 0.0, "Volume": 0.0, "MC_Prob": 0.0}
        configured = strategy.score_batch(features, StrategyContext(risk=_risk()))
        overridden = strategy.score_batch(
            features, StrategyContext(risk=_risk(), weights=learned)
        )
        assert any(
            configured[s].score != pytest.approx(overridden[s].score)
            for s in configured
        )


# --------------------------------------------------------------------------
# The missing Monte Carlo stays a refusal, not a bad score
# --------------------------------------------------------------------------


class TestMonteCarloAbsence:
    def test_no_buy_without_a_monte_carlo_result(self, strategy, features):
        """Deliberate and unchanged: a compliance gate with no evidence either
        way refuses. T20 fixes the *score*, not the gate."""
        signals = strategy.score_batch(features, StrategyContext(risk=_risk()))
        assert not any(s.signal == "BUY" for s in signals.values())

    def test_the_rationale_says_why_rather_than_implying_a_bad_score(
        self, strategy, features
    ):
        signals = strategy.score_batch(features, StrategyContext(risk=_risk()))
        best = max(signals.values(), key=lambda s: s.score)
        assert "no MC result" in best.rationale
        assert best.score > 0.0

    def test_the_missing_component_redistributes_rather_than_scoring_zero(self):
        """`combine_weighted`'s `unavailable` argument, which already did the
        right thing and was simply never reached with usable weights."""
        from portfolio_agent.strategies.weighting import combine_weighted

        components = {"Trend": 0.8, "Breakout": 0.8, "Volume": 0.8, "MC_Prob": 0.0}
        with_mc_missing, _ = combine_weighted(
            components, DEFAULT_COMPONENT_WEIGHTS, unavailable=["MC_Prob"]
        )
        counted_as_zero, _ = combine_weighted(components, DEFAULT_COMPONENT_WEIGHTS)
        assert with_mc_missing > counted_as_zero


# --------------------------------------------------------------------------
# The contract is written down
# --------------------------------------------------------------------------


def test_the_context_documents_which_caller_supplies_what():
    """A field that is present on some paths and absent on others is the whole
    hazard; the table is the mitigation."""
    from portfolio_agent.strategies.types import StrategyContext

    doc = StrategyContext.__doc__ or ""
    assert "backtest" in doc and "evaluate" in doc
    assert "weights" in doc and "mc_result" in doc
    assert "treat every optional field as absent" in doc.lower()
