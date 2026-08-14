"""One definition of which features a run trains on.

Two modules hardcoded the same eight names and a test asserted them equal —
which caught drift but institutionalised the copy. The duplication had a real
cause: the supervised list lived behind `import torch` and the boosting trainer
must not need one. A shared torch-free home removes the reason for the copy
rather than the copy alone.

What the copy was hiding matters more than the copy. The registry holds 22
features; both lists held the same 8, and **none of `mom_9m_skip1m`,
`realized_vol_60`, `adx_14` or the tradability screens was among them.** A
model could not be trained on the inputs the cross-sectional strategies rank
on without editing source, so "does the model beat the rule" was never a
question about the model.
"""

from __future__ import annotations

import pytest

from portfolio_agent.features.registry import list_features
from portfolio_agent.features.sets import (
    CROSS_SECTIONAL_FEATURES,
    DEFAULT_TRAINING_FEATURES,
    FEATURE_SETS,
    TRADABILITY_FEATURES,
    features_for_strategy,
    list_feature_sets,
    resolve_feature_set,
)


# --------------------------------------------------------------------------
# The copy is gone, and nothing moved
# --------------------------------------------------------------------------


class TestOneDefinition:
    def test_both_trainers_share_the_same_object(self):
        """Not merely equal — the same list. Equality is what the old test
        asserted, and it is satisfied by two copies that happen to match."""
        from portfolio_agent.agents.trainer import TRAINING_FEATURE_NAMES
        from portfolio_agent.training.trainers.gbm import DEFAULT_GBM_FEATURES

        assert TRAINING_FEATURE_NAMES is DEFAULT_TRAINING_FEATURES
        assert DEFAULT_GBM_FEATURES is DEFAULT_TRAINING_FEATURES

    def test_the_default_set_is_unchanged(self):
        """A faithful home for what was already there — no existing run moves."""
        assert DEFAULT_TRAINING_FEATURES == [
            "sma_20", "sma_50", "rsi_14", "macd",
            "bollinger_pct_b", "atr_14", "return_1d", "return_5d",
        ]

    def test_no_module_redefines_the_list(self):
        """Searched by shape, the way T10 does for RSI and T12 for rank IC."""
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent.parent
        literal = re.compile(r"\[\s*['\"]sma_20['\"]\s*,\s*['\"]sma_50['\"]", re.S)

        offenders = [
            str(path.relative_to(repo))
            for path in (repo / "portfolio_agent").rglob("*.py")
            if "tests" not in path.parts
            and path.name != "sets.py"
            and literal.search(path.read_text())
        ]
        assert offenders == [], offenders

    def test_the_shared_home_needs_neither_optional_extra(self):
        """The reason the copy existed. `features/sets.py` must import without
        torch or scikit-learn, or the boosting trainer is back where it was."""
        import subprocess
        import sys

        script = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['sklearn'] = None;"
            "from portfolio_agent.features.sets import DEFAULT_TRAINING_FEATURES;"
            "print(len(DEFAULT_TRAINING_FEATURES))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "8"


# --------------------------------------------------------------------------
# What the copy was hiding
# --------------------------------------------------------------------------


class TestTheStrategiesInputsWereUnreachable:
    @pytest.mark.parametrize(
        "feature", ["mom_9m_skip1m", "realized_vol_60", "adx_14"]
    )
    def test_they_are_registered_but_were_not_trainable(self, feature):
        """Registered all along, and absent from the only list a trainer read."""
        assert feature in list_features()
        assert feature not in DEFAULT_TRAINING_FEATURES

    def test_the_cross_sectional_set_now_reaches_them(self):
        resolved = resolve_feature_set("cross_sectional")
        assert "mom_9m_skip1m" in resolved
        assert "realized_vol_60" in resolved
        assert "adx_14" in resolved

    def test_a_strategys_own_inputs_are_resolvable(self):
        """The seam: train on exactly what the rule reads, so the comparison
        is about the model rather than about which columns someone typed."""
        momentum = features_for_strategy("momentum")
        assert "mom_9m_skip1m" in momentum
        assert set(momentum) >= {"close", "atr_14"}

    def test_two_strategies_declare_different_inputs(self):
        assert features_for_strategy("momentum") != features_for_strategy("rule_based")

    def test_an_unknown_strategy_raises_with_the_registry_listed(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            features_for_strategy("no_such_strategy")

    def test_the_result_is_deduplicated_and_ordered(self):
        """A checkpoint records its feature order and inference rebuilds the
        matrix in it, so a permuted set feeds a model shuffled columns."""
        names = features_for_strategy("low_volatility_idio")
        assert len(names) == len(set(names))
        assert names == features_for_strategy("low_volatility_idio")


# --------------------------------------------------------------------------
# Resolving a set
# --------------------------------------------------------------------------


class TestResolution:
    def test_every_named_set_resolves(self):
        for name in list_feature_sets():
            assert resolve_feature_set(name)

    def test_every_name_in_every_set_is_registered(self):
        """A set naming a feature the registry lacks fails deep in the
        pipeline, as a KeyError one frame below anything that says 'feature'."""
        registered = set(list_features())
        for name, names in FEATURE_SETS.items():
            missing = sorted(set(names) - registered)
            assert missing == [], f"{name}: {missing}"

    def test_all_tracks_the_registry_rather_than_a_frozen_copy(self):
        """The failure mode a hardcoded list has, and the one this removes."""
        assert resolve_feature_set("all") == sorted(list_features())

    def test_an_unknown_set_raises_and_lists_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown feature set"):
            resolve_feature_set("momentum_features")

    def test_resolution_is_order_preserving(self):
        assert resolve_feature_set("cross_sectional") == CROSS_SECTIONAL_FEATURES

    def test_the_tradability_set_is_the_exclusion_screens(self):
        assert set(TRADABILITY_FEATURES) < set(list_features())
        assert "traded_value_60" in TRADABILITY_FEATURES


# --------------------------------------------------------------------------
# Choosing one from config
# --------------------------------------------------------------------------


class TestConfig:
    def test_the_default_is_unchanged_behaviour(self):
        from portfolio_agent.config.schema import FeaturesConfig

        assert FeaturesConfig().training_set == "default"

    def test_a_named_set_validates(self):
        from portfolio_agent.config.schema import FeaturesConfig

        assert FeaturesConfig(training_set="cross_sectional").training_set == (
            "cross_sectional"
        )

    def test_a_typo_raises_rather_than_falling_back(self):
        """Falling back would train on eight columns while the manifest
        recorded a different intent."""
        from portfolio_agent.config.schema import FeaturesConfig

        with pytest.raises(ValueError, match="training_set must be one of"):
            FeaturesConfig(training_set="cross-sectional")

    def test_the_shipped_config_still_loads(self):
        from pathlib import Path

        from portfolio_agent.config.loader import load_config

        repo = Path(__file__).resolve().parent.parent.parent
        assert load_config(str(repo / "config.yaml")) is not None

    @pytest.mark.parametrize(
        "module,symbol",
        [
            ("portfolio_agent.agents.trainer", "resolve_feature_set"),
            ("portfolio_agent.training.trainers.gbm", "resolve_feature_set"),
        ],
    )
    def test_both_trainers_read_the_config(self, module, symbol):
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module))
        assert "training_set" in source, module
        assert symbol in source, module
