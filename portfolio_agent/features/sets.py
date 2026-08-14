"""Which features a run trains on, defined once.

There were two hardcoded copies of the same eight names —
`agents/trainer.py::TRAINING_FEATURE_NAMES` and
`training/trainers/gbm.py::DEFAULT_GBM_FEATURES` — kept in step by a test
asserting them equal. The duplication had a real cause, stated in `gbm.py`:
the supervised list lives in a module behind `import torch`, and the point of
the boosting trainer is that it needs no torch. So the fix is not to delete one
copy but to give both a home that neither PyTorch nor scikit-learn gates. Same
move `features/labels.py` made for the label definitions in T06.

The larger problem the duplication was hiding
---------------------------------------------
Those eight names are `sma_20`, `sma_50`, `rsi_14`, `macd`, `bollinger_pct_b`,
`atr_14`, `return_1d`, `return_5d`. The registry holds 22. **None of
`mom_9m_skip1m`, `realized_vol_60`, `adx_14`, or any of the tradability screens
is in either list** — which is to say a model could not be trained on the
features the cross-sectional strategies actually rank on without editing
source. A platform whose strategies and whose models read different inputs is
comparing two things that were never the same experiment.

`features_for_strategy` closes that: a trainer can ask for exactly what a
strategy declares it needs, so "does a model beat the rule that inspired it"
becomes a question about the model rather than about which eight columns
somebody typed first.

Named sets, not a free-for-all
------------------------------
A run records its feature list in its manifest either way, so nothing is lost
by allowing an arbitrary list — and `TrainerConfig.features` already does.
Named sets exist because most runs want one of a few coherent answers, and a
name is what makes two runs comparable at a glance.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

#: The eight the supervised pipeline has always trained on. Unchanged, so no
#: existing run moves — this module starts by being a faithful home for what
#: was already there, and the wider sets are opt-in.
DEFAULT_TRAINING_FEATURES: List[str] = [
    "sma_20", "sma_50", "rsi_14", "macd",
    "bollinger_pct_b", "atr_14", "return_1d", "return_5d",
]

#: What the cross-sectional strategies rank on. Momentum's 9-month formation
#: skipping the most recent month, realized volatility, and the trend strength
#: the regime filter reads — none of which a model could previously be trained
#: against.
CROSS_SECTIONAL_FEATURES: List[str] = [
    "mom_9m_skip1m", "realized_vol_60", "adx_14",
    "sma_50", "sma_200", "atr_14", "return_5d", "traded_value_60",
]

#: The liquidity and circuit-lock screens. Not predictive features in the usual
#: sense — they are the reason a name is *excluded* — but a model that never
#: sees them cannot learn that a suppressed variance is illiquidity rather than
#: stability, which is the exact trap the low-volatility sort walks into.
TRADABILITY_FEATURES: List[str] = [
    "traded_value_60", "zero_return_fraction_60",
    "circuit_lock_fraction_60", "operator_trap_fraction_60",
]

#: Resolved at call time rather than frozen here, so a newly registered feature
#: is included without this module being edited — the failure mode a hardcoded
#: list has, and the one this module exists to remove.
_ALL = "all"

FEATURE_SETS: Dict[str, List[str]] = {
    "default": DEFAULT_TRAINING_FEATURES,
    "cross_sectional": CROSS_SECTIONAL_FEATURES,
    "tradability": TRADABILITY_FEATURES,
}


def list_feature_sets() -> List[str]:
    """Named sets a config or CLI may ask for, including `all`."""
    return sorted([*FEATURE_SETS, _ALL])


def resolve_feature_set(name: str) -> List[str]:
    """Names in a set, de-duplicated and order-preserving.

    Args:
        name: A key of `FEATURE_SETS`, or `"all"` for every registered feature.

    Raises:
        ValueError: On an unknown name, listing what is available. A typo that
            silently fell back to the default set would train a model on eight
            columns while its manifest recorded a different intent.
    """
    if name == _ALL:
        from .registry import list_features

        return sorted(list_features())

    if name not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature set {name!r}; available: {list_feature_sets()}"
        )
    return _unique(FEATURE_SETS[name])


def features_for_strategy(strategy: str) -> List[str]:
    """Exactly what a registered strategy declares it needs.

    The point of the seam: training a model on a strategy's own inputs is what
    makes "does the model beat the rule" a question about the model. Anything
    the strategy needs comes through unchanged, because `required_features()`
    is the strategy's own statement of its inputs and this does not
    second-guess it.

    Raises:
        ValueError: If the strategy is not registered, or declares a feature
            that is not. Either is a configuration error that would otherwise
            surface as a `KeyError` inside the feature pipeline, one frame
            below anything that names the strategy.
    """
    from portfolio_agent.config.schema import StrategyConfig
    from portfolio_agent.strategies.registry import STRATEGY_REGISTRY, load_strategy

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown strategy {strategy!r}; registered: {sorted(STRATEGY_REGISTRY)}"
        )

    names = _unique(load_strategy(StrategyConfig(type=strategy, params={})).required_features())

    from .registry import is_feature_registered

    unknown = [n for n in names if not is_feature_registered(n)]
    if unknown:
        raise ValueError(
            f"strategy {strategy!r} needs unregistered feature(s) {unknown}. "
            "A strategy's required_features() must name registry entries."
        )
    return names


def _unique(names: Sequence[str]) -> List[str]:
    """Order-preserving de-duplication.

    Order matters more than it looks: a trained checkpoint records its feature
    names and inference rebuilds the matrix in that order, so a set that came
    back permuted would feed a model its columns shuffled.
    """
    seen = set()
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
