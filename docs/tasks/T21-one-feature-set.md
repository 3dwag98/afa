# T21 — One feature set

**Status:** done · **Effort:** ~1 day · **Depends on:** nothing
**Review reference:** round-three plan, finding C (feature list)

> **Scope note.** The plan's T21 bundled the feature list with the panel-builder
> merge. They are separable — `prepare_panel` already takes `feature_names` as
> an argument — and the list is the base the builder consumes, so it ships
> first. The builder merge is [T23](T23-one-panel-policy.md).

## Goal

One definition of which features a run trains on, reachable from config, and
derivable from a strategy.

## Why

Two hardcoded copies of the same eight names:

- `agents/trainer.py::TRAINING_FEATURE_NAMES`
- `training/trainers/gbm.py::DEFAULT_GBM_FEATURES`

kept in step by a test asserting them equal. The duplication had a stated,
real cause — `gbm.py` says so: the supervised list lives behind `import torch`
and the point of the boosting trainer is that it needs none. So the fix is not
to delete one copy but to give both a home that neither PyTorch nor
scikit-learn gates. Same move `features/labels.py` made in T06.

## What the copy was hiding

The registry holds **22** features. Both lists held the same **8**:

    sma_20, sma_50, rsi_14, macd, bollinger_pct_b, atr_14, return_1d, return_5d

**None of `mom_9m_skip1m`, `realized_vol_60`, `adx_14`, or any tradability
screen was among them.** Those are what the cross-sectional strategies rank on.
So a model could not be trained on the inputs its own strategies read without
editing source, and "does the model beat the rule that inspired it" was never a
question about the model — it was a question about which eight columns somebody
typed first.

## Approach

`features/sets.py`, importable without either optional extra (asserted by a
subprocess test that stubs both out):

| set | contents |
| --- | --- |
| `default` | the original eight, **unchanged**, so no existing run moves |
| `cross_sectional` | what momentum and low-volatility actually rank on |
| `tradability` | the liquidity and circuit-lock screens |
| `all` | resolved from the registry at call time, not frozen |

`all` resolves live on purpose: a frozen copy is the failure mode this module
exists to remove, so a newly registered feature is included without editing it.

`features_for_strategy(name)` returns exactly what a registered strategy
declares. That is the seam — train on the rule's own inputs and the comparison
is about the model.

`features.training_set` in config picks a set, validated: a typo raises rather
than falling back, because falling back would train on eight columns while the
manifest recorded a different intent. Both trainers read it; an explicit
`features` list on a trainer config still wins.

## Acceptance criteria

- [x] Both trainers reference the **same object**, not two equal lists — the
      old test asserted equality, which two copies that happen to match satisfy.
- [x] The default set is byte-identical, so nothing that ran before moves.
- [x] A shape search finds no third copy, the way T10 does for RSI.
- [x] `features/sets.py` imports with torch and scikit-learn stubbed out.
- [x] Every name in every set is registered — a set naming a missing feature
      fails as a `KeyError` one frame below anything that says "feature".
- [x] `all` tracks the registry rather than a frozen list.
- [x] A strategy's own inputs are resolvable, de-duplicated and order-stable —
      a checkpoint records feature order and inference rebuilds in it.
- [x] Config validates the name; the shipped config still loads.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/features/sets.py` | New — the sets, `resolve_feature_set`, `features_for_strategy` |
| `portfolio_agent/agents/trainer.py` | Re-exports; reads `features.training_set` |
| `portfolio_agent/training/trainers/gbm.py` | Re-exports; reads the same config |
| `portfolio_agent/config/schema.py` | `FeaturesConfig.training_set`, validated |
| `portfolio_agent/tests/test_feature_sets.py` | New — 24 tests |

## What this enables

```yaml
features:
  training_set: cross_sectional    # train on what the strategies rank on
```

The comparison that was previously impossible:

```bash
portfolio-agent train --trainer gbm            # on momentum's own inputs
portfolio-agent evaluate --strategy momentum   # the rule those inputs came from
```

## Not done here

**`features_for_strategy` is not yet wired to a CLI flag.** The function is the
hard part and is tested; `--features from:momentum` is a parser change that
belongs with T23's builder work, where the feature list stops being passed
separately by each caller.
