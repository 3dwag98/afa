# T25 — Declare the strategy contract

**Status:** done · **Effort:** ~1 day · **Depends on:** T24
**Review reference:** round-three plan, finding F

## Goal

Everything a new strategy has to satisfy is visible from `BaseStrategy` and the
registry, rather than discoverable by reading call sites.

## Why

The surface is small — three abstract members — and the floor for actually
writing one was not. Four separate things a strategy author had to learn the
hard way.

### 1. Registration was a function call in a second file

The strategy registry was the last of four still using `register(name, cls)`.
Features, models and trainers all use a decorator. The consequences:

- a new strategy had to be edited into two files, the class and the registry;
- **`rule_based` was registered twice** — once in `registry.py`, once in
  `strategies/__init__.py` — the second call silently overwriting the first
  with the same class, which is harmless only by luck;
- nothing rejected a duplicate name, so two classes under one key meant
  whichever module imported last won.

### 2. `__init__(config)` and `load()` were required and declared nowhere

`load_strategy` constructs every strategy as `cls(config)`. Every subclass
happened to accept one; nothing said it had to.

`load()` is worse, because the callers show what they thought of it:

```python
# agents/backtester.py
if hasattr(strategy, "load") and not strategy.load():

# strategies/ensemble.py
if hasattr(member, "load") and not member.load():
```

A `hasattr` probe is the shape of a contract that exists in practice and not in
the type.

### 3. `_rank_and_select_decile` was module-private

175 lines holding the tradability rejections, the minimum-universe abstention,
the percentile score, the volatility-targeted `position_scale`, and the
reward:risk gate — which is to say, **everything a second cross-sectional
strategy needs**. Phase 3 adds four of them. Each would have had to import an
underscore-prefixed name or reimplement the routine, and the second is how two
strategies come to disagree about what "top decile" means.

### 4. Two `extra` keys were untyped strings across five modules

`extra["position_scale"]` reaches position sizing in **both** the backtest
engine and the live orchestrator. `extra["tradability_reject_reason"]` decides
`ModelVerdict.liquidity_pass`. Neither had a constant, a type, or an accessor.

And the sizing logic was duplicated byte-for-byte:

```python
# execution/orchestrator.py::_scaled_quantity
"""Mirrors BacktestEngine._apply_position_scale so live and backtested
sizing cannot drift..."""
```

A comment asserting two copies cannot drift is not a mechanism.

## What shipped

**`@register_strategy("momentum")` on the class**, matching `register_trainer`
and `register_feature`. A duplicate name raises; re-registering the same class
does not, because import order can legitimately do that.

Registration is **lazy**, for the reason `training/registry.py` gives: importing
a registry must not drag in PyTorch, since rule-based backtests are supported
without the `gpu` extra. It also resolves the circularity a decorator
introduces — a strategy module imports `register_strategy` from the registry, so
the registry cannot import the strategy modules at its top.
`unavailable_strategies()` reports the torch-gated built-ins by name, so a
missing one says what to install rather than looking like a typo.

**`__init__(config)` and `load()` on `BaseStrategy`.** `load()` defaults to
`True` rather than being abstract — a rule-based strategy should not have to
implement a method to say it needs nothing. Both `hasattr` probes are gone.

**`rank_and_select` is public.** A new cross-sectional strategy now reduces to:
compute one number per symbol, call this with `higher_is_better` set the right
way. A test exercises exactly that path with a synthetic metric.

**`POSITION_SCALE_KEY` / `TRADABILITY_REJECT_KEY` constants**, plus
`StrategySignal.position_scale`, `.tradability_reject_reason`, `.is_tradable`
and `.scaled_quantity(qty)`. Both sizing paths call the last one, so they cannot
drift rather than merely being asked not to. A parametrized test runs the two
call sites against each other across ten scale values.

## A behaviour deliberately not changed

Both sizing call sites have always treated an **unparseable** `position_scale`
as "no scale", i.e. size fully. That is preserved.

It is arguably wrong: a corrupt multiplier is a bug, and sizing fully on a bug
is the more expensive of the two failures — sizing *nothing* is the safe
direction. But tightening it is a risk decision rather than an ergonomics one,
and burying it in a refactor is how a behaviour change escapes review. It is
recorded in `StrategySignal.position_scale`'s docstring and asserted in a test
that says explicitly that it is preserved, not chosen.

## What changed

- `strategies/registry.py` — rewritten: decorator, lazy built-ins,
  `get_strategy`, `list_strategies`, `is_strategy_registered`,
  `unavailable_strategies`.
- `strategies/__init__.py` — re-exports only; the duplicate `rule_based`
  registration is gone.
- `strategies/base.py` — `__init__(config)`, `load()`.
- `strategies/{rule_based,cross_sectional,ml_strategy,india_sac,ensemble}.py` —
  `@register_strategy` at each class.
- `strategies/cross_sectional.py` — `rank_and_select`; writes through the key
  constants.
- `strategies/types.py` — the constants and the four accessors.
- `src/backtest_engine.py`, `execution/orchestrator.py` — both delegate.
- `agents/backtester.py`, `strategies/ensemble.py` — probes removed.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_strategy_contract.py -q
python -m pytest portfolio_agent/tests/test_strategies.py -q
```
