# T20 — One `StrategyContext` contract

**Status:** done · **Effort:** ~1 day · **Depends on:** nothing
**Review reference:** round-three plan, finding A

## Goal

A strategy must not need a field that only one caller happens to set.

## Why

`StrategyContext` has one required field and six optional ones, and which of
the six arrive depends entirely on the caller:

| field | backtest, per-ticker | backtest, batched | evaluate | live |
| --- | --- | --- | --- | --- |
| `risk` | yes | yes | yes | yes |
| `weights` | yes | yes | **no** | yes |
| `mc_result` | yes | **no** | **no** | yes |
| `benchmark_close` / `_ohlcv` | **no** | yes | yes | yes |
| `regime_label` | yes | yes | **no** | yes |

Nothing wrote that table down, and nothing enforced it.

## What it cost

`rule_based` read its component weights from `context.weights` and nowhere
else. `evaluation/harness.py:572` builds a context with `risk` and the two
benchmark fields. So under `evaluate`:

    normalize_weights({})  ->  {}
    combine_weighted(components, {})  ->  0.0

**Every `rule_based` score the harness ever produced was zero.** Measured on a
40-name synthetic cross-section with the harness's own metric:

| | `score_dispersion` |
| --- | ---: |
| weights never supplied (pre-T20) | **0.025** |
| strategy self-supplies (post-T20) | **1.000** |

The published finding in `docs/tasks/README.md` — *"score dispersion is 0.016:
one floor value for 98% of the universe, and no cross-section left to rank"* —
was that floor. It has been retracted there and points here.

## The deeper defect

The constructor docstring has always claimed the rules file supplies "the
weights the method is applied to". It did not: only `scoring.method` was read.
**`scoring.weights` in every strategy YAML was dead config**, and the strategy
was entirely dependent on `AgentBrain`'s defaults arriving through a context
field — defaults which happen to duplicate the YAML's values, so nothing ever
looked wrong in a backtest.

That is the pluggability failure underneath the arithmetic one. A strategy that
cannot score itself without a caller filling in its parameters is not a unit
anyone can reuse.

## Approach

- `RuleBasedStrategy._load_weights` reads `scoring.weights` from the rules
  file, mapping the file's vocabulary (`trend`, `model_probability`) to the
  scoring code's (`Trend`, `MC_Prob`) through one stated table. An unknown key
  **raises** — silently weighting a component at zero is the failure being
  removed.
- `_effective_weights(context)` returns `context.weights` when the caller has
  them, else the configured set. The backtest evolves weights across a run and
  still overrides; `evaluate` has no learning loop and scores the strategy as
  configured.
- `DEFAULT_COMPONENT_WEIGHTS` equals `AgentBrain`'s defaults, asserted by a
  test, so an evaluation and day one of a backtest weigh components identically
  rather than by accident of which path ran.
- `StrategyContext`'s docstring now carries the table above and the rule it
  implies: **treat every optional field as absent and degrade to something
  defensible.**

## What deliberately did not change

**`rule_based` still cannot emit BUY under `evaluate`.** Its probability gate
fails closed with no Monte Carlo result, which is correct — a compliance gate
with no evidence either way should refuse, not wave the trade through. T20
fixes the *score*, which is what rank IC is computed on; the gate is a separate
and defensible decision, and the rationale already says "prob(no MC result):
FAIL" rather than implying the stock scored badly.

`combine_weighted`'s `unavailable` argument already redistributed a missing
component's weight instead of counting it as zero — the T04 principle working
as designed. It was simply never reached with usable weights.

## Acceptance criteria

- [x] The strategy scores without `context.weights`.
- [x] Dispersion measured both ways with the harness's own metric, so the
      retracted number is comparable to the original.
- [x] The zeroing mechanism isolated in a test, so the cause stays legible.
- [x] Configured weights actually take effect; editing the YAML changes scores.
- [x] A misspelled weight key raises rather than zeroing a component.
- [x] `context.weights` still overrides, so the backtest's learning loop works.
- [x] Defaults match `AgentBrain`, asserted.
- [x] The missing-MC refusal is unchanged and still explains itself.
- [x] The per-caller table is documented on `StrategyContext` and tested for.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/strategies/rule_based.py` | `_load_weights`, `_effective_weights`, `DEFAULT_COMPONENT_WEIGHTS`, `_YAML_WEIGHT_KEYS` |
| `portfolio_agent/strategies/types.py` | `StrategyContext` per-caller contract |
| `portfolio_agent/tests/test_strategy_context_contract.py` | New — 14 tests |
| `docs/tasks/README.md` | The `rule_based` finding retracted |

## Still open

The table documents the divergence; it does not remove it. `evaluate` still
sets no `regime_label`, so a strategy gated on regime behaves as "not assessed"
there while the backtest gates it — the cross-sectional strategies happen to
re-derive their own regime from `benchmark_close`, which the harness does
supply, so momentum's crash filter works under `evaluate` while a UMA-level
regime map would not. That belongs with the panel unification in T21, not here.
