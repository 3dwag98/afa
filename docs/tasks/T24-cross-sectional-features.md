# T24 — A feature registry that can see the universe

**Status:** done · **Effort:** ~1 day · **Depends on:** T21, T23
**Review reference:** round-three plan, finding D

## Goal

A registered feature can be a function of the whole cross-section, not only of
one ticker's own history.

## Why

`features/registry.py` binds exactly one shape:

```python
Series = f(one_ticker_ohlcv)
```

No date argument, no universe. That is correct for a moving average and
impossible for anything peer-relative — and *peer-relative is what a
cross-sectional book ranks on*. Idiosyncratic volatility is the residual
against the cross-section's own market. Beta is measured against it. Gross
profitability ranked within sector needs every peer on the date.

The evidence that the shape was missing is in the tree, not in an argument.
`features/market_relative.py` (T14):

- lives **outside** every registry,
- is **not imported** by `features/__init__.py`, so it registers nothing and
  appears in no `list-features` output,
- **re-implements the lag convention by hand** (`idiosyncratic_vol_from_closes`
  takes a `lag` parameter and shifts internally), and
- is reached by **importing it directly inside a strategy method**.

Every one of those is a symptom of a feature that had nowhere to be declared.

**This is the binding constraint on Phase 4.** No fundamental characteristic
worth having is expressible one ticker at a time.

## What shipped

`features/cross_section.py` adds the second shape:

```python
DataFrame(date x symbol) = f(CrossSectionPanel)
```

```python
@register_cross_sectional_feature("market_beta_60", inputs=("close",))
def market_beta_60(panel: CrossSectionPanel) -> pd.DataFrame:
    returns = panel.get("close").pct_change()   # already lagged
    return rolling_beta(returns, panel.benchmark, window=60)
```

`inputs` is declared rather than discovered, so the builder knows what to pivot
and a caller who cannot supply a column fails before any computation rather
than after.

### Two things the decorator enforces

**The lag.** `technical.py` opens with a capitalized declaration that every
feature shifts its input, and then each of twenty-two functions calls
`.shift(1)` itself. That is twenty-two chances to forget, and `market_relative`
already had a second convention. Here the decorator shifts every input frame
*and the benchmark* before the body runs, so a feature **cannot** read the
session it is used to decide.

`lag=0` remains available for a quantity genuinely known at the decision — the
per-ticker registry has exactly one such feature, `close`, the reference price.
A test enumerates every `lag=0` cross-sectional feature; the list is currently
empty, so an unlagged one is a decision someone has to make and defend rather
than an omission.

**The warm-up.** Measured on a synthetic universe rather than declared, for the
reason T23 established: a constant is right until someone registers a longer
window, and then it is silently wrong. "Defined" here means defined *for the
median symbol* — a feature can resolve early for one lucky name while the
cross-section it is meant to rank is still mostly NaN, and ranking a handful of
names against each other is the thin-cross-section failure
`MIN_CROSS_SECTION_NAMES` exists to refuse.

### The window belongs to the name

Registered: `idiosyncratic_vol_{20,60,120,252}` and
`market_beta_{20,60,120,252}`.

This matches what the per-ticker registry already does — `sma_20`, `sma_50`,
`sma_200`, `realized_vol_60` all carry their window in the name. A window
outside the family raises rather than rounding to a neighbour, because **a sort
measured over the wrong window looks exactly like a sort measured over the right
one.**

That was not hypothetical. `LowVolatilityStrategy` accepts an
`idiosyncratic_window` param, and the existing suite asserted only that
`entry_rules()["vol_window"]` *reported* the configured number. Nothing asserted
the residual was measured over it, so a first pass at this task silently pinned
every run to 60 sessions while the strategy went on reporting 120. The
strategy's configured window now selects the registry name, and a test covers
the gap.

### `rolling_beta` moved

From `evaluation/neutralize.py` to `features/market_relative.py`. It was
importing `market_composite` *from the feature layer* to do its work, and beta
is a characteristic of a stock rather than a piece of evaluation machinery —
betting-against-beta ranks on beta rather than neutralizing by it.
`neutralize.rolling_beta` delegates, so its callers and its tests are unchanged.

### The strategy contract

`BaseStrategy.required_cross_sectional_features()` returns `[]` by default. A
strategy that declares any must also report `requires_full_batch`: scored one
ticker at a time, a cross-sectional feature degenerates to a universe of one,
which is precisely the failure that flag exists to prevent. A test asserts the
implication across every loadable strategy.

## What changed

- `features/cross_section.py` (new) — `CrossSectionPanel`,
  `register_cross_sectional_feature`, `get_cross_sectional_feature`,
  `list_cross_sectional_features`, `is_cross_sectional_feature`,
  `required_columns`, `panel_from_frames`, `build_cross_section`,
  `latest_values`, `warmup_rows`.
- `features/market_relative.py` — `rolling_beta`; the registered window family;
  `idiosyncratic_vol_feature` / `market_beta_feature` name resolvers.
- `features/__init__.py` — exports the registry and imports `market_relative`
  for its registration side effect.
- `evaluation/neutralize.py` — `rolling_beta` delegates.
- `strategies/base.py` — `required_cross_sectional_features`.
- `strategies/cross_sectional.py` — `LowVolatilityStrategy` routes through the
  registry, and its window selects the feature name.

## What this does not do

The harness, the backtest engine and the trainers do not yet *build*
cross-sectional features for a strategy that declares them — `LowVolatilityStrategy`
still assembles its own panel inside `score_batch`, now through
`build_cross_section` rather than by hand. That is enough to remove the
mixed-convention hazard and to prove the seam, and it is deliberately less than
threading a second build stage through all three paths: doing that before a
strategy needs it there would be building the wiring against a guess about how
it will be used. The four Phase 3 strategies are that test.

`latest_values` omits symbols whose value is NaN on the decision date rather
than filling them. A strategy that substituted a fallback would rank a name it
could not measure alongside names it could — T14's rule, applied to the new
seam.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_cross_section_registry.py -q
python -m pytest portfolio_agent/tests/test_idiosyncratic_volatility.py -q
```
