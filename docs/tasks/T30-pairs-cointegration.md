# T30 — Pairs and cointegration

**Status:** done · **Effort:** ~1 day · **Depends on:** T24, T25
**Review reference:** round-three plan, Phase 3 (closes it)

## Goal

Make a *relationship between two tickers* expressible, and find out whether
T24's and T25's seams were real.

## Why this was the test of the seams

`docs/QUANT_RESEARCH.md` §7 had scoped this out with an unusually precise
diagnosis:

> **Why it isn't implemented yet — an architectural gap, not a research gap.**
> Every strategy in this platform (`BaseStrategy::score()`/`score_batch()`)
> scores *one ticker at a time* against a shared context; pairs trading
> fundamentally needs a *relationship between two tickers* … Supporting it
> properly needs: (a) a pair-selection step (cointegration screening across
> O(n²) candidate pairs) that doesn't fit `BaseStrategy`'s per-ticker
> interface, and (b) either accepting long-only spread trades … or a
> deliberate, explicit exception to the short-selling guardrail.

Half of that was right, and the other half was informative in being wrong.

### (a) The diagnosis was in the wrong layer

The obstruction was attributed to `BaseStrategy`. It was not there.

T24's cross-sectional registry receives the whole `(date × symbol)` panel, so a
**feature** can screen pairs internally and emit a per-symbol number — how
cheap each name is against its own partner. `BaseStrategy` never had to change,
and `CointegrationPairsStrategy` is a `MomentumStrategy` subclass overriding
`_formation_metric` and nothing else.

So the honest finding is that §7 correctly identified an architectural gap and
placed it one layer away from where it lived. It was the *feature* layer that
could not express the question, and T24 is what fixed it — three tasks before
anyone tried to write this strategy.

### (b) The short leg is foregone, not solved

Textbook pairs trading is market-neutral because it shorts the expensive leg
against the cheap one. This platform does not short, so `pairs` buys the cheap
leg and stops.

**That is a real cost, not a technicality.** The signal carries full market
exposure, and its results are **not comparable** with published market-neutral
pairs returns. `PAIRS_NOT_NEUTRAL_NOTE` travels into every selection's notes
and the strategy reports `market_neutral: False`, so a report cannot use the
words "pairs trading" and leave a reader assuming neutrality.

§7's third option — "a deliberate, explicit exception to the short-selling
guardrail" — was not taken. A guardrail with one exception is a guardrail that
will acquire a second.

## The two ways a pairs backtest lies

Both are handled in `features/cointegration.py` and asserted on constructed
data rather than argued.

### Look-ahead through pair selection

Screening for cointegration on the whole sample and then trading the pairs it
found is severe and very easy to commit without noticing: **the pairs are
chosen because their spread mean-reverted over the period being evaluated.**

`rolling_pair_scores` walks the panel forward. Each screen sees only the
`formation` sessions **ending at** its refresh point, and its pairs apply only
to the `refresh_every` sessions **after** it. The test perturbs every price
after a cut date and asserts every score before it is byte-identical.

### Multiple testing

Screening every pair of an *n*-name universe is *n(n−1)/2* tests — 190 for 20
names, 1,225 for 50. Measured on twenty **independent random walks**, where
nothing is cointegrated:

| screen | pairs found | expected by chance |
| --- | --- | --- |
| uncorrected, p<0.05 | several | **9.5** |
| Bonferroni | ≤1 | **0.05** |

An uncorrected screen reliably finds cointegration in noise, and finds a lot of
it. The correction is on by default; `expected_false_positives` is **reported**
rather than merely applied, because it is the number that says whether a screen
finding forty pairs found anything. A separate test confirms the correction is
conservative rather than blind, by checking it still recovers a constructed
pair.

## A third trap, in the z-score

The spread's z-score uses the **formation** window's mean and standard
deviation, not the trading window's.

Re-estimating on the trading window would centre the z-score at zero *by
construction*, so it could never say the spread is stretched — which is the one
thing the number exists to say. Pinned by a test with a spread held constant at
+5 formation standard deviations.

## Scoring

For a pair with spread `P_left − β·P_right` and z-score `z`:

- `z` strongly **negative** → `left` is cheap → `left` scores `−z`
- `z` strongly **positive** → `right` is cheap → `right` scores `+z`

A symbol in several pairs takes the **mean** of its scores, not the extreme:
one stretched pair out of five is as likely to be that pair breaking down as it
is an opportunity, and the max would rank a name entirely on its single most
extreme relationship.

A symbol in no surviving pair is **absent** rather than zero — zero would read
as "fairly valued", a claim the screen did not make.

## Dependency

`statsmodels` moves from a transitive dependency of `arch` to a declared one.
MacKinnon's critical values are the reason not to hand-roll the ADF test, and
relying on another package's dependency for something load-bearing is how a
working install breaks on an unrelated upgrade.

## What changed

- `features/cointegration.py` (new) — `Pair`, `PairSelection`,
  `engle_granger`, `select_pairs`, `pair_scores`, `rolling_pair_scores`,
  `pair_cheapness_{126,252}`, `PAIRS_NOT_NEUTRAL_NOTE`.
- `features/__init__.py` — imports it for registration.
- `strategies/cross_sectional.py` — `CointegrationPairsStrategy`.
- `pyproject.toml` — `statsmodels` declared.
- `docs/QUANT_RESEARCH.md` §7 — rewritten; it said "not yet implemented".

## What this does not do

**No sector restriction on candidates.** `select_pairs(candidates=...)` accepts
one, and cointegration between two banks has an economic story where a bank and
a cement maker usually does not. The sector map is T32's deliverable, so the
hook exists and is unused.

**Not market-neutral**, per (b) above. This is the single most important thing
to carry into any reading of its results.

**Re-screening is quarterly, not adaptive.** A pair that breaks down mid-window
keeps being traded until the next refresh. Adaptive exit on a
cointegration-breakdown test is a real refinement and is not here.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_cointegration.py -q
portfolio-agent evaluate --strategy pairs --slippage-bps 25
```
