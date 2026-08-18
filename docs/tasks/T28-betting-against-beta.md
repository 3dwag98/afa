# T28 — Betting against beta, and when it pays

**Status:** done · **Effort:** ~1 day · **Depends on:** T24, T25, T27
**Review reference:** round-three plan, Phase 3

## Goal

Long the low-beta decile, and a measurement that can say the effect is
concentrated in one market state rather than reporting an average of two
different things.

## Why the strategy

Frazzini & Pedersen's account is a funding-constraint one: investors who want
more risk than they can borrow to obtain bid up high-beta stocks instead, so
beta is overpriced and the security market line is flatter than CAPM predicts.
A beta-sorted long-short book earns a positive alpha; the long-only half is the
low-beta decile, which is what this platform can hold.

**Indian evidence is favourable and specific.** NSE 2001-2016 finds the effect
positive across capitalizations *after controlling for size, value and
momentum*. That control matters: the naive objection to a low-beta sort is that
it is a size bet wearing a different name.

## Why the conditional split

2025 Asian work finds the effect **concentrated in downturns**. A pooled IC
made of a strong down-market number and a flat up-market one describes neither
state, and the platform had no way to report the difference.

`evaluation/conditional.py` splits IC and decile spread by market state. It
distinguishes two conditioners that are easy to conflate, and the distinction
is the module's main content:

| conditioner | splits on | answers | tradable |
| --- | --- | --- | --- |
| `realized` (default) | market return **over the label horizon** | when did the signal pay? | **no** |
| `trailing` | market state **as of the decision date** | can I condition on this? | yes |

Reading a `realized` split as though it were `trailing` turns an attribution
into an imaginary timing rule — on the decision date nobody knows which bucket
the date will land in. So the conditioner travels into every result and every
note, and `trailing` **refuses to run without a market series** rather than
falling back to `realized` and silently answering a different question.

## A flaw the verification found in the measurement

`is_conditional` originally tested *"significant in one state and not the
other"* — the shape the low-risk anomaly is usually described as having.

That misses the stronger case: **significant in both, with opposite signs.** It
is the easier one to miss precisely because both halves look healthy in
isolation. On the verification panel the low-beta signal scores **+0.32 in
falling markets and −0.35 in rising ones** — pooling gives roughly zero, and
"no skill" is the one description that is wrong about both halves.

Both cases now trip the flag, and `signs_disagree` reports the second
separately. `conditional_notes` says which one occurred.

The flag is deliberately crude, and says so: comparing two Newey-West
t-statistics is *not* a test of their difference. Putting a p-value on it would
be claiming more than the calculation supports.

## One more extraction from `MomentumStrategy`

T27 extracted `_formation_metric`. T28 needed one more: the sort direction was
a literal `higher_is_better=True` inside `score_batch`, and a strategy ranking
on a **risk** measure rather than a return has to flip it. It is now a
`higher_metric_is_better` property.

That is the whole of what `BettingAgainstBetaStrategy` overrides beyond the
metric — tradability screen, regime filter, volatility targeting and decile
selection are inherited, so the comparison is clean:

```bash
portfolio-agent compare --strategies bab,low_volatility_idio
```

Worth running: beta and idiosyncratic volatility are the two halves total
volatility mixes together, and T14 found the residual sort survives where the
total sort does not.

## The beta window

Defaults to **252 sessions**, not the registry's 60. Frazzini & Pedersen
estimate correlations on five years and volatilities on one; a one-year beta is
the shortest window on which the ranking is about a stock's exposure rather
than about the last quarter's news. Configurable via `beta_window`, resolved
through T24's `market_beta_feature` — so an unregistered window fails **at
construction**, not on the first scored date, and never by rounding to a
neighbour.

## What changed

- `evaluation/conditional.py` (new) — `ConditionalIC`, `conditional_ic`,
  `realized_states`, `trailing_states`, `market_return_by_date`,
  `conditional_notes`.
- `strategies/cross_sectional.py` — `MomentumStrategy.higher_metric_is_better`;
  `BettingAgainstBetaStrategy`; `DEFAULT_BAB_BETA_WINDOW`.

## What this does not do

The split is two-state. A "flat" third bucket sounds more careful but needs a
threshold nobody can justify, and it thins both real buckets to buy a third
whose interpretation is "the market did approximately nothing".

The conditional machinery is not wired into `evaluate`'s default output — it is
a deliberate question to ask, and asking it of every strategy by default would
put two more columns in every report for the one strategy in four where the
question is live.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_betting_against_beta.py -q
```

```python
from portfolio_agent.evaluation import conditional_ic, conditional_notes
result = conditional_ic(panel, horizon=5)
print("\n".join(conditional_notes(result)))
```
