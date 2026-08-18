# T27 — Residual momentum

**Status:** done · **Effort:** ~1 day · **Depends on:** T24, T25
**Review reference:** round-three plan, Phase 3

## Goal

Momentum measured on the CAPM residual rather than the raw return, and the
first real test of whether T24's and T25's seams hold.

## Why

Price momentum's return is substantially a bet on whatever the market has been
rewarding. That is why it crashes the way it does: the exposure that pays
during a trend is the exposure that reverses violently at a turn.

This is not an imported concern. **Round two measured this platform's own
momentum at 58% factor loading.**

Blitz, Huij & Martens (2011) rank on the residual's *information ratio* and
report roughly double the risk-adjusted profit of price momentum with
materially shallower drawdowns.

## The formation measure

    RESMOM = mean(residual over formation) / std(residual over formation)

evaluated as of `t - skip`, with `residual_t = r_t - beta_t * r_m,t`.

**The standardization is the substance, not a tidying step.** Ranking on raw
cumulated residuals still puts high-residual-volatility names on top, which
reintroduces exactly the risk exposure residualizing was meant to remove. A
test asserts the two orderings differ — if they did not, there would be nothing
to the argument.

Windows match `technical.mom_9m_skip1m` exactly — 189-day formation, 21-day
skip — because the point is to be *the same momentum measured differently*. A
different formation window would make the comparison a comparison of two things
at once.

### The intercept is a trap

If beta is fitted **with an intercept** over exactly the window the residuals
are then cumulated over, the cumulative residual is **identically zero** — OLS
residuals sum to zero by construction, so a strategy ranking on it would be
ranking floating-point noise.

BHM avoid it by estimating over 36 months and forming over 12. Here beta is
estimated on a rolling **year** ending at each date and applied **without an
intercept**, so the residual carries the alpha rather than having it
differenced away. A test demonstrates both halves on a noiseless construction,
so the constant that avoids the trap has a reason attached rather than a number.

Without an intercept, a market with a non-zero mean lets beta absorb a little
of the alpha. The test pins that too: the recovered alpha is most of the true
one, and never more than it.

### Half a window is fine for an estimate and not for a formation window

Everything else in `market_relative.py` floors `min_periods` at half the
window. That is right for a **parameter estimate** — a beta or a volatility
degrades gracefully as observations are lost — and wrong for a **formation
window**, whose length is part of what the signal *is*.

At half, the opening months of every evaluation would have ranked on
4.5-month momentum under a name that says nine, mixing two formation lengths
inside one backtest. The formation window is now required in full; the beta
keeps the half-window convention. The asymmetry is deliberate and recorded.

Warm-up is therefore **337 rows** = 126 (half the beta window) + 189
(formation) + 21 (skip) + 1.

## Measured

Synthetic 40-name panel, betas 0.4 → 1.8, five names with a true positive daily
alpha and five negative:

| | raw return | CAPM residual |
| --- | --- | --- |
| mean \|corr with market\| | **0.715** | **0.019** |

Rank correlation between true alpha and the formation measure is above 0.5, and
the positive-alpha group outscores the flat group, which outscores the
negative-alpha group. Beta-rank loading is lower for the residual sort than for
raw price momentum on the same panel.

## The seam held, and it had a hole

`ResidualMomentumStrategy` subclasses `MomentumStrategy` and overrides one
method. Everything else — tradability screen, regime filter, volatility
targeting, decile selection via T25's now-public `rank_and_select` — is
inherited, so:

```bash
portfolio-agent compare --strategies momentum,residual_momentum
```

compares the formation measure and nothing else. That required extracting a
`_formation_metric` hook from `MomentumStrategy`, which is the only change to
the incumbent.

**But T27 found a gap T24 had re-opened.** `_required_history_rows` in the
backtest engine and `min_history` in the evaluation harness both consulted the
**per-ticker** registry only. This strategy needs 337 rows for its formation
window and nothing per-ticker beyond 62 — so either path would have admitted
tickers whose ranking key was still NaN. That is precisely the defect T23
removed, re-created by the second registry three tasks later. Both now take the
maximum across both registries.

This is what Phase 3 was for. A seam is not proven by the code that defines it.

## What changed

- `features/market_relative.py` — `residual_returns`, `residual_momentum`, the
  three window constants, and the registered
  `residual_momentum_9m_skip1m`.
- `strategies/cross_sectional.py` — `MomentumStrategy._formation_metric` hook
  and `trigger_name`; `ResidualMomentumStrategy`.
- `src/backtest_engine.py`, `evaluation/harness.py` — warm-up across both
  registries.
- `docs/QUANT_RESEARCH.md` §12(c) — corrected; it recorded this as blocked on
  §8's un-ingested factor data.

## What this does not do

The residual is against the market alone, not Fama-French. §8's data would let
the residual be taken against a multi-factor model, which is what BHM actually
do — a refinement, and now a visible one rather than a blocker.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_residual_momentum.py -q
portfolio-agent compare --strategies momentum,residual_momentum --neutralize beta,size
```
