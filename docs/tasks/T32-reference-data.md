# T32 — Reference data adapters

**Status:** done · **Effort:** ~1 day · **Depends on:** T31
**Review reference:** round-three plan, Phase 4 (closes it)

## Goal

Close the last three inputs the platform prints caveats about, and with them
the last `QUANT_RESEARCH.md` section that said *"not yet implemented"*.

## Free float, not shares outstanding

`evaluation/neutralize.py` has been printing this since T05:

> size is a proxy: log rolling-median traded value, not log market cap. Market
> cap needs shares outstanding, which this platform does not have, and for
> Indian equities the correct figure is **free float** — promoter holdings run
> 50-75%, so total capitalisation is not what trades.

That parenthesis is why this is a separate adapter from T31's
`shares_outstanding`, and it is not a refinement.

**Two firms with identical issued shares and identical prices are the same size
on paper and 2.7x apart to anyone who has to trade them**, if one is 30% floated
and the other 80%. A total-capitalisation size sort on Indian equities ranks by
promoter stake as much as by size. NSE's own indices are free-float weighted for
exactly this reason. A test measures the 2.7x.

`add_exposures(free_float=...)` swaps log free-float market cap in and replaces
the note with one saying so.

**No blending.** A store covering none of the universe falls back to the proxy
and says it did, rather than using real caps for covered names and the proxy for
the rest. A size column built from two definitions ranks partly on *which
definition applied* — the mixed-measure failure T14 removed from the volatility
sort.

### The wrong-file case

`total_shares` is optional and earns its keep: it makes the free-float
*fraction* checkable, and the fraction is where the errors are. A file that has
quietly put total shares in the free-float column is **undetectable from the
float alone**, and produces a size sort wrong by exactly the promoter stake —
largest where promoter holdings are largest, which is the opposite of a size
correction. The validator flags a median float above 95% of issued shares.

## Sector: coverage, not existence

`src/sectors.py` has loaded these files since the concentration limits were
written. Nothing ships a CSV, so `--neutralize sector` has never had anything to
work with — and Indian momentum concentrates hard by sector, which makes this
the single most likely place for an apparent alpha to be a sector bet.

What was missing is not only the file. **A map resolving 60% of a universe
produces a "sector-neutral" result in which the other 40% were neutralized
against a pool called `UNKNOWN`** — that is, against each other rather than
against their real peers. `sector_coverage()` reports the fraction and the
largest sector's share, so the claim can be read at its real strength rather
than at its nominal one.

## FII/DII flows: the subtraction is the point

Unlike everything else here, this describes the **market** rather than the names
in it, so it conditions a result rather than entering a ranking.

`FlowSeries.net` is **FII minus DII**. Domestic institutions systematically buy
into foreign selling, so the two series are strongly negatively correlated and
neither leg alone says whether the market was under pressure. A file read one
leg at a time would report "heavy FII selling" on days when domestic buying
fully absorbed it.

`states()` labels each date `inflow`/`outflow` on a trailing 63-session window,
shaped to drop into T28's `conditional_ic` as an alternative conditioner. The
window **ends at the date it labels**, so a flow-conditioned split is *tradable*
rather than only attribution — the distinction T28 turns on, and the one that
separates a regime filter from a post-hoc explanation.

## Point-in-time, again

All three are keyed on an effective date and read with `<= date`. Share counts
move on splits, bonuses, buybacks and QIPs; promoter stakes move on pledges and
secondary sales; a stock moves sector on reclassification.

Free float is forward-filled and **never** back-filled: back-filling would apply
a post-buyback share count to the years before it, restating every market cap in
the sample in one direction.

## What changed

- `data_quality/reference.py` (new) — `FreeFloatStore`, `validate_free_float`,
  `sector_coverage`, `FlowSeries`, `load_free_float`, `load_flows`, and the
  three notes.
- `evaluation/neutralize.py` — `add_exposures(free_float=...)`.
- `docs/OBTAINING_DATA.md` — formats, validation rationale, sources.
- `docs/QUANT_RESEARCH.md` §10 — rewritten; it said "not yet implemented".

**With this, no section of `QUANT_RESEARCH.md` says "not yet implemented".**
Every one that recorded a research idea as blocked on data now has an adapter,
a schema, validation and tests — and says so in its own notes when run without
the data.

## What this does not do

**No data ships**, per the chosen scope. `OBTAINING_DATA.md` carries the
sources: NSE index methodology for free-float factors, BSE shareholding
patterns for promoter holdings, NSE/BSE constituent files for sectors, NSE/SEBI
for daily provisional flows.

**The flow-based sizing filter §10 anticipated is not wired.** The states exist
and `CrashProtection` already accepts an exposure scalar, so it is one step
away — but the daily FII/DII figures are *provisional and later revised*, and
whether to hand a live sizing decision to a number that gets restated is a risk
judgement rather than an implementation detail.

**No CLI flags.** `add_exposures(free_float=...)` exists at the API; wiring
`--free-float`, `--sector-map` and `--flows` through the commands belongs with
T33's CLI pass, alongside `--fundamentals`.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_reference_data.py -q
```
