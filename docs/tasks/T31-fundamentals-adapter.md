# T31 — Point-in-time fundamentals

**Status:** done · **Effort:** ~1 day · **Depends on:** T24
**Review reference:** round-three plan, Phase 4

## Goal

Make §8 and §9 expressible — schema, validation and tests now, data supplied
later. The T15 pattern, chosen deliberately.

## Why

`QUANT_RESEARCH.md` §8 and §9 both stopped at the same sentence — *the data
isn't ingested* — and between them they cover size, value, profitability,
investment and quality. That is most of the published cross-section of equity
returns, and none of it is expressible from OHLCV.

T24 removed the *other* obstruction three tasks earlier: every one of these
characteristics is peer-relative, and before the cross-sectional registry
existed none of them could have been written at all.

## The bias this exists to prevent

A company's results for the quarter ending **31 March** are published in late
May. A backtest that uses them from 31 March has read six to eight weeks into
the future — on every stock, every quarter, forever.

This is the most common way a fundamentals backtest lies, and it is invisible
in the output: **the numbers are real, the dates are real, and the strategy
simply knew them early. It does not look like a bug. It looks like alpha.**

So every fact carries two dates and the store is keyed on the second:

| field | meaning |
| --- | --- |
| `fiscal_date` | the period the number describes |
| `report_date` | when it was published |

`as_of(date)` filters on `report_date <= date`, **never** `fiscal_date <= date`.
`report_date` is a **required** column rather than an optional one: the only
available fallback is the fiscal date, and that fallback *is* the look-ahead.

Verified directly — on 30 June the Q2 figures are invisible; on 8 August, their
publication day, they appear.

## Validation, calibrated to Indian filing law

SEBI **LODR Regulation 33**: quarterly results within **45 days** of the period
end, annual within **60**.

| check | severity | why |
| --- | --- | --- |
| `report_date` missing | **error** | the only fallback is the look-ahead |
| `report_date` < `fiscal_date` | **error** | not a late filing — a column swap, so every value is suspect |
| lag < 15 days | warning | faster than an audited filing plausibly is |
| lag > 120 days | warning | a late filer, or a fiscal date in the wrong column |
| **every lag identical** | warning | see below |

The last is the sharpest check in the module. A file whose report dates were
produced by adding a constant to the fiscal date is **worse than having no file
at all**: it has the shape of point-in-time data and none of the content, so a
backtest on it looks rigorous and is not.

**Restatements keep the earliest report date for a period.** A restatement
published in November was not knowable in May, and taking the latest would
reintroduce the look-ahead by the back door.

## The characteristics

Six, registered through T24:

| name | formula | notes |
| --- | --- | --- |
| `book_to_price` | equity / market cap | HML. Non-positive equity dropped |
| `earnings_to_price` | net income / market cap | loss-makers **kept** |
| `gross_profitability` | (revenue − COGS) / assets | Novy-Marx. RMW |
| `asset_growth` | assets / assets(−252) − 1 | CMA |
| `accruals` | (NI − CFO) / assets | Sloan |
| `leverage` | debt / equity | a **control**, not a signal |

**The denominator is the design decision**, and Novy-Marx's point generalizes:
gross profit over *assets* ranks firms by how productively they use capital;
over *sales* it ranks them by margin, and margin is close to an industry label.
Each docstring states which it computes.

### Two sign traps, recorded because they look plausible backwards

`asset_growth` is signed as **growth**, so CMA's conservative leg is the
*bottom* of the sort. `accruals` is signed so **high means more accrual**, so
Sloan's leg is also the bottom. A long-only book takes the bottom of both.

### One asymmetry that is deliberate

`book_to_price` drops non-positive book equity; `earnings_to_price` keeps
negative earnings.

That is not inconsistency. A firm whose accumulated losses exceed its paid-in
capital has a *negative* B/P, which sorts it to the extreme **growth** end —
the opposite end from where a distressed balance sheet belongs, so one end of
the value decile would be populated by exactly the firms the characteristic
cannot describe. Fama and French apply this screen. A loss-making firm's
negative earnings yield, by contrast, sorts where a reader would expect, and
screening it out would be a quality filter wearing a value label.

**This was found by writing the tests, not by design.** `_safe_ratio` guards
the denominator, and for book-to-price the equity is the numerator — so the
first version ranked distressed firms as extreme growth stocks.

## What a run without the data says

Same contract T15 gave survivorship:

> No fundamentals data was supplied, so this result controls for no accounting
> characteristic. Size, value, profitability, investment and quality exposures
> are uncontrolled, and any of them could be producing the ranking measured
> here.

A file that *was* supplied has its validation warnings surfaced in the same
place — a synthetic-lag file should say so where the result can be seen, not
only where someone thought to look.

## What changed

- `data_quality/fundamentals.py` (new) — `FundamentalsStore`,
  `validate_fundamentals`, `load_fundamentals`, `FUNDAMENTALS_NOTE`.
- `features/characteristics.py` (new) — the six, plus
  `computable_characteristics`.
- `evaluation/harness.py` — `fundamentals=` and `fundamentals_notes`.
- `docs/OBTAINING_DATA.md` — format, validation rationale, sources.
- `docs/QUANT_RESEARCH.md` §8 and §9 — both said "not yet implemented".

## What this does not do

**No data ships.** That is the chosen scope, and `OBTAINING_DATA.md` carries
the acquisition options. One is worth repeating: `yfinance` quarterly
statements are already a dependency and will produce a file that **validates
cleanly and is silently wrong**, because a restated figure is served for the
date the original was published. Use it to check the pipeline runs, not to
produce a result.

**QMJ's earnings-growth leg is absent.** It needs a multi-year history of the
same field rather than one snapshot, so it is genuinely blocked on data depth
rather than data existence.

**No CLI flag yet.** `evaluate(fundamentals=...)` exists at the API; wiring
`--fundamentals` through the command belongs with T33's CLI pass.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_fundamentals.py -q
```
