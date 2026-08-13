# T15 — Point-in-time index membership

**Status:** done (mechanism); the data itself is not acquirable here
**Effort:** ~1 day · **Depends on:** T02 (data quality), T04 (harness)
**Review reference:** `docs/architecture_review_2.html`, "Is the data
sufficient?"

## Goal

Rank each date against the names that were actually in the index on that date.

## Why

Every result this platform has produced was computed on the tickers that
survived to be downloaded. A stock in the Nifty 500 in 2012 that was delisted
in 2015 has no parquet file, so it never enters a cross-section, so the 2012
deciles are formed from a universe that excludes exactly the names that went on
to fail. The bias is one-directional and it compounds across every date.

The 2026 study of Nifty constituents puts it at **4.94 percentage points of
annual return overstatement** against **82.5% membership turnover**. Both
strategies' neutralized rank IC — momentum +0.026, low volatility +0.018 — is
smaller than that. **The largest number in the pipeline was the one nobody was
measuring.**

`src/universe.py::select_universe` already said so in a docstring: "A random
sample of an alphabetical cache is still not a point-in-time index membership."
That comment has now been turned into something the code can act on.

## Approach

The interval file is CSV because that is the shape this data arrives in and
because a human has to be able to fix a row by hand:

    symbol,index_name,start_date,end_date
    RELIANCE.NS,NIFTY50,2000-01-01,
    YESBANK.NS,NIFTY50,2017-03-27,2020-03-19

Both ends inclusive, matching NSE change announcements. A symbol may appear
more than once — names leave an index and come back — and a re-entry is a
second interval rather than an extension. Treating it as an extension would
silently cover the gap the name was out, which is the same hindsight error at a
finer grain.

## Three decisions that are the point of the task

**A missing file is an error, not an empty filter.** A typo'd path that quietly
disabled the filter would restore the survivorship bias *inside a run that
claims to have corrected for it* — a wrong number wearing a correct label. It
raises.

**Dates outside the file's coverage are left unfiltered and counted, not
dropped.** Emptying them would silently shorten the evaluation window, and a
shorter window that looks clean is worse than a longer one that admits which
part of it is uncorrected. The count travels into the result and the note.

**Overlapping stays for one symbol are rejected at load.** They read as a
single longer membership, which would make a name look eligible through a
period it had left.

## When there is no membership file

The run prints:

> No point-in-time index membership was supplied, so every date was ranked
> against the names that survived to be downloaded. Indian index membership
> turns over heavily — a published study puts it at 82.5% over its sample, with
> roughly 4.94pp of annual return overstatement — so this result is biased
> upward by an amount plausibly larger than the alpha it reports.

Same pattern as the sector-map caveat T05 established (finding A8): the default
state is stated on every result rather than living in a document nobody reads
next to the number. A test asserts the note keeps its magnitudes — a caveat
without a number is a caveat people learn to skip.

## When there is one

    Note: Point-in-time membership removed 4,812 of 51,300 observations (9.4%)
    that were not index constituents on their own date, across 137 symbol(s).

`membership_removed_share` goes into the manifest as a metric and the file path
goes in as provenance — which membership file a number was computed against is
the first thing to check when two runs of the "same" strategy disagree.

## Acceptance criteria

- [x] `members_on(date)` with inclusive ends, re-entries as separate stays.
- [x] Corrupt files rejected: overlaps, end-before-start, unparseable dates,
      missing columns, empty symbols.
- [x] A missing file raises rather than disabling the filter.
- [x] Uncovered dates left unfiltered and reported separately.
- [x] The filter reports what it removed, so the correction is auditable.
- [x] Every run without one says so, with the magnitude in the sentence.
- [x] `--membership` and `--index-name` on the CLI.
- [x] Acquisition documented, including that no free feed ships it.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/data_quality/membership.py` | New — `IndexMembership`, `apply_membership`, `SURVIVORSHIP_NOTE` |
| `portfolio_agent/evaluation/harness.py` | Filter applied before scoring; `membership` field and provenance |
| `portfolio_agent/cli_forecast.py` | `--membership`, `--index-name` |
| `docs/OBTAINING_DATA.md` | Format, acquisition routes, what each costs |
| `portfolio_agent/tests/test_membership.py` | New — 39 tests |

## What is not done, and why

**No membership file ships with the repository.** This is the one input here
that cannot be reconstructed from prices. The routes, all documented in
`OBTAINING_DATA.md`: NSE index-maintenance circulars (free, authoritative,
days of manual assembly for 20 years), niftyhistory.in (free, pre-assembled,
verify against the circulars), or a paid vendor — Prowess is the usual Indian
academic source. This environment has no access to any of them.

**Delisted names still need prices.** The membership file alone removes names
from dates they were not in the index, which corrects *part* of the bias. It
cannot resurrect the price history of a delisted stock, and yfinance will not
serve it. Full correction needs a vendor that carries both. The filter's
reported `removed_share` measures what was corrected; it does not claim to
measure what remains.
