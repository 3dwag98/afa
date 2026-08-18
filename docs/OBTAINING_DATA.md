# Obtaining market data and model checkpoints

Neither is tracked in git any more. This is how you get them.

## Why they were removed

2,397 parquet files, 112 MB tracked and about 80 MB in the pack, plus one
`.pt` checkpoint. Three problems, in increasing order of seriousness:

1. **Every clone pays for it**, forever, whether or not the person cloning
   wants five-year-old bars.
2. **It cannot be refreshed without bloating history.** A re-download rewrites
   most of 2,397 files, and git stores every version. The practical consequence
   was that nobody refreshed it, so the cache went stale silently — and stale
   market data looks exactly like current market data.
3. **The checkpoint implied a canonical model.** `models/lstm_best.pt` was in
   the repository with no record of the universe, config, or code revision that
   produced it. Anyone finding it would reasonably assume it was *the* model.
   It was a file. Run manifests (`runs/`) exist so that a checkpoint can carry
   that record; a tracked binary from before they existed cannot.

**The history is not rewritten.** The ~80 MB already in the pack is still
there; removing the files stops the bleeding but does not reclaim it.
Reclaiming it needs a history rewrite, which invalidates every existing clone
and open branch, and that is a separate decision with its own disruption. It is
noted here rather than bundled into this change.

## Market data

```bash
portfolio-agent download-data
```

Reads `data.source` from `config.yaml`. The default is the versioned
HuggingFace dataset, which needs the `hf` extra:

```bash
uv sync --extra hf
portfolio-agent download-data --years 20
```

`--years` overrides `data.default_history_years`. Twenty is the default now,
and worth keeping: five years of history begins *after* the COVID crash, so a
regime model trained on it has never seen one.

Pin `data.hf_revision` in `config.yaml` if you want two machines to agree
byte-for-byte on what they downloaded.

To use yfinance instead:

```bash
portfolio-agent download-data --source yfinance
```

### Check what you actually got

```bash
portfolio-agent data status      # span, coverage, gaps, corporate actions
portfolio-agent data validate    # exits non-zero on a structural violation
```

Run `data status` after the first download. It reports the span in one line,
which is the check that would have caught the five-year window years earlier.

## Model checkpoints

Train one:

```bash
portfolio-agent train --trainer gbm --universe-size 500
portfolio-agent train --strategy india_sac
```

Every run writes a manifest under `runs/` recording the universe fingerprint,
config hash, code revision and settings that produced the checkpoint. Render
one with:

```bash
portfolio-agent report --run <id>
```

A checkpoint without its manifest is a file of numbers nobody can account for,
which is the state the deleted `lstm_best.pt` was in.

## Where things live

| Path | Contents | Tracked |
| --- | --- | --- |
| `data/market_data/` | One parquet per ticker | no |
| `models/` | Checkpoints and their sidecars | no |
| `runs/` | Run manifests and rendered notes | no |
| `output/` | Generated reports | no |

## Point-in-time index membership (the largest uncorrected bias)

Everything above downloads the tickers that **exist today**. A stock that was
in the Nifty 500 in 2012 and was delisted in 2015 has no parquet file, so it
never enters a cross-section, so the 2012 deciles are formed from a universe
that excludes exactly the names that went on to fail.

This is not a rounding error. The 2026 study of Nifty constituents reports
**82.5% membership turnover** over its sample and roughly **4.94 percentage
points of annual return overstatement**. Both strategies' neutralized rank IC
is smaller than that.

Every evaluation run without a membership file prints a note saying so. To
supply one:

```bash
portfolio-agent evaluate --strategy momentum \
    --membership universe/nifty500_membership.csv \
    --index-name NIFTY500
```

### The file format

CSV, one row per stay. `end_date` empty means "still a member". Both ends are
inclusive, matching the convention NSE change announcements use.

```csv
symbol,index_name,start_date,end_date
RELIANCE.NS,NIFTY50,2000-01-01,
YESBANK.NS,NIFTY50,2017-03-27,2020-03-19
YESBANK.NS,NIFTY500,2020-03-20,
```

A symbol may appear more than once — names leave an index and come back, and a
re-entry is a second interval rather than an extension of the first.
Overlapping stays for one symbol are rejected at load: they read as a single
longer membership, which would make a name look eligible through a period it
had left.

### Where to get it

No feed ships this for free, and it is the one input here that cannot be
reconstructed from prices. The options, in descending order of quality:

| Source | Cost | Notes |
| --- | --- | --- |
| NSE index-maintenance circulars | free | Authoritative. Every reconstitution is announced; assembling ~20 years of them into intervals is manual work measured in days, not hours. |
| [niftyhistory.in](https://niftyhistory.in/) | free | Already-assembled historical constituents. Verify a few dates against the circulars before trusting it. |
| Bloomberg / Refinitiv / CMIE Prowess | paid | Point-in-time membership is a standard field. Prowess is the usual Indian academic source. |

Whichever route, delisted symbols also need **prices**, and yfinance will not
serve them. Prowess or a paid vendor covers both; otherwise the membership file
alone still helps — it removes names from dates they were not in the index,
which corrects part of the bias even when the failed names remain absent.

The filter reports what it removed, so the correction is auditable rather than
assumed:

```
Note: Point-in-time membership removed 4,812 of 51,300 observations (9.4%)
that were not index constituents on their own date, across 137 symbol(s).
```

Dates outside the file's coverage are left **unfiltered** rather than dropped,
and counted separately — a shorter window that looks clean is worse than a
longer one that says which part of it is uncorrected.

---

## Point-in-time fundamentals (the bias that looks like alpha)

`docs/QUANT_RESEARCH.md` §8 and §9 both stop at the same sentence — the data
isn't ingested — and between them they cover size, value, profitability,
investment and quality. None of it is expressible from OHLCV.

### Why the *report* date is the whole problem

A company's results for the quarter ending **31 March** are published in late
May. A backtest that uses them from 31 March has read six to eight weeks into
the future, on every stock, every quarter, forever.

This is the single most common way a fundamentals backtest lies, and it is
invisible in the output: the numbers are real, the dates are real, and the
strategy simply knew them early. **It does not look like a bug. It looks like
alpha.**

So every row carries two dates, and the store is keyed on the second:

- `fiscal_date` — the period the number describes.
- `report_date` — when it was published.

`report_date` is a **required** column. A file without it is rejected rather
than defaulted, because the only available default is the fiscal date and that
is precisely the look-ahead.

### The file format

```csv
symbol,fiscal_date,report_date,total_assets,total_equity,revenue,cost_of_goods_sold,net_income,cash_flow_operating,total_debt,shares_outstanding
RELIANCE.NS,2023-03-31,2023-05-12,1650000,720000,220000,150000,19300,28000,310000,6765
RELIANCE.NS,2023-06-30,2023-08-08,1680000,738000,214000,146000,16000,25000,315000,6765
```

Amounts in whatever unit you like, as long as it is **consistent within a
column** — every characteristic is a ratio, so the units cancel. Only `symbol`,
`fiscal_date` and `report_date` are required; a file with three fact columns
yields the characteristics those three support and says which. Restatements are
fine as extra rows: the store keeps the **earliest** report date for a period,
because a restatement published later was not knowable at the original
announcement.

```bash
portfolio-agent evaluate --strategy momentum \
    --fundamentals universe/nifty500_fundamentals.csv
```

### What validation checks, and why those numbers

SEBI **LODR Regulation 33** requires quarterly results within **45 days** of the
period end and annual results within **60 days**. That fixes what a plausible
lag looks like:

| Check | Severity | Why |
| --- | --- | --- |
| `report_date` column missing | **error** | The only fallback is the fiscal date, which is the look-ahead |
| `report_date` < `fiscal_date` | **error** | Not a late filing — a column swap, so every value is suspect |
| lag < 15 days | warning | Faster than an audited filing plausibly is |
| lag > 120 days | warning | A late filer, or a fiscal date in the report-date column |
| **every lag identical** | warning | See below |

That last one is the sharpest check in the module. A file whose report dates
were produced by adding a constant to the fiscal date is **worse than having no
file at all**: it has the shape of point-in-time data and none of the content,
so a backtest on it looks rigorous and is not. If a vendor gives you fiscal
dates only, do not synthesize report dates — run without fundamentals and let
the result say so.

### Where to get it

| Source | Cost | Notes |
| --- | --- | --- |
| BSE / NSE corporate-filings archives | free | Authoritative and carries the true announcement timestamp, which is the field everything else gets wrong. Machine-readable coverage thins going back before ~2015. |
| Screener.in, Tijori, Trendlyne | low | Convenient and broadly accurate on *values*. Check what they give for announcement dates — several expose only fiscal periods, which is the one field that cannot be reconstructed. |
| Refinitiv, Bloomberg, CMIE Prowess | paid | Prowess is the standard for Indian academic work and carries genuine point-in-time snapshots. |
| `yfinance` quarterly statements | free | Already a dependency. Convenient for a smoke test, **not** point-in-time — it serves the current view of history, restatements and all. |

The last row is worth its own sentence: `yfinance` is the easiest thing to
reach for and will give you a file that validates cleanly and is silently
wrong, because a restated figure is served for the date the original was
published. Use it to check the pipeline runs, not to produce a result.

### Running without it

Every evaluation without a fundamentals file prints:

> No fundamentals data was supplied, so this result controls for no accounting
> characteristic. Size, value, profitability, investment and quality exposures
> are uncontrolled, and any of them could be producing the ranking measured
> here.

That is the same contract the survivorship note has. It is a statement about
the number sitting next to it, not a chore someone is meant to remember.

---

## Reference data: free float, sector, and institutional flows

Three inputs that are neither prices nor accounts. Each closes a caveat the
platform is already printing.

### Free float (replaces the size proxy)

Every neutralized result currently prints:

> size is a proxy: log rolling-median traded value, not log market cap. Market
> cap needs shares outstanding, which this platform does not have, and for
> Indian equities the correct figure is free float — promoter holdings run
> 50–75%, so total capitalisation is not what trades.

**Free float, not total shares**, and that is not a refinement. Two firms with
identical issued shares and identical prices are the same size on paper and
2.7× apart to anyone who has to trade them, if one is 30% floated and the other
80%. A total-capitalisation size sort on Indian equities ranks by promoter stake
as much as by size. NSE's own indices are free-float weighted for this reason.

```csv
symbol,effective_date,free_float_shares,total_shares
RELIANCE.NS,2020-01-01,3380000000,6765000000
RELIANCE.NS,2023-08-15,3420000000,6765000000
```

`total_shares` is optional but earns its keep: it is what makes the free-float
*fraction* checkable, and the fraction is where the errors are. A file that has
quietly put total shares in the free-float column is **undetectable from the
float alone** and produces a size sort wrong by exactly the promoter stake —
largest where promoter holdings are largest, which is the opposite of a size
correction. The validator flags a median float above 95% of issued shares for
that reason.

Rows are keyed on `effective_date` and read with `<= date`. Share counts move
on splits, bonuses, buybacks and QIPs; promoter stakes move on pledges and
secondary sales. Applying today's float to a 2015 date restates a decade of
market caps in one direction.

**Where to get it:** NSE publishes free-float factors alongside its index
methodology, and BSE's shareholding-pattern filings give promoter holdings
quarterly. Both are free and both need assembling into a time series.

### Sector map

`src/sectors.py` has loaded these files since the concentration limits were
written. Nothing ships a CSV, so `--neutralize sector` has never had anything
to neutralize with — and Indian momentum concentrates hard by sector, which
makes this the single most likely place for an apparent alpha to be a sector
bet.

```csv
symbol,sector
HDFCBANK.NS,Financials
INFY.NS,Information Technology
```

Column names are flexible (`ticker`/`symbol`, `sector`/`industry`/`gics_sector`).

**Coverage is the thing to check, not existence.** A map resolving 60% of a
universe produces a "sector-neutral" result in which the other 40% were
neutralized against a pool called `UNKNOWN` — that is, against each other
rather than against their real peers. `sector_coverage()` reports the fraction
and the largest sector's share so the claim can be read at its real strength.

**Where to get it:** NSE's index constituent files carry a sector column;
so does BSE's list of listed companies. Both are free downloads.

### FII / DII flows (§10)

Unlike the two above, this describes the **market** rather than the names in
it, so it conditions a result rather than entering a cross-sectional ranking.

```csv
date,fii_net,dii_net
2023-01-02,-1245.30,982.11
2023-01-03,-88.42,301.55
```

Units are free as long as they are consistent — every use is a sign, a ratio or
a z-score.

The adapter exposes `fii_net − dii_net` directly, because **neither leg alone
says whether the market was under pressure.** Domestic institutions
systematically buy into foreign selling, so the two series are strongly
negatively correlated; their difference is what carries the information.

`FlowSeries.states()` labels each date `inflow`/`outflow` on a trailing
63-session window, shaped to drop into `evaluation/conditional.py` as an
alternative conditioner. The window ends at the date it labels, so a
flow-conditioned split is **tradable** rather than only attribution — the
distinction T28 turns on.

**Where to get it:** NSE and SEBI both publish daily provisional FII/DII
figures. NSDL publishes the settled monthly series. The provisional daily
numbers are revised, so a file assembled from them is approximately right
rather than exactly right — which is fine for a regime read and not for
anything that needs the level.
