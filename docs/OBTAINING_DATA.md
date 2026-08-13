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
