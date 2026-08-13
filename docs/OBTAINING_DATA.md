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
