# Status against the quantitative review

An item-by-item record of what the August 2026 review asked for and what this
repository now does. Written to be checked rather than believed: every "done"
below names the code and the test that pins it, and every "not done" says what
blocks it rather than leaving it implied.

The review's own sequencing principle is followed — *fix the measurement before
the model, the sizing before the signal, and the data before the architecture* —
which is why the unglamorous phases are complete and the architectural ones are
not.

## Headline findings

| # | Finding | Status | Where |
|---|---|---|---|
| D1 | Kelly `f*` is a binary-bet stake fraction used as an allocation | **Done** | `src/risk.py::kelly_allocation_fraction` |
| D2 | MC `probability_profit` dominated by drift-estimation noise | **Done** | `src/monte_carlo.py::shrink_drift` |
| D3 | No portfolio covariance anywhere | **Done**, wired into sizing | `src/portfolio.py`, `BacktestEngine._apply_portfolio_risk_cap` |
| D4 | Neural target is absolute, not cross-sectional | **Done** (target); feature normalization not done | `agents/trainer.py::apply_cross_sectional_target` |
| D5 | GJR-GARCH unusable at platform scale | **Not done** | — |
| D6 | No Markov chains / regime-switching models | **Done** as a module; not switched on | `src/markov_regime.py` |
| D7 | Sharpe mis-specified; no PSR/DSR/PBO | **Done** | `src/performance_stats.py` |
| D8 | Adaptive weighting unshrunk, in-sample | **Partly done** — shrunk and gated, still in-sample | `strategies/weighting.py` |
| D9 | Universe is an alphabetical slice, not point-in-time | **Not done** — data problem | — |
| D10 | `MC_Prob` holds 25% of weight, discriminates nothing | **Partly done** — see below | `strategies/rule_based.py` |

### Where a "partly" needs spelling out

**D8.** The win rate is now Beta-shrunk, floored at 30 trades per component and
gated on an exact binomial test, so the loop no longer moves on noise. It is
still a *closed* loop: weights are fitted on realized outcomes that the weights
themselves influenced. Making it out-of-sample means fitting on walk-forward
training folds and freezing for the test fold, as model checkpoints already are.
Any backtest run with adaptation enabled remains contaminated to that extent.

**D10.** Two separate problems live under this heading and only one is fixed.

- *The level shift* — a `rule_based` member inside a batched UMA saw no
  per-ticker Monte Carlo result and scored `MC_Prob` at 0, so the identical
  stock scored ~12 points lower in an ensemble than standalone against absolute
  60/45 thresholds. Fixed: unavailable components are renormalized away.
- *The near-constant component* — `MC_Prob` has a realized standard deviation
  around 0.05 and consumes a quarter of the score budget while separating
  almost nothing. **The rank composite does not fix this**, despite being the
  review's proposed remedy for it: ranking ties hands every name the same
  percentile, so the component contributes a flat number under either method.
  Making influence track discrimination requires dispersion- or IC-weighted
  weights — a change to the weight learner, not the combination rule. A test
  asserts this non-property so it cannot be quietly forgotten.

The rank composite itself is implemented (`scoring.method: rank_composite`) and
does fix what it can: commensurability, and invariance to each component's
marginal distribution. It is off by default because it converts the entry
threshold from an absolute quality bar into a percentile — under it a roughly
fixed share of the universe always clears 60, whatever the market is doing.

## Smaller findings (D11)

| Issue | Status |
|---|---|
| Overlapping-label Sharpe uncorrected | **Done** — Newey–West t-statistic at lag H−1 in `evaluate_predictions` |
| `f*` clamp to [0,1] becomes load-bearing | **Done** — replaced by an explicit leverage constraint |
| Isotonic calibration pooled across regimes | Not done |
| Calibrating `p` but not the payoff `b` for Kelly | Not done |
| Training label gross of costs | Not done |
| No ASM/GSM/ESM/T2T awareness | Not done — inferred from price data, not ingested |
| No lot sizes, no T+1 cash settlement | Not done |
| No turnover or capacity model | Partly — the optimizer has a turnover penalty; no ADV participation cap |

## India-specific strategy problems (§5)

None of these are addressed. Recorded because they are the difference between a
platform that models Indian equities and one that models equities and trades
them in India.

| § | Issue | Blocked on |
|---|---|---|
| 5.1 | Circuit limits break the return-generating assumption; a locked day is a censored observation, not a price | A Tobit-style censored GARCH likelihood; at minimum, excluding locked days from volatility estimation |
| 5.2 | Gap risk reaches volatility estimation but not the stop fill or position sizing | `fill = min(open, stop)` for longs, and adding the overnight component to risk-per-share |
| 5.3 | SEBI surveillance frameworks are published, objective and change tradability | A daily scraper for the NSE ASM/GSM/ESM/T2T lists (Phase 2 data) |
| 5.4 | No tax-lot accounting; no 365-day LTCG boundary optimization | FIFO lots per demat; a real India-specific alpha source left unclaimed |
| 5.5 | Momentum and low-vol run without factor neutralization | Residual momentum needs only the sector map from Phase 2 |

## Roadmap phases (§7)

| Phase | Status |
|---|---|
| 0 — Correctness | 6 of 8: Kelly, drift shrinkage, arithmetic Sharpe, weighting guards, rank composite (opt-in), leverage constraint. **Missing:** GARCH refit scheduling (0.5), gap-aware stop fills (0.7), net-of-cost training label (0.8) |
| 1 — Measurement | Essentially complete: PSR, DSR, trial log, PBO, rank IC/ICIR, Newey–West. See the note on 1.6 below |
| 2 — Data | **Not started.** Every downstream number is limited by this |
| 3 — Portfolio construction | Estimators, optimizer and HRP built and wired as a volatility cap. Per-trade sizing is **not** retired (3.4) — the cap sits on top of it rather than replacing it |
| 4 — Signal | Only 4.1 (cross-sectional target). No residual momentum, feature expansion, conditional autoencoder, Kronos features, or seed ensembling |
| 5 — Regime | 5.1 and 5.5 done (HMM + honest validation). No TVTP (5.2), no MS-GARCH (5.3). `sleeve_weights` exists but the orchestrator is not switched to it (5.4) |
| 6 — Uncertainty | Not started. Isotonic calibration is unchanged |
| 7 — Execution and capacity | Not started |
| 8 — Governance | Not started. `paper_trading_mode` remains the default |

### One place the review is imprecise about the existing code

Item 1.6 asks for purged K-fold CV, describing the current implementation as
handling "the left boundary correctly and ignoring the right". The walk-forward
here is strictly expanding-window — training is always `index < train_end_date`
and the test block is `[train_end_date, test_end_date)` — so *all* training data
precedes the test period and there is no right boundary to purge. The
right-boundary purge is a requirement of K-fold, where test blocks sit in the
middle of the sample. Moving to purged K-fold is a legitimate proposal (it uses
more of the data), but it is a methodology change, not a leak being fixed.

## What has not changed by default

Backtest numbers move because of three changed defaults, all reversible:
`training.target_transform`, the Monte Carlo drift shrinkage, and the weighting
guards. Deliberately unchanged: `risk.portfolio_volatility_target` is off,
`scoring.method` is `weighted_sum`, `src/regime.py` is untouched, and
`paper_trading_mode` is true.

## The caveat that outlives all of this

The review's warning about D9 stands and is worth repeating: the universe is an
alphabetical slice of a parquet cache with no point-in-time index membership, so
it carries survivorship and look-ahead selection simultaneously. **Until that is
fixed, no cross-sectional backtest number from this repository should be
believed in either direction** — including any number that improves because of
the changes above. The measurement layer added in Phase 1 exists to make that
judgeable rather than assumed; it cannot manufacture a clean sample.

---

## Appendix: defects found by running the platform

Six issues reported from a real run — a Windows machine with a 6 GB GPU, a
populated cache, and a UMA containing an untrained member. None came from the
review; all are fixed.

| Symptom | Cause | Fix |
|---|---|---|
| NaN training loss **on CPU**, surviving the earlier mixed-precision fix | Features were standardized and clipped to ±10σ; the *target* never was. One cached close printed at 0.001 turns a 5-day forward return into 111,300, and one gradient step against a loss that size leaves every later batch NaN. The cause was the label, not fp16 | `training.max_abs_target` (default 5.0) drops the poisoned rows — dropped, not clipped, since clipping piles a spike of samples at the bound |
| Windows paging / apparent hang while training | `ProcessPoolExecutor(max_workers=None)` spawns one worker per CPU, and Windows has no `fork` — each is a fresh interpreter re-importing torch, pandas and pyarrow at 300–800 MB. Twelve of those pages a 16 GB machine. The 6 GB GPU is irrelevant; the exhausted resource is host RAM | `utils/workers.py` caps process pools (2 on Windows, 8 elsewhere) and uses in-process DataLoading on Windows, printing the plan at startup |
| Re-downloads data already on disk | `sync_hf_to_cache` had no existence check and re-fetched all ~2,400 symbols every invocation | `skip_existing=True` by default via `DataStore.has_ticker_data`, which also rejects zero-byte files left by an interrupted write; `--force` restores the old behaviour |
| Download is serial | One symbol at a time over the network | `ThreadPoolExecutor` (8 by default, `--workers` to change). Threads not processes: the work is network-bound, and processes would re-import pandas per worker — the same Windows problem again |
| Training and backtesting use identical tickers | `tickers[:n]` on an alphabetically sorted cache | `data.universe_selection: random` with `universe_seed`, and the seed offset by purpose so train and backtest draw different names. Seeded so one config reproduces one universe |
| `ensemble` fails with "missing trained model checkpoint?" | `EnsembleStrategy.load()` returned False if *any* member failed, and the message named none of them | The error now names the failing members, the path searched, and both remedies; `drop_unavailable_members: true` runs the loadable subset (with a warning that it is no longer the configured strategy) |

Two of these are worth separating from the rest. The NaN was a **correctness**
bug that silently defined which rows a model trained on, and the universe
truncation meant every model was being evaluated on the exact names it was
fitted on — neither is a usability complaint.
