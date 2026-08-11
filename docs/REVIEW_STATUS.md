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
| D5 | GJR-GARCH unusable at platform scale | **Done** — refit scheduling, measured 22x | `src/volatility_models.py::forecast_volatility` |
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

**The net-of-cost label (0.8).** `training.cost_adjust_target` is on by default
and charges the platform's own round-trip friction (~0.8%, read from
`execution_sim.py` rather than hardcoded) against the forward-return label, so
a +0.4% five-day forecast stops being a positive label and a losing trade.
What it does *not* do is worth stating plainly, because the default
configuration hides it: the cost is a constant, and both cross-sectional
transforms are invariant to a constant — demeaning subtracts it back out,
ranking never sees it. So under the shipped `cross_sectional_rank` target this
changes no ordering and therefore nothing at all. It binds on `absolute`.
Neither piece is at fault: a relative label already asks "which name is
better", which a uniform fee cannot answer. Making friction discriminate
*between* names requires a per-ticker liquidity-scaled cost, which is Phase 7
capacity work. A test pins the invariance so it cannot be quietly assumed away.

## Smaller findings (D11)

| Issue | Status |
|---|---|
| Overlapping-label Sharpe uncorrected | **Done** — Newey–West t-statistic at lag H−1 in `evaluate_predictions` |
| `f*` clamp to [0,1] becomes load-bearing | **Done** — replaced by an explicit leverage constraint |
| Isotonic calibration pooled across regimes | Not done |
| Calibrating `p` but not the payoff `b` for Kelly | Not done |
| Training label gross of costs | **Done** — `training.cost_adjust_target`, but see the note below |
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
| 5.2 | **Half done.** Gap risk now reaches the stop fill — a level the market gapped through fills at the open, both directions. It still does not reach position *sizing* | Adding the overnight component to risk-per-share: `(entry − stop) + z·σ_gap·entry`. The gap sigma already exists in `volatility_models.py`; it is not plumbed to `_kelly_quantity` |
| 5.3 | SEBI surveillance frameworks are published, objective and change tradability | A daily scraper for the NSE ASM/GSM/ESM/T2T lists (Phase 2 data) |
| 5.4 | No tax-lot accounting; no 365-day LTCG boundary optimization | FIFO lots per demat; a real India-specific alpha source left unclaimed |
| 5.5 | Momentum and low-vol run without factor neutralization | Residual momentum needs only the sector map from Phase 2 |

## Roadmap phases (§7)

| Phase | Status |
|---|---|
| 0 — Correctness | **Complete.** Kelly units, drift shrinkage, arithmetic Sharpe, weighting guards, rank composite (opt-in), leverage constraint, GARCH refit scheduling (0.5), gap-aware stop fills (0.7), net-of-cost label (0.8) |
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

### What made GARCH affordable (D5 / 0.5)

The review's diagnosis was that `use_garch_volatility: false` was not a
considered default but the only setting under which the backtest terminates:
one MLE per ticker per bar is ~4.5 million fits for the documented run.

Option 1 from the review is what shipped — refit on a schedule, cache the
coefficients, run the recursion forward daily. Two details were not obvious
going in and are worth recording:

- **The fitted window is anchored to the history length, not to call order.**
  The natural implementation ("refit if N bars have passed since we last fitted
  this symbol") makes the answer depend on the sequence calls arrive in, and
  both the backtest and the orchestrator dispatch per-ticker scoring to worker
  processes holding independent caches. Different workers would hit different
  refit points and produce different volatility paths for the same ticker on
  the same date. Anchoring to `floor(n / interval) * interval` makes the cache
  pure memoization: it changes how long the answer takes, never what it is.
  `test_result_does_not_depend_on_call_order` pins this.
- **The forward step is exact, not an approximation.** One step ahead the sign
  of the last residual is known so the leverage term applies exactly; beyond
  that symmetric innovations put it below zero half the time, giving the
  familiar `alpha + gamma/2 + beta`. `test_matches_arch_forecast_exactly_at_a_refit_boundary`
  checks the closed-form path against `arch`'s own analytic forecast and it
  agrees to ~1e-16 relative. So the only approximation is the *staleness of the
  coefficients* between refits — every new bar still updates conditional
  volatility.

Measured at 22x on a 63-bar walk-forward for one ticker, which is the expected
saving for a 21-bar interval. `use_garch_volatility` remains `false`: making it
affordable is a separate question from whether it improves results, and that is
a judgement Phase 1's measurement layer exists to make rather than assume.

## What has not changed by default

Backtest numbers move because of four changed defaults, all reversible:
`training.target_transform`, the Monte Carlo drift shrinkage, the weighting
guards, and `training.cost_adjust_target` (which, per the note above, moves
nothing under the default rank target). Deliberately unchanged:
`risk.portfolio_volatility_target` is off, `scoring.method` is `weighted_sum`,
`simulation.use_garch_volatility` is still false, `src/regime.py` is untouched,
and `paper_trading_mode` is true.

One default change is **not** opt-in and cannot be reverted by config: gap-aware
stop fills. A level the market gapped through now fills at the open rather than
at the level, in both directions. There is no flag for it because the old
behaviour was not a modelling choice with a defensible other side — it credited
the book with liquidity that did not exist, asymmetrically, since the gaps that
blow through a long's stop are the adverse ones. Expect reported drawdowns and
realized losses to get slightly worse and more honest; the same correction also
removes an upward bias in Kelly's payoff ratio `b`, which had been fed an
average loss smaller than the one actually taken.

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
