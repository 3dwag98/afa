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
| D5 | GJR-GARCH unusable at platform scale | **Done** — scheduled refits, measured 16.9x | `src/volatility_models.py::forecast_volatility_scheduled` |
| D6 | No Markov chains / regime-switching models | **Done** as a module; not switched on | `src/markov_regime.py` |
| D7 | Sharpe mis-specified; no PSR/DSR/PBO | **Done** | `src/performance_stats.py` |
| D8 | Adaptive weighting unshrunk, in-sample | **Partly done** — shrunk and gated, still in-sample | `strategies/weighting.py` |
| D9 | Universe is an alphabetical slice, not point-in-time | **Not done** — data problem | — |
| D10 | `MC_Prob` holds 25% of weight, discriminates nothing | **Partly done** — see below | `strategies/rule_based.py` |

### What D5's fix does and does not claim

The review's option 1 — refit on a schedule, run the recursion forward daily.
`fit_garch_parameters` does the maximum likelihood; `filter_conditional_variance`
and `forecast_from_parameters` do the arithmetic. So between refits the
*parameter vintage* goes stale and the *conditional variance* does not, which is
the right way round: GARCH parameters move on the scale of weeks and sigma^2
moves daily.

Measured on a simulated GJR series, 40 consecutive scoring days for one ticker:

| | per-call refit | scheduled (21) |
|---|---|---|
| wall clock | 81.4 ms/day | 4.8 ms/day |
| fits for the documented backtest | 4,468,044 | 212,764 |
| extrapolated optimizer time | ~101 h | ~5 h |

**16.9x**, for volatility paths that differ from the per-bar refit by a median
of 0.64% and at worst 1.3%. An `arch`-gated test asserts the scheduled path
agrees with `forecast_volatility()` on a refit boundary, so the recursion is
checked against `arch`'s own forecaster rather than only against itself.

The fit window is truncated to a multiple of the interval, which makes the
cached parameters a pure function of `(symbol, anchor)` rather than of which
worker process saw the ticker first — without that the cache would break
`test_parallel_determinism.py`. Failed fits are cached too, or a ticker that
never converges pays the full optimizer cost every day, which is the cost this
exists to avoid concentrated on the worst names.

What it does **not** do: `use_garch_volatility` is still `false` by default.
The change makes it affordable to turn on; it is not evidence that turning it on
improves anything, and that is a question for a backtest on a point-in-time
universe.

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
| Training label gross of costs | **Done** — `training.cost_adjusted_target`, on by default |
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
| 5.2 | Gap risk reaches volatility estimation, and now the stop fill — but still not position sizing | **Fill done** (`BacktestEngine._gap_aware_fill`). Risk-per-share still omits the overnight component `z * sigma_gap * entry` |
| 5.3 | SEBI surveillance frameworks are published, objective and change tradability | A daily scraper for the NSE ASM/GSM/ESM/T2T lists (Phase 2 data) |
| 5.4 | No tax-lot accounting; no 365-day LTCG boundary optimization | FIFO lots per demat; a real India-specific alpha source left unclaimed |
| 5.5 | Momentum and low-vol run without factor neutralization | Residual momentum needs only the sector map from Phase 2 |

## Roadmap phases (§7)

| Phase | Status |
|---|---|
| 0 — Correctness | **Complete.** Kelly units (0.1), drift shrinkage (0.2), arithmetic Sharpe and time-varying `r_f` (0.3), weighting guards (0.4), GARCH refit scheduling (0.5), rank composite (0.6, opt-in), gap-aware stop fills (0.7), net-of-cost training label (0.8) |
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

### Two notes on 0.7 and 0.8

**0.7 — the gap fill is one-signed, and that is the point.** A stop is a
resting order, not a guaranteed price: when the session opens below it, the
first available price is the open. Filling at the stop regardless never
overstates a loss, only ever understates it, and understates it most on exactly
the gap days the drawdown breaker exists to catch. It also feeds back into
sizing — a recorded average loss smaller than the real one biases Kelly's `b`
upward and sizes the next position too large. The take-profit leg gets the same
treatment in the other direction (a session that gaps *above* the target fills
at the open, which is better), because correcting the loss leg alone would
trade one bias for another. **This makes reported backtest returns worse, and
that is the correct direction.**

**0.8 — most of a round-trip cost is invisible to the default target.** The
statutory part — brokerage, STT on both legs, exchange and SEBI charges, GST,
stamp duty — is identical for every name, so it is a pure level shift.
Cross-sectional demeaning subtracts a level shift back out and ranking is
invariant to it, which means subtracting a flat 0.8% from every label under
`cross_sectional_rank` would change precisely nothing. What survives is the
part that varies by name: slippage, estimated as `0.5 * ATR / price`, the same
bid-ask proxy `ExecutionSimulator` charges realized fills. That is the whole
economic content of the adjustment for a micro-cap-tilted universe — a
wide-spread small cap has to move materially further than a liquid large cap to
deliver the same return to the portfolio, and a model ranking on gross returns
cannot see the difference. A test asserts the charge survives cross-sectional
ranking, so the no-op case cannot reappear unnoticed.

## What has not changed by default

Backtest numbers move because of five changed defaults, all reversible:
`training.target_transform`, `training.cost_adjusted_target`, the Monte Carlo
drift shrinkage, the weighting guards, and the gap-aware stop fill (which has no
switch — the old behaviour was wrong rather than optional). Deliberately
unchanged: `risk.portfolio_volatility_target` is off, `scoring.method` is
`weighted_sum`, `simulation.use_garch_volatility` is still false, `src/regime.py`
is untouched, and `paper_trading_mode` is true.

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
