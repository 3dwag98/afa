# Audit Response and Engineering Roadmap

**Scope.** This document responds to an external quantitative audit of AFA and converts it
into a prioritized engineering plan. Every claim in the audit was re-checked against the
code at `8db845c`. The headline result is that **most of the audit describes a version of
this repository that no longer exists**: seven of its ten findings are already implemented,
one is misdiagnosed, and two are genuinely open.

That matters for planning. Acting on the audit as written would mean re-implementing
finished work (circuit filters, Indian tax friction, Student-t innovations, PatchTST,
pinball loss) while the one real mathematical bug it identified stays in production.

**Baseline.** 709 tests pass, 1 fails (`test_example_uma_yaml_loads`), and that failure is
an environment artifact — `torch` is not installed in the default `uv sync`, so the `lstm`
strategy never registers. It is not a code defect.

---

## 1. Audit triage

| # | Audit claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Monte Carlo applies Itô's correction twice | **CONFIRMED — real bug** | `monte_carlo.py:383,396,402`; measured below |
| 2 | MC draws Gaussian shocks off a Student-t GARCH fit | **Already fixed** | `_standardized_shocks()`, `innovation_df` threaded via `run_monte_carlo_garch` |
| 3 | No circuit-limit handling | **Already implemented** | `liquidity.py` — full 1/2/5/10/20% ASM/GSM ladder |
| 4 | No STT / stamp duty / spread modelling | **Already implemented** | `execution_sim.py` — STT, stamp duty, GST, SEBI, exchange, STCG/LTCG |
| 5 | Trained on `MSELoss` | **Already fixed** | `loss: quantile` is the shipped default; `QuantileLoss` is pinball |
| 6 | Vanilla LSTM, should be PatchTST/TFT | **Already implemented** | `PatchTSTForecaster` in `models/pytorch_models.py` |
| 7 | Kelly is regime-fragile | **Partly mitigated; residual risk real** | Beta shrinkage, κ ≤ 0.25 cap, net-of-cost inputs — but no regime conditioning |
| 8 | "Standard differencing wipes out memory" | **Misdiagnosed** | No differencing is applied *at all*; the real problem is raw price levels |
| 9 | No HMM regime filter | **Genuinely open** (a deterministic filter exists) | `regime.py` is threshold-based, not probabilistic |
| 10 | No fractional differentiation | **Genuinely open** | No `fracdiff` anywhere in the tree |

So the actionable set is **#1, #7, #8, #9, #10** — and #8 is a different problem from the
one the audit named.

---

## 2. The one real bug: double Itô correction

### What the code does

`monte_carlo.py` estimates drift from log returns:

```python
log_returns = np.log1p(returns_arr)      # line 354
mu = np.mean(log_returns)                # line 356  <- already a LOG-space drift
...
daily_drift_path = mu - 0.5 * sigma_path ** 2   # lines 383, 396, 402
```

In GBM, log returns are distributed `N(μ_arith − ½σ², σ²)`. Since `mu` is estimated
*directly from log returns*, it already **is** `μ_arith − ½σ²`. Subtracting `½σ²` a second
time drives the simulated log drift to `μ_arith − σ²`. All three shock methods share the
line, so all three inherit it.

### Measured impact

Simulating series with a known log-drift of +0.0005/day, horizon 20 days, 200k paths:

| Ticker profile | Daily σ | Correct E[20d] | MC reports | Bias | P(profit) error |
|---|---|---|---|---|---|
| Large-cap (~24% ann) | 1.5% | +0.02% | −0.18% | **−0.20%** | −1.2 pp |
| Mid-cap (~48% ann) | 3.0% | +2.21% | +1.33% | **−0.87%** | −2.6 pp |
| Small-cap (~71% ann) | 4.5% | +2.40% | +0.38% | **−2.02%** | −3.9 pp |

### Why this is worse than a level bias

The error is `½σ²H` — **proportional to variance**. It is therefore not a constant haircut
that washes out of a ranking; it is a *volatility-graded penalty* applied to exactly the
names where it does the most damage. Two consequences:

1. **It corrupts a hard gate.** `rule_based.py:172` reads
   `passed_prob = prob_profit >= context.risk.target_prob_profit`, with
   `target_prob_profit = 0.55`. A small-cap is being assessed ~3.9 pp below its true
   probability of profit. Signals are being rejected by an arithmetic error.
2. **It silently duplicates the low-volatility strategy.** A variance-proportional penalty
   on probability-of-profit is a low-vol tilt. The platform already *has* an explicit
   `LowVolatilityStrategy`; this bug applies a second, undocumented one to every other
   strategy that consumes `mc_result`.

### The fix

Delete the `- 0.5 * sigma_path ** 2` term at all three sites. The drift becomes `mu`, which
makes simulated `E[Σ log r] = H·μ_log`, matching the empirical distribution the simulation
is meant to resample.

Two things to preserve:

- **Keep the jump compensator** at line 396. That one is correct — it stops jump risk from
  shifting the mean, which is exactly the property `test_jump_drift_is_compensated…` pins.
- **`daily_drift_path` stops being a path.** With the correction gone the drift is a
  scalar; the GARCH `sigma_path` continues to govern shock *scale* only. That is the right
  separation of concerns and simplifies the three branches.

### ⚠️ This fix is not behaviourally free

Removing the bias **raises every probability-of-profit**, so more signals will clear the
0.55 gate — disproportionately high-volatility ones. `target_prob_profit = 0.55` was tuned
against biased numbers, so it must be **re-calibrated in the same change**, or the fix will
read as a sudden loosening of risk discipline. Ship the correction and the re-calibration
together, with a before/after backtest on an identical universe and window.

---

## 3. The real stationarity problem (audit #8, corrected)

The audit assumed the platform applies standard differencing and warned that this destroys
long-term memory. **No differencing is applied at all.** The actual defect is the opposite
and more serious.

The shipped `config.yaml` feature set is:

```yaml
features:
  normalize: false          # <- rolling causal z-score is OFF
  feature_sets:
    core:       [sma_20, sma_50, sma_200]        # rupees
    momentum:   [rsi_14, macd, return_1d, return_5d]   # macd is also rupees
    volatility: [atr_14, bollinger_pct_b, donchian_upper_20]  # rupees
```

**Six of ten features are in rupee units** (`sma_20/50/200`, `macd`, `atr_14`,
`donchian_upper_20`). The only normalization on the default path is
`FeatureScaler` — a *single global* mean/std fitted across the entire stacked panel, then
clipped to ±10σ.

One global mean/std cannot serve a universe spanning ₹20 to ₹5,000. Measured on a
three-ticker panel (₹25 / ₹400 / ₹3,000), after `FeatureScaler`:

```
sma_20 z-scores       PENNY  mean −0.762  std 0.030
                      MID    mean +0.450  std 0.559
                      LARGE  mean +4.093  std 1.077

spread of per-ticker MEANS : 2.063
avg within-ticker STD      : 0.555
identity : signal ratio    : 3.7x
```

Two failures, both fatal to generalization:

1. **The feature encodes identity, not behaviour.** Between-ticker variance is 3.7× the
   within-ticker variance, so `sma_20` functions mainly as a price-level dummy telling the
   network *which stock this is*. On a real 4,000-name NSE universe the ratio is far worse
   than this three-name toy.
2. **Sensitivity varies 36× by ticker.** `PENNY`'s within-ticker std is 0.030 against
   `LARGE`'s 1.077. No single learned weight can be correct for both.

There is also a pure non-stationarity failure on top: a stock that triples during training
has `sma_200` drifting monotonically, so test-period values fall outside the training
distribution and get pinned at the ±10σ clip.

### Fix ladder — cheapest first

This is where the audit's priority ordering is most wrong. It puts `fracdiff` first;
fracdiff is step 3, and steps 1–2 capture most of the benefit for a fraction of the effort.

1. **Ratio-transform the price-level features** — the 80/20. Replace absolutes with
   dimensionless equivalents: `close/sma_200 − 1`, `atr_14/close`,
   `(donchian_upper_20 − close)/close`, `macd/close`. Scale-free, stationary, directly
   interpretable, and no new dependency. Do this first.
2. **Make normalization cross-sectional.** Z-score each feature *across the universe on
   each date* rather than pooling all tickers and dates. This is what Qlib's
   `CSZScoreNorm` does and what NSE's own factor indices do when building composite
   scores. It removes the identity leak by construction.
3. **Then evaluate fractional differentiation** for whatever genuine long-memory signal
   remains. Evidence supports it — fractionally differenced series improve neural forecast
   error versus integer differencing, and the minimum stationary `d` is typically well
   below 1, so most memory survives. But it is only worth measuring *after* steps 1–2,
   because it addresses a residual that ratio features have already largely removed.

---

## 4. HMM regimes — do it, but not the way the audit says

The audit recommends `regime_filter.py` "using the Baum-Welch algorithm." Implemented
literally, that is a **lookahead-bias generator**.

Baum-Welch is an EM procedure whose E-step produces *smoothed* posteriors
`P(S_t | y_1…y_T)` — conditioned on the **entire** sample, including the future. Published
practitioner work isolating this effect finds the same allocation rule scores **Sharpe 0.78
evaluated honestly and 1.74 evaluated on smoothed probabilities**. Notably, *parameter*
lookahead is nearly free while *probability* lookahead is decisive — so the danger is not
fitting on history, it is labelling with smoothed states.

Non-negotiable requirements if this is built:

- **Filtered, never smoothed.** Use forward-algorithm `P(S_t | y_1…y_t)` only. A backtest
  consuming Viterbi paths or `predict_proba` over the full series is invalid.
- **Walk-forward refit with embargo**, matching the discipline `trainer.py` already
  enforces (`embargo_days`, date-based folds, causal scaling — `QUANT_RESEARCH.md` §18).
  The existing infrastructure is the reason this is even feasible; reuse it.
- **Benchmark against the incumbent, don't assume improvement.** `regime.py` already
  implements a well-designed deterministic filter (200-DMA trend + realized-vol multiple +
  ADX chop test, fail-neutral on thin history). It must be the control arm. A 2-state HMM
  that cannot beat a 200-DMA out-of-sample is a large increase in machinery for nothing,
  and that is a common outcome.

**Indian calibration anchor.** Markov-switching studies on the Nifty 50 report transition
probabilities of ~0.9694 (bear) and ~0.9893 (bull) — highly persistent states with expected
durations of roughly 33 and 93 days. Use these as priors and as a sanity check: a fitted
HMM producing 3-day regimes has found noise, not states.

**Recommended shape.** Implement as an alternative provider behind the existing
`MarketRegime` interface, so `classification` can be sourced from either the deterministic
rules or the HMM, and the two can be A/B tested through the same backtest harness. Do not
replace `regime.py`.

**MS-GARCH** is the higher-value variant of the same idea and needs no new heavy
dependency — `statsmodels` provides `MarkovRegression` / `MarkovAutoregression`. It targets
a documented weakness: `volatility_models.py` notes that a single-regime GJR-GARCH fit
drags persistence toward the stationarity bound and destabilizes across refits. Letting the
GARCH parameters switch is the textbook fix for precisely that pathology.

---

## 5. Kelly under regime shift (audit #7)

The audit's concern is legitimate but the code is further along than it credits. `risk.py`
already applies Beta-Binomial shrinkage toward a no-edge prior, caps the fractional-Kelly
multiplier at κ ≤ 0.25, enforces a 50-trade floor, and — importantly — restates the trade
history **net of friction** before estimating `p` and `b`.

The residual risk is real and unaddressed: `p` and `b` are estimated over a trailing window
with **no regime conditioning**, so a bull-market sample sizes up into a structural break.

Two options, in order of cost:

1. **Condition Kelly inputs on the regime label** once §4 lands — estimate `(p, b)`
   separately per regime, falling back to the pooled estimate when a regime has too few
   trades. This composes with the existing shrinkage rather than replacing it.
2. **Drawdown-scaled κ** — reduce the multiplier as realized drawdown deepens. Cheaper,
   independent of the HMM work, and directly targets the "catastrophic drawdown" failure
   the audit describes.

Option 2 is the better first move: it is a small, self-contained change that does not block
on regime infrastructure.

---

## 6. Reference systems

| System | What to take from it | Caution |
|---|---|---|
| [**Microsoft Qlib**](https://github.com/microsoft/qlib) | Closest architectural analogue — full pipeline from data to execution, loosely coupled modules, point-in-time data handling, cross-sectional normalization (`CSZScoreNorm`), model zoo | China A-share defaults; data layer assumes its own binary format |
| [**mlfinlab / Hudson & Thames**](https://hudsonthames.org/fractional-differentiation/) | Reference implementations of exactly the open items — fractional differentiation, plus triple-barrier labelling, meta-labelling, `PurgedKFold` | Now partly commercial; `mlfinpy` is an open fork |
| **`statsmodels`** | `MarkovRegression` / `MarkovAutoregression` — MS-GARCH without a new heavy dependency | Already an indirect dependency via `arch` |
| **`hmmlearn`** | Standard Gaussian HMM; must use `filter()` not `predict_proba()` over the full series | Defaults invite the smoothing trap in §4 |
| **`arch`** (Sheppard) | Already in use for GJR-GARCH — the right choice, keep it | — |
| [**FinRL**](https://github.com/AI4Finance-Foundation/FinRL) | Reference only, for the DRL question in §7 | Research-grade; see §7 for why this is not a near-term target |
| [**NSE Nifty200 Momentum 30**](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30-index/nifty200_momentum_30_index_whitepaper_sep_20.pdf) | The *investable benchmark* the momentum strategy must beat — there are index funds tracking it | Methodology differs from AFA's; see below |

### The benchmark divergence worth fixing

NSE's Nifty200 Momentum 30 builds a **Normalised Momentum Score from 6-month *and*
12-month price returns, each divided by daily-return volatility**, then weights by
free-float cap × NMS with a 5% cap, rebalanced semi-annually.

AFA uses **raw 9-month return, skip-1-month**, equal-weighted, with no volatility
adjustment (`features/technical.py::mom_9m_skip1m`).

These are materially different factors. Volatility-adjusting momentum is what stops the
score from being dominated by high-variance names — which is the same population the
circuit-limit and zombie screens in `liquidity.py` already exist to catch, and the same
population the double-Itô bug is currently over-penalizing from the other direction.

**Recommendation:** implement NSE's NMS as a registered strategy variant alongside the
Jegadeesh-Titman one. It costs little, and it converts backtests from "does this look good"
into "does this beat a fund you could actually buy." That is the single highest-value
addition to the *financial* model set in this document.

### Cost realism cross-check

Indian factor-backtest practitioners model roughly **0.11% per trade** in statutory charges
plus ~0.05% slippage. AFA's estimator assumes 25 bps of slippage *per side* and lands near
**0.8% per round trip** — materially more conservative. That is defensible for mid/small
caps and the docs justify it, but the gap should be a documented, configurable assumption
rather than an accident, because it directly determines which signals clear
`min_reward_risk`. Published work also finds **monthly rebalancing beats weekly** in Indian
equities once costs bite — worth pinning as a default and a test.

---

## 7. What the audit recommends that you should *not* build yet

Honest pushback matters more than a longer roadmap.

- **Deep RL (PPO/SAC) for position sizing.** The audit's Phase 3 centrepiece. DRL is
  sample-hungry, and OHLCV daily bars over a few thousand NSE names is a small, noisy,
  non-stationary sample with no execution feedback loop to learn against. The platform
  cannot observe fill quality, so an agent "learning to avoid circuit traps" has no
  gradient signal for the thing it is supposed to learn — `liquidity.py`'s explicit
  detectors do that job with far less machinery and full auditability. Revisit only if
  intraday data and a real fill model are ingested.
- **Temporal Fusion Transformer.** TFT's advantages are static covariates and *known
  future inputs* (earnings dates, index-rebalance dates). This platform is deliberately
  OHLCV-only, so both are unavailable and TFT reduces to a heavier PatchTST. PatchTST is
  already implemented and is the correct choice for this data.
- **Rough volatility (Rough Bergomi).** Estimating the roughness parameter needs
  high-frequency intraday data. On daily OHLCV the estimate is dominated by microstructure
  noise and discretization error. Not actionable without a data-source change.

Each of these becomes reasonable only after a specific *data* upgrade — which makes the
data layer, not the model layer, the real gate on the audit's Phase 3.

---

## 8. Prioritized roadmap

### P0 — Correctness (ship first, small, high confidence)

1. **Fix the double-Itô drift** in `monte_carlo.py` (3 lines), keeping the jump
   compensator. Add a regression test pinning `E[Σ log r] ≈ H·μ_log` across all three
   methods.
2. **Re-calibrate `target_prob_profit`** in the same change, with a before/after backtest.
   Non-optional — see §2.
3. Make the `torch`-absent test failure explicit (skip rather than fail) so the suite is
   green in a default `uv sync`.

### P1 — Feature integrity (largest model-quality win per unit effort)

4. **Ratio-transform the six rupee-denominated features.**
5. **Switch to cross-sectional (per-date) normalization.**
6. Re-run walk-forward and compare against the current baseline. Steps 4–5 are the plan's
   highest expected improvement in out-of-sample forecast quality.

### P2 — Indian financial-model depth

7. **Implement NSE's Normalised Momentum Score** as a strategy variant; benchmark against
   Nifty200 Momentum 30.
8. **Drawdown-scaled Kelly κ** (§5, option 2).
9. Document and pin the slippage assumption; add a monthly-vs-weekly rebalance test.

### P3 — Regime modelling (only with §4's guardrails)

10. **MS-GARCH via `statsmodels`** — targets the documented persistence instability, no new
    heavy dependency.
11. **Filtered-probability HMM** behind the `MarketRegime` interface, walk-forward refit,
    A/B against the deterministic filter. Ship only if it wins out-of-sample.
12. **Regime-conditioned Kelly inputs** (§5, option 1), once 10–11 land.

### P4 — Evaluate, don't assume

13. **Fractional differentiation**, measured against the P1 baseline rather than adopted on
    principle.

**Explicitly deferred:** DRL, TFT, rough volatility — all blocked on data-layer
capabilities, not model choices (§7).

---

## 9. Sources

- Politis & Romano, "The Stationary Bootstrap", *JASA* 89(428), 1994
- Merton, "Option pricing when underlying stock returns are discontinuous", *JFE* 3(1-2), 1976
- [Microsoft Qlib](https://github.com/microsoft/qlib) · [Qlib paper](https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/)
- [Hudson & Thames — Fractional Differentiation](https://hudsonthames.org/fractional-differentiation/) · [Fractional differentiation in ML (Springer)](https://link.springer.com/article/10.1007/s12572-021-00299-5)
- [HMM regime detection: the lookahead ladder (Sharpe 0.78 vs 1.74)](https://github.com/dmitridefreitas-dev/regime-detection)
- [Regime switching volatility, Indian market (MDPI *Mathematics* 9(14), 1595)](https://www.mdpi.com/2227-7390/9/14/1595) · [Identifying regime shifts in the Indian stock market (MPRA)](https://mpra.ub.uni-muenchen.de/37174/1/MPRA_paper_37174.pdf)
- [NSE Nifty200 Momentum 30 index whitepaper](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30-index/nifty200_momentum_30_index_whitepaper_sep_20.pdf)
- [BacktestIndia — NSE factor backtests, cost assumptions](https://backtestindia.com/)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
