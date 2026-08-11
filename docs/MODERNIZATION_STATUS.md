# Modernization plan — implementation status

Tracks the quantitative review's defect list (D1–D11) and phased roadmap
against what is actually in the codebase. Each entry says what shipped, where
it lives, and — where a fix changed behaviour — what changed and why.

Anything not listed as **Done** is not implemented. This file is the honest
answer to "which of the plan's items can I rely on?"

---

## Defects

| # | Finding | Status | Where |
|---|---|---|---|
| D1 | Kelly `f*` applied as an allocation fraction without dividing by loss-given-stop | **Done** | `src/risk.py::kelly_allocation_fraction` |
| D2 | Monte Carlo `probability_profit` dominated by drift-estimation noise | **Done** | `src/monte_carlo.py::DriftPrior` |
| D3 | No portfolio covariance anywhere; positions sized independently | **Not done** | — (Phase 3) |
| D4 | Neural target is an absolute return, not a cross-sectional excess | **Done** | `agents/trainer.py::apply_cross_sectional_target` |
| D5 | GJR-GARCH architecturally unusable — one MLE fit per ticker per day | **Done** | `src/volatility_models.py::GARCH_REFIT_INTERVAL` |
| D6 | No Markov / regime-switching model; the "classifier" is a threshold cascade | **Not done** | — (Phase 5) |
| D7a | `Sharpe = (CAGR − r_f)/σ` mixes geometric and arithmetic measurement spaces | **Done** | `src/risk_analytics.py::calculate_sharpe_ratio` |
| D7b | No Probabilistic or Deflated Sharpe despite an extensive config search | **Done** | `src/performance_stats.py`, `src/trial_log.py` |
| D7c | Hardcoded constant 6.5% risk-free rate | **Done** | `RiskAnalyzer(risk_free_rate=...)` accepts a `pd.Series` |
| D8 | Adaptive component weighting uses a raw, unshrunk win rate at a 5-trade floor | **Done** | `strategies/weighting.py` |
| D9 | Universe is an alphabetical slice with no point-in-time index membership | **Not done** | — (Phase 2, a data problem) |
| D10 | `MC_Prob` holds 25% of the score weight but is near-constant | **Done** | `strategies/weighting.py::combine_rank_composite` |
| D11 | Overlapping-label Sharpe not corrected | **Done** | `src/performance_stats.py::newey_west_variance` |
| D11 | Training label is gross of costs | **Done** | `training.target_net_of_costs` |
| D11 | Isotonic calibration pooled across regimes | **Not done** | — (Phase 6) |
| D11 | Calibrating `p` but not the payoff distribution `b` | **Not done** | — (Phase 6) |
| D11 | No ASM/GSM/ESM/T2T awareness | **Not done** | — (Phase 2) |
| D11 | No lot sizes, no T+1 settlement lag | **Not done** | — |
| D11 | No turnover or ADV capacity model | **Not done** | — (Phase 7) |

---

## Phase 0 — Correctness · **complete**

**0.1 Kelly units.** `f* = p − (1−p)/b` is the growth-optimal *stake* in a bet
that loses the whole stake. A stop-loss trade loses the distance to the stop —
about 6% of the position — so spending that figure as an allocation fraction
under-bets by `1/l`, roughly 17× at Indian stop widths, and every safety
mechanism above it was guarding a quantity already an order of magnitude too
small. `kelly_allocation_fraction` implements
`f* = (p·g − (1−p)·l)/(g·l)`. `estimate_kelly_inputs` returns a `KellyInputs`
carrying both averages rather than only their ratio, and both sizing call
sites pass the trade's *own* distance to the stop — the correction scales as
`1/l`, so it re-sorts the book by stop width rather than rescaling it. The
`min(1.0, f_star)` clamp became an explicit `MAX_GROSS_EXPOSURE`.

> **Behavioural consequence, stated plainly.** With the units right, an
> ordinary 55–60% / 2:1 edge saturates every cap above it, so
> `max_single_position_pct` is now what sizes the book. That is the honest
> reading of Kelly for a bet that can only lose 5–6%, and it is the strongest
> argument for Phase 3: a per-name cap is not a risk model.

**0.2 Drift shrinkage.** `DriftPrior` estimates `(μ̄, τ²)` across the panel by
method of moments and shrinks each ticker's sample mean toward the universe
mean. The prior is estimated only from history visible at T (every 21 scoring
rounds in the backtest, once per run live). `MonteCarloResult` additionally
reports `probability_profit_lower` — the same paths re-scored with the drift
at its one-sided 95% bound — and `compliance.gate_on_probability_lower_bound`
(default on) makes that the number the BUY gate reads.

Measured on a 120-ticker zero-drift panel: names clearing the 0.55 gate fall
from 2.5% to 0.0%, and the spread of `probability_profit` falls by over a
third.

**0.3 Sharpe.** Arithmetic mean daily excess return over its own standard
deviation, annualized by `√252`. Sortino received the same correction.
`risk_free_rate` accepts a `pd.Series` (the 91-day T-bill) reindexed onto the
equity curve.

**0.4 Adaptive weighting.** Shrinks the win rate with the same Beta prior the
Kelly path uses, requires 30 realized trades *per component* rather than 5 in
total, and moves a weight only when a two-sided binomial test rejects "coin
flip" at α = 0.05.

**0.5 GARCH refit scheduling.** Parameters are fitted every 21 observations
per ticker and cached; the conditional variance recursion — which is what
actually depends on today's bar — runs on every call.
`forecast_from_parameters` reproduces `arch`'s own multi-step forecast to
4e-16 relative error, so this is an optimization and not a silent
approximation. ~12× faster over 60 successive days.

**0.6 Rank composite.** `combine_rank_composite` ranks each component within
the date's cross-section before combining, so components are commensurable by
construction and a weight buys discrimination rather than level: given the
whole weight budget, `MC_Prob` moves the weighted sum by 3 points and the rank
composite by 60. Opt in with `scoring_mode: rank_composite`; the weighted sum
remains the default.

This required a batched scoring path, which also fixed the documented
standalone/UMA divergence — a `rule_based` member inside a batched UMA
received no per-ticker Monte Carlo result and scored `MC_Prob` at zero, a
silent ~12-point level shift. `StrategyContext.mc_results` now carries
per-symbol results and the engine populates them.

**0.7 Gap-aware stop fills.** Stops fill at `min(open, stop)`. A stop is an
intraday construct; when the market gaps through it, the price never trades
there. This also un-biases Kelly, which was reading losses at the modelled
stop and therefore understating `l`. The favourable side of the gap is *not*
credited on the target — modelling that would be assuming gaps always go your
way.

**0.8 Net-of-cost training label.** `training.target_net_of_costs`. Note the
interaction: friction is the same fraction for every ticker, so under either
cross-sectional transform it is a level shift the ranks are invariant to. It
changes the label only under `target_transform: absolute` — which is itself an
argument for the cross-sectional target.

---

## Phase 1 — Measurement layer · **complete**

- **1.1 PSR and DSR** — `src/performance_stats.py`, surfaced in the analytics
  report. PSR carries the skewness/kurtosis correction, so negative skew and
  fat tails reduce confidence rather than being invisible.
- **1.2 Trial log** — `src/trial_log.py`. Append-only JSONL, keyed by config
  hash, written by the backtester *before* the Excel export so a run that dies
  during reporting still counts toward N. Repeat runs of one configuration
  count once. With no log, the DSR reports `computable: False` rather than
  pretending there was one trial.
- **1.3 PBO** — combinatorially-symmetric cross-validation. Tests the
  *selection procedure*, not the strategy.
- **1.4 Rank IC and ICIR** — reported by `evaluate_predictions`.
- **1.5 Newey–West** — Bartlett-kernel long-run variance, and
  `strategy_sharpe_overlap_adjusted` alongside the naive figure.
- **1.6 Purge and embargo** — `purge_and_embargo` drops training rows whose
  label window touches the test period on *either* side (the previous code
  handled only the left boundary and would leak on a non-contiguous split),
  with an optional `walk_forward_embargo_days` gap afterwards. The same purge
  now applies between each fold's inner training block and the inner
  validation block that drives early stopping — a selection set leaks exactly
  as a test set does.

---

## Phase 4 — Signal layer · **partial**

- **4.1 Cross-sectional target** — **Done**.
  `training.target_transform: cross_sectional_rank` (default) maps each date's
  cross-section to [-1, 1] by rank; `cross_sectional_demean` subtracts the
  universe mean. `absolute` restores the previous behaviour.
- **4.2 Residual momentum** — not done (needs the Phase 2 sector map).
- **4.3 Expanded feature set** — not done.
- **4.4 Conditional autoencoder** — not done.
- **4.5 Kronos as a frozen feature extractor** — not done.
- **4.6 Seed ensembling** — not done.

Per-date cross-sectional *feature* normalization (the second half of D4) is
also not done: features are still standardized with a single global mean/σ per
feature across the pooled panel.

---

## Not started

**Phase 2 (data foundation)** — point-in-time index membership, ASM/GSM/ESM
lists, per-day circuit bands, sector map, India VIX, the T-bill series, FII/DII
flows, free float and ADV, corporate actions. This is the binding constraint,
and it is a data problem rather than a code problem: until it is fixed, no
cross-sectional backtest number from this repository should be believed in
either direction.

**Phase 3 (portfolio construction)** — covariance estimation, constrained
mean-variance with a turnover penalty, HRP. The plan's largest single lever,
and now the most conspicuous gap: fixing D1 made the per-name concentration
cap the binding sizing constraint, and a per-name cap does not model
correlation.

**Phase 5 (Markov regime layer)**, **Phase 6 (conformal uncertainty)**,
**Phase 7 (execution and capacity)**, **Phase 8 (governance)** — not started.

---

## Reproducing the numbers quoted above

```bash
python -m pytest portfolio_agent/tests/test_performance_stats.py \
                 portfolio_agent/tests/test_trial_log.py \
                 portfolio_agent/tests/test_monte_carlo.py \
                 portfolio_agent/tests/test_risk_compliance.py \
                 portfolio_agent/tests/test_volatility_models.py
```

The D1 factor, the D2 false-positive rate, the GARCH recursion's agreement
with `arch`, and the rank composite's dispersion are each pinned by a named
test rather than quoted from a one-off script.
