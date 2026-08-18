# Quantitative Research Basis for AFA's Strategies

This document is the mathematical/research foundation behind the platform's strategies. For each model: the academic evidence (with an emphasis on studies that specifically test Indian equities, not just US/global markets), the exact formulation used, and — since this platform only ingests OHLCV data (no fundamentals, no institutional-flow data) — an honest assessment of what's implementable today versus what needs a new data source.

## Contents

1. [Cross-sectional momentum](#1-cross-sectional-momentum) — implemented (`strategies/cross_sectional.py::MomentumStrategy`)
2. [Low-volatility anomaly](#2-low-volatility-anomaly) — implemented (`strategies/cross_sectional.py::LowVolatilityStrategy`)
3. [Conditional volatility (GARCH family)](#3-conditional-volatility-garch-family) — implemented (`src/volatility_models.py`)
4. [Kelly criterion position sizing](#4-kelly-criterion-position-sizing) — implemented (`src/risk.py::calculate_kelly_quantity`)
5. [Trend + breakout + volume + Monte Carlo](#5-trend--breakout--volume--monte-carlo) — implemented (`strategies/rule_based.py`, predates this document)
6. [Time-series momentum](#6-time-series-momentum) — partially implemented (subsumed by #5's trend component)
7. [Cointegration-based pairs trading](#7-cointegration-based-pairs-trading-not-yet-implemented) — **not implemented** (architectural gap)
8. [Fama-French multi-factor models](#8-fama-french-multi-factor-models-not-yet-implemented) — **not implemented** (data gap)
9. [Quality (QMJ) factor](#9-quality-qmj-factor-not-yet-implemented) — **not implemented** (data gap)
10. [FII/DII institutional flows](#10-fiidii-institutional-flows-not-yet-implemented) — **not implemented** (data gap)
11. [Calendar anomalies](#11-calendar-anomalies-deliberately-not-implemented) — deliberately not implemented (weak/inconsistent evidence)
12. [Momentum crash protection](#12-momentum-crash-protection) — implemented (`src/regime.py`)
13. [Transaction costs and concentration limits](#13-transaction-costs-and-concentration-limits) — implemented (`src/execution_sim.py`, `src/sectors.py`)
14. [Fat tails in the forward simulation](#14-fat-tails-in-the-forward-simulation) — implemented (`src/monte_carlo.py`)
    - [14.1 The drift must not be Itô-converted twice](#141-the-drift-is-estimated-in-log-space-so-it-must-not-be-converted-again) — fixed (`src/monte_carlo.py`)
15. [Circuit limits and the illiquidity illusion](#15-circuit-limits-and-the-illiquidity-illusion) — implemented (`src/liquidity.py`)
16. [Overnight gaps and GARCH stationarity](#16-overnight-gaps-and-garch-stationarity) — implemented (`src/volatility_models.py`)
17. [The long-only constraint](#17-the-long-only-constraint) — a known, accepted limitation
18. [Making the walk-forward measurement honest](#18-making-the-walk-forward-measurement-honest) — implemented (`agents/trainer.py`)
19. [Combining models: arbitration, not averaging](#19-combining-models-arbitration-not-averaging) — implemented (`src/trigger_engine.py`)
20. [Forecasting a distribution instead of a point](#20-forecasting-a-distribution-instead-of-a-point) — implemented (`models/pytorch_models.py`, `src/calibration.py`)
21. [The drift is the noisiest input in the simulation](#21-the-drift-is-the-noisiest-input-in-the-simulation) — implemented (`src/monte_carlo.py::shrink_drift`)
22. [Portfolio covariance and constrained allocation](#22-portfolio-covariance-and-constrained-allocation) — implemented (`src/portfolio.py`)
23. [Measuring a Sharpe ratio that means something](#23-measuring-a-sharpe-ratio-that-means-something) — implemented (`src/performance_stats.py`)
24. [Predicting the cross-section, not the market](#24-predicting-the-cross-section-not-the-market) — implemented (`agents/trainer.py::apply_cross_sectional_target`)
25. [Regimes as estimated states, not hand-set thresholds](#25-regimes-as-estimated-states-not-hand-set-thresholds) — implemented (`src/markov_regime.py`)
26. ["Cannot compute" is not "scores zero"](#26-cannot-compute-is-not-scores-zero) — implemented (`strategies/weighting.py::combine_weighted`)
27. [Reinforcement learning, scoped to what one trajectory supports](#27-reinforcement-learning-scoped-to-what-one-trajectory-supports) — implemented (`src/rl.py`)

Closing: [what to combine into a UMA](#summary-what-to-combine-into-a-uma)

---

## 1. Cross-sectional momentum

**Evidence.** Jegadeesh & Titman (1993) established that stocks with high returns over the past 3–12 months continue outperforming over the next 3–12 months. For India specifically: Joshipura & Mankar's NSE working paper on CNX 100 constituents found momentum profitability; a portfolio-based study of the Indian market found momentum profits over 6-month and 1-year horizons, with contrarian (reversal) returns emerging only over a 3-year horizon; sector-level studies confirm the effect is not limited to specific industries; liquidity research shows momentum is *strongest* among the most liquid stocks (favorable for backtesting realism — liquid names have less slippage). One caveat: short-term reversal coexists with momentum in India, motivating the standard skip-month convention below.

**The skip is now measurable rather than assumed (T29).** The 21 sessions `mom_9m_skip1m` drops are exactly what the `reversal` strategy ranks on, so the convention this platform inherited from the literature can be checked against this data in one command:

```bash
portfolio-agent compare --strategies momentum,reversal --slippage-bps 25
```

A flat or negative reversal spread means the skip is buying nothing here and those 21 sessions could be folded back into the formation return; a positive one means it is earning its keep. **Costs decide it, not the spread.** A one-month formation window implies replacing most of the decile every month against a round trip of 79.4 bps on the shipped Indian schedule (40.5 to buy, 39.0 to sell) — 9.53% of capital a year at full monthly turnover, which is larger than most published gross reversal spreads. `breakeven_round_trip_cost` (§13, T13) is the figure that settles it. A reversal sort also concentrates in precisely the names the tradability screen exists for (§15): a stock that fell hard on no volume prints the return it ranks highest while offering nothing to buy.

**Formulation.** For each stock \(i\) at formation date \(t\), with formation window \(J\) months and a skip period of 1 month (to avoid short-term reversal contamination — the most recent month's return behaves differently from the momentum-driving 2–12 month window):

$$
\text{MOM}_i(t) = \frac{P_i(t - 1\text{mo})}{P_i(t - 1\text{mo} - J)} - 1
$$

Rank all stocks in the universe by \(\text{MOM}_i(t)\) descending. Long the top decile (configurable percentile). This platform never shorts, so the bottom decile (which academic long-short studies short) is simply avoided rather than shorted.

**India-specific tuning.** \(J = 6\) to \(12\) months per the evidence above; this platform defaults to \(J=9\) months (splitting the difference, since both 6- and 12-month formation showed profitability and the effect reverses only past 3 years — far outside this range). Rebalance monthly.

**Sources:**
- [Momentum as an investment strategy in the Indian stock market](https://www.researchgate.net/publication/289034010_Momentum_as_an_investment_strategy_in_the_Indian_stock_market-an_evaluative_study)
- [Momentum Effect in Indian Stock Market: A Sectoral Study](https://journals.sagepub.com/doi/abs/10.1177/0972150915569940)
- [Momentum returns: A portfolio-based empirical study — Indian stock market](https://www.sciencedirect.com/science/article/pii/S0970389617301647)
- [Momentum, reversals and liquidity: Indian evidence](https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640)
- [Physical Momentum in the Indian Stock Market (NSE 500)](https://arxiv.org/abs/2302.13245)

---

## 2. Low-volatility anomaly

**Evidence.** Contrary to CAPM's prediction that higher risk should earn higher return, low-volatility stocks have historically matched or beaten high-volatility stocks on a risk-adjusted (and sometimes absolute) basis. This is one of the most robust anomalies found in Indian-market-specific research: a Nifty-500-based study (2001–2011, 2016–2023 extensions) confirms low-volatility portfolios beat both high-volatility portfolios and the market on a risk-adjusted basis; a broad 4,400-company, 19-year study finds the anomaly survives after controlling for standard factors; one study reports an 18.82% CAGR for low-volatility portfolios versus 17.22% for high-volatility.

**Formulation.** Trailing realized volatility over window \(W\) (default 60 trading days):

$$
\sigma_i(t) = \sqrt{\frac{252}{W-1}\sum_{k=0}^{W-1}\big(r_i(t-k) - \bar r_i\big)^2}
$$

Rank ascending; long the bottom decile (lowest realized volatility). This is the mirror image of momentum's ranking machinery, so both strategies share the `rank_and_select()` helper in `strategies/cross_sectional.py` (public since T25 — it was `_rank_and_select_decile`).

**Three sorts, not one.** Total volatility mixes two things a cross-sectional book should keep apart: how much a stock moves *with* the market, and how much it moves on its own. The formula above is the total, and the platform now implements all three decompositions as separate strategies so the comparison is one command rather than an argument:

| strategy | ranks on | registry feature |
| --- | --- | --- |
| `low_volatility` | total realized volatility (the formula above) | `realized_vol_60` |
| `low_volatility_idio` | volatility of the CAPM **residual** (T14) | `idiosyncratic_vol_60` |
| `bab` | rolling market **beta** (T28) | `market_beta_252` |

```bash
portfolio-agent compare --strategies low_volatility,low_volatility_idio,bab
```

T14 recorded the reason the second exists: 2025 work on the low-risk anomaly finds idiosyncratic-volatility sorts survive out of sample where beta sorts largely do not. `bab` is included so that claim is measured on this data rather than assumed from it.

**Betting against beta (`bab`).** Frazzini & Pedersen's account is a funding-constraint one: investors who want more risk than they can borrow to obtain bid up high-beta stocks instead, so beta is overpriced and the security market line is flatter than CAPM predicts. Indian evidence is favourable and specific — NSE 2001–2016 finds the effect positive across capitalizations *after controlling for size, value and momentum*, which matters because the naive objection to a low-beta sort is that it is a size bet wearing a different name.

**It is conditional, and must be reported that way.** 2025 Asian work finds the effect concentrated in downturns. A pooled IC made of a strong down-market number and a flat up-market one describes neither state, so `evaluation/conditional.py` splits IC and decile spread by market state. Two conditioners, and conflating them turns an attribution into an imaginary timing rule:

- `realized` splits on the market's return **over the label horizon** — it says *when the signal paid*, and is not tradable, because on the decision date nobody knows which bucket the date will land in.
- `trailing` splits on the market's state **as of the decision date** — tradable, and a weaker conditioner for exactly that reason.

**Sources:**
- [The Volatility Effect: Recent Evidence from Indian Markets](https://www.scirp.org/html/28-1501934_94780.htm)
- [An Investigation of Low Volatility Anomaly in Indian Stock Market](https://www.researchgate.net/publication/305621746_An_Investigation_of_Low_Volatility_Anomaly_in_Indian_Stock_Market)
- [Low-Risk Anomaly: Evidence from India](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4398656)
- Frazzini & Pedersen, ["Betting Against Beta"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X13002675), *Journal of Financial Economics* 111(1), 2014

---

## 3. Conditional volatility (GARCH family)

**Evidence.** The platform's original Monte Carlo simulation (`src/monte_carlo.py`) assumes *constant* volatility — a single historical standard deviation applied uniformly to every simulated day. Indian-market research consistently rejects this: NSE Nifty 50 and BSE Sensex returns exhibit strong volatility clustering (ARCH effects) and, more specifically, an **asymmetric leverage effect** — negative return shocks raise future volatility more than positive shocks of the same size (bad news is more impactful than good news). Studies comparing GARCH(1,1), EGARCH, TGARCH/GJR-GARCH, and FIGARCH on Nifty/Sensex find the asymmetric variants fit better than plain GARCH(1,1) specifically *because* of this leverage effect.

**Formulation.** GJR-GARCH(1,1) (Glosten-Jagannathan-Runkle), chosen over EGARCH for numerical simplicity (no log-variance transform) while still capturing the leverage effect:

$$
r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,\quad z_t \sim t_\nu
$$

$$
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \gamma\,\varepsilon_{t-1}^2\,\mathbb{1}[\varepsilon_{t-1}<0] + \beta\,\sigma_{t-1}^2
$$

- \(\omega > 0\), \(\alpha, \beta \ge 0\), \(\alpha + \gamma \ge 0\); stationarity requires \(\alpha + \tfrac{\gamma}{2} + \beta < 1\).
- \(\gamma > 0\) is the leverage term: negative shocks (\(\varepsilon_{t-1}<0\)) get an *extra* \(\gamma\varepsilon_{t-1}^2\) of variance on top of \(\alpha\varepsilon_{t-1}^2\).
- Student-\(t\) innovations (\(z_t \sim t_\nu\)) rather than Gaussian, since equity returns — especially in an emerging market with retail-heavy participation and circuit-limit dynamics — are fat-tailed.
- Fit via maximum likelihood using the `arch` package (Sheppard et al.) rather than hand-rolled optimization — this is standard practice and avoids numerical-optimization bugs.

**Use in the platform:** `src/volatility_models.py::forecast_volatility()` fits this model per ticker and produces a forward conditional volatility path, used as a (better) input to the lognormal Monte Carlo machinery in `monte_carlo.py` — enable it with `simulation.use_garch_volatility: true`, which is off by default because per-ticker MLE is much slower than the closed-form flat-vol path. It falls back to the flat historical standard deviation when there isn't enough history to fit reliably (~250+ observations for stable convergence) or when the `arch` package isn't installed.

Note what it does *not* currently drive: the volatility-targeting scalar in §12 keys off trailing realized volatility (`realized_vol_60`), not off this forecast. Feeding the GARCH path into position sizing is the natural next step and the literature below supports it, but it is not what the code does today.

**Sources:**
- [Modeling and Forecasting the Volatility of NIFTY 50 Using GARCH and RNN Models](https://www.mdpi.com/2227-7099/10/5/102)
- [Modelling time-varying volatility using GARCH models: evidence from the Indian stock market](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758444/)
- [Forecasting Stock Market Volatility using GARCH Models: Evidence from the Indian Stock Market](https://www.researchgate.net/publication/305803327_Forecasting_Stock_Market_Volatility_using_GARCH_Models_Evidence_from_the_Indian_Stock_Market)
- [Modelling Stock Market Volatility in India: A GARCH Analysis of the Nifty 50 Index](https://www.researchgate.net/publication/400666876_Modelling_Stock_Market_Volatility_in_India_A_GARCH_Analysis_of_the_Nifty_50_Index)
- Volatility targeting (position-sizing side): [The Impact of Volatility Targeting](https://www.man.com/insights/the-impact-of-volatility-targeting) (Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert, *Journal of Portfolio Management*, 2018); [Understanding Risk Parity (AQR)](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf)

---

## 4. Kelly criterion position sizing

**Evidence.** The Kelly criterion (Kelly, 1956) gives the capital fraction that maximizes long-run logarithmic (compound) growth for a repeated bet with known win probability and payoff ratio. Universally used in quantitative position sizing, but **full Kelly is not used in practice** — win probability and payoff ratio are estimated with error, and full Kelly is extremely sensitive to that error (and to return skewness). The standard mitigation, per both practitioner sources and the platform's own risk posture (paper trading, capped positions), is fractional Kelly: quant funds commonly use 1/4 to 1/2 Kelly; half-Kelly captures roughly 75% of the theoretical growth rate at meaningfully lower drawdown risk, quarter-Kelly roughly 50% of growth at even lower volatility.

**Formulation, and the measurement space it lives in.** The familiar Kelly result, for realized win probability \(p\) and reward:risk ratio \(b\) (average win ÷ average loss):

$$
f = p - \frac{1-p}{b}
$$

is the growth-optimal **stake** fraction for a *binary, all-or-nothing bet* — one that returns \(b\) times the stake with probability \(p\) and loses the **entire** stake otherwise. A stop-loss equity trade is not that bet. A loser gives back the distance from entry to stop, a few percent of the position, not the position.

For a position that gains a fraction \(g\) of its value with probability \(p\) and loses a fraction \(l\) otherwise, the growth-optimal **allocation** — the fraction of *wealth* to commit — is:

$$
f^{*}_{\text{alloc}} = \frac{p\,g - (1-p)\,l}{g\,l} = \frac{p - (1-p)/b}{l}, \qquad b = g/l
$$

The binary form is that expression with \(l = 1\). Using it as an allocation therefore under-allocates by a factor of \(1/l\): at \(p = 0.55\), \(b = 1.8\) and a 6% loss-given-stop it returns 0.30 where the allocation is 5.0, a factor of 16.7. The error is also **signal-dependent rather than a constant shrink** — \(l\) varies per trade, so a wide-stop signal that should receive a *smaller* allocation received the same one, which is why no adjustment to \(\kappa\) could have recovered it. `src/risk.py::kelly_allocation_fraction` does the conversion; `calculate_kelly_fraction` retains the binary form, which is the correct quantity in its own right and the numerator here.

**The constraint stack, and what actually binds.** Position size applies \(\kappa f^{*}_{\text{alloc}}\), then an explicit unlevered-cash-book leverage limit, then `max_single_position_pct`. \(\kappa\) is configurable through `risk.kelly_fraction` but **hard-capped at 0.25 (quarter-Kelly) at the point of use**, so no config, environment override or test fixture can size above it.

Worth stating plainly, because it follows from the corrected arithmetic rather than from any choice: with the units right, \(f^{*}_{\text{alloc}}\) for any trade with a real edge is of order 1–10, so the leverage and concentration limits bind almost always and **the concentration cap, not Kelly, is what sizes a position**. Kelly still binds where it matters — as the edge approaches zero, \(f^{*}_{\text{alloc}}\) collapses continuously to 0 and sizes the position down. The consequence is that growth-optimal sizing across a book is a portfolio-level problem, not something a per-trade formula can solve; see §26.

**Kelly is applied as a ceiling, not a licence.** `calculate_position_quantity` takes the *smaller* of the Kelly allocation and the fixed-fractional risk budget. The budget is priced off this trade's own stop distance, which is known exactly, while Kelly's \(l\) is a historical average across trades with different stops — where they disagree, the known quantity wins. It also means correcting the units could not land as a silent 17× increase in position size across the whole book. Enabling Kelly can therefore only size *down*: it de-risks as the estimated edge approaches zero and is otherwise inactive.

\(p\), \(g\) and \(l\) are estimated from realized trade history, gated by a 50-trade floor and with \(p\) shrunk toward 0.5 by a Beta prior (§12 of this section's rationale applies: over-betting off a noisy estimate costs far more long-run growth than under-betting). The magnitudes travel alongside the ratio deliberately — a caller handed only \(b = g/l\) cannot recover \(l\), which is exactly how the allocation lost its \(1/l\) factor. With too few realized trades, Kelly sizing falls back to the existing ATR/fixed-fractional sizing (`src/risk.py::calculate_quantity`).

**Both inputs are measured net of friction, on both paths.** Stored trade logs are deliberately gross — `return_pct` is the price move, which is what a report should show — and that is the wrong input here for two reasons that both push the same way:

- **\(b\) is overstated.** With ~0.8% of round-trip friction, a trade whose gross reward:risk is 2.0 realizes nearer 1.8. \(f^*\) is more sensitive to \(b\) than to anything except \(p\).
- **\(p\) is overstated.** A trade that gained 0.3% gross *lost* money. Counting it as a win adds a phantom win and simultaneously drags the average win magnitude down by less than the loss it actually was.

Both call sites therefore restate before estimating, and do it identically. The backtest engine divides `net_pnl` — already net of brokerage, STT, exchange and SEBI charges, GST, stamp duty and capital-gains tax — by the cost basis (`BacktestEngine._net_return_pct`). The live orchestrator restates its stored SQLite history through `src/risk.py::to_net_realized_trades()` as it loads it into `brain.trade_history`, which also keeps the weight learner and the sizer using one definition of a winning trade.

The quarter-Kelly cap exists on top of all that because a lower-circuit lock (§15) realizes far worse than the modelled stop, biasing the measured \(b\) upward in a way no estimator can see.

**Sources:**
- Kelly, J. L. (1956), "A New Interpretation of Information Rate", *Bell System Technical Journal*
- [Kelly Criterion and Position Sizing: From Formula to Quant Practice](https://coriva.eu.org/en/kelly-criterion-position-sizing/)
- [The Kelly Criterion: A retail trader's guide to position sizing](https://experts.deriv.com/insights/kelly-criterion-position-sizing)

---

## 5. Trend + Breakout + Volume + Monte Carlo

The platform's original strategy (`strategies/rule_based.py`, predating this document) — retained as the default. Component-weighted score combining: price above rising 50/200-day moving averages (trend-following, related to §6 below), a 20-day Donchian channel breakout (a classical technical/momentum-adjacent signal), a volume-confirmation filter, and Monte Carlo probability-of-profit. Component weights self-adapt based on realized win rate per trigger (`strategies/weighting.py`). This is closer to a practitioner heuristic than an academically-isolated factor, but it's been the platform's live/backtested baseline and gives a useful comparison point for the newer factor-based strategies in a UMA ensemble.

---

## 6. Time-series momentum

**Evidence.** Moskowitz, Ooi & Pedersen (2012), *"Time Series Momentum"* (*Journal of Financial Economics* 104(2), 228–250): a security's own past 1–12 month return sign predicts its own future return sign, across 58 futures markets, with persistence reversing only at longer horizons — distinct from cross-sectional momentum (§1), which ranks stocks *against each other* rather than against their own history.

**Status in this platform.** Not implemented as a standalone strategy — the existing rule-based strategy's trend component (price above rising SMA-50/SMA-200) is a simplified expression of the same idea (an absolute trend filter on the security's own price history, not a cross-sectional rank). A dedicated time-series-momentum strategy (e.g., sign of trailing 12-month return, volatility-scaled) is a natural, low-effort future addition once there's appetite for it, reusing the volatility model from §3 for the scaling.

**Sources:**
- [Time Series Momentum (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
- [Time Series Momentum (ScienceDirect / JFE)](https://www.sciencedirect.com/science/article/pii/S0304405X11002613)

---

## 7. Cointegration-based pairs trading

**Evidence.** Strong, India-specific support: an arXiv paper "Designing Efficient Pair-Trading Strategies Using Cointegration for the Indian Stock Market" and multiple QuantInsti EPAT projects (e.g., a 25-stock market-neutral pairs strategy across Banking/IT/Pharma/Cement/Auto) validate cointegration-based (not merely correlation-based) pairs trading on NSE large-caps. Standard approach: Engle-Granger two-step cointegration test to find pairs, trade the z-scored spread \(z_t = \frac{(P_A - \beta P_B) - \mu_{\text{spread}}}{\sigma_{\text{spread}}}\), entering when \(|z_t|\) exceeds a threshold (e.g. 2) and exiting at mean reversion.

**Implemented as the `pairs` strategy (T30).** This section previously read *"Why it isn't implemented yet — an architectural gap, not a research gap"*, and named two blockers. Both are now resolved, and one of them is resolved by *conceding* rather than solving. The original diagnosis is kept below because half of it was right and the other half was informative in being wrong.

**(a) The per-ticker interface — the diagnosis was in the wrong layer.** The gap was described as `BaseStrategy::score()` scoring one ticker at a time while a pair is a relationship between two. True, and not where the obstruction was: T24's cross-sectional feature registry receives the whole `(date × symbol)` panel, so a *feature* screens pairs internally and emits a per-symbol number — how cheap each name is against its own partner. The scoring interface never had to change. It was the feature layer that could not express the question.

**(b) The short leg is foregone, not solved.** Textbook pairs trading is market-neutral because it shorts the expensive leg against the cheap one. This platform does not short, so `pairs` buys the cheap leg and stops. That is a real cost rather than a technicality — the signal carries full market exposure, and **its results are not comparable with published market-neutral pairs returns.** The strategy reports `market_neutral: False` and carries that sentence in its own entry rules, so a report cannot use the words "pairs trading" and leave a reader assuming neutrality.

**Two failure modes make pairs backtests famously optimistic, and both are handled in `features/cointegration.py`.**

*Look-ahead through pair selection.* Screening for cointegration on the whole sample and then trading the pairs it found is severe and easy to commit without noticing: the pairs are chosen **because** their spread mean-reverted over the period being evaluated. `rolling_pair_scores` walks the panel forward — each screen sees only the formation sessions ending at its refresh point, and its pairs apply only to sessions after it. A test perturbs every price after a cut date and asserts every score before it is unchanged.

*Multiple testing.* Screening every pair of an \(n\)-name universe is \(\binom{n}{2}\) hypothesis tests — 190 for 20 names, 1,225 for 50. Measured on twenty independent random walks, where nothing is cointegrated:

| screen | pairs found | expected by chance |
| --- | --- | --- |
| uncorrected, p<0.05 | several | **9.5** |
| Bonferroni | ≤1 | **0.05** |

An uncorrected screen reliably finds cointegration in noise. The correction is on by default, and `expected_false_positives` is reported rather than merely applied, because it is the number that says whether a screen finding forty pairs found anything.

The spread's z-score uses the **formation** window's mean and standard deviation, not the trading window's. A z-score computed against its own window's mean is centred at zero by construction and can never say the spread is stretched — the one thing it exists to say.

```bash
portfolio-agent evaluate --strategy pairs --slippage-bps 25
```

**Sources:**
- [Designing Efficient Pair-Trading Strategies Using Cointegration for the Indian Stock Market (arXiv)](https://arxiv.org/abs/2211.07080)
- [Cointegrated Pairs Trading Strategy in Indian Equity Market (2015–2025), QuantInsti EPAT](https://blog.quantinsti.com/cointegrated-pairs-trading-indian-equity-market-epat-project/)
- [Mean-Reversion Statistical Arbitrage Pair Trading Across Sectors, QuantInsti EPAT](https://blog.quantinsti.com/epat-project-mean-reversion-statistical-arbitrage-pair-trading-strategy-indian-market-sectors/)

---

## 8. Fama-French multi-factor models

**Evidence.** Well-validated for India: multiple studies (2005–2015, 2016–2023, CNX 500 2000–2015) find the three- and five-factor models outperform CAPM, with the market factor dominant and a documented size effect; the five-factor extension (adding profitability RMW and investment CMA) improves on the three-factor version in several studies.

**The adapter is implemented; the data is supplied by the user (T31).** This section previously read *"Why it isn't implemented"* and listed what each factor needs: SMB market capitalization, HML book value, RMW operating profitability, CMA total asset growth — none of it in OHLCV. That diagnosis was correct, and `data_quality/fundamentals.py` plus `features/characteristics.py` now close it in the shape T15 established for membership: **schema, validation and tests now; data supplied later.**

| Factor | Characteristic | Registry name |
| --- | --- | --- |
| SMB (size) | market capitalization | `shares_outstanding` × `close` |
| HML (value) | book-to-price | `book_to_price` |
| RMW (profitability) | gross profitability | `gross_profitability` |
| CMA (investment) | asset growth | `asset_growth` |
| — | earnings yield | `earnings_to_price` |
| — | accruals (Sloan) | `accruals` |
| control | leverage | `leverage` |

**The report date is the whole problem.** Results for the quarter ending 31 March are published in late May, so a backtest using them from 31 March has read six to eight weeks into the future on every stock, every quarter, forever. It is invisible in the output — the numbers are real, the dates are real, and the strategy simply knew them early. **It does not look like a bug; it looks like alpha.** Every fact therefore carries both a `fiscal_date` and a `report_date`, and `as_of(D)` filters on the second. `report_date` is a required column rather than an optional one, because the only available fallback is the fiscal date.

Validation is calibrated to Indian filing law — SEBI LODR Regulation 33 allows 45 days for quarterly results and 60 for annual — so a lag under 15 days or over 120 is flagged, and a negative lag is an error rather than a warning: a row reporting before the period it describes is a column swap, and every value in the file is then suspect. The sharpest check is for a file whose lags are all *identical*, which means the report dates were reconstructed by adding a constant. That is worse than having no file: it has the shape of point-in-time data and none of the content.

Two sign conventions are worth stating because they would look plausible backwards. `asset_growth` is signed as **growth**, so CMA's conservative leg is the *bottom* of the sort; `accruals` is signed so that **high means more accrual**, so Sloan's leg is also the bottom. Non-positive book equity is dropped from `book_to_price` (Fama and French's own screen — a negative B/P sorts a distressed firm to the extreme growth end), while `earnings_to_price` **keeps** loss-makers, because a negative earnings yield sorts where a reader would expect and screening it would be a quality filter wearing a value label.

Acquisition, formats and the trap in `yfinance`'s quarterly statements are in [`docs/OBTAINING_DATA.md`](OBTAINING_DATA.md). Every run without a fundamentals file says in its notes that it controlled for no accounting characteristic.

**Sources:**
- [Empirical Applicability of the Fama-French Three-Factor Model in the Indian Equity Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5010405)
- [Fama-French five-factor asset pricing model: empirical evidence from Indian stock market](https://www.inderscience.com/info/inarticle.php?artid=111959)
- [An Empirical Evaluation of the Fama-French Five-Factor Model in the Indian Equity Market: Evidence from NSE-Listed Stocks](https://www.researchgate.net/publication/391388581_An_Empirical_Evaluation_of_the_Fama-French_Five-Factor_Model_in_the_Indian_Equity_Market_Evidence_from_NSE-Listed_Stocks)

---

## 9. Quality (QMJ) factor

**Evidence.** "Quality Minus Junk" (profitability + safety/low-leverage) shows the highest average monthly returns among factors tested in recent Indian research (1.69% vs. 1.33% for the market factor) with *lower* volatility than the momentum factor, and a significant four-factor alpha.

**The adapter is implemented; the data is supplied by the user (T31).** Same root cause as §8 and the same resolution: `gross_profitability` covers the profitability leg and `leverage` the safety leg, both registered as cross-sectional characteristics and both requiring only a fundamentals file with the relevant columns.

QMJ's third leg — earnings **growth** — is not implemented. It needs a multi-year history of the same field rather than one snapshot, so it is the one part of this section still genuinely blocked on data depth rather than on data existence.

`leverage` is registered as a **control** rather than as a signal, and the distinction matters here. Leverage mechanically raises equity beta, so a quality sort that is silently a low-leverage sort has found `bab` (§2) with extra steps. Having it in the registry means a run can neutralize against it instead of wondering.

**Sources:**
- [Machine learning-enhanced quality minus junk (QMJ) factor and stock returns: Evidence from the Indian equity market](https://www.sciencedirect.com/science/article/pii/S3050700625000416)
- [Performance of quality factor in Indian Equity Market (IIMA working paper)](https://www.iima.ac.in/sites/default/files/2022-12/Wp-2022-11-01.pdf)
- [Performance of Factor Strategies in India, QuantPedia](https://quantpedia.com/performance-of-factor-strategies-in-india/)

---

## 10. FII/DII institutional flows (not yet implemented)

**Evidence.** FII (foreign) and DII (domestic) daily net flows are widely documented as a major driver of Indian index-level returns and volatility — FIIs tend to dominate price momentum and are correlated with global risk appetite and rate cycles, while DII flows (mutual funds, insurers) provide offsetting structural support during FII selloffs; large-cap, high-foreign-ownership sectors (banks, IT) are most sensitive.

**Why it isn't implemented.** FII/DII net-flow data isn't in OHLCV and isn't currently ingested; NSE/SEBI publish daily provisional figures that would need a new scraper/ingestion path (similar scoping decision to §8/§9). A flow-based regime filter (e.g., dampen position sizing during sustained FII selling) is a reasonable future addition once that data source exists.

**Sources:**
- [Why Do FII and DII Investment Flows Significantly Impact Indian Stock Market Movements?](https://www.gwcindia.in/blog/why-do-fii-and-dii-investment-flows-significantly-impact-indian-stock-market-movements/)
- [An Empirical Study on the Impact of FII and DII on Volatility, Leverage and Long-Term Returns of the Indian Stock Index](https://doi.org/10.9734/ajeba/2025/v25i41739)
- [FII Flows & Indian Stock Market Momentum Explained](https://www.mnclgroup.com/how-fiis-drive-momentum-in-indian-stocks)

---

## 11. Calendar anomalies (deliberately not implemented)

**Evidence is weak and unstable.** Day-of-week and month-of-year effects are reported in Indian-market research, but findings are inconsistent across studies and time periods: one study finds a Wednesday effect, another finds Monday+Wednesday positive effects, another finds day-of-week effects *only* in BSE Smallcap (not large/mid-cap); the "month of the year" effect's location itself has drifted across three different sample periods (February in 1990–98, November in 1999–2006, April in 2007–15) in the same line of research. An effect that relocates every few years as soon as it's found and published is a textbook sign of a spurious/arbitraged-away pattern rather than a stable, tradeable anomaly — not worth a strategy slot.

**Sources:**
- [Calendar Anomalies in the Indian Stock Markets: Monsoon Effect](https://www.abacademies.org/articles/calendar-anomalies-in-the-indian-stock-markets-monsoon-effect-8015.html)
- [The Existence of Day of the Week Effect in Indian Stock Market](https://www.abacademies.org/articles/the-existence-of-day-of-the-week-effect-in-indian-stock-market-9827.html)
- [DAY OF THE WEEK EFFECT IN INDIAN STOCK MARKET WITH REFERENCE TO NSE NIFTY INDEX](https://www.researchgate.net/publication/315891601_DAY_OF_THE_WEEK_EFFECT_IN_INDIAN_STOCK_MARKET_WITH_REFERENCE_TO_NSE_NIFTY_INDEX)

---

## 12. Momentum crash protection

**The problem.** Momentum is the factor most prone to catastrophic, fat-tailed failure, and its crashes are not random draws from a fat tail — they cluster in an identifiable *panic state*: after a bear market, during the rebound, when the recent losers a momentum book is underweight rally hardest and market volatility is elevated. In Indian equities pure momentum has drawn down roughly 45–55% in 2011, 2018 and the COVID crash, and the Nifty 200 Momentum 30 index's worst drawdown (−70.25%) is materially deeper than the Nifty's (−55.12%). A long-only implementation with no volatility scaling and no regime awareness inherits all of that.

**Three fixes exist in the literature; two need only OHLCV and are implemented.**

**(a) Constant volatility scaling.** Scale exposure by the ratio of a target risk budget to realized volatility, so a position's risk contribution stays roughly constant instead of ballooning exactly when volatility spikes:

$$
w_i(t) = \min\left(w_{\max},\ \max\left(w_{\min},\ \frac{\sigma_{\text{target}}}{\sigma_i(t)}\right)\right)
$$

with \(\sigma_{\text{target}} = 20\%\) annualized by default. \(w_{\max} = 1\), so this can only ever *reduce* a position — the platform's existing position caps stay the binding constraint in calm markets, and no configuration turns this into leverage.

**(b) Dynamic scaling / regime filter.** Classify the market state and cut momentum exposure in the panic state:

| State | Condition | Exposure |
|---|---|---|
| `crash_risk` | market below its 200-day MA **and** realized vol > 1.5× target | `bear_exposure` (0 by default: no new momentum entries) |
| `elevated_vol` | exactly one of those holds | volatility-scaled, halved again if the trigger is the downtrend |
| `risk_on` | neither holds | volatility-scaled only |

Both conditions are required for the hard stand-down because the crash literature puts the danger in the *rebound out of* a bear market, not in the decline itself — a quiet downtrend is a warning, so it dampens rather than halts.

**A second, separate classification answers a different question.** The table above says *how much exposure*; it does not say *what kind of market this is*, and those come apart — a directionless chop and a calm uptrend can call for an identical exposure scalar while suiting completely different strategies. `regime.py::classify_regime()` therefore also labels the state, and the meta-orchestrator (§19) keys its model-permission map off that label:

| Classification | Condition |
|---|---|
| `BULL_RISK_ON` | index above its 200-day MA, realized vol below target |
| `BEAR_CRASH_RISK` | index below its 200-day MA, **or** realized vol above 1.5× target |
| `SIDEWAYS_CHOP` | index within 2% of its 200-day MA **and** ADX below 20 |
| `NEUTRAL` | above the MA, vol between target and the crash multiple |
| `UNKNOWN` | not enough history to judge |

Those definitions overlap as stated, so the order of evaluation is load-bearing rather than incidental:

1. **A volatility spike wins first.** Vol above 1.5× target is unambiguous panic wherever price sits, and misreading a crash as a chop would leave mean reversion buying into it.
2. **Chop is checked before the bear branch's trend clause**, because it is the more specific condition and because an index sitting 1% below its 200-day average in a calm market is a range, not a bear market. Reading "below the MA" literally there would mute the trend sleeves during exactly the drift they handle fine.
3. Then the trend clauses. What is left — above the average but jittery — is named `NEUTRAL` rather than forced into one of the three, since the strategies suited to a calm uptrend are not the ones suited to a nervous one.

`SIDEWAYS_CHOP` is the only branch needing ADX, and ADX is the only statistic here needing the daily high/low range. A benchmark cached as closes alone falls back to a close-only directional index (`src/indicators.py::calculate_adx`), which is well defined — true range collapses to \(|\Delta \text{close}|\) — but coarser, and reads somewhat higher than the OHLC version on the same market.

**The market series.** When `data.benchmark_symbol` (default `^NSEI`, the Nifty 50) is cached, the trend and volatility tests key off the real index — "the market is below its 200-day average" is a statement about the index the research studied. Without one, `src/regime.py::build_market_proxy()` falls back to an equal-weighted composite of the traded universe, which needs no extra data but is only a proxy: it reflects whatever is in today's universe, and idiosyncratic noise diversifies out of it in a way real index volatility does not.

That fallback is not a rare edge case, and treating it as one was a live defect. An index only reaches the cache if someone downloads one, so on a default install the classification was `UNKNOWN`, every model was permitted in every regime, and the gating quietly did nothing — with nothing in the logs to say so. The backtest engine and the live orchestrator now both derive the composite when no index is cached, so the map works out of the box. On the repository's own 120-ticker cache it produces a regime series that tracks the market: `BEAR_CRASH_RISK` through the 2022 small-cap drawdown, `BULL_RISK_ON` across 2023, `NEUTRAL` from mid-2024.

The composite averages **daily returns** and cumulates them, rather than averaging rebased price levels. Real universes have ragged start dates and the eligible set changes daily during a backtest; averaging rebased levels makes a newly-listed ticker join at its own base of 1.0 while incumbents sit at 1.4, printing a double-digit synthetic drop on a day every constituent rose — and the filter would read that construction artifact as a trend break and as realized volatility. A return average has no such discontinuity.

Nifty VIX would be a better volatility gauge than realized volatility, but it is not in the OHLCV cache; see §10 for the flow-data equivalent.

**(c) Residual momentum** — ranking on residual rather than total returns — is the third documented fix, and it is **implemented** as the `residual_momentum` strategy (T27).

This entry previously read *"it needs a factor model to residualize against, which needs §8's data. Not implemented."* That was wrong on the premise. A CAPM residual needs a market return and a rolling beta, and T14 built both; §8's multi-factor data would let the residual be taken against Fama-French rather than the market alone, which is a refinement rather than a prerequisite. The claim is corrected here rather than quietly deleted, because it is the kind of premise that stops work from being attempted.

The ranking statistic is the residual's **information ratio** — mean residual over the formation window divided by its own dispersion — not the raw cumulated residual. Blitz, Huij & Martens attribute roughly double the risk-adjusted profit of price momentum, with materially shallower drawdowns, to that standardization: ranking on raw cumulated residuals still puts high-residual-volatility names on top, reintroducing exactly the risk exposure residualizing was meant to remove.

One implementation detail is a genuine trap. If beta is fitted **with an intercept** over exactly the window the residuals are then cumulated over, the cumulative residual is *identically zero* — OLS residuals sum to zero by construction. BHM avoid it by estimating over 36 months and forming over 12. This platform estimates beta on a rolling year ending at each date and applies it without an intercept, so the residual carries the alpha rather than having it differenced away.

Comparable against the incumbent in one command, since everything except the formation measure is shared:

```bash
portfolio-agent compare --strategies momentum,residual_momentum --neutralize beta,size
```

**Fail-neutral by design.** With too little history to judge the regime, exposure is left unscaled rather than blocked: there is no evidence of a panic state, and inventing one would silently disable the strategy on a short cache rather than protect it.

**Sources:**
- Daniel & Moskowitz, ["Momentum Crashes"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16301490), *Journal of Financial Economics* 122(2), 2016
- Barroso & Santa-Clara, ["Momentum has its moments"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002323), *Journal of Financial Economics* 116(1), 2015
- Blitz, Huij & Martens, ["Residual momentum"](https://www.sciencedirect.com/science/article/abs/pii/S0927539811000247), *Journal of Empirical Finance* 18(3), 2011
- Harvey et al., ["The Impact of Volatility Targeting"](https://www.man.com/insights/the-impact-of-volatility-targeting), *Journal of Portfolio Management*, 2018

---

## 13. Transaction costs and concentration limits

### 13.1 The Indian cost stack

`src/execution_sim.py` charges the full delivery (CNC) schedule on every simulated fill:

| Component | Rate | Legs |
|---|---|---|
| Brokerage | min(₹20, 0.03% of turnover) | both |
| STT | 0.1% of turnover | both |
| Exchange transaction charges | 0.00345% | both |
| SEBI turnover fees | 0.0001% (₹10/crore) | both |
| GST | 18% of (brokerage + exchange + SEBI) | both |
| Stamp duty | 0.015% | buy only |
| STCG / LTCG | 20% / 12.5% above the ₹1.25L exemption | on realized gains |

STT is 0.1% on *both* legs, not 0.025%: the 0.025% figure is the intraday sell-side rate, and a platform that holds overnight never qualifies for it.

### 13.2 Costs must reach the *signal*, not just the fill

Charging friction at fill time makes the equity curve honest but leaves the decision unchanged — the strategy still selects trades as if they were free. A momentum book rebalancing monthly into a top-decile portfolio gives up roughly 0.5–1.5% per round trip, which is the same order as the premium being harvested.

So reward:risk is now reported **net of estimated round-trip friction** (`src/risk.py::net_reward_risk`):

$$
\text{RR}_{\text{net}} = \frac{P_{\text{target}}(1-c_{\text{sell}}) - P_{\text{entry}}(1+c_{\text{buy}})}{P_{\text{entry}}(1+c_{\text{buy}}) - P_{\text{stop}}(1-c_{\text{sell}})}
$$

with \(c\) from `cost_fraction_per_side()` — the statutory rates above plus an assumed 25 bps/side of slippage, ~0.8% per round trip in total. Brokerage enters at its percentage rate rather than min(₹20, 0.03%), because the flat cap only ever *lowers* the effective rate, making the percentage the conservative upper bound.

**A consequence worth stating plainly, and what was done about it.** The old rule-based exit rules (1.5× ATR stop, 2.0× ATR target) give a *gross* reward:risk of 1.33 regardless of ATR. Charging ~0.79% of round-trip friction against an ATR-scale move takes that to 0.72–1.11 depending on how large ATR is relative to price — so against `compliance.min_reward_risk: 1.5` the flagship strategy could not emit a single BUY, and could not before costing either (1.33 < 1.5 gross).

Rather than document a config that cannot fire, the geometry was widened to match the economics: the take-profit multiple is now **3.0× ATR** (2.0 gross, ~1.2–1.7 net across realistic ATR levels) and `min_reward_risk` is **1.2**, a threshold good setups clear and marginal ones fail. Signals carry both `reward_risk` (net) and `extra["gross_reward_risk"]`, so the size of the friction haircut stays visible instead of implicit. `src/compliance.py::check_risk_reward_ratio` measures the same net quantity, so the strategy gate and the compliance gate cannot disagree about what a trade's reward:risk is.

### 13.3 Sector concentration

Ranking on one characteristic and buying the extreme decile has no term in the objective that cares what those stocks *are*. In Indian equities that reliably produces a portfolio which is nominally 10 names and economically one bet — momentum concentrated in IT through 2020-21, then Banking/PSU through 2022-23. The factor exposure is intended; the sector exposure is an accident, and it is what turns a factor drawdown into a portfolio drawdown.

`src/sectors.py` caps any one sector at `risk.max_sector_pct` (25% by default), enforced at *order-creation* time in both the backtest engine and the live orchestrator, and accounting for orders queued in the same round — five BUYs in one sector each fit under the cap individually and blow straight through it together.

The map comes from a `ticker,sector` CSV at `paths.sector_map_csv`, and unmapped tickers need their own treatment, because both obvious answers are wrong:

- **Capping a pooled `UNKNOWN` bucket at `max_sector_pct`** looks conservative and is a different constraint entirely. With no map at all every holding is UNKNOWN, so a 25% *sector* limit silently becomes a 25% limit on total invested capital and leaves three quarters of the portfolio in cash forever.
- **Exempting `UNKNOWN` outright** opens a bypass. Indian sector maps are chronically incomplete in exactly the small- and micro-cap segment where concentration risk is worst; a 500-name universe with 200 unmapped tickers could put the entire book into unmapped micro-caps and satisfy every cap.

So the two cases are separated. With **no map at all** the cap is unenforceable: it is reported inactive and both engines log a WARNING, rather than freezing the portfolio. With a **partial map**, each mapped sector gets `max_sector_pct` and the whole unmapped pool gets its own wider budget, `risk.max_unknown_sector_pct` (30%) — finite, so an incomplete map is not a route around the limit. Unmapped exposure is never charged against a mapped sector's allowance either way.

### 13.4 Drawdown circuit breaker

`risk.max_portfolio_drawdown_pct` (15%) halts *new* entries once peak-to-trough drawdown reaches it; buying resumes at `drawdown_reentry_pct` (10%). Two thresholds rather than one, so the breaker cannot flicker on and off as equity wobbles across a single line — and a config validator rejects `drawdown_reentry_pct >= max_portfolio_drawdown_pct`, since that setting re-creates the single-threshold flicker the pair exists to prevent. Open positions keep their stops and targets by default: force-liquidating a whole book at a drawdown trough is how a bad quarter becomes a permanent loss. `risk.liquidate_on_drawdown_halt` turns that into a full liquidation for mandates where a hard equity floor outranks recovery potential.

**Recovery alone is not a sufficient re-arming condition**, and assuming it was produced a silent deadlock. Halting buys does not freeze the book — open positions keep exiting through their own stops and targets until nothing is left but cash, and cash does not appreciate. Equity is then pinned at its trough, permanently below a re-entry threshold measured against a peak it can no longer approach. In a five-year backtest the breaker tripped in month seven and the strategy sat in cash for the remaining four years, reporting a 15.04% maximum drawdown that actually meant "stopped trading". `risk.drawdown_halt_max_days` (60 trading days) re-arms regardless after a cooldown and resets the equity peak to current equity — resetting matters as much as re-arming, since leaving the old peak in place puts the very next bar back over the trip threshold.

### 13.5 Exit triggers

Stops and targets assume a fill is available near them. Two conditions invalidate that assumption rather than merely arguing against holding, so they force an exit outside the normal rules:

- **Locked at the lower circuit** (`risk.exit_on_lower_circuit_lock`, on by default). On a lock there is no bid, so the modelled stop is not a stop — it is a hope. Every further locked session realizes a loss the sizing never priced, which is the same asymmetry that biases the \(b\) Kelly reads (§4). The exit is queued for the next session, the earliest a real order could work.
- **The drawdown breaker tripping**, when `liquidate_on_drawdown_halt` is set.

### 13.6 The exit plan has to reach the fill

A defect worth recording because of how quietly it corrupted everything upstream: the backtest engine used to overwrite every filled position's stop and target with a hardcoded 5% / 10% pair, discarding whatever the strategy computed. It was found by an experiment that should have changed the results and did not — widening `atr_stop_multiplier` from 1.5 to 6.0 produced a byte-identical backtest.

The damage was not confined to the exit. `compliance.min_reward_risk` screened signals on a net-of-cost reward:risk derived from ATR levels the engine then ignored, so the platform was gating on one exit plan and trading another; the quantile model's distribution-derived levels (§20) went the same way; and Kelly's \(b\) was estimated from trades exited under the 5/10 rule while the signals feeding it were screened under the ATR rule. On Indian small- and mid-caps a flat 5% stop is roughly one session of noise, which is why backtests showed a 2-day median holding period with 82% of exits at the stop.

Fills now carry the signal's own levels, as *distances* re-applied to the price actually paid — the signal's levels are computed off T−1's close while the fill lands at T+1's open plus slippage, so copying absolute levels across would put a gapped-up entry immediately through its own target.

**The measured consequence, and the configuration lesson.** With the exit plan actually reaching the fill, the same cross-sectional momentum signal over the same universe and window behaves completely differently depending on the stop width:

| `atr_stop_multiplier` | Round trips | Median hold | Stop-loss exits | Gross return on deployed |
|---|---|---|---|---|
| 1.5 | 590 | 5 days | 71% | **−2.02%** |
| 6.0 | 113 | **94 days** | 31% | **+4.44%** |

Momentum forms on a 9-month window (§1). At 1.5× ATR it was exiting in a working week — a multi-month factor traded on a two-day exit rule, paying ~0.8% of round-trip friction each time. The signal has edge; the stop was cutting the thesis short rather than protecting it. **Match the exit horizon to the signal's formation horizon**; the default stays 1.5 because it suits the per-ticker trend/breakout strategy, not because it suits everything.

---

## 14. Fat tails in the forward simulation

The original Monte Carlo drew i.i.d. Gaussian shocks around a historical mean and standard deviation. That model assumes stationary parameters, thin tails and independent days — all three fail in an emerging market with circuit limits, retail participation spikes and policy shocks, and the failure is one-directional: it *understates* tail risk, which is exactly the quantity `compliance.target_prob_profit` gates on.

`simulation.method` now selects the shock process:

- **`block_bootstrap`** (default) — the stationary bootstrap of Politis & Romano. At each simulated day, either continue the current block (probability \(1 - 1/L\)) or jump to a fresh random start (probability \(1/L\)), giving geometric block lengths with mean \(L\). Contiguous days travel together, so both the empirical fat tails and the volatility clustering survive resampling; drawing days independently would destroy exactly the serial dependence that makes a drawdown a drawdown. Geometric rather than fixed block lengths avoid imprinting an artificial periodicity on the paths.
- **`jump_diffusion`** — Merton: Gaussian diffusion plus a compound-Poisson jump component, \(N \sim \text{Poisson}(\lambda/252)\) jumps per day of size \(\sim \mathcal{N}(\mu_J, \sigma_J^2)\) with \(\mu_J < 0\). Diffusion cannot produce a gap; jumps can. The drift is jump-compensated, so adding jump risk widens the tails without also quietly shifting every expected return downward.
- **`gaussian`** — the original, retained for comparison.

All three compose with the GARCH volatility path (§3, §16): GARCH sets how large tomorrow's shocks should be, `method` sets what shape they take.

**The innovation distribution has to travel with the volatility path.** GJR-GARCH is fitted with `dist="t"` precisely because Indian returns are fat-tailed — and the fitted degrees of freedom \(\nu\) are the estimate that fitting produces. Dropping \(\nu\) and simulating Gaussian shocks off a fat-tailed fit throws that estimate away and leaves VaR and CVaR optimistic in exactly the 5% tail `compliance.target_prob_profit` reads. `GarchForecast.distribution_df` now carries \(\nu\) through to the simulation, where shocks are drawn as standardized Student-t:

$$
z = \frac{t_\nu}{\sqrt{\nu / (\nu - 2)}}
$$

The rescaling matters: \(t_\nu\) has variance \(\nu/(\nu-2)\), so an unrescaled draw would inflate every path's volatility *as well as* widening its tails, and the two effects would be impossible to separate. A fit landing at \(\nu \le 2\) has infinite variance and is discarded in favour of Gaussian draws. This applies to the `gaussian` method and to `jump_diffusion`'s diffusion leg; `block_bootstrap` inherits the empirical tail shape by construction and ignores it. A method whose preconditions do not hold — a block bootstrap over fewer than 60 observations just re-prints the same few days — degrades to the Gaussian path rather than failing the ticker, since a missing MC result is read downstream as zero probability-of-profit.

**Sources:**
- Politis & Romano, ["The Stationary Bootstrap"](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870), *JASA* 89(428), 1994
- Merton, ["Option pricing when underlying stock returns are discontinuous"](https://www.sciencedirect.com/science/article/abs/pii/0304405X76900222), *Journal of Financial Economics* 3(1-2), 1976

---

### 14.1 The drift is estimated in log space, so it must not be converted again

**The defect.** The simulator estimated drift as

```python
log_returns = np.log1p(returns_arr)
sample_mu   = np.mean(log_returns)      # already a LOG-space drift
```

and then applied, in all three shock branches,

```python
daily_drift_path = mu - 0.5 * sigma_path ** 2
```

For a lognormal return, \(\log(1+r) \sim \text{Normal}(\mu_a - \tfrac12\sigma^2,\ \sigma^2)\). Since `mu` is estimated *directly from log returns*, it already **is** \(\mu_a - \tfrac12\sigma^2\). Subtracting \(\tfrac12\sigma^2\) a second time drives the simulated log drift to \(\mu_a - \sigma^2\).

**Why it was worse than a level bias.** The error is \(\tfrac12\sigma^2 H\) — proportional to *variance*. It is therefore not a constant haircut that washes out of a cross-sectional ranking; it is a penalty graded by volatility, applied hardest to exactly the names it damages most. Two consequences:

1. **It corrupted a hard gate.** `RuleBasedStrategy` tests `prob_profit >= compliance.target_prob_profit`. Signals were being rejected by an arithmetic error, and high-volatility names disproportionately.
2. **It was a second, undocumented low-volatility tilt** applied to every strategy consuming `mc_result` — in a platform that already ships an explicit `LowVolatilityStrategy`. Two low-vol tilts, one of them invisible and unintended.

**The sharpest test.** A driftless log random walk is a coin flip over any horizon, whatever its volatility — the median terminal price is the starting price. Under the double correction P(profit) fell to \(\Phi(-\tfrac12\sigma\sqrt{H})\): below a half, and further below the more volatile the name. That is now asserted directly at three volatilities.

A companion property, easy to mistake for a contradiction: the same driftless walk has a *positive* expected return, \(e^{H\sigma^2/2} - 1\), because the mean of a lognormal sits above its median. The double correction cancelled that convexity exactly, collapsing the expected return to zero — which is part of why the bug survived: the number looked reasonable.

**What was kept.** The jump compensator, \(-\lambda_{\text{daily}}\mu_J\), is a *different* correction and is correct. It offsets a shift that the jump shocks themselves introduce, so that adding jump risk widens the tails without also moving the mean. Removing it alongside the Itô term would have traded one drift bug for another.

**Measured impact.** On 212 cached NSE names under the default configuration:

| | Median P(profit) | Share clearing 0.55 |
|---|---|---|
| Before | 0.4660 | 0.5% |
| After | 0.4950 | 3.3% |

A median of 0.466 was the bug's signature: for a universe of real equities that on average drift upward, the whole distribution sat below a coin flip.

**On recalibrating the gate.** Removing a downward bias raises every probability, so `compliance.target_prob_profit = 0.55` was tuned against biased numbers and the question of re-tuning it is real. It was left at 0.55 deliberately, for two measured reasons. First, the gate exists to reject noise, and its noise rejection is *unchanged*: on a zero-drift universe under the default empirical prior, 0.0% clear it both before and after. Second, re-tuning the threshold to restore the old pass rate would preserve the effect of an arithmetic error after removing its cause — encoding the bug into the config. 0.55 now means what it says; before, the bias made it silently demand about 0.58.

For anyone who needs the old selectivity for continuity, the lever is explicit rather than implicit: `target_prob_profit: 0.60` reproduces the pre-fix pass rate (0.5%) on the same universe.

**Sources:**
- Itô's lemma as applied to geometric Brownian motion — the \(-\tfrac12\sigma^2\) term is the drift adjustment for the *arithmetic-to-log* conversion, and applies only in that direction

---

## 15. Circuit limits and the illiquidity illusion

Every ranking formula in this platform assumes continuous price discovery and that a printed close is a price you could have transacted at. On the NSE/BSE mid- and small-cap segments neither holds, in two specific ways that are both detectable from OHLCV alone.

**Circuit locks.** Stocks lock at statutory price bands. An operator-driven pump locks a stock in the upper circuit for consecutive sessions: \(P(t)/P(t-J)\) registers a huge formation return and momentum screams BUY, while there is no offer to lift. On the way back down the stock locks in the lower circuit and the position cannot be exited at all, so the realized loss blows through the modelled stop — which in turn inflates the payoff ratio \(b\) that Kelly sizes off (§4). §13.5's exit trigger addresses the down-lock directly.

The ladder is **1/2/5/10/20%**, matched within a tick-rounding tolerance rather than screened by a single "at least 5%" floor. The 1% and 2% bands are the ones exchanges impose ad hoc through the ASM and GSM surveillance frameworks on volatile, news-driven or operator-suspected scrips — that is, on precisely the names most likely to be operator-driven, which makes them the bands that matter most. A floor set at 5% waves them straight through: a small-cap pinned at a 1% upper circuit prints +1% on zero volume, and momentum reads it as strength for a stock nobody can buy.

The matching tolerance narrows with the band. A flat 40 bp window is proportionate around 5% and absurd around 1%, where it would accept anything from 0.6% to 1.4% — half the ordinary quiet days on the tape. Tick rounding on a circuit price is worth a few basis points, not tens. Moves beyond the widest band still count, since nothing legitimate moves 20% with no intraday range.

**Two lock signatures, not one.** The original detector required a **zero intraday range** (high == low) alongside the band-sized move. Both conditions are needed for *that* signature — a zero range alone is an untraded day, a band-sized move alone is an ordinary volatile session — but it only describes the stock that gapped straight to its limit and never traded off it.

The more common footprint is weaker and was being missed entirely: a stock walked up through the session that locks late, closing pinned at the limit (**high == close**) after printing a perfectly real intraday range. `operator_trap_days()` detects that separately. Both are untradeable *at the close*, which is the price every signal in this platform is generated from. The condition is asymmetric on purpose — a lower-circuit close (`low == close`) is a different problem, an exit that cannot be taken rather than an entry that cannot be filled, and is reported by `lower_circuit_locked_days()`.

**The illiquidity illusion.** Low realized volatility is supposed to proxy for a stable business. In India it frequently proxies for *nothing trading*: an illiquid stock carries yesterday's close forward, printing \(r = 0\) rather than a small return, which mechanically suppresses \(\sigma_i\) and walks the stock into the low-volatility buy decile. The anomaly the strategy is trying to harvest is not the one it ends up holding — these "quiet" names carry exactly the governance risk that ends in a forensic audit or a SEBI suspension.

`src/liquidity.py` screens on four trailing-60-session statistics — median rupee turnover, the share of unchanged closes, the share of circuit-locked sessions and the share of upper-circuit closes — plus two "on the decision date" checks for whether an order could be filled at all. Screened names are **removed from the ranking**, not merely marked un-buyable: leaving an untradeable stock in the cross-section would still shift every other name's percentile. Statistics that cannot be computed (short cache) pass, since the screen's job is to exclude on positive evidence of untradeability, not on absence of evidence.

**A related bias this does not fix.** \(p\) and \(b\) estimated only from currently-listed stocks ignore suspensions and delistings, inflating the apparent win rate. The backtest engine does force-liquidate delisted holdings and books the loss, so a *backtest's* Kelly inputs see them; a universe list built from today's constituents still cannot.

---

## 16. Overnight gaps and GARCH stationarity

GJR-GARCH (§3) is a model of a continuous trading process, and close-to-close returns are not one — they bundle two mechanisms with different dynamics:

- the **overnight gap**, \(\text{open}_t / \text{close}_{t-1} - 1\), which reprices global cues, FII decisions and policy news while the market is shut, arriving as a single instantaneous jump with no within-gap conditional variance to track;
- the **intraday session**, \(\text{close}_t / \text{open}_t - 1\), which is the continuous trading the recursion actually describes.

Feeding close-to-close returns to the recursion makes it attribute every gap to yesterday's session shock. That inflates \(\alpha\) and \(\gamma\), drags the persistence estimate \(\alpha + \gamma/2 + \beta\) toward (and sometimes past) the stationarity bound, and makes the fitted parameters unstable across refits — a pronounced problem for NSE, which opens after both the US close and the Asian session.

`forecast_volatility_gap_aware()` fits GJR-GARCH to the session leg only and adds gap risk as a separate component:

$$
\sigma_{\text{daily},t}^2 = \sigma_{\text{intraday},t}^2 + \sigma_{\text{gap}}^2
$$

with \(\sigma_{\text{gap}}\) the unconditional standard deviation of the gap series. Independence is the standard simplification and a conservative one here: positive correlation between a gap and the session following it would only widen the total. Enabled by `simulation.separate_overnight_gaps`; each fallback is independent, so a failed gap-aware fit drops to close-to-close GARCH rather than all the way to constant volatility.

---

## 17. The long-only constraint

Momentum and the low-volatility anomaly are defined in the literature as **long-short** portfolios. Shorting the loser decile is not incidental — it is what makes the factor market-neutral (\(\beta \approx 0\)) and what isolates the anomaly from the market return. This platform never shorts, so its momentum sleeve carries \(\beta \approx 1\) and will draw down with the Nifty in any broad selloff, whereas the academic long-short construction would earn on the short leg.

This is an accepted limitation, not an oversight, for two reasons. First, India's Securities Lending and Borrowing mechanism is thin outside large caps, so the short leg of a mid-cap momentum book is not reliably borrowable at any price — the academic construction is not merely disallowed here but largely unimplementable. Second, a paper-trading decision-support tool that quietly modelled unborrowable shorts would report returns nobody could have earned.

What the platform does instead is manage the resulting beta directly rather than hedge it away: the §12 regime filter cuts exposure when the market is in the state where a long-only factor book suffers most, and §13.4's circuit breaker caps how far a systemic drawdown can run before new risk stops being added. Index-futures hedging (a Nifty short against the long book) is the natural way to close the remaining gap and would need a futures data path plus an explicit exception to the no-shorting guardrail — a scoping decision, flagged here rather than half-built.

---

## 18. Making the walk-forward measurement honest

Section 12–17's controls are only as good as the evidence that they help, and the LSTM's walk-forward validation (`agents/trainer.py`) is the only place this platform produces a deploy/don't-deploy number. Three properties are what make that number mean anything, and each was a live defect at some point in this work:

**The target must be a forecast.** `config.training.target` defaults to `return_5d`, which is *also* a registered feature (the trailing 5-day return). The original "create the target if it doesn't exist" branch therefore never fired, and the model was trained to reproduce a quantity already determined by the price history it was being shown. Training loss looked excellent and forecast nothing. The target is now always recomputed as a forward return under a namespaced column (`target_return_5d`), so a same-named feature can never silently become the label.

**Folds must split by date, not by row.** The stacked training panel is ordered *[every ticker's train block][every val block][every test block]*, an ordering chosen so `create_dataloaders()`'s single 70/15/15 index split represents every ticker. A row-index split of that same panel is not chronological: fold 1's "future" test rows and its training rows cover the *same calendar dates* for different tickers. Walk-forward therefore works from per-ticker frames (`load_panel_by_ticker()`) and splits each by date, concatenating the blocks afterwards — which keeps each ticker's rows contiguous, as the sequence windows require, while guaranteeing no training row is dated at or after its fold's test period.

**Forward-looking labels need an embargo.** With a 5-day forward target, the last five training rows before a boundary carry labels computed from prices *inside* the test period. Those rows are dropped, or the model is fitted against labels encoding the moves it is about to be scored on.

**Normalization must be causal.** Fitting a scaler over the whole panel before splitting is the classic route to a 2.0 walk-forward Sharpe that dies live, because every training row then knows the future's volatility and price level. `features/pipeline.py::_normalize_features` uses `shift(1)` plus a rolling window, so each row's z-score is computed from strictly prior observations — appending future bars does not change a single earlier value, which is what `tests/test_features.py::TestNormalizationIsCausal` pins. (`features.normalize` is also `false` in the shipped config, so the default path does not normalize at all.)

Reported metrics are directional accuracy plus the annualized Sharpe of a long-only signal-following rule, always against an always-long benchmark on the identical sample. A model that cannot beat always-long is adding turnover, not alpha, and the trainer says so in as many words. The held-out test evaluation reloads the best checkpoint first, so the numbers describe the weights `MLStrategy` will actually load rather than whatever the last epoch of overfitting produced.

---

## 19. Combining models: arbitration, not averaging

Every section above produces *one* opinion. Combining several is its own problem, and the obvious answer is wrong in a specific and expensive way.

**The failure.** A weighted blend maps each member's signal to a strength and takes a weighted mean. Give it a momentum model at BUY with 0.90 conviction and a mean-reversion model at SELL with 0.85, and it reports a mild BUY. But those two models do not disagree *mildly*. They disagree maximally, which is the strongest available evidence that nobody knows what this stock is about to do — and it is precisely the setup that produces whipsaw: entered on a blended signal neither member would have taken alone, then stopped out by whichever of them was right.

Averaging is the correct operation for **estimates of the same quantity**. It is the wrong operation for **votes on a decision**. `src/trigger_engine.py` treats them as votes.

**A common contract.** `StrategySignal` is a trade plan, and its 0-100 `score` means a momentum percentile in one strategy and a forecast probability in another — averaging those is meaningless before it is dangerous. Each member is first flattened into a `ModelVerdict`: a direction, a conviction on a comparable 0-1 scale, an expected value, and two admissibility flags. Conviction maps from the native score (a model emitting SELL at score 15 is 85% convinced, not 15%), and expected value is derived in "R units" from the reward:risk the signal already carries:

$$
\text{EV}_R = p\,b - (1-p), \qquad \text{EV}_\% = \text{EV}_R \cdot \frac{P_{\text{entry}} - P_{\text{stop}}}{P_{\text{entry}}} \cdot 100
$$

Deriving it from \(b\) rather than re-costing target and stop is deliberate: \(b\) is already net of the §13 friction stack, and charging it twice would double-count. A model with no probability estimate reports `None`, not zero — otherwise the expected-value hurdle would bite hardest on the models most honest about their uncertainty.

**The conflict penalty.** Buy-side conviction is discounted multiplicatively by the strongest opposing conviction:

$$
c_{\text{eff}} = c_{\text{buy}} \cdot \big(1 - \max_j c_{\text{opposing},j}\big)
$$

The 0.90-against-0.85 case leaves 0.135 of usable conviction, not a positive average. An opposing conviction above `conflict_veto_confidence` blocks outright, so the decision's stated reason is "models disagree" rather than a confidence number that happened to land low. An `AVOID` verdict is an *abstention* and carries no opposing weight — a model declining to call a name is not evidence the name is bad.

**Vetoes before arithmetic.** A veto is not a low score to be outweighed. Untradeable instruments (§15), models the regime does not permit (§12), and trades whose expected value misses the hurdle are blocked outright, in that order, before any confidence is combined.

**Firing rules and size.** A trade needs either one genuinely strong model (`strong_single`) or several independent models agreeing (`consensus`) — "several models each mildly positive" is evidence, "one model mildly positive" is noise. What survives is a **position-size multiplier** rather than a boolean: a trade that only just clears its threshold is a trade the evidence only just supports, so it is taken at half size and scales to full size at full conviction.

**Regime-incompatible models are muted, not vetoing.** A veto reading means any single out-of-season sleeve stands the entire book down, which makes the §12 regime map useless — one sleeve being out of season must not stop the sleeve that is in it. `regime_policy: veto` restores the strict reading for callers who want it.

---

## 20. Forecasting a distribution instead of a point

**Why squared error fails here.** MSE is minimized by the conditional mean, and the conditional mean of a 5-day equity return is very close to a constant. A network trained on it converges to a near-constant output that scores excellently on the loss curve and forecasts nothing — the mean-reversion trap, and the most common way a financial deep-learning result is fooled. It also produces a bare point estimate, which §19's expected-value calculation cannot use without inventing a distribution around it.

**Pinball loss.** For quantile \(q\), prediction \(\hat y\) and realized \(y\):

$$
L_q(y, \hat y) = \max\big(q\,(y - \hat y),\ (q-1)(y - \hat y)\big)
$$

The penalty is asymmetric, so the minimizer of \(L_{0.9}\) is the true 90th percentile rather than the mean. Fitting \(q \in \{0.1, 0.5, 0.9\}\) jointly means a constant answer cannot satisfy all three at once, and the outer pair is a confidence interval that falls out of the fit rather than being bolted on afterwards — `MLStrategy` derives its stop and target from those percentiles, so a name the model reads as wide gets a wide stop. Quantile crossing is repaired by sorting at inference, which is exact and free, rather than penalized during training, where the penalty would distort the quantiles it was protecting.

**Architecture.** A vanilla LSTM compresses a 60-day multi-feature window into a single hidden vector and predicts from the final timestep, so everything the sequence contained must survive one bottleneck. **PatchTST** (Nie et al.) instead cuts the window into 5-day patches and attends over them: a single day's return is nearly pure noise, a week of them carries a shape worth attending to, and attention costs \(12^2\) rather than \(60^2\). Encoding is **channel-independent** — every feature passes through the same encoder weights separately — because mixing channels inside attention lets the model fit spurious cross-feature relationships that a few years of daily bars cannot support. Per-window instance normalization is what lets one set of weights serve a ₹30 small-cap and a ₹3,000 large-cap.

**Calibration.** A raw network score is not a probability, and networks on noisy financial data are systematically overconfident — the score at which the model says 80% is typically won far less than 80% of the time. That matters more here than in most settings, because the number feeds Kelly (§4) and the §19 expected-value hurdle, both far more sensitive to an optimistic \(p\) than a pessimistic one.

`src/calibration.py` fits an **isotonic** (monotone) map from score to realized win rate by the Pool-Adjacent-Violators Algorithm:

$$
\min_f \sum_i w_i\,(y_i - f(x_i))^2 \quad \text{subject to } f \text{ non-decreasing}
$$

Monotonicity is the whole trick: it preserves the model's *ranking*, which is where the alpha is and which walk-forward actually measured, while discarding its *scale*, which nothing measured. The fit runs on the **walk-forward test folds** — the only genuinely out-of-sample scores a training run produces. Fitting on training predictions would measure memorization and hand back a map that makes an overfitted model look perfectly calibrated. Training reports the expected calibration error before and after, so the correction is auditable rather than a black box.

**Sources:**
- Nie, Nguyen, Sinthong & Kalagnanam, ["A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"](https://arxiv.org/abs/2211.14730), ICLR 2023 (PatchTST)
- Koenker & Bassett, "Regression Quantiles", *Econometrica* 46(1), 1978 (the pinball loss)
- Zadrozny & Elkan, ["Transforming Classifier Scores into Accurate Multiclass Probability Estimates"](https://dl.acm.org/doi/10.1145/775047.775151), KDD 2002 (isotonic calibration)
- Guo, Pleiss, Sun & Weinberger, ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), ICML 2017

---

## 21. The drift is the noisiest input in the simulation

**The problem.** The forward Monte Carlo estimated each ticker's drift as the in-sample mean log return over its whole history and propagated it forward as if it were known. The sample mean of daily returns is the noisiest statistic in finance: its standard error is \(\sigma/\sqrt{T}\), which for a 2%/day Indian mid-cap over five years is about 0.057%/day — roughly **14% a year**. The 20-day probability of profit is essentially \(\Phi(\hat\mu\sqrt{H}/\sigma)\), so it inherits that noise directly.

Measured: simulating tickers whose true drift is *exactly zero*, **16%** of them cleared the 0.55 `compliance.target_prob_profit` gate on estimation error alone. On a 3,800-name universe that is ~610 zero-edge tickers passing the gate every day.

That number is worth stating with its construction. Closed form, the share is

$$
1 - \Phi\!\left(\Phi^{-1}(0.55)\sqrt{T/H}\right)
$$

**\(\sigma\) does not appear.** The drift a name needs to fake the gate scales linearly in \(\sigma\), and so does the standard error of \(\hat\mu\), so volatility cancels exactly. Only the length of the history and the forecast horizon matter:

| \(T\) | \(H\) | False positives |
|---|---|---|
| 750 | 20 | 22.1% |
| 1250 | 20 | **16.0%** |
| 2500 | 20 | 8.0% |
| 1250 | 10 | 8.0% |

Shorter histories are worse and longer horizons are worse, both because they widen \(\hat\mu\) relative to what the gate demands. A Monte Carlo run of the headline row measures 16.9%.

**This figure used to read 8.9%, and the correction is instructive.** The simulator was applying the Itô conversion to a drift already estimated in log space (see §14.1), which cost every path \(\tfrac12\sigma^2\) per day and pushed roughly half of these false positives back under the gate. Two errors were partially cancelling: the drift-noise problem looked half as bad as it was, and the \(\sigma\)-dependence in the old closed form above was the second bug showing through the first. Removing the Itô term unmasks the rest — which is why the two changes belong in the same discussion even though they landed separately.

The bias is also not independent of the rest of the platform. \(\hat\mu\) is largest precisely for stocks that have already run, so an unshrunk drift makes the Monte Carlo gate a noisy restatement of the momentum signal it is supposed to corroborate — it adds no independent information while appearing to confirm.

**Formulation.** Under a \(\text{Normal}(\mu_0, \tau^2)\) prior on the true daily drift and a sample mean with variance \(\sigma^2/T\):

$$
\mu_{\text{post}} = \frac{\tau^2\hat\mu + (\sigma^2/T)\mu_0}{\tau^2 + \sigma^2/T}, \qquad
\text{Var}_{\text{post}} = \frac{\tau^2 \cdot \sigma^2/T}{\tau^2 + \sigma^2/T}
$$

An annual drift is 252 daily drifts, so a prior dispersion stated per year converts as \(\tau = \tau_{\text{annual}}/252\). `simulation.prior_annual_drift_std` defaults to 10%, which is deliberately generous — it says a genuinely exceptional name might compound 20% a year faster than the market.

**Estimating the prior instead of asserting it.** The posterior above is only as good as \(\tau\) and \(\mu_0\), and both were assumed. The cross-section can supply them. Writing \(\hat\mu_i \sim \text{Normal}(\mu_i, \sigma_i^2/T_i)\) with \(\mu_i \sim \text{Normal}(\bar\mu, \tau^2)\), the observed variance of the sample means is the sum of the true dispersion and the average sampling noise, so the method of moments gives

$$
\bar\mu = \frac{1}{N}\sum_i \hat\mu_i, \qquad
\tau^2 = \max\!\left(0,\; \operatorname{Var}_i(\hat\mu_i) - \frac{1}{N}\sum_i \frac{\sigma_i^2}{T_i}\right)
$$

and each name is then shrunk toward \(\bar\mu\) rather than toward zero — the James-Stein form. The subtraction is the whole idea: what survives it is the dispersion the panel actually evidences. `simulation.use_empirical_drift_prior` (on by default) switches to this, falling back to the fixed prior when fewer than 20 names have usable history, since two moments cannot be told apart on a handful of names.

Shrinking toward \(\bar\mu\) rather than zero matters for a long-only book. Zero is the conservative choice for *relative* drift but it also discards the one thing a cross-section does evidence — the common level. On Indian equities that level is not zero, and pushing every name to zero understates every probability of profit uniformly.

**What it measures on real data.** Earlier versions of this section asserted that method-of-moments estimates on daily equity panels "typically come out at or below zero". On 66 cached NSE names that is not what happens:

| Quantity | Value |
|---|---|
| \(\bar\mu\) | 3.28e-4 /day (≈ 8.3%/year) |
| \(\tau^2\) | 4.15e-8 |
| \(\tau\) | 2.04e-4 /day (≈ **5.1%/year**) |
| Noise floor \(\overline{\sigma_i^2/T_i}\) | 7.91e-7 |
| Shrinkage intensity | **0.95** |

So the panel does evidence a real spread of true drifts — but a small one, about **half** the assumed 10%/year, with 95% of the observed spread in sample means attributable to estimation noise. The hardcoded prior was therefore *under*-shrinking on this data, which is the concrete argument for measuring it. On the synthetic zero-drift panel where there is genuinely nothing to find, the estimator returns \(\tau^2\) within a rounding error of zero and the false-positive rate goes to 0.0%.

**Posterior predictive, not plug-in.** Each simulated path draws its own drift from the posterior rather than sharing the posterior mean (`simulation.propagate_drift_uncertainty`), so `probability_profit` is the probability *accounting for the drift being estimated*. That is the quantity the compliance gate should read, and it widens the tails the VaR/CVaR cells report rather than leaving the simulation confident about the one input it has least right. Under both changes, the same zero-drift experiment passes **0.0%** of tickers.

---

## 22. Portfolio covariance and constrained allocation

**The gap.** Every position was sized independently: a signal scored, a stop set, a quantity derived from that trade's own risk, no reference to what else was in the book. For a long-only portfolio of 20–40 Indian equities that is the largest misstatement of risk in the platform, because the dominant term in realized portfolio volatility is the covariance:

$$
\sigma_p^2 = \sum_i \sum_j w_i w_j \sigma_i \sigma_j \rho_{ij}
$$

At twenty 3% positions and 30% single-name annualized volatility:

| Average pairwise correlation | Portfolio volatility | vs. independence |
|---|---|---|
| 0.00 | 4.02% | ×1.00 |
| 0.35 | 11.13% | ×2.77 |
| 0.60 | 14.17% | ×3.52 |
| 0.85 | 16.67% | ×4.14 |

Indian equity pairwise correlation sits around 0.3–0.4 in calm markets and rises toward 0.6–0.85 in exactly the panic states the drawdown breaker (§12) exists to survive. **The risk model was most wrong precisely when being wrong cost most.** `max_sector_pct` was the only concentration control, and it requires a `ticker,sector` CSV the repository does not ship — with no map, the cap is inactive.

This also explains an observation already recorded in the README: a meta-orchestrator that underperformed its own momentum sleeve. Four sleeves ranking one universe on correlated signals produce a book far less diversified than "four sleeves" suggests, so the extra sleeves added correlated risk without adding independent return — and nothing in the stack could measure it.

**Covariance estimation.** With \(N\) names and \(T\) days, the sample covariance needs \(T > N\) even to be invertible, and its largest eigenvalues are biased up and smallest biased down — exactly the distortion a mean-variance optimizer amplifies. `ledoit_wolf_covariance` shrinks toward a constant-correlation target \(F_{ij} = \bar{r}\sqrt{s_{ii}s_{jj}}\) with the analytically optimal intensity \(\delta = \max(0, \min(1, (\pi - \rho)/\gamma/T))\). That target is wrong in detail but right about what matters — that the names move together — and on a panel where it is well specified, shrinkage cuts estimation error by roughly 30% at the sample sizes this platform has. `exponentially_weighted_covariance` adds a half-life so the matrix responds to a correlation regime shift instead of averaging across one; `single_factor_covariance` estimates \(2N\) parameters instead of \(N(N+1)/2\) and is positive-definite by construction on a wide universe.

**Allocation.** `optimize_long_only` solves

$$
\max_w \; w^\top\mu - \frac{\lambda}{2}w^\top\Sigma w - c^\top|w - w_{\text{prev}}|
\quad \text{s.t.} \quad \mathbf{1}^\top w \le 1, \; 0 \le w_i \le w_{\max}, \; \sqrt{w^\top\Sigma w} \le \sigma_{\text{tgt}}
$$

by projected subgradient ascent over a capped simplex, so every constraint holds by construction rather than by tolerance. **The turnover penalty is load-bearing, not decoration.** Expected returns are estimated with enormous error (§21), so an unpenalized optimizer re-solves to a materially different book every day and hands the entire Indian friction stack (§13) the edge it was trying to capture.

**Where this reaches a decision.** `BacktestEngine` estimates the covariance once per rebalance round over the names that could be in the book by the end of it — strictly from returns dated *before* the decision date — and applies `risk.portfolio_volatility_target` as a final trim after the position, sector and risk-per-trade limits. Portfolio variance in the candidate's size is a convex quadratic, so the feasible set is an interval containing zero and bisection on its upper endpoint is exact rather than a heuristic; when the book is already over target the answer is to add nothing.

The **measurement runs unconditionally, the constraint is opt-in**. `portfolio_volatility_target` defaults to 0 because choosing a volatility ceiling is a risk-policy decision rather than a defect fix, but the book's volatility, its independence-assumed volatility, the ratio between them, the diversification ratio and the largest single risk contribution are sampled every 20 sessions and reported regardless. That ratio is the finding: a report that omits it is the state the platform was already in. Note also that the largest *risk contribution* routinely exceeds the largest *weight*, which is why a concentration cap is not a risk cap.

`hierarchical_risk_parity` is the allocation to use when \(\mu\) is not trustworthy — which, given §21, is most of the time. It clusters the correlation matrix and recursively bisects, splitting capital by inverse variance, so it uses the covariance structure without ever inverting it and without needing expected returns at all.

**Weighting and shrinkage are complementary, not alternatives.** Exponential weighting fixes *when* the matrix is measured — an equally weighted five-year window averages calm and crisis and is wrong in both. Shrinkage fixes *whether it can be used at all*. Weighting alone makes conditioning strictly worse: a 60-day half-life has an effective sample size of \(1/\sum_i w_i^2 \approx 87\) regardless of how much history is available, so the weighted matrix over any wide universe is massively rank-deficient. `shrunk_ewma_covariance` composes them.

That singularity is the mechanism behind crash-state blowups, and it is worth being precise about. A mean-variance optimizer inverts \(\Sigma\), and \(\Sigma^{-1}\) is dominated by the *smallest* eigenvalues — which in a near-singular sample estimate are pure noise. The optimizer reads that noise as a near-riskless arbitrage between two almost-collinear names and takes a large offsetting position in both. Shrinkage lifts the small eigenvalues off the floor, which is what bounds the condition number and makes the weights a function of the data rather than of the noise. On a 60-asset, 40-period panel the sample covariance is singular (smallest eigenvalue \(< 10^{-10}\)); the composed estimator is positive definite with a condition number six orders of magnitude lower.

**Group constraints need a different solver.** The subgradient optimizer is exact about its feasible set precisely *because* that set is a capped simplex with a closed-form projection. Add a sector limit \(S w \le c\) and the exact projection no longer exists, leaving only an approximation or a post-hoc clip — and a clipped solution is not the optimum of anything. `src/portfolio_optimizer.py` states the same objective as a quadratic program instead (cvxpy, optional extra):

$$
\max_w \; w'\mu - \tfrac{\lambda}{2} w'\Sigma w - c'u
\quad\text{s.t.}\quad
u \ge w - w_{\text{prev}},\; u \ge -(w - w_{\text{prev}}),\;
\mathbf{1}'w \le B,\; 0 \le w \le \bar w,\; Sw \le c_{\text{sector}}
$$

The auxiliary \(u\) is what makes the L1 turnover term expressible: \(|w - w_{\text{prev}}|\) is not differentiable and cannot enter a QP directly, but since \(u\) appears in the objective only with a positive cost, the solver drives it down until one of its two constraints is tight — at which point \(u_i\) is *exactly* \(|w_i - w_{\text{prev},i}|\). The reformulation is therefore equivalent rather than a relaxation, and the test suite asserts the equality rather than assuming it.

Sector limits are the constraint that actually binds in an Indian book. A momentum signal concentrates in whatever sector has been running, and a book with no group limit expresses a single macro bet through twenty tickers while reporting itself as diversified — the same failure this section exists to measure, arriving through a different door. Given a synthetic sector with overwhelming alpha the cap binds at exactly 25%; without it the same book puts over 90% there.

One numerical detail that is not optional: a shrunk covariance is positive semi-definite in exact arithmetic but carries eigenvalues around \(-10^{-18}\) after floating point, and cvxpy rejects a non-PSD quadratic form outright. The matrix is symmetrized and its eigenvalues clipped at zero before use, which changes it by less than the error already in it.

**Sources:**
- Ledoit & Wolf (2003), "Honey, I Shrunk the Sample Covariance Matrix", *Journal of Portfolio Management*
- López de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample", *Journal of Portfolio Management*
- Boyd & Vandenberghe (2004), *Convex Optimization*, §4.4 — the epigraph reformulation of an L1 penalty

---

## 23. Measuring a Sharpe ratio that means something

**Two problems, both in the measurement rather than the strategy.**

**(a) The definition was mixed.** `(CAGR − r_f)/σ` divides a *geometric* mean by an *arithmetic* standard deviation. Since \(\text{CAGR} \approx \mu - \sigma^2/2\), that returns approximately

$$
\frac{\mu - r_f}{\sigma} - \frac{\sigma}{2}
$$

biased low by \(\sigma/2\) — a flat −0.10 at 20% annualized volatility, growing with volatility, so it charged volatile strategies twice and was not comparable to the conventional 1.2 target it was being read against. Sharpe and Sortino are now arithmetic on both sides, on per-period excess returns, with the risk-free rate compounded rather than divided by 252. The old figure is retained as `geometric_sharpe_ratio` so a report carrying both makes the gap visible.

**(b) Nothing was adjusted for selection.** The platform searches: two stop multipliers, five strategies, three UMA combination methods, three simulation methods, two model architectures, and a regime map with four hand-set thresholds. Reporting the best result of that search reports the maximum of many draws — and the maximum of many draws from a zero-edge process is comfortably positive.

`src/performance_stats.py` implements the statistics that price that in. The Probabilistic Sharpe Ratio,

$$
\text{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{n-1}}{\sqrt{1 - \gamma_3\widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right)
$$

whose denominator is the point: negative skew and fat tails — endemic to equity strategies and to Indian small caps especially — inflate the standard error of a Sharpe estimate, so the same headline number is weaker evidence than a Gaussian assumption implies. The Deflated Sharpe Ratio evaluates PSR against the Sharpe the *best of N trials* would show by luck alone,

$$
SR^*_0 = \sqrt{V[\widehat{SR}]}\left[(1-\gamma)\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right)\right]
$$

with \(\gamma\) the Euler–Mascheroni constant. DSR is undefined without \(N\), which is exactly the quantity a research process forgets — hence the append-only trial log, a change with more scientific value than any model in the repository.

Also implemented: **PBO** by combinatorially-symmetric cross-validation, which asks not "is this strategy good" but "is my selection procedure predictive" (PBO > 0.5 means the configuration that looked best in sample is more likely than not to be *below* average out of sample); **cross-sectional rank IC and ICIR**, which is the right primary metric for a ranking system and separates signal quality from portfolio construction; and a **Newey–West standard error** for the overlapping labels a daily-sampled \(H\)-day target always has — adjacent observations share \(H-1\) days, and treating them as independent understates the standard error by roughly \(\sqrt{H}\), in the direction that manufactures significance.

One calibrating result from the tests: five years of daily data showing an annualized Sharpe of −0.4 still leaves roughly a **one-in-five chance the true Sharpe is positive**. A backtest is a small sample even when it covers a long calendar.

**\(N\) has to mean "configurations", not "runs".** DSR deflates against the expected maximum of \(N\) trials, so \(N\) is the input the whole adjustment turns on — and it is the one a research process is worst at recording. Two failure modes, opposite in sign:

- *Under-counting.* The trial log recorded a hand-enumerated parameter dict, so any knob nobody thought to list made two materially different runs record as the same trial. Trial identity is now a SHA-256 hash of the whole resolved config (plus the backtest window and strategy selection, which arrive by CLI rather than config), so extending the search space cannot silently escape the count. Paths are excluded deliberately — writing the same backtest to a different filename is not a search step, and including it would make every timestamped run unique and defeat the deduplication entirely.
- *Over-counting.* Re-running one configuration is not a new trial. Determinism is enforced, so a repeat reproduces its Sharpe; counting it again deflates against a search that never happened. Trials are deduplicated on that hash before the variance and count are taken.

A detail that matters more than it looks: the hash is SHA-256, never Python's builtin `hash()`. `str.__hash__` is salted per process unless `PYTHONHASHSEED` is pinned, so a log keyed on it would count every re-run as fresh, inflate \(N\) monotonically, and slowly deflate every reported Sharpe — a drift that reads as a result rather than as a bug.

**The risk-free rate is an input, not a constant.** The Sharpe numerator is the arithmetic mean of *excess* daily returns, which requires a rate to subtract. That rate used to be a `0.065` default argument on `RiskAnalyzer`, and the backtester constructed the analyzer without passing it at all — so every reported Sharpe was measured against a 6.5% hurdle no call site acknowledged. It is now a required argument, sourced from `risk.risk_free_rate` or, preferably, from a dated 91-day T-bill series at `paths.risk_free_rate_csv`, aligned to the return index and de-annualized so the excess return is taken day by day against the rate that actually prevailed. India's policy rate moved materially across 2021–2025; a single constant is wrong at both ends of any multi-year window, and which of the two was used is logged rather than left to be inferred from the absence of a file.

**Sources:**
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio Management* 40(5)
- Bailey, Borwein, López de Prado & Zhu (2016), "The Probability of Backtest Overfitting", *Journal of Computational Finance*
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns" — the \(t > 3.0\) hurdle

---

## 24. Predicting the cross-section, not the market

**The problem.** The neural stack was trained to predict each stock's *absolute* forward return. In an equity panel the overwhelming majority of the variance of a 5-day return is the common market factor — a typical Indian mid-cap runs an \(R^2\) against the Nifty of 0.35–0.55 at daily frequency, and the common component dominates further over a week.

So the network spent most of its capacity learning to forecast the market, which is (a) nearly unforecastable and (b) not what this system needs, because it cannot act on a market view at all: long-only, no shorting, no index hedge (§17). The residual, idiosyncratic component — the only part the platform monetizes, by choosing *between* stocks — was a minority of the training signal.

**Formulation.** The convention in empirical asset pricing since Gu, Kelly & Xiu is a cross-sectionally demeaned or rank-transformed target:

$$
y_{i,t} = r_{i,t\to t+H} - \frac{1}{N_t}\sum_{j \in \mathcal{U}_t} r_{j,t\to t+H}
\qquad\text{or}\qquad
y_{i,t} = \frac{2\,\text{rank}_t(r_{i,t\to t+H})}{N_t + 1} - 1
$$

The rank form is the default (`training.target_transform`) and is the more robust of the two on Indian data: a circuit-limited +20% print dominates the cross-sectional mean and drags every other name's label with it, while moving a ranking by exactly one place.

**This adds no look-ahead.** The transform mixes only labels dated at the same decision point \(t\), each of which is already realized at \(t+H\); it introduces no information the raw forward return did not already contain. Nothing about inference changes either — the model still scores one ticker at a time, and its output was always consumed as a relative score.

**The evaluation had to follow.** A rank is not a return: +0.4 is a position in the ordering, not 40%, so a Sharpe computed on ranks is a confident number about the wrong quantity. Under a relative target the walk-forward folds report **rank IC and its information ratio** instead — which is the better metric regardless, being far less noisy than backtested P&L and free of any confounding with position sizing, costs or the covariance of the book.

**The companion change: cross-sectional features.** Normalizing the label without normalizing the inputs is half a fix. A feature z-scored against a pooled five-year mean answers "is this RSI high for this stock over the sample"; a model choosing *between* stocks needs "is this RSI high relative to what else I could buy today". Worse, the pooled form puts the market factor straight back into every input column that the label transform above just removed: on a day the whole market gapped down, every name's return feature reads extreme against a five-year mean, and the network is handed the one state a long-only book cannot act on.

`training.feature_normalization: cross_sectional` (the default) z-scores each feature across the universe per date. Unlike the pooled scaler it **cannot leak by construction** — it fits no state and carries nothing across dates, so the transform for date \(t\) reads only rows dated \(t\). That is a stronger guarantee than "the statistics were fitted on the training split", which is a property of the calling code rather than of the transform, and it is directly testable: the suite rewrites every later row and asserts the earlier dates come back bit-for-bit identical.

**The trap this walked into, recorded because it is not obvious.** An earlier version of this section called the change infeasible on the grounds that "the fitted scaler ships inside the model checkpoint and is applied per ticker at inference, where no cross-section is available". That objection is real, and enabling the cross-sectional pass at training time *without* addressing it produces a silent, severe failure rather than a loud one:

1. Training z-scores each feature across the universe, so the values the global `FeatureScaler` is then fitted on are already \(\approx N(0,1)\).
2. That scaler therefore comes out with mean \(\approx 0\) and standard deviation \(\approx 1\) — a near-identity transform.
3. At inference it is applied to *raw* features. A price level of ₹1,500 passes through essentially unchanged and saturates the \(\pm 10\sigma\) clip.
4. Every price-level feature for every name collapses to the same ceiling. Measured on a 20-name synthetic panel: the model trains on inputs spanning \([-2.40, 1.94]\) and receives a constant \(+10.00\) at inference — one distinct value where training saw 594.

Nothing about that fails loudly. The scaler is a valid scaler, applied correctly; what diverges is the *pipeline*. So the mode travels in the checkpoint metadata exactly as the scaler's constants do, and `MLStrategy.score_batch` redoes the cross-sectional pass when the checkpoint says training used one — the cross-section it needs is precisely the batch it was handed. A checkpoint trained under `cross_sectional` also reports `requires_full_batch`, since a cross-section of one has no dispersion to standardize by. Checkpoints written before the mode existed default to `global` and are scored exactly as before.

PatchTST's per-window instance normalization partially compensates within a window; the plain LSTM path has nothing, which is why this matters more for the default architecture than for the recommended one.

**Sources:**
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies*
- Gu, Kelly & Xiu (2021), "Autoencoder Asset Pricing Models", *Journal of Econometrics*


---

## 25. Regimes as estimated states, not hand-set thresholds

**What was there.** `src/regime.py` classifies the market with a deterministic if/elif cascade over three hand-thresholded statistics — distance from a 200-day moving average, 60-day realized volatility against a 20% target, and ADX-14 — and emits a hard string. The branch ordering is well reasoned and the fail-neutral behaviour is right, but as a statistical object it has four defects:

1. **No state probability.** The output is a label, so everything downstream is a step function of a noisy continuous input. A benchmark oscillating around its 200-day average flips the entire sleeve-permission map day to day, with no hysteresis and no smoothing.
2. **No persistence model.** Real regimes are strongly persistent — empirical daily self-transition probabilities for equity volatility states run 0.95–0.99. A threshold rule's implicit persistence is whatever the autocorrelation of its inputs happens to be, which is neither the same thing nor estimable.
3. **The thresholds are free parameters** (200 days, 1.5×, ADX 20, ±2%), never fitted or tested — four researcher degrees of freedom inflating every backtest run through them.
4. **The states are defined, not discovered.** Model selection over the number of states is not even expressible in that formulation.

**Formulation.** A K-state Gaussian hidden Markov model on benchmark returns:

$$
r_t = \mu_{S_t} + \sigma_{S_t} z_t, \qquad \Pr(S_t = j \mid S_{t-1} = i) = p_{ij}, \qquad \sum_j p_{ij} = 1
$$

estimated by Baum–Welch (EM) with a scaled forward–backward recursion, and \(K\) chosen by BIC over candidates rather than fixed at four by hand. Every defect above has a direct answer: a filtered probability vector instead of a label, an estimated transition matrix instead of implicit persistence, maximum-likelihood parameters instead of thresholds, and states discovered rather than named.

| Property | Threshold cascade | This model |
|---|---|---|
| Output | hard label | filtered probability `P(S_t \| F_t)` |
| Persistence | implicit, unestimable | transition matrix, estimated |
| Hysteresis | none — flips on crossing | built into the forward filter |
| Number of states | fixed at 4 by hand | selected by BIC |
| Forecasting | none | `P(S_{t+h}) = pi_t P^h` |
| Uncertainty | none | full distribution over states |

**Two look-ahead distinctions, both enforced by test.**

*Filtered vs. smoothed.* The forward recursion gives \(P(S_t \mid \mathcal{F}_t)\) — what an observer standing at \(t\) would have believed — and is the only state estimate a trading decision may use. The smoothed \(P(S_t \mid \mathcal{F}_T)\) conditions on the whole sample, produces a visibly cleaner state sequence, and is unusable: a backtest wired to it looks excellent and is worthless. Both are provided, named to make the difference obvious, and a test asserts that corrupting every observation after date \(t\) leaves every filtered row up to \(t\) bit-identical while the smoothed rows move.

*Parameters vs. filtering.* A model fitted on the full sample has seen the test period even though the recursion has not — a subtler leak, and just as real. `assess_markov_regime` therefore fits on the leading `train_fraction` of history and filters the whole series forward from those frozen parameters; a test corrupts everything past the training window and asserts the fitted volatilities and transition probabilities do not move.

**Identifiability.** HMM states are exchangeable: relabelling them permutes the parameters without changing the likelihood. States are therefore always ordered by ascending fitted volatility, and EM is initialized deterministically from quantiles of \(|r|\) — otherwise two runs on identical data could produce the same model with the labels swapped, and which sleeves may trade would not be reproducible.

**Consuming probabilities instead of labels.** `sleeve_weights` computes

$$
w_{\text{sleeve},t} = \sum_k \pi_{t,k}\, a_{\text{sleeve},k}
$$

with \(a\) a sleeve-by-state affinity matrix. Because \(\pi_t\) varies smoothly, the permission map does too — which removes the day-to-day flipping entirely while preserving the existing "mute, don't veto" semantics, a good design that should survive. The practical gain is visible in one number: `expected_volatility` is the probability-weighted average across states, so a 60/40 split between calm and crisis reads as materially riskier than a confident calm, where a hard label reports both as "calm".

**Validation.** On a simulated two-state process with 0.985 persistence, the fit recovers per-state volatilities within 15%, self-transition probabilities of 0.983/0.985, BIC selection of \(K = 2\) over {1,2,3,4}, and filtered state identification agreeing with the true hidden state on more than 85% of days.

**Not yet done.** Time-varying transition probabilities (\(p_{ij,t}\) as a softmax over India VIX, FII flow and trend distance) and MS-GARCH — a regime-dependent conditional variance, which would subsume the existing GJR-GARCH rather than sit beside it and would address the per-ticker fitting cost as a side effect. Both need the Phase 2 data that is not cached here.

**Sources:**
- Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", *Econometrica*
- Baum, Petrie, Soules & Weiss (1970), the forward–backward / EM algorithm for HMMs


---

## 26. "Cannot compute" is not "scores zero"

**The bug.** `strategies/rule_based.py` combines four components onto a 0–100 scale and gates on absolute thresholds — BUY at 60, WATCH at 45. A component whose input was missing scored 0, which does not merely add no information: it moves the total across those fixed thresholds for a reason unrelated to the stock.

The live case is the one the README already noted. `context.mc_result` is per-ticker, and the callers that batch build a single context for the whole round, so a `rule_based` member inside a UMA sees no Monte Carlo result and scored `MC_Prob` at 0. The identical stock on the identical day therefore scored roughly 12 points lower inside an ensemble than standalone — a silent, systematic level shift between two code paths that are supposed to agree.

`combine_weighted` now takes the names of unavailable components and renormalizes the remaining weights to 100, so the score stays on a scale the thresholds still mean something on.

**The asymmetry that makes this subtle.** There are two reasons a component can be unmeasurable, and treating them alike introduces a worse bug than the one being fixed:

| Reason | Example | Correct handling |
|---|---|---|
| The **pipeline** did not compute it | no `mc_result` in a batched UMA | renormalize the weight away — nothing about the stock differs |
| The **stock** lacks the data | no SMA-200, i.e. a recent listing | keep the conservative zero — the absence *is* information |

Renormalizing a missing SMA-200 away would scale the remaining components up and score a young, illiquid name *higher* than a seasoned one: least caution exactly where an alphabetically-sliced Indian micro-cap universe (§9 of the review's findings) warrants most. Only `MC_Prob` is passed as unavailable, and a test pins the distinction in both directions.

**What deliberately did not change.** With no Monte Carlo result the probability-of-profit gate still fails closed, so a `rule_based` member cannot issue BUY inside a batched UMA. A compliance gate with no evidence either way should refuse rather than wave a trade through untested; the rationale now says `prob(no MC result):FAIL` instead of a `prob(0.00)` that reads like a measurement, so the limitation is legible rather than silent.

### The rank composite

The second half of D10: weighted-summing an ordinal (trend, three levels), a binary (breakout), a right-skewed continuous (volume) and a near-constant (`MC_Prob`, standard deviation ≈ 0.05 once the drift is shrunk) onto a shared 0–100 scale is not a scoring function with a probabilistic interpretation. `scoring.method: rank_composite` converts each component to its percentile rank across the eligible universe on that date before weighting:

$$
S_{i,t} = \sum_k w_k \cdot \text{pct-rank}_t\!\left(c_{k,i,t}\right)
$$

Percentile rather than the review's \(\Phi^{-1}(\text{rank}/(N_t+1))\) form, deliberately: it keeps the composite on 0–100, so the existing `score >= 60` and `>= 45` gates stay syntactically valid and gain a clean reading — "60th percentile of the weighted composite" — instead of requiring every YAML to be rewritten against a z-scale. Ties take the average rank, which is the common case here rather than an edge case, since breakout is binary and trend has three levels.

**What it fixes.** Commensurability, and invariance to each component's marginal distribution. Express volume as a ratio or as any monotone transform of one and the composite is unchanged; under the weighted sum it moves. A component's influence stops depending on the accident of its units.

**What it does not fix, despite being the proposed remedy for it.** A component that is near-constant across the universe still consumes its full share of the budget. Ranking ties hands every name the *same* percentile — 0.6 in a five-name universe, whatever the constant is — so `MC_Prob` contributes a flat number under rank scoring exactly as it did under the weighted sum. A different flat number, but still flat, and still 30% of the budget spent on a component that separates nobody. Making influence track discrimination requires the *weights* to respond to realized dispersion or information coefficient, which is a change to the weight learner (§ D8's path) rather than to the combination rule. Lowering `model_probability`'s configured weight is the blunt version available today. A test asserts this non-property directly, so the limitation cannot quietly be forgotten.

**What it changes about the strategy's meaning.** Under `weighted_sum`, `score >= 60` is an absolute quality bar that nothing need clear on a bad day. Under `rank_composite` it is a percentile, so a roughly fixed share of the universe clears it whatever the market is doing: the score stops being a quality filter and becomes a ranker, leaving the absolute gates — probability of profit, net reward:risk, minimum price, and the regime layer — as what remains to say "not today". On a worked five-name example the two agree on ordering exactly while the scores compress from a 23–90 range to 42–86, and a name sitting exactly on the 60 threshold moves to 69.

That is a change of strategy rather than a defect fix, so `weighted_sum` remains the default and `rank_composite` is one YAML line away. `requires_full_batch` becomes True under rank scoring, since ranking a universe of one is not a degraded answer but a meaningless one.

### The probit composite

A percentile is a uniform variate, and that is the limitation `rank_composite` cannot escape: a weighted sum of uniforms has a spread that depends on how many components were measurable and on how correlated they happened to be that day. The composite is therefore comparable *within* a date and not *across* dates — fine for ordering names, useless as a magnitude. Anything that consumes the score as a number rather than as a rank (the expected-return input to §22's optimizer, most obviously) needs more than that.

`scoring.method: probit_composite` applies the inverse normal CDF the review originally proposed, then standardizes the combination:

$$
z_{k,i,t} = \Phi^{-1}\!\left(\frac{r_{k,i,t}}{N_t + 1}\right), \qquad
C_{i,t} = \sum_k w_k z_{k,i,t}, \qquad
S_{i,t} = \frac{C_{i,t} - \bar C_t}{\operatorname{sd}_t(C)}
$$

so the composite is mean-zero and unit-variance on **every** date by construction, whatever the universe size or the correlation structure that day.

**The plotting position is load-bearing, not a rounding detail.** Ranks are divided by \(N_t + 1\), not by \(N_t\). Under \(r/N\) the best name in the universe ranks at exactly 1.0 and \(\Phi^{-1}(1) = +\infty\) — and that does not fail loudly on that one ticker. It propagates into the weighted sum, then into \(\bar C_t\) and \(\operatorname{sd}_t(C)\), and nans **every** score on the date. The \(r/(N+1)\) convention (Van der Waerden) keeps the argument strictly inside \((0,1)\) for every \(N\).

**Keeping the 0–100 gates.** The obvious move — publish \(S_{i,t}\) as the score — would break the platform silently: `_build_signal` gates on `score >= 60` and `>= 45`, and a z-score never clears 60, so the system would simply stop issuing BUY. The reported score is therefore \(100\,\Phi(S_{i,t})\). \(\Phi\) is monotone, so the ordering is exactly the z-ordering and the existing thresholds keep working while acquiring a cleaner reading — `>= 60` is "in the top 40% of today's cross-section". The raw \(z\) is published alongside as `signal.extra["composite_z"]` for consumers that want the magnitude.

It inherits `rank_composite`'s percentile-threshold caveat, and it does **not** fix the near-constant component either: ties still hand every name the same quantile, so a flat percentile becomes a flat \(z\). Also off by default, for the same reason.


---

## 27. Reinforcement learning, scoped to what one trajectory supports

**Why the obvious version does not work.** The instinct is a deep network that
observes prices and emits buy/sell. On this data it will produce a spectacular
backtest and no edge, for a reason that is arithmetic rather than pessimism:

- **There is one episode.** Supervised learning here has ~1,250 daily
  observations per ticker, already thin. RL's unit of learning is a
  *trajectory*, and the market has produced exactly one — 2021 to 2025 happened
  once. An agent with a few thousand parameters memorises that single path.
  Atari agents train on millions of episodes because the environment can be
  re-run; the market cannot.
- **Naive rewards learn leverage.** Reward cumulative return and the optimal
  policy is "hold the maximum position always" — a beta exposure with extra
  steps, not a strategy.
- **The reward is measured through the same instruments.** If costs are
  understated, the agent finds and exploits exactly that gap, because that is
  what RL does.

**What is actually estimable.** Rather than *what* to hold, RL is applied to
*how much*:

> Given the regime probabilities and how the book is currently positioned, what
> fraction of capital should be deployed today?

A small discrete action space over a low-dimensional state, which one
trajectory can support in a way a price-level policy cannot. It is also the
formulation the literature supports — regime states from a hidden Markov model
(§25) with an allocation policy on top.

**Formulation.** State at \(t\) is the filtered regime probability vector, the
current exposure, trailing volatility and a bias — all strictly backward-looking.
Action is a target exposure from \(\{0, 0.25, 0.5, 0.75, 1\}\). The reward
uses the return the market *subsequently* delivered:

$$
R_t = w_t r_{t+1} - c\,|w_t - w_{t-1}| - \frac{\lambda}{2}\left(w_t r_{t+1}\right)^2
$$

The turnover charge \(c\) and the quadratic risk term \(\lambda\) are both on
by default, because they are what decide whether the agent learns a strategy or
learns leverage. Filtered probabilities only — smoothed ones condition on the
whole sample and would hand the agent the future inside its own state vector.

**Policy.** Linear softmax, `n_features × n_actions` parameters — typically
under 40 — trained by REINFORCE with a standardized return-to-go baseline and
an entropy bonus. Deliberately tiny: a neural policy here would not be more
capable, it would be more confident. The baseline matters more than usual
because daily rewards are noisy and near zero-mean; without it the gradient is
dominated by whatever the market happened to do rather than by the action's
contribution.

**Validation.** On a synthetic two-regime path where the regime is observable,
fitted on the leading 60% and evaluated frozen and greedy on the held-out 40%:

| | Sharpe | Return | Turnover |
|---|---|---|---|
| RL policy | **+1.25** | +43.7% | 2.8 over 599 steps |
| Always invested | +0.88 | +55.0% | — |

Lower return, materially better risk-adjusted return, almost no churn — which
is the shape a working exposure policy should have. **This demonstrates the
machinery learns, not that RL works on markets**: the regime is handed to the
agent noiselessly, which no real regime estimate is.

The score function is checked against a numerical gradient in the tests. A sign
error there trains the policy to do the opposite of what works, which presents
as "RL does not work on finance" rather than as a bug.

**How to treat a result.** As one trial in a search, not as a finding. Feed it
to `src/performance_stats.py` and log it: an RL agent that beat the baseline on
one split of one trajectory is exactly what the deflated Sharpe ratio exists to
discount. And nothing here has a point-in-time universe yet (D9), so a good
backtest remains not evidence.

**Sources:**
- Sutton & Barto, *Reinforcement Learning: An Introduction* — REINFORCE and baselines
- *Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning*, arXiv:2605.27848


---

## Summary: what to combine into a UMA

Given the above, a reasonable evidence-backed UMA blends the fully-implemented, roughly orthogonal signals. `config/strategies/uma_meta_orchestrator.yaml` is that configuration; `config/strategies/example_uma.yaml` is the minimal two-member version if you want to see the YAML mechanics alone.

| Strategy | Signal type | Data required | Signal horizon |
|---|---|---|---|
| `rule_based` | Trend/breakout/volume/MC (technical) | OHLCV | Days |
| `momentum` | Cross-sectional 9-month momentum | OHLCV | Months |
| `low_volatility` | Cross-sectional trailing volatility | OHLCV | Months |
| `lstm` / `patchtst` | Learned sequence pattern | OHLCV | 5-day forward return |

Momentum and low-volatility are close to orthogonal in the literature — momentum profits from continuation, low-vol from an entirely different risk-pricing anomaly — which makes them complementary sleeves; `rule_based` and the neural forecaster add faster-moving signals on top. Use `method: trigger` to combine them (§19), not `weighted_blend`: sleeves this different *will* disagree, and averaging disagreement is how you get a trade none of them wanted.

**The column that is deliberately not here is "rebalance cadence."** The academic constructions rebalance monthly; this platform has no rebalance cadence at all. Every strategy is re-scored every trading day, and a name leaving the top decile simply stops being bought while the position already open rides its stop. That is workable — but only as long as the *exit* horizon matches the signal's, which is what §13.6 measures and why `atr_stop_multiplier` is the single most consequential setting for a months-horizon sleeve.
