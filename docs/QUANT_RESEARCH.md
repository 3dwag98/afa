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
15. [Circuit limits and the illiquidity illusion](#15-circuit-limits-and-the-illiquidity-illusion) — implemented (`src/liquidity.py`)
16. [Overnight gaps and GARCH stationarity](#16-overnight-gaps-and-garch-stationarity) — implemented (`src/volatility_models.py`)
17. [The long-only constraint](#17-the-long-only-constraint) — a known, accepted limitation
18. [Making the walk-forward measurement honest](#18-making-the-walk-forward-measurement-honest) — implemented (`agents/trainer.py`)

---

## 1. Cross-sectional momentum

**Evidence.** Jegadeesh & Titman (1993) established that stocks with high returns over the past 3–12 months continue outperforming over the next 3–12 months. For India specifically: Joshipura & Mankar's NSE working paper on CNX 100 constituents found momentum profitability; a portfolio-based study of the Indian market found momentum profits over 6-month and 1-year horizons, with contrarian (reversal) returns emerging only over a 3-year horizon; sector-level studies confirm the effect is not limited to specific industries; liquidity research shows momentum is *strongest* among the most liquid stocks (favorable for backtesting realism — liquid names have less slippage). One caveat: short-term reversal coexists with momentum in India, motivating the standard skip-month convention below.

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

Rank ascending; long the bottom decile (lowest realized volatility). This is the mirror image of momentum's ranking machinery, so both strategies share a `_rank_and_select_decile()` helper.

**Sources:**
- [The Volatility Effect: Recent Evidence from Indian Markets](https://www.scirp.org/html/28-1501934_94780.htm)
- [An Investigation of Low Volatility Anomaly in Indian Stock Market](https://www.researchgate.net/publication/305621746_An_Investigation_of_Low_Volatility_Anomaly_in_Indian_Stock_Market)
- [Low-Risk Anomaly: Evidence from India](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4398656)

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

**Use in the platform:** `src/volatility_models.py::forecast_volatility()` fits this model per ticker and produces a one-step-ahead conditional volatility forecast, used as a (better) input to the same lognormal Monte Carlo machinery in `monte_carlo.py`, and to volatility-targeted position sizing (`risk.py`) — position size scaled inversely to *forecasted* (not just trailing historical) volatility, per the volatility-targeting literature below. Falls back to the existing flat historical-std approach when there isn't enough history to fit GARCH reliably (needs ~250+ observations for stable MLE convergence) or when the `arch` package isn't available.

**Sources:**
- [Modeling and Forecasting the Volatility of NIFTY 50 Using GARCH and RNN Models](https://www.mdpi.com/2227-7099/10/5/102)
- [Modelling time-varying volatility using GARCH models: evidence from the Indian stock market](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758444/)
- [Forecasting Stock Market Volatility using GARCH Models: Evidence from the Indian Stock Market](https://www.researchgate.net/publication/305803327_Forecasting_Stock_Market_Volatility_using_GARCH_Models_Evidence_from_the_Indian_Stock_Market)
- [Modelling Stock Market Volatility in India: A GARCH Analysis of the Nifty 50 Index](https://www.researchgate.net/publication/400666876_Modelling_Stock_Market_Volatility_in_India_A_GARCH_Analysis_of_the_Nifty_50_Index)
- Volatility targeting (position-sizing side): [The Impact of Volatility Targeting](https://www.man.com/insights/the-impact-of-volatility-targeting) (Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert, *Journal of Portfolio Management*, 2018); [Understanding Risk Parity (AQR)](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf)

---

## 4. Kelly criterion position sizing

**Evidence.** The Kelly criterion (Kelly, 1956) gives the capital fraction that maximizes long-run logarithmic (compound) growth for a repeated bet with known win probability and payoff ratio. Universally used in quantitative position sizing, but **full Kelly is not used in practice** — win probability and payoff ratio are estimated with error, and full Kelly is extremely sensitive to that error (and to return skewness). The standard mitigation, per both practitioner sources and the platform's own risk posture (paper trading, capped positions), is fractional Kelly: quant funds commonly use 1/4 to 1/2 Kelly; half-Kelly captures roughly 75% of the theoretical growth rate at meaningfully lower drawdown risk, quarter-Kelly roughly 50% of growth at even lower volatility.

**Formulation.** For a strategy with realized win probability \(p\) and reward:risk ratio \(b\) (average win ÷ average loss, in the same units as this platform's existing `reward_risk` field):

$$
f^{*} = p - \frac{1-p}{b}
$$

Position size uses a fractional Kelly \(f = \kappa f^{*}\) with \(\kappa \in [0.25, 0.5]\) (configurable; default 0.5), clamped to \([0, \text{max\_single\_position\_pct}]\) so it can never exceed the platform's existing hard position cap, and clamped to \(\ge 0\) (a negative \(f^*\) — an unprofitable edge — falls back to the existing fixed-fractional sizing rather than sizing a "negative" position, since this platform never shorts).

\(p\) and \(b\) are estimated from `AgentBrain.trade_history`, gated by the same `min_trades_for_learning` sample-size threshold already used by the weight-adaptation logic (`strategies/weighting.py`) — with too few realized trades, Kelly sizing falls back to the existing ATR/fixed-fractional sizing (`risk.py::calculate_quantity`) rather than sizing off noisy estimates.

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

## 7. Cointegration-based pairs trading (not yet implemented)

**Evidence.** Strong, India-specific support: an arXiv paper "Designing Efficient Pair-Trading Strategies Using Cointegration for the Indian Stock Market" and multiple QuantInsti EPAT projects (e.g., a 25-stock market-neutral pairs strategy across Banking/IT/Pharma/Cement/Auto) validate cointegration-based (not merely correlation-based) pairs trading on NSE large-caps. Standard approach: Engle-Granger two-step cointegration test to find pairs, trade the z-scored spread \(z_t = \frac{(P_A - \beta P_B) - \mu_{\text{spread}}}{\sigma_{\text{spread}}}\), entering when \(|z_t|\) exceeds a threshold (e.g. 2) and exiting at mean reversion.

**Why it isn't implemented yet — an architectural gap, not a research gap.** Every strategy in this platform (`BaseStrategy::score()`/`score_batch()`) scores *one ticker at a time* against a shared context; pairs trading fundamentally needs a *relationship between two tickers* (a rolling hedge ratio and spread) and, in its academically-standard form, a short leg — which conflicts with this platform's no-short-selling guardrail. Supporting it properly needs: (a) a pair-selection step (cointegration screening across \(O(n^2)\) candidate pairs) that doesn't fit `BaseStrategy`'s per-ticker interface, and (b) either accepting long-only spread trades (long the undervalued leg only, foregoing the short leg's hedge) or a deliberate, explicit exception to the short-selling guardrail. Flagged here as a scoped-out future strategy family rather than folded in half-implemented.

**Sources:**
- [Designing Efficient Pair-Trading Strategies Using Cointegration for the Indian Stock Market (arXiv)](https://arxiv.org/abs/2211.07080)
- [Cointegrated Pairs Trading Strategy in Indian Equity Market (2015–2025), QuantInsti EPAT](https://blog.quantinsti.com/cointegrated-pairs-trading-indian-equity-market-epat-project/)
- [Mean-Reversion Statistical Arbitrage Pair Trading Across Sectors, QuantInsti EPAT](https://blog.quantinsti.com/epat-project-mean-reversion-statistical-arbitrage-pair-trading-strategy-indian-market-sectors/)

---

## 8. Fama-French multi-factor models (not yet implemented)

**Evidence.** Well-validated for India: multiple studies (2005–2015, 2016–2023, CNX 500 2000–2015) find the three- and five-factor models outperform CAPM, with the market factor dominant and a documented size effect; the five-factor extension (adding profitability RMW and investment CMA) improves on the three-factor version in several studies.

**Why it isn't implemented.** SMB (size) needs market capitalization (shares outstanding × price — shares outstanding isn't in OHLCV), HML (value) needs book value per share, RMW (profitability) needs operating profitability, CMA (investment) needs total asset growth — all balance-sheet/fundamental data this platform doesn't ingest. Implementing this properly requires a new fundamentals data source (e.g., NSE/BSE corporate filings, or a paid fundamentals API) and a new data-ingestion pipeline parallel to `data_store.py`'s OHLCV cache — a genuine scoping decision, not a quick add.

**Sources:**
- [Empirical Applicability of the Fama-French Three-Factor Model in the Indian Equity Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5010405)
- [Fama-French five-factor asset pricing model: empirical evidence from Indian stock market](https://www.inderscience.com/info/inarticle.php?artid=111959)
- [An Empirical Evaluation of the Fama-French Five-Factor Model in the Indian Equity Market: Evidence from NSE-Listed Stocks](https://www.researchgate.net/publication/391388581_An_Empirical_Evaluation_of_the_Fama-French_Five-Factor_Model_in_the_Indian_Equity_Market_Evidence_from_NSE-Listed_Stocks)

---

## 9. Quality (QMJ) factor (not yet implemented)

**Evidence.** "Quality Minus Junk" (profitability + safety/low-leverage) shows the highest average monthly returns among factors tested in recent Indian research (1.69% vs. 1.33% for the market factor) with *lower* volatility than the momentum factor, and a significant four-factor alpha.

**Why it isn't implemented.** Same root cause as §8 — needs fundamental data (profitability ratios, leverage, balance-sheet safety metrics) not present in OHLCV. Same future path: a fundamentals data source would unlock this alongside Fama-French.

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

## Summary: what to combine into a UMA

Given the above, a reasonable evidence-backed UMA (see `config/strategies/example_uma.yaml` for the YAML mechanics) blending fully-implemented, orthogonal signals:

| Strategy | Signal type | Data required | Rebalance cadence |
|---|---|---|---|
| `rule_based` | Trend/breakout/volume/MC (technical) | OHLCV | Daily |
| `momentum` | Cross-sectional 9-month momentum | OHLCV | Monthly |
| `low_volatility` | Cross-sectional trailing volatility | OHLCV | Monthly |
| `lstm` | Learned sequence pattern | OHLCV | Daily |

Momentum and low-volatility are close to orthogonal in the literature (momentum profits from continuation, low-vol profits from an entirely different risk-pricing anomaly), making them a good `weighted_blend` pair; `rule_based` and `lstm` add faster-moving, daily-refreshed signals on top.

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

**The market series.** When `data.benchmark_symbol` (default `^NSEI`, the Nifty 50) is cached, the trend and volatility tests key off the real index — "the market is below its 200-day average" is a statement about the index the research studied. Without one, `src/regime.py::build_market_proxy()` falls back to an equal-weighted composite of the traded universe, which needs no extra data but is only a proxy: it reflects whatever is in today's universe, and idiosyncratic noise diversifies out of it in a way real index volatility does not.

The composite averages **daily returns** and cumulates them, rather than averaging rebased price levels. Real universes have ragged start dates and the eligible set changes daily during a backtest; averaging rebased levels makes a newly-listed ticker join at its own base of 1.0 while incumbents sit at 1.4, printing a double-digit synthetic drop on a day every constituent rose — and the filter would read that construction artifact as a trend break and as realized volatility. A return average has no such discontinuity.

Nifty VIX would be a better volatility gauge than realized volatility, but it is not in the OHLCV cache; see §10 for the flow-data equivalent.

**(c) Idiosyncratic momentum** — ranking on residual rather than total returns — is the third documented fix. It needs a factor model to residualize against, which needs §8's data. Not implemented.

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

The map comes from a `ticker,sector` CSV at `paths.sector_map_csv`. **Without one the cap is inactive, and both engines say so at WARNING level.** Pooling unmapped names into a single `UNKNOWN` bucket and capping *that* looks like the conservative choice and is in fact a completely different constraint: with no map at all, every holding is UNKNOWN, so a 25% "sector" limit silently becomes a 25% limit on total invested capital and leaves three quarters of the portfolio in cash forever. An unmapped ticker is likewise never charged against a mapped sector's allowance. A cap that cannot be computed is reported as unenforceable rather than quietly reinterpreted.

### 13.4 Drawdown circuit breaker

`risk.max_portfolio_drawdown_pct` (15%) halts *new* entries once peak-to-trough drawdown reaches it; buying resumes at `drawdown_reentry_pct` (10%). Two thresholds rather than one, so the breaker cannot flicker on and off as equity wobbles across a single line — and a config validator rejects `drawdown_reentry_pct >= max_portfolio_drawdown_pct`, since that setting re-creates the single-threshold flicker the pair exists to prevent. Open positions keep their stops and targets: force-liquidating a whole book at a drawdown trough is how a bad quarter becomes a permanent loss.

---

## 14. Fat tails in the forward simulation

The original Monte Carlo drew i.i.d. Gaussian shocks around a historical mean and standard deviation. That model assumes stationary parameters, thin tails and independent days — all three fail in an emerging market with circuit limits, retail participation spikes and policy shocks, and the failure is one-directional: it *understates* tail risk, which is exactly the quantity `compliance.target_prob_profit` gates on.

`simulation.method` now selects the shock process:

- **`block_bootstrap`** (default) — the stationary bootstrap of Politis & Romano. At each simulated day, either continue the current block (probability \(1 - 1/L\)) or jump to a fresh random start (probability \(1/L\)), giving geometric block lengths with mean \(L\). Contiguous days travel together, so both the empirical fat tails and the volatility clustering survive resampling; drawing days independently would destroy exactly the serial dependence that makes a drawdown a drawdown. Geometric rather than fixed block lengths avoid imprinting an artificial periodicity on the paths.
- **`jump_diffusion`** — Merton: Gaussian diffusion plus a compound-Poisson jump component, \(N \sim \text{Poisson}(\lambda/252)\) jumps per day of size \(\sim \mathcal{N}(\mu_J, \sigma_J^2)\) with \(\mu_J < 0\). Diffusion cannot produce a gap; jumps can. The drift is jump-compensated, so adding jump risk widens the tails without also quietly shifting every expected return downward.
- **`gaussian`** — the original, retained for comparison.

All three compose with the GARCH volatility path (§3, §16): GARCH sets how large tomorrow's shocks should be, `method` sets what shape they take. A method whose preconditions do not hold — a block bootstrap over fewer than 60 observations just re-prints the same few days — degrades to the Gaussian path rather than failing the ticker, since a missing MC result is read downstream as zero probability-of-profit.

**Sources:**
- Politis & Romano, ["The Stationary Bootstrap"](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870), *JASA* 89(428), 1994
- Merton, ["Option pricing when underlying stock returns are discontinuous"](https://www.sciencedirect.com/science/article/abs/pii/0304405X76900222), *Journal of Financial Economics* 3(1-2), 1976

---

## 15. Circuit limits and the illiquidity illusion

Every ranking formula in this platform assumes continuous price discovery and that a printed close is a price you could have transacted at. On the NSE/BSE mid- and small-cap segments neither holds, in two specific ways that are both detectable from OHLCV alone.

**Circuit locks.** Stocks lock at 5%/10%/20% bands. An operator-driven pump locks a stock in the upper circuit for consecutive sessions: \(P(t)/P(t-J)\) registers a huge formation return and momentum screams BUY, while there is no offer to lift. On the way back down the stock locks in the lower circuit and the position cannot be exited at all, so the realized loss blows through the modelled stop — which in turn inflates the payoff ratio \(b\) that Kelly sizes off (§4).

A locked session is identified by a **zero intraday range** (high == low) together with a move from the prior close at or beyond the smallest statutory band. Both conditions are needed: a zero range alone is an untraded day, a large move alone is an ordinary volatile session.

**The illiquidity illusion.** Low realized volatility is supposed to proxy for a stable business. In India it frequently proxies for *nothing trading*: an illiquid stock carries yesterday's close forward, printing \(r = 0\) rather than a small return, which mechanically suppresses \(\sigma_i\) and walks the stock into the low-volatility buy decile. The anomaly the strategy is trying to harvest is not the one it ends up holding — these "quiet" names carry exactly the governance risk that ends in a forensic audit or a SEBI suspension.

`src/liquidity.py` screens on three trailing-60-session statistics — median rupee turnover, the share of unchanged closes, and the share of circuit-locked sessions — plus a "locked on the decision date" check for whether an order could be filled at all. Screened names are **removed from the ranking**, not merely marked un-buyable: leaving an untradeable stock in the cross-section would still shift every other name's percentile. Statistics that cannot be computed (short cache) pass, since the screen's job is to exclude on positive evidence of untradeability, not on absence of evidence.

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

Reported metrics are directional accuracy plus the annualized Sharpe of a long-only signal-following rule, always against an always-long benchmark on the identical sample. A model that cannot beat always-long is adding turnover, not alpha, and the trainer says so in as many words. The held-out test evaluation reloads the best checkpoint first, so the numbers describe the weights `MLStrategy` will actually load rather than whatever the last epoch of overfitting produced.
