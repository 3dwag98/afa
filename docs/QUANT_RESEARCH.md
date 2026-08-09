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
