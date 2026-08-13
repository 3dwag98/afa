"""Configuration schema for the Autonomous Financial Advisor (AFA) portfolio agent."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Configuration for data paths and settings."""

    data_dir: str = Field(default="data", description="Base directory for data files")
    market_data_dir: str = Field(
        default="data/market_data", description="Directory for market data"
    )
    source: Literal["huggingface", "yfinance"] = Field(
        default="huggingface",
        description="Where historical OHLCV comes from. 'huggingface' pulls a versioned Hub "
        "dataset (see src/hf_dataset.py) — reproducible when hf_revision is pinned, and one "
        "columnar download instead of thousands of per-ticker requests. 'yfinance' keeps the "
        "original per-ticker download path. Either way the bars land in the same parquet cache.",
    )
    hf_dataset_id: str = Field(
        default="vishnun0027/indian-market-historical-ohlcv",
        description="HuggingFace Hub dataset repo id used when source='huggingface'.",
    )
    hf_revision: Optional[str] = Field(
        default=None,
        description="Git revision (branch, tag or commit SHA) of the Hub dataset to pin. Leave "
        "unset to track the default branch; pin it for a reproducible backtest, since an "
        "unpinned dataset can be updated underneath a running series of experiments.",
    )
    hf_asset_dir: str = Field(
        default="stocks",
        description="Directory within the Hub dataset to ingest equities from "
        "(stocks | indices | etfs | commodities | forex). One parquet per symbol.",
    )
    hf_adjust_prices: bool = Field(
        default=True,
        description="Back-adjust OHLC by adj_close/close on ingest. Leave on: an unadjusted "
        "1:10 split prints as a -90% daily return, which cross-sectional momentum reads as a "
        "crash and the circuit-lock detector reads as a limit move. The previous yfinance path "
        "used auto_adjust=True, so this also keeps the cache internally consistent.",
    )
    benchmark_symbol: str = Field(
        default="^NSEI",
        description="Index symbol (in the Hub dataset's indices/ directory) used as the market "
        "benchmark for the momentum crash filter. ^NSEI is the Nifty 50. When its history is "
        "cached, src/regime.py keys the trend and volatility filters off the real index; "
        "otherwise it falls back to an equal-weighted composite of the traded universe.",
    )
    default_history_years: int = Field(
        default=20,
        description="Years of historical data to keep. Applied to both sources: the Hub dataset "
        "is trimmed to this window on ingest rather than cached in full. "
        "This was 5, and the consequence was not obvious: every cached file spanned exactly "
        "five years, so the sample began *after* the COVID crash and contained one bull run, "
        "one rate-hike correction, and no crisis. Every tail estimate, regime model and "
        "drawdown forecast was therefore fitted on data with no crash in it. The source is "
        "trimmed to whatever it actually holds, so a value larger than the available history "
        "costs nothing — which makes a low value pure downside. Raise it further rather than "
        "lower it; the ingest reports the span it actually obtained.",
    )
    universe_size: int = Field(
        default=10, description="Number of securities in the trading universe"
    )
    universe_selection: Literal["alphabetical", "random"] = Field(
        default="random",
        description="How universe_size names are chosen from the parquet cache. The cache is "
        "scanned in sorted filename order, so 'alphabetical' returns whatever sits at the front "
        "of the alphabet — the same few hundred names every run, for training and backtesting "
        "alike. That is a sample of the alphabet rather than of the market, and it means a model "
        "is evaluated on the very tickers it was fitted on, however carefully the dates are "
        "split. 'random' draws a seeded sample instead, and offsets the seed by purpose so the "
        "training and backtest draws are different names. Seeded, not truly random: two runs of "
        "one config must produce the same universe or nothing is reproducible. Note this does "
        "not fix survivorship bias — a random sample of a cache is still not point-in-time index "
        "membership (see docs/REVIEW_STATUS.md, D9).",
    )
    universe_seed: int = Field(
        default=42,
        description="Base seed for data.universe_selection='random'. Change it to draw a "
        "different sample and re-run; keep it fixed to reproduce one.",
    )
    tickers: List[str] = Field(
        default_factory=list,
        description="Explicit ticker override list for the live agent; empty means auto-discover from cache",
    )
    min_history_days: int = Field(
        default=250, description="Minimum number of historical days required to consider a ticker tradeable"
    )
    allow_synthetic_fallback: bool = Field(
        default=True, description="Whether to fall back to synthetic OHLCV data when real data is unavailable"
    )
    download_workers: int = Field(
        default=4,
        description="Concurrent chunk downloads when fetching market data. Downloading is "
        "network-bound, so threads (not processes) are the right tool. Set to 1 to download "
        "strictly one chunk at a time if the data provider rate-limits you.",
    )
    parallel_ticker_prep: bool = Field(
        default=True,
        description="Compute per-ticker indicators, Monte Carlo and features across a CPU "
        "process pool during the live agent run instead of one ticker at a time. Results are "
        "reassembled in universe order, so this changes speed only, never recommendations.",
    )
    ticker_prep_workers: Optional[int] = Field(
        default=None,
        description="Max worker processes for parallel_ticker_prep (default: CPU count).",
    )


class FeaturesConfig(BaseModel):
    """Configuration for feature engineering."""

    lookbacks: Dict[str, int] = Field(
        default_factory=lambda: {
            "short": 5,
            "medium": 20,
            "long": 60,
        },
        description="Lookback periods for different feature types",
    )
    normalize: bool = Field(
        default=True, description="Whether to normalize features"
    )
    normalize_window: int = Field(
        default=252, description="Window size for feature normalization"
    )
    feature_sets: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "price": ["open", "high", "low", "close", "volume"],
            "technical": ["rsi", "macd", "bollinger"],
            "fundamental": ["pe_ratio", "market_cap", "dividend_yield"],
        },
        description="Named sets of features to compute",
    )


class StrategyConfig(BaseModel):
    """Configuration for trading strategies."""

    enabled: bool = Field(default=True, description="Whether strategies are enabled")
    type: str = Field(
        default="rule_based",
        description="Registered strategy type (see strategies/registry.py), e.g. 'rule_based' or 'lstm'",
    )
    module: str = Field(
        default="portfolio_agent.strategies",
        description="Python module path for strategy implementations",
    )
    config_path: str = Field(
        default="config/strategies/trend_breakout.yaml",
        description="Path to strategy configuration file",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    model: str = Field(
        default="lstm",
        description="Registered model architecture (see models/registry.py). 'patchtst' is the "
        "recommended one: it attends over 5-day patches rather than squeezing a 60-day window "
        "through a single LSTM hidden state, and encodes each feature channel with shared weights "
        "so attention cannot fit spurious cross-feature relationships.",
    )
    target: str = Field(
        default="return_5d", description="Target variable for prediction"
    )
    feature_normalization: Literal["global", "cross_sectional"] = Field(
        default="cross_sectional",
        description="How model inputs are standardized before training "
        "(features/scaling.py::apply_cross_sectional_scaling). 'global' fits one mean and "
        "standard deviation per feature over the pooled training rows, which answers 'is this RSI "
        "high for this stock over the sample'. 'cross_sectional' z-scores each feature across the "
        "universe separately on every date, which answers 'is this RSI high relative to what else "
        "I could buy today' — the question a model that chooses between stocks is actually being "
        "asked. The cross-sectional form also strips the market factor out of the inputs, the "
        "same way target_transform strips it out of the label, and cannot leak across dates by "
        "construction since it fits no state. The global scaler still runs afterwards either way: "
        "it is what ships in the checkpoint metadata and guarantees inference reproduces training.",
    )
    target_transform: Literal[
        "absolute", "cross_sectional_demean", "cross_sectional_rank"
    ] = Field(
        default="cross_sectional_rank",
        description="How the forward-return label is measured before training "
        "(agents/trainer.py::apply_cross_sectional_target). 'absolute' predicts the raw forward "
        "return, most of whose variance in an equity panel is the common market factor — which "
        "is both nearly unforecastable and unusable by a long-only book with no index hedge, so "
        "the network spends its capacity on the one component it cannot act on. The two "
        "cross-sectional forms measure each name against the rest of the universe on the same "
        "date, leaving the idiosyncratic part the system actually monetizes by choosing between "
        "stocks. 'cross_sectional_rank' maps to [-1, 1] and is the more robust of the two on "
        "Indian data, where a circuit-limited print dominates the cross-sectional mean but moves "
        "a rank by one place. Falls back to 'absolute' automatically when the universe is too "
        "small to rank. Note this changes what the model predicts, so a checkpoint trained under "
        "one setting should not be scored under another.",
    )
    max_abs_target: float = Field(
        default=5.0,
        gt=0.0,
        description="Largest |forward return| accepted as a training label; rows above it are "
        "dropped (agents/trainer.py::prepare_features). Input features are standardized and "
        "clipped before training but the target never was, and a single bad cached bar is enough "
        "to poison a whole run: one close printed at 0.001 turns a 5-day forward return into "
        "111,300, and one gradient step against a loss that size moves the weights somewhere "
        "every later batch evaluates to NaN. This is why NaN losses outlived the mixed-precision "
        "fix and still appear on CPU — the cause was the label, not fp16. 5.0 (+500%) admits any "
        "genuinely reachable move — five consecutive 20% upper circuits compound to +149% — and "
        "rejects only arithmetic that cannot be a price.",
    )
    sequence_length: int = Field(
        default=60, description="Length of input sequences"
    )
    train_fraction: float = Field(
        default=0.8, description="Fraction of data to use for training"
    )
    batch_size: int = Field(default=128, description="Training batch size (larger for GPU efficiency)")
    epochs: int = Field(default=100, description="Number of training epochs")
    learning_rate: float = Field(default=0.003, description="Learning rate")
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto", description="Device for model training"
    )
    use_mixed_precision: bool = Field(
        default=True, description="Whether to use mixed precision training"
    )
    num_workers: int = Field(
        default=2, description="Number of data loading workers (lower on Windows)"
    )
    use_synthetic_data: bool = Field(
        default=False,
        description="If True, train on generated synthetic OHLCV data instead of real cached tickers "
        "(offline/CI testing only; real training should leave this False)",
    )
    parallel_data_loading: bool = Field(
        default=True,
        description="Load and featurize per-ticker training data across a CPU process pool "
        "instead of sequentially. Recommended when training on the full cached ticker universe.",
    )
    data_load_workers: Optional[int] = Field(
        default=None,
        description="Max worker processes for parallel data loading (default: CPU count).",
    )
    use_torch_compile: bool = Field(
        default=False,
        description="If True, wrap the model with torch.compile() for faster training (PyTorch 2.0+; "
        "biggest benefit on CUDA). Off by default since compile overhead isn't worth it for very "
        "short runs and isn't supported on every platform.",
    )
    walk_forward_splits: int = Field(
        default=5,
        description="Number of expanding-window walk-forward folds used to validate the model "
        "(agents/trainer.py::run_walk_forward_validation). A single chronological 70/15/15 split "
        "measures one regime and one initialization; walk-forward re-trains on an expanding "
        "history and tests on the next contiguous block, so the reported out-of-sample numbers "
        "average over several regimes. Set to 0 to skip validation and train on the single split "
        "only (faster, much weaker evidence).",
    )
    walk_forward_epochs: Optional[int] = Field(
        default=None,
        description="Epoch budget per walk-forward fold. Folds exist to estimate generalization, "
        "not to produce the shipped weights, so this defaults to min(20, training.epochs) to keep "
        "validation affordable; set explicitly to train folds to full length.",
    )
    loss: Literal["quantile", "mse"] = Field(
        default="quantile",
        description="Training objective. 'quantile' fits pinball loss over `quantiles`, so the "
        "model predicts a distribution of the forward return rather than a point. This is the "
        "default because squared error is minimized by the conditional mean, the conditional mean "
        "of a 5-day equity return is nearly constant, and a network trained on MSE therefore "
        "collapses to a near-constant output that validates beautifully and forecasts nothing. "
        "Quantile outputs also give the trigger engine a native confidence interval instead of a "
        "bare number. 'mse' restores the single-output point forecast.",
    )
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="Quantile levels predicted when loss='quantile', in ascending order. The "
        "median is the point forecast; the outer pair is the confidence interval. Levels must lie "
        "strictly inside (0, 1) — the 0th and 100th percentiles of a return distribution are not "
        "estimable from a few years of daily bars.",
    )
    calibrate_confidence: bool = Field(
        default=True,
        description="Fit an isotonic map from raw model score to realized win rate on the "
        "walk-forward test folds (src/calibration.py) and ship it beside the checkpoint. Networks "
        "on noisy financial data are systematically overconfident, and the score feeds Kelly "
        "sizing and the trigger engine's expected-value hurdle — both far more sensitive to an "
        "optimistic probability than to a pessimistic one. Calibration preserves the model's "
        "ranking (which walk-forward actually measured) and discards its scale (which nothing "
        "measured).",
    )
    walk_forward_embargo: int = Field(
        default=0,
        ge=0,
        description="Extra sessions excluded from training *after* each walk-forward test "
        "fold, on top of the purge (validation/purged.py). The purge removes samples whose "
        "label window reaches into the fold — a 5-day label observed daily means the last 5 "
        "training samples are partly made of the test period — and that has always been "
        "applied. The embargo is a different guard: it removes samples immediately after the "
        "fold, whose *features* are nearly identical to the test fold's because financial "
        "series are persistent, so a model that memorizes them recovers test-period "
        "information through the inputs rather than the labels. Defaults to 0 because it "
        "only binds when training data exists on the far side of a fold, which an expanding "
        "window never produces; set it when moving to combinatorial splits.",
    )
    walk_forward_min_train_fraction: float = Field(
        default=0.4,
        description="Fraction of the panel used to train the first walk-forward fold. Subsequent "
        "folds expand into the data the previous fold tested on.",
    )


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    initial_capital: float = Field(
        default=1000000.0, description="Initial capital for backtesting"
    )
    start_years_ago: int = Field(
        default=5, description="How many years ago to start the backtest"
    )
    use_trained_model: bool = Field(
        default=False, description="Whether to use a trained model for backtesting"
    )
    model_path: str = Field(
        default="", description="Path to the trained model file"
    )
    parallel: bool = Field(
        default=False, description="Whether to parallelize per-ticker signal generation across CPU workers"
    )
    max_workers: Optional[int] = Field(
        default=None, description="Max worker processes when parallel=True (default: CPU count)"
    )


class RiskConfig(BaseModel):
    """Configuration for risk management."""

    portfolio_value_inr: float = Field(
        default=308733.0, description="Total portfolio value in INR"
    )
    risk_per_trade_pct: float = Field(
        default=0.01, description="Risk per trade as a percentage"
    )
    risk_free_rate: float = Field(
        default=0.065,
        ge=0.0,
        description="Annualized risk-free rate subtracted from strategy returns before the "
        "Sharpe and Sortino ratios (decimal, so 0.065 is 6.5%). Stated here rather than defaulted "
        "inside RiskAnalyzer so that whatever rate a reported Sharpe was computed against is "
        "visible in the config a reviewer reads. This is a constant over the whole backtest "
        "window, which is wrong for any multi-year run — India's policy rate moved materially "
        "over 2021-2025 — so prefer paths.risk_free_rate_csv, which overrides this with a dated "
        "series.",
    )
    max_single_position_pct: float = Field(
        default=0.03, description="Maximum allocation to a single position"
    )
    atr_stop_multiplier: float = Field(
        default=1.5,
        description="ATR multiple below entry for the stop. **Match this to the signal's own "
        "horizon.** A measured example: cross-sectional momentum forms on a 9-month window, and "
        "at 1.5x ATR its positions exited at a 5-day median holding period with 71% of exits at "
        "the stop, turning a multi-month factor into day-trading and losing 2.0% gross on "
        "deployed capital. At 6.0x the same signal over the same universe and window held for a "
        "94-day median and returned +4.4% gross. The stop was cutting the thesis short, not "
        "protecting it. 1.5 is retained as the default because it suits the per-ticker "
        "trend/breakout strategy this platform started with; raise it for anything with a "
        "formation window measured in months.",
    )
    atr_target_multiplier: float = Field(
        default=2.0,
        description="ATR multiple above entry for the target. Scale it with atr_stop_multiplier "
        "— the ratio between them is the gross reward:risk every signal is screened on.",
    )
    use_kelly_sizing: bool = Field(
        default=False,
        description="If True, size positions with fractional-Kelly once enough realized trade "
        "history exists (see risk.py::calculate_kelly_quantity), falling back to fixed-fractional "
        "sizing otherwise. Off by default since it needs real trade history to be meaningful.",
    )
    kelly_fraction: float = Field(
        default=0.25,
        ge=0.0,
        description="Fractional-Kelly multiplier kappa. Values above 0.25 are silently clamped to "
        "quarter-Kelly by risk.py::calculate_kelly_quantity — the ceiling lives at the point of "
        "use, so no config, env override or fixture can size above it, and an existing config "
        "carrying the old 0.5 default still loads. Kelly assumes a well-estimated, roughly "
        "symmetric payoff distribution and Indian small/mid-caps offer neither: a 'loss' that "
        "locks at the lower circuit for days realizes far worse than the modelled stop, which "
        "inflates the payoff ratio b and f* with it. Quarter-Kelly captures roughly half of "
        "full-Kelly's growth rate at a small fraction of its drawdown risk.",
    )
    kelly_min_trades: int = Field(
        default=50,
        description="Minimum realized (WIN/LOSS) trades required before Kelly sizing is trusted; "
        "below this, sizing falls back to fixed-fractional. 50 is a floor, not a target: at "
        "smaller samples the win-rate standard error is ~5-7 percentage points, and Kelly "
        "penalizes over-betting off an optimistic estimate far more than under-betting.",
    )
    kelly_shrinkage_strength: float = Field(
        default=20.0,
        description="Beta-prior strength (in pseudo-trades) used to shrink the realized win rate "
        "toward 0.5 before it is fed to Kelly (see risk.py::shrink_win_probability). 20 means a "
        "coin-flip prior worth 20 trades of evidence; 0 disables shrinkage and uses the raw rate.",
    )
    portfolio_volatility_target: float = Field(
        default=0.0,
        ge=0.0,
        description="Annualized volatility ceiling for the whole book, enforced at order time "
        "against a shrunk covariance estimate (src/portfolio.py). 0 disables the constraint but "
        "NOT the measurement — book risk is reported either way. This is the only limit here that "
        "is not per-position: risk-per-trade, max_single_position_pct and max_sector_pct are all "
        "blind to the fact that twenty 3% positions correlated 0.6 carry ~3.5x the volatility of "
        "twenty independent ones, and Indian equity correlations run to 0.6-0.85 in exactly the "
        "drawdowns the circuit breaker exists to survive. Left off by default because choosing a "
        "volatility target is a risk-policy decision, not a defect fix; 0.15-0.25 is the usual "
        "range for a long-only equity book.",
    )
    covariance_lookback_days: int = Field(
        default=252,
        ge=60,
        description="Trailing window, in trading days, used to estimate the covariance behind "
        "portfolio_volatility_target. Only returns dated strictly before the decision date enter "
        "it. One year balances responsiveness against the conditioning problem: over a shorter "
        "window the covariance of 20-60 names is dominated by estimation noise, which is what the "
        "Ledoit-Wolf shrinkage is there to contain.",
    )
    max_sector_pct: float = Field(
        default=0.25,
        description="Maximum share of portfolio value allowed in any single sector "
        "(see src/sectors.py). Requires a ticker,sector CSV at paths.sector_map_csv; without "
        "one every holding counts as UNKNOWN and the cap applies to that single pooled bucket. "
        "Set to 0 (or >= 1) to disable.",
    )
    max_unknown_sector_pct: float = Field(
        default=0.30,
        description="Aggregate cap on holdings whose ticker is missing from the sector map, as a "
        "fraction of portfolio value. Wider than max_sector_pct because the pool spans many real "
        "sectors, but finite: Indian sector maps are chronically incomplete in exactly the "
        "small/micro-cap segment where concentration risk is worst, so an exempt pool would let "
        "the entire book concentrate there and satisfy every cap. Only applies when a map is "
        "loaded — with no map at all every holding is unmapped and the cap is inactive instead.",
    )
    max_portfolio_drawdown_pct: float = Field(
        default=0.15,
        description="Circuit breaker: once peak-to-trough portfolio drawdown reaches this "
        "fraction, no new BUY orders are created until equity recovers to within "
        "drawdown_reentry_pct of the peak. Existing positions keep their stops and targets. "
        "Set to 0 (or >= 1) to disable.",
    )
    drawdown_reentry_pct: float = Field(
        default=0.10,
        description="Drawdown level at which the circuit breaker re-arms and buying resumes. "
        "Kept below max_portfolio_drawdown_pct so the breaker cannot flicker on and off at "
        "the trip point.",
    )
    drawdown_halt_max_days: int = Field(
        default=60,
        description="Trading days after which a halted drawdown breaker re-arms regardless of "
        "recovery, resetting the equity peak to current equity. Recovery-only re-arming "
        "deadlocks and does so silently: the breaker halts buying, the open positions exit "
        "through their own stops, the book is now all cash — and cash cannot appreciate back "
        "toward a peak it is measured against, so the halt is permanent. A 5-year backtest hit "
        "exactly this, tripping in month 7 and sitting in cash for four years, which reads in "
        "the report as a flat equity curve rather than a stuck flag. Set to 0 to disable the "
        "cooldown and restore recovery-only behaviour.",
    )
    exit_on_lower_circuit_lock: bool = Field(
        default=True,
        description="Queue an immediate exit when a holding closes pinned at its lower circuit. "
        "The modelled stop assumes a fill is available near it; on a lock there is no bid, so "
        "waiting for the stop means holding through however many further locked sessions it takes "
        "to find one. Every one of those realizes a loss the position sizing never priced — the "
        "same asymmetry that biases the payoff ratio Kelly estimates from. The exit is queued for "
        "the next session, the earliest a real order could work.",
    )
    liquidate_on_drawdown_halt: bool = Field(
        default=False,
        description="Also sell every open position when the drawdown circuit breaker trips, "
        "instead of only suppressing new BUYs. Off by default: open positions already carry stops "
        "and targets, and force-liquidating an entire book at a drawdown trough is how a bad "
        "quarter becomes a permanent loss. Turn it on for mandates where a hard equity floor "
        "outranks recovery potential.",
    )
    slippage_pct_per_side: float = Field(
        default=0.0025,
        description="Assumed per-side slippage, as a fraction of turnover, used when charging "
        "estimated round-trip costs against a strategy's reward:risk before a quantity exists "
        "(src/execution_sim.py::cost_fraction_per_side). Realized fills price slippage off each "
        "ticker's own ATR and traded volume instead; this only affects signal gating.",
    )

    @model_validator(mode="after")
    def _check_drawdown_thresholds(self) -> "RiskConfig":
        """The breaker needs distinct trip and re-arm levels to be a breaker.

        With reentry >= max, equity sitting just past the trip point satisfies
        both conditions, so the breaker halts on one bar and resumes on the
        next, churning the book at the drawdown trough — the exact behaviour
        two thresholds exist to prevent.
        """
        if self.max_portfolio_drawdown_pct <= 0:
            return self
        if self.drawdown_reentry_pct >= self.max_portfolio_drawdown_pct:
            raise ValueError(
                f"risk.drawdown_reentry_pct ({self.drawdown_reentry_pct}) must be below "
                f"risk.max_portfolio_drawdown_pct ({self.max_portfolio_drawdown_pct}); "
                f"otherwise the circuit breaker trips and re-arms on alternating bars."
            )
        return self


class LearningConfig(BaseModel):
    """Configuration for the self-learning weight adaptation."""

    learning_rate: float = Field(
        default=0.15, description="Rate at which strategy component weights adapt to realized win rate"
    )
    min_trades_for_learning: int = Field(
        default=5, description="Minimum number of realized trades required before weights are adjusted"
    )
    min_trades_per_component: int = Field(
        default=30,
        description="Minimum realized trades attributed to a single component before that "
        "component's weight may move. The overall floor above is not a substitute: with only a "
        "total-trade floor, a component credited with three trades could move on the strength of "
        "a fifty-trade sample it barely contributed to. At 30 trades a win rate still carries a "
        "~9 percentage point standard error, so this is a floor, not a comfort.",
    )
    shrinkage_strength: float = Field(
        default=20.0,
        description="Beta-prior strength (in pseudo-trades) used to shrink a component's realized "
        "win rate toward 0.5 before it moves that component's weight — the same prior the Kelly "
        "path applies in src/risk.py. Weight adaptation is a feedback loop (weights change which "
        "trades are taken, which changes the outcomes the next adaptation sees), so an unshrunk "
        "win rate makes noise self-reinforcing. 0 disables shrinkage.",
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="One-sided alpha a component's win rate must clear, on an exact binomial test "
        "against a coin flip, before its weight moves at all. Without it weights moved on every "
        "evaluation whether or not the difference was distinguishable from zero. Set to 1.0 to "
        "adapt on every evaluation regardless of significance.",
    )


class SimulationConfig(BaseModel):
    """Configuration for per-symbol Monte Carlo forward simulation (feeds strategy scoring)."""

    mc_horizon_days: int = Field(default=20, description="Forward simulation horizon in trading days")
    mc_simulations: int = Field(default=1000, description="Number of Monte Carlo simulation paths")
    random_seed: int = Field(default=42, description="Random seed for reproducible simulations")
    method: Literal["gaussian", "block_bootstrap", "jump_diffusion"] = Field(
        default="block_bootstrap",
        description="Shock-generating process for the forward simulation (src/monte_carlo.py). "
        "'gaussian' draws i.i.d. normal shocks (thin-tailed, no serial dependence — understates "
        "tail risk); 'block_bootstrap' resamples contiguous blocks of the ticker's own returns, "
        "preserving both the empirical fat tails and volatility clustering; 'jump_diffusion' adds "
        "a compound-Poisson jump component to the Gaussian diffusion for gap/circuit-limit risk. "
        "Defaults to block_bootstrap as the most faithful to Indian equity return behaviour.",
    )
    block_size_days: int = Field(
        default=5,
        description="Mean block length for the stationary block bootstrap. Blocks are drawn with "
        "geometric lengths averaging this many days, so serial dependence within a block survives "
        "resampling while the resulting series stays stationary.",
    )
    jump_intensity_per_year: float = Field(
        default=12.0,
        description="Expected number of jumps per year for method='jump_diffusion' (Merton "
        "lambda). 12 ~ one policy/earnings/flow shock a month.",
    )
    jump_mean: float = Field(
        default=-0.02,
        description="Mean jump size in log-return terms for method='jump_diffusion'. Negative by "
        "default: equity jump risk is asymmetric, and downside gaps are what a risk model needs "
        "to capture.",
    )
    jump_volatility: float = Field(
        default=0.05,
        description="Standard deviation of jump size in log-return terms for method='jump_diffusion'.",
    )
    separate_overnight_gaps: bool = Field(
        default=True,
        description="When GARCH volatility is enabled, fit the recursion to intraday session "
        "returns (close/open) and add overnight gap risk (open/prev_close) as a separate "
        "component, instead of feeding it close-to-close returns. NSE opens after both the US "
        "close and the Asian session, so gaps are frequent and large; attributing them to the "
        "previous session's shock inflates alpha/gamma and destabilizes the persistence estimate. "
        "Falls back to close-to-close GARCH when open prices or history are unavailable.",
    )
    use_garch_volatility: bool = Field(
        default=False,
        description="If True, forecast each ticker's forward volatility with GJR-GARCH(1,1) "
        "(src/volatility_models.py) instead of assuming a flat historical standard deviation. "
        "Falls back to the flat assumption automatically when there isn't enough history to fit "
        "GARCH reliably. Off by default since per-ticker GARCH fitting is much slower than the "
        "closed-form flat-vol path.",
    )
    prior_annual_drift_std: float = Field(
        default=0.10,
        ge=0.0,
        description="Prior standard deviation (annualized, in log-return terms) of the "
        "cross-sectional spread of *true* drifts, used to shrink each ticker's estimated drift "
        "toward zero before simulating (src/monte_carlo.py::shrink_drift). The sample mean of "
        "daily returns has a standard error of sigma/sqrt(T) — roughly 14% a year for a 2%/day "
        "name over five years — so an unshrunk drift makes probability-of-profit mostly "
        "estimation noise: at T=1250 and a 20-day horizon, 16% of tickers with exactly zero true "
        "drift clear a 0.55 gate on noise alone. The rate is 1 - Phi(Phi^-1(gate) * sqrt(T/H)) — "
        "sigma cancels — so only the history length and the horizon move it; see "
        "docs/QUANT_RESEARCH.md section 21. Raise this toward infinity to recover the raw sample "
        "mean; set it to 0 to credit no ticker with any drift edge at all.",
    )
    use_empirical_drift_prior: bool = Field(
        default=True,
        description="If True, estimate the drift prior from the cross-section of the active "
        "universe each scoring round (src/monte_carlo.py::estimate_cross_sectional_drift_prior) "
        "instead of using the fixed prior_annual_drift_std, and shrink each ticker toward the "
        "universe mean rather than toward zero. The method of moments splits the observed spread "
        "of sample means into true dispersion and estimation noise — tau^2 = max(0, Var(mu_hat) - "
        "mean(sigma_i^2/T_i)) — so the amount of shrinkage is measured rather than assumed. Falls "
        "back to the fixed prior automatically when the usable cross-section is too thin to "
        "estimate two moments from. Set False to restore the fixed-prior behaviour.",
    )
    propagate_drift_uncertainty: bool = Field(
        default=True,
        description="If True, each simulated path draws its own drift from the posterior instead "
        "of sharing the posterior mean, so probability_profit is the posterior *predictive* "
        "probability — the one that accounts for the drift being estimated rather than known. "
        "Set False to reproduce the older plug-in behaviour, which reports a confident number "
        "about the least reliable input in the simulation.",
    )


class ComplianceConfig(BaseModel):
    """Configuration for trade compliance/eligibility checks."""

    min_price_inr: float = Field(default=20.0, description="Minimum share price to be eligible")
    target_prob_profit: float = Field(
        default=0.55,
        description="Minimum Monte Carlo probability-of-profit required for a BUY signal. Left at "
        "0.55 through the Ito fix (docs/QUANT_RESEARCH.md section 14.1) rather than re-tuned, and "
        "the choice was measured rather than assumed. Removing that bias raises every probability, "
        "so on 212 cached NSE names the share clearing this gate went from 0.5% to 3.3% and the "
        "median probability from 0.466 to 0.495. It was kept because the gate's job is rejecting "
        "noise and that is unchanged — on a zero-drift universe under the default empirical drift "
        "prior, 0.0% clear it both before and after — while re-tuning to restore the old pass rate "
        "would preserve the effect of an arithmetic error after removing its cause. 0.55 now means "
        "what it says; the bias previously made it silently demand about 0.58. Set 0.60 to "
        "reproduce the pre-fix selectivity if continuity matters more.",
    )
    min_reward_risk: float = Field(
        default=1.2,
        description="Minimum reward:risk ratio required for a BUY signal, measured NET of "
        "estimated round-trip friction (src/risk.py::net_reward_risk) rather than gross. "
        "Lowered from 1.5 when costing was introduced: the same trade scores roughly 0.6x its "
        "gross ratio once ~0.79% of round-trip cost is charged against an ATR-scale move, so "
        "the old threshold was unreachable for any ATR-derived stop/target.",
    )
    paper_trading_mode: bool = Field(
        default=True, description="Must remain True; this system never places real trades"
    )


class PathsConfig(BaseModel):
    """Configuration for file/directory locations."""

    brain_file: str = Field(default="data/agent_brain.json", description="Path to the agent's learned weights JSON")
    sqlite_path: str = Field(default="data/portfolio_agent.db", description="Path to the SQLite state database")
    excel_output: str = Field(
        default="output/Agent_Orchestrator_Output.xlsx", description="Path to the live-agent Excel report"
    )
    backtest_excel_output: str = Field(
        default="output/Backtest_Report.xlsx", description="Path to the backtest Excel report"
    )
    sector_map_csv: str = Field(
        default="data/sector_map.csv",
        description="CSV mapping tickers to sectors (columns: ticker,sector) used to enforce "
        "risk.max_sector_pct. Optional, but note what 'absent' means: with no map at all the "
        "sector cap is reported INACTIVE and both engines log a warning, rather than falling back "
        "to capping a single pooled UNKNOWN bucket — that fallback would constrain total invested "
        "capital instead of sector concentration, leaving most of the portfolio in cash forever. "
        "A partial map gives each mapped sector max_sector_pct and the unmapped pool its own "
        "max_unknown_sector_pct.",
    )
    risk_free_rate_csv: str = Field(
        default="data/risk_free_rate.csv",
        description="Optional CSV of the dated risk-free rate — columns date,annualized_yield — "
        "typically the 91-day T-bill. When the file exists it is aligned to the return index, "
        "forward-filled across non-trading days and de-annualized, so the excess return is "
        "computed day by day against the rate that actually prevailed. When it is absent, "
        "risk.risk_free_rate is used as a constant across the whole window and the run logs that "
        "it did so. Values may be given as decimals (0.068) or percent (6.8); both are read "
        "correctly.",
    )
    trial_log: str = Field(
        default="output/trials.jsonl",
        description="Append-only JSONL log of every backtest configuration tried and the Sharpe "
        "it produced (src/performance_stats.py). This is what makes the Deflated Sharpe Ratio "
        "computable: DSR adjusts a reported Sharpe for the number of trials behind it, and N is "
        "exactly the quantity a research process forgets. Search enough configurations of a "
        "strategy with no edge and the best one still prints a respectable Sharpe; without the "
        "count, there is no way to tell that from a real result.",
    )
    log_file: str = Field(default="logs/agent.log", description="Path to the log file")
    log_dir: str = Field(default="logs", description="Directory for log files")
    output_dir: str = Field(default="output", description="Directory for generated reports")


class AppConfig(BaseModel):
    """Root application configuration composing all sub-configurations."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
