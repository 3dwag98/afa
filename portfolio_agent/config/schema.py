"""Configuration schema for the Autonomous Financial Advisor (AFA) portfolio agent."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data paths and settings."""

    data_dir: str = Field(default="data", description="Base directory for data files")
    market_data_dir: str = Field(
        default="data/market_data", description="Directory for market data"
    )
    default_history_years: int = Field(
        default=5, description="Default number of years of historical data to use"
    )
    universe_size: int = Field(
        default=10, description="Number of securities in the trading universe"
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

    model: str = Field(default="lstm", description="Model architecture to use")
    target: str = Field(
        default="return_5d", description="Target variable for prediction"
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
    max_single_position_pct: float = Field(
        default=0.03, description="Maximum allocation to a single position"
    )
    use_kelly_sizing: bool = Field(
        default=False,
        description="If True, size positions with fractional-Kelly once enough realized trade "
        "history exists (see risk.py::calculate_kelly_quantity), falling back to fixed-fractional "
        "sizing otherwise. Off by default since it needs real trade history to be meaningful.",
    )
    kelly_fraction: float = Field(
        default=0.5,
        description="Fractional-Kelly multiplier kappa in [0, 1] (0.5 = half-Kelly, the common "
        "practitioner default that captures ~75% of full-Kelly's growth rate at lower drawdown risk).",
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
    max_sector_pct: float = Field(
        default=0.25,
        description="Maximum share of portfolio value allowed in any single sector "
        "(see src/sectors.py). Requires a ticker,sector CSV at paths.sector_map_csv; without "
        "one every holding counts as UNKNOWN and the cap applies to that single pooled bucket. "
        "Set to 0 (or >= 1) to disable.",
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
    slippage_pct_per_side: float = Field(
        default=0.0025,
        description="Assumed per-side slippage, as a fraction of turnover, used when charging "
        "estimated round-trip costs against a strategy's reward:risk before a quantity exists "
        "(src/execution_sim.py::cost_fraction_per_side). Realized fills price slippage off each "
        "ticker's own ATR and traded volume instead; this only affects signal gating.",
    )


class LearningConfig(BaseModel):
    """Configuration for the self-learning weight adaptation."""

    learning_rate: float = Field(
        default=0.15, description="Rate at which strategy component weights adapt to realized win rate"
    )
    min_trades_for_learning: int = Field(
        default=5, description="Minimum number of realized trades required before weights are adjusted"
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
    use_garch_volatility: bool = Field(
        default=False,
        description="If True, forecast each ticker's forward volatility with GJR-GARCH(1,1) "
        "(src/volatility_models.py) instead of assuming a flat historical standard deviation. "
        "Falls back to the flat assumption automatically when there isn't enough history to fit "
        "GARCH reliably. Off by default since per-ticker GARCH fitting is much slower than the "
        "closed-form flat-vol path.",
    )


class ComplianceConfig(BaseModel):
    """Configuration for trade compliance/eligibility checks."""

    min_price_inr: float = Field(default=20.0, description="Minimum share price to be eligible")
    target_prob_profit: float = Field(
        default=0.55, description="Minimum Monte Carlo probability-of-profit required for a BUY signal"
    )
    min_reward_risk: float = Field(default=1.5, description="Minimum reward:risk ratio required for a BUY signal")
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
        "risk.max_sector_pct. Optional — when absent, every holding is pooled into a single "
        "UNKNOWN sector and the cap applies to that pool.",
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
