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
