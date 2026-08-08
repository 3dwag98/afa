"""Configuration schema for the Autonomous Financial Advisor (AFA) portfolio agent."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

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
    module: str = Field(
        default="portfolio_agent.strategies",
        description="Python module path for strategy implementations",
    )
    config_path: str = Field(
        default="strategies/default.yaml",
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


class AppConfig(BaseModel):
    """Root application configuration composing all sub-configurations."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
