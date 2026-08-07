"""Configuration module for portfolio agent."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache

import yaml


REQUIRED_FIELDS = [
    "portfolio_value_inr",
    "risk_per_trade_pct",
    "max_single_position_pct",
    "min_price_inr",
    "target_prob_profit",
    "min_reward_risk",
    "learning_rate",
    "min_trades_for_learning",
    "mc_horizon_days",
    "mc_simulations",
    "random_seed",
    "tickers",
    "brain_file",
    "sqlite_path",
    "excel_output",
    "log_file",
    "paper_trading_mode",
    "min_history_days",
]


@dataclass
class AppConfig:
    """Application configuration dataclass."""

    portfolio_value_inr: float
    risk_per_trade_pct: float
    max_single_position_pct: float
    min_price_inr: float
    target_prob_profit: float
    min_reward_risk: float
    learning_rate: float
    min_trades_for_learning: int
    mc_horizon_days: int
    mc_simulations: int
    random_seed: int
    tickers: List[str]
    brain_file: str
    sqlite_path: str
    excel_output: str
    log_file: str
    paper_trading_mode: bool
    min_history_days: int
    allow_synthetic_fallback: bool = True
    # Scheduler settings (Apache Airflow)
    scheduler_enabled: bool = False
    schedule_time_ist: str = "15:45"
    schedule_outcome_time_ist: str = "16:00"
    airflow_ui_enabled: bool = True
    airflow_webserver_port: int = 8080
    airflow_timezone: str = "Asia/Kolkata"

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create AppConfig from dictionary."""
        return cls(
            portfolio_value_inr=data["portfolio_value_inr"],
            risk_per_trade_pct=data["risk_per_trade_pct"],
            max_single_position_pct=data["max_single_position_pct"],
            min_price_inr=data["min_price_inr"],
            target_prob_profit=data["target_prob_profit"],
            min_reward_risk=data["min_reward_risk"],
            learning_rate=data["learning_rate"],
            min_trades_for_learning=data["min_trades_for_learning"],
            mc_horizon_days=data["mc_horizon_days"],
            mc_simulations=data["mc_simulations"],
            random_seed=data["random_seed"],
            tickers=data["tickers"],
            brain_file=data["brain_file"],
            sqlite_path=data["sqlite_path"],
            excel_output=data["excel_output"],
            log_file=data["log_file"],
            paper_trading_mode=data["paper_trading_mode"],
            min_history_days=data["min_history_days"],
            allow_synthetic_fallback=data.get("allow_synthetic_fallback", True),
            scheduler_enabled=data.get("scheduler_enabled", False),
            schedule_time_ist=data.get("schedule_time_ist", "15:45"),
            schedule_outcome_time_ist=data.get("schedule_outcome_time_ist", "16:00"),
            airflow_ui_enabled=data.get("airflow_ui_enabled", True),
            airflow_webserver_port=data.get("airflow_webserver_port", 8080),
            airflow_timezone=data.get("airflow_timezone", "Asia/Kolkata"),
        )


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in the same directory as this module or project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or invalid")

    return config


def _validate_config(config: dict) -> None:
    """Validate that all required fields are present."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Get cached application configuration.

    Args:
        config_path: Optional path to config.yaml. If None, uses default location.

    Returns:
        AppConfig instance with validated configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is missing required fields.
    """
    config_dict = _load_config(config_path)
    _validate_config(config_dict)
    return AppConfig.from_dict(config_dict)


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing)."""
    get_config.cache_clear()
