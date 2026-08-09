"""Tests for configuration module."""

import os

import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.config.schema import AppConfig


class TestConfigLoading:
    """Test suite for configuration loading."""

    def test_config_loads_successfully(self):
        """Test that config loads from the default location."""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.risk.portfolio_value_inr > 0

    def test_config_has_all_sections(self):
        """Test that all configuration sections are present."""
        config = load_config()

        assert hasattr(config, "data")
        assert hasattr(config, "features")
        assert hasattr(config, "strategy")
        assert hasattr(config, "training")
        assert hasattr(config, "backtest")
        assert hasattr(config, "risk")
        assert hasattr(config, "learning")
        assert hasattr(config, "simulation")
        assert hasattr(config, "compliance")
        assert hasattr(config, "paths")

    def test_paper_trading_mode_is_true(self):
        """Test that paper trading mode is enabled by default."""
        config = load_config()
        assert config.compliance.paper_trading_mode is True

    def test_strategy_config_path_exists(self):
        """The default strategy YAML referenced by config.yaml must actually exist."""
        config = load_config()
        candidate = config.strategy.config_path
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        workspace_root = os.path.dirname(package_root)
        assert os.path.exists(os.path.join(workspace_root, candidate)) or os.path.exists(
            os.path.join(package_root, candidate)
        )


class TestAppConfigDefaults:
    """Test AppConfig default construction and validation."""

    def test_default_construction(self):
        """AppConfig should construct with sensible defaults with no input."""
        config = AppConfig()
        assert config.risk.portfolio_value_inr > 0
        assert config.compliance.paper_trading_mode is True
        assert config.learning.learning_rate > 0
        assert config.simulation.mc_simulations > 0

    def test_validate_from_nested_dict(self):
        """AppConfig should validate a nested dict matching the schema."""
        valid_dict = {
            "risk": {"portfolio_value_inr": 500000, "risk_per_trade_pct": 0.02},
            "data": {"tickers": ["RELIANCE.NS"]},
            "compliance": {"paper_trading_mode": True},
        }

        config = AppConfig.model_validate(valid_dict)
        assert config.risk.portfolio_value_inr == 500000
        assert config.data.tickers == ["RELIANCE.NS"]
        assert config.compliance.paper_trading_mode is True


class TestConfigEnvOverrides:
    """Test environment variable overrides via the AFA_ prefix."""

    def test_env_override_applied(self, monkeypatch):
        monkeypatch.setenv("AFA_RISK__PORTFOLIO_VALUE_INR", "999999")
        config = load_config()
        assert config.risk.portfolio_value_inr == 999999.0

    def test_missing_config_file_uses_defaults(self):
        """A nonexistent config path should fall back to schema defaults, not raise."""
        config = load_config(path="/nonexistent/path/config.yaml")
        assert isinstance(config, AppConfig)
