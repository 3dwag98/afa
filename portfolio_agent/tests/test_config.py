"""Tests for configuration module."""

import os
import tempfile
import pytest
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_config, clear_config_cache, _validate_config, AppConfig


class TestConfigLoading:
    """Test suite for configuration loading."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_config_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_config_cache()

    def test_config_loads_successfully(self):
        """Test that config loads from default location."""
        config = get_config()
        assert config is not None
        assert isinstance(config.portfolio_value_inr, (int, float))
        assert config.portfolio_value_inr > 0

    def test_default_tickers_exist(self):
        """Test that default tickers are present and valid."""
        config = get_config()
        assert config.tickers is not None
        assert len(config.tickers) > 0
        
        # Check expected tickers are present
        expected_tickers = ["NIFTYBEES.NS", "RELIANCE.NS", "TCS.NS", 
                          "HDFCBANK.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS"]
        
        for ticker in expected_tickers:
            assert ticker in config.tickers, f"Expected ticker {ticker} not found"

    def test_config_has_required_fields(self):
        """Test that all required fields are loaded."""
        config = get_config()
        
        assert hasattr(config, 'portfolio_value_inr')
        assert hasattr(config, 'risk_per_trade_pct')
        assert hasattr(config, 'max_single_position_pct')
        assert hasattr(config, 'min_price_inr')
        assert hasattr(config, 'target_prob_profit')
        assert hasattr(config, 'min_reward_risk')
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'mc_horizon_days')
        assert hasattr(config, 'mc_simulations')
        assert hasattr(config, 'random_seed')
        assert hasattr(config, 'tickers')
        assert hasattr(config, 'brain_file')
        assert hasattr(config, 'sqlite_path')
        assert hasattr(config, 'excel_output')
        assert hasattr(config, 'log_file')
        assert hasattr(config, 'paper_trading_mode')
        assert hasattr(config, 'min_history_days')

    def test_paper_trading_mode_is_true(self):
        """Test that paper trading mode is enabled by default."""
        config = get_config()
        assert config.paper_trading_mode is True


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_invalid_config_raises_error(self):
        """Test that missing required fields raise ValueError."""
        invalid_config = {
            "portfolio_value_inr": 100000,
            # Missing other required fields
        }
        
        with pytest.raises(ValueError) as exc_info:
            _validate_config(invalid_config)
        
        assert "Missing required configuration fields" in str(exc_info.value)

    def test_empty_config_raises_error(self):
        """Test that empty config raises ValueError."""
        with pytest.raises(ValueError):
            _validate_config({})


class TestAppConfigFromDict:
    """Test AppConfig creation from dictionary."""

    def test_from_dict_creates_valid_config(self):
        """Test creating AppConfig from valid dictionary."""
        valid_dict = {
            "portfolio_value_inr": 308733,
            "risk_per_trade_pct": 0.01,
            "max_single_position_pct": 0.03,
            "min_price_inr": 20,
            "target_prob_profit": 0.55,
            "min_reward_risk": 1.5,
            "learning_rate": 0.15,
            "min_trades_for_learning": 5,
            "mc_horizon_days": 20,
            "mc_simulations": 1000,
            "random_seed": 42,
            "tickers": ["RELIANCE.NS"],
            "brain_file": "data/brain.json",
            "sqlite_path": "data/test.db",
            "excel_output": "output/test.xlsx",
            "log_file": "logs/test.log",
            "paper_trading_mode": True,
            "min_history_days": 250,
        }
        
        config = AppConfig.from_dict(valid_dict)
        assert config.portfolio_value_inr == 308733
        assert config.tickers == ["RELIANCE.NS"]
        assert config.paper_trading_mode is True


class TestConfigFileNotFound:
    """Test behavior when config file is missing."""

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        clear_config_cache()
        
        with pytest.raises(FileNotFoundError):
            get_config(config_path="/nonexistent/path/config.yaml")
