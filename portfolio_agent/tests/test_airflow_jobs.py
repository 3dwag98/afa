"""Tests for airflow_jobs module."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRunDailyJob:
    """Tests for run_daily_job function."""

    @patch('src.airflow_jobs.get_config')
    @patch('src.airflow_jobs.run_orchestrator')
    def test_run_daily_job_calls_mocked_orchestrator(self, mock_run_orchestrator, mock_get_config):
        """Test that run_daily_job calls mocked orchestrator."""
        from src.airflow_jobs import run_daily_job

        # Setup mocks
        mock_config = MagicMock()
        mock_config.paper_trading_mode = True
        mock_get_config.return_value = mock_config

        mock_run_orchestrator.return_value = "/path/to/output.xlsx"

        # Call function
        result = run_daily_job()

        # Assertions
        assert result["status"] == "success"
        assert result["job"] == "DAILY_RUN"
        assert result["excel_path"] == "/path/to/output.xlsx"
        mock_run_orchestrator.assert_called_once_with(force_refresh=False)

    @patch('src.airflow_jobs.get_config')
    def test_run_daily_job_raises_error_if_paper_trading_false(self, mock_get_config):
        """Test that run_daily_job raises error if paper_trading_mode=false."""
        from src.airflow_jobs import run_daily_job

        # Setup mock with paper_trading_mode = False
        mock_config = MagicMock()
        mock_config.paper_trading_mode = False
        mock_get_config.return_value = mock_config

        # Assert RuntimeError is raised
        with pytest.raises(RuntimeError) as exc_info:
            run_daily_job()

        assert str(exc_info.value) == "Live trading is disabled. Set paper_trading_mode=true."


class TestRunRelearnJob:
    """Tests for run_relearn_job function."""

    @patch('src.airflow_jobs.get_config')
    @patch('src.airflow_jobs.load_brain')
    @patch('src.airflow_jobs.save_brain')
    @patch('src.airflow_jobs.get_trade_history')
    @patch('src.airflow_jobs.init_db')
    @patch('src.airflow_jobs.evaluate_and_learn')
    def test_run_relearn_job_calls_mocked_learning_function(
        self, mock_evaluate_and_learn, mock_init_db, mock_get_trade_history,
        mock_save_brain, mock_load_brain, mock_get_config
    ):
        """Test that run_relearn_job calls mocked learning function."""
        from src.airflow_jobs import run_relearn_job

        # Setup mocks
        mock_config = MagicMock()
        mock_config.paper_trading_mode = True
        mock_config.sqlite_path = "/tmp/test.db"
        mock_config.brain_file = "/tmp/brain.json"
        mock_get_config.return_value = mock_config

        mock_brain = MagicMock()
        mock_brain.trade_history = []
        mock_brain.weights = {"Trend": 25.0, "Breakout": 25.0}
        mock_brain.learning_log = [{"entry": "Learning complete"}]
        mock_load_brain.return_value = mock_brain

        mock_get_trade_history.return_value = []

        mock_updated_brain = MagicMock()
        mock_updated_brain.weights = {"Trend": 26.0, "Breakout": 24.0}
        mock_updated_brain.learning_log = [{"entry": "Learning complete"}]
        mock_evaluate_and_learn.return_value = mock_updated_brain

        # Call function
        result = run_relearn_job()

        # Assertions
        assert result["status"] == "success"
        assert result["job"] == "RELEARN"
        assert result["weights"] == {"Trend": 26.0, "Breakout": 24.0}

        mock_init_db.assert_called_once_with("/tmp/test.db")
        mock_load_brain.assert_called_once_with("/tmp/brain.json")
        mock_evaluate_and_learn.assert_called_once()
        mock_save_brain.assert_called_once()

    @patch('src.airflow_jobs.get_config')
    def test_run_relearn_job_raises_error_if_paper_trading_false(self, mock_get_config):
        """Test that run_relearn_job raises error if paper_trading_mode=false."""
        from src.airflow_jobs import run_relearn_job

        # Setup mock with paper_trading_mode = False
        mock_config = MagicMock()
        mock_config.paper_trading_mode = False
        mock_get_config.return_value = mock_config

        # Assert RuntimeError is raised
        with pytest.raises(RuntimeError) as exc_info:
            run_relearn_job()

        assert str(exc_info.value) == "Live trading is disabled. Set paper_trading_mode=true."


class TestRunUpdateOutcomesJob:
    """Tests for run_update_outcomes_job function."""

    @patch('src.airflow_jobs.get_config')
    @patch('src.airflow_jobs.init_db')
    @patch('src.airflow_jobs.get_open_trades')
    def test_run_update_outcomes_job_no_open_trades(
        self, mock_get_open_trades, mock_init_db, mock_get_config
    ):
        """Test run_update_outcomes_job when there are no open trades."""
        from src.airflow_jobs import run_update_outcomes_job

        # Setup mocks
        mock_config = MagicMock()
        mock_config.paper_trading_mode = True
        mock_config.sqlite_path = "/tmp/test.db"
        mock_get_config.return_value = mock_config

        mock_get_open_trades.return_value = []

        # Call function
        result = run_update_outcomes_job()

        # Assertions
        assert result["status"] == "success"
        assert result["job"] == "UPDATE_OUTCOMES"
        assert result["updated_outcomes"] == 0

    @patch('src.airflow_jobs.get_config')
    @patch('src.airflow_jobs.init_db')
    @patch('src.airflow_jobs.get_open_trades')
    @patch('src.airflow_jobs.update_outcomes_from_market')
    def test_run_update_outcomes_job_handles_missing_module_gracefully(
        self, mock_update_outcomes_from_market, mock_get_open_trades, mock_init_db, mock_get_config
    ):
        """Test that run_update_outcomes_job handles missing outcome module gracefully."""
        from src.airflow_jobs import run_update_outcomes_job

        # Setup mocks
        mock_config = MagicMock()
        mock_config.paper_trading_mode = True
        mock_config.sqlite_path = "/tmp/test.db"
        mock_get_config.return_value = mock_config

        # Mock an open trade
        mock_trade = MagicMock()
        mock_trade.symbol = "TEST.NS"
        mock_get_open_trades.return_value = [mock_trade]

        # Make update_outcomes_from_market raise AttributeError (simulating missing module)
        mock_update_outcomes_from_market.side_effect = AttributeError("module not found")

        # Also need to patch yfinance download since we have open trades
        with patch('yfinance.download') as mock_yf_download:
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.columns = ['open', 'high', 'low', 'close', 'volume']
            mock_df.sort_index.return_value.dropna.return_value = mock_df
            mock_yf_download.return_value = mock_df

            # Call function
            result = run_update_outcomes_job()

            # Should return skipped status due to AttributeError
            assert result["status"] == "skipped"
            assert result["job"] == "UPDATE_OUTCOMES"
            assert "reason" in result

    @patch('src.airflow_jobs.get_config')
    def test_run_update_outcomes_job_raises_error_if_paper_trading_false(self, mock_get_config):
        """Test that run_update_outcomes_job raises error if paper_trading_mode=false."""
        from src.airflow_jobs import run_update_outcomes_job

        # Setup mock with paper_trading_mode = False
        mock_config = MagicMock()
        mock_config.paper_trading_mode = False
        mock_get_config.return_value = mock_config

        # Assert RuntimeError is raised
        with pytest.raises(RuntimeError) as exc_info:
            run_update_outcomes_job()

        assert str(exc_info.value) == "Live trading is disabled. Set paper_trading_mode=true."
