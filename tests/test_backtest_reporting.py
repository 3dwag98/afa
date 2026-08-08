"""Tests for backtest reporting module - Trade Log normalization."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest_reporting import _normalize_trade_log, EXPECTED_COLUMNS


class TestTradeLogNormalization:
    """Test class for trade log normalization functionality."""

    def test_trade_log_columns(self):
        """
        Test that _normalize_trade_log produces exactly 16 columns.
        
        Create a sample trade_log with 3 flat dicts and verify:
        - The resulting DataFrame has exactly 16 columns
        - Column names match EXPECTED_COLUMNS
        """
        # Create sample trade log with 3 flat dicts
        trade_log = [
            {
                'trade_id': 'T000001',
                'ticker': 'RELIANCE.NS',
                'entry_date': '2020-01-15',
                'entry_price': 1500.0,
                'exit_date': '2020-02-15',
                'exit_price': 1600.0,
                'quantity': 100,
                'side': 'LONG',
                'signal_trigger': 'Trend',
                'gross_pnl': 10000.0,
                'transaction_costs': 50.0,
                'taxes': 10.0,
                'net_pnl': 9940.0,
                'return_pct': 6.67,
                'holding_days': 31,
                'exit_reason': 'target'
            },
            {
                'trade_id': 'T000002',
                'ticker': 'TCS.NS',
                'entry_date': '2020-01-20',
                'entry_price': 3200.0,
                'exit_date': '2020-02-10',
                'exit_price': 3100.0,
                'quantity': 50,
                'side': 'LONG',
                'signal_trigger': 'Breakout',
                'gross_pnl': -5000.0,
                'transaction_costs': 80.0,
                'taxes': 0.0,
                'net_pnl': -5080.0,
                'return_pct': -3.125,
                'holding_days': 21,
                'exit_reason': 'stop_loss'
            },
            {
                'trade_id': 'T000003',
                'ticker': 'HDFCBANK.NS',
                'entry_date': '2020-02-01',
                'entry_price': 1400.0,
                'exit_date': None,
                'exit_price': None,
                'quantity': 75,
                'side': 'LONG',
                'signal_trigger': 'Volume',
                'gross_pnl': 0.0,
                'transaction_costs': 52.5,
                'taxes': 0.0,
                'net_pnl': -52.5,
                'return_pct': 0.0,
                'holding_days': 0,
                'exit_reason': None
            }
        ]
        
        # Call _normalize_trade_log
        result_df = _normalize_trade_log(trade_log)
        
        # Assert the resulting DataFrame has exactly 16 columns
        assert result_df.shape[1] == 16, \
            f"Expected 16 columns, got {result_df.shape[1]}"
        
        # Assert column names match EXPECTED_COLUMNS
        assert list(result_df.columns) == EXPECTED_COLUMNS, \
            f"Column names do not match EXPECTED_COLUMNS.\nGot: {list(result_df.columns)}\nExpected: {EXPECTED_COLUMNS}"
        
        # Verify we have 3 rows
        assert result_df.shape[0] == 3, f"Expected 3 rows, got {result_df.shape[0]}"

    def test_normalize_handles_nested_dict(self):
        """
        Test that nested dict entries are skipped without crashing.
        
        Create one malformed entry (a nested dict) and assert it is skipped.
        """
        trade_log = [
            {
                'trade_id': 'T000001',
                'ticker': 'RELIANCE.NS',
                'entry_date': '2020-01-15',
                'entry_price': 1500.0,
                'exit_date': '2020-02-15',
                'exit_price': 1600.0,
                'quantity': 100,
                'side': 'LONG',
                'signal_trigger': 'Trend',
                'gross_pnl': 10000.0,
                'transaction_costs': 50.0,
                'taxes': 10.0,
                'net_pnl': 9940.0,
                'return_pct': 6.67,
                'holding_days': 31,
                'exit_reason': 'target'
            },
            # Malformed entry with nested dict
            {
                'trade_id': 'T000002',
                'ticker': 'TCS.NS',
                'entry_date': '2020-01-20',
                'nested_data': {'foo': 'bar'},  # This should be skipped
                'entry_price': 3200.0,
                'exit_date': '2020-02-10',
                'exit_price': 3100.0,
                'quantity': 50,
                'side': 'LONG',
                'signal_trigger': 'Breakout',
                'gross_pnl': -5000.0,
                'transaction_costs': 80.0,
                'taxes': 0.0,
                'net_pnl': -5080.0,
                'return_pct': -3.125,
                'holding_days': 21,
                'exit_reason': 'stop_loss'
            },
            # Entry with nested list value
            {
                'trade_id': 'T000003',
                'ticker': 'INFY.NS',
                'entry_date': '2020-02-01',
                'entry_price': 1400.0,
                'exit_date': '2020-03-01',
                'exit_price': 1500.0,
                'quantity': 75,
                'side': 'LONG',
                'signal_trigger': ['Volume', 'Trend'],  # List value should become None
                'gross_pnl': 7500.0,
                'transaction_costs': 52.5,
                'taxes': 0.0,
                'net_pnl': 7447.5,
                'return_pct': 7.14,
                'holding_days': 29,
                'exit_reason': 'target'
            }
        ]
        
        # Call _normalize_trade_log - should not crash
        result_df = _normalize_trade_log(trade_log)
        
        # Should still have 16 columns
        assert result_df.shape[1] == 16
        
        # Should have 3 rows (malformed entries are normalized, not skipped)
        assert result_df.shape[0] == 3
        
        # Check that nested dict value became None
        assert pd.isna(result_df.iloc[1]['signal_trigger']) or result_df.iloc[1]['signal_trigger'] is None
        
        # Check that nested list value became None
        assert pd.isna(result_df.iloc[2]['signal_trigger']) or result_df.iloc[2]['signal_trigger'] is None

    def test_normalize_empty_trade_log(self):
        """Test handling of empty trade log."""
        result_df = _normalize_trade_log([])
        
        assert result_df.shape[0] == 0
        assert result_df.shape[1] == 16
        assert list(result_df.columns) == EXPECTED_COLUMNS

    def test_normalize_none_trade_log(self):
        """Test handling of None trade log."""
        result_df = _normalize_trade_log(None)
        
        assert result_df.shape[0] == 0
        assert result_df.shape[1] == 16
        assert list(result_df.columns) == EXPECTED_COLUMNS

    def test_normalize_dict_input(self):
        """Test handling of dict input (converts to list of values)."""
        trade_log_dict = {
            'trade1': {
                'trade_id': 'T000001',
                'ticker': 'RELIANCE.NS',
                'entry_date': '2020-01-15',
                'entry_price': 1500.0,
                'exit_date': '2020-02-15',
                'exit_price': 1600.0,
                'quantity': 100,
                'side': 'LONG',
                'signal_trigger': 'Trend',
                'gross_pnl': 10000.0,
                'transaction_costs': 50.0,
                'taxes': 10.0,
                'net_pnl': 9940.0,
                'return_pct': 6.67,
                'holding_days': 31,
                'exit_reason': 'target'
            }
        }
        
        result_df = _normalize_trade_log(trade_log_dict)
        
        assert result_df.shape[0] == 1
        assert result_df.shape[1] == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
