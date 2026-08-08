"""Tests for backtest reporting module - Trade Log normalization."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest_reporting import _normalize_trade_log, EXPECTED_COLUMNS, _normalize_daily_log, EXPECTED_DAILY_COLUMNS


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
        Test that nested dict entries are handled without crashing.
        
        Create entries with nested dict/list values and assert they become None.
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
            # Entry with extra nested dict key (not in EXPECTED_COLUMNS - will be ignored)
            {
                'trade_id': 'T000002',
                'ticker': 'TCS.NS',
                'entry_date': '2020-01-20',
                'nested_data': {'foo': 'bar'},  # Extra key not in EXPECTED_COLUMNS
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
            # Entry with nested list value in signal_trigger
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
        
        # Should have 3 rows (entries are normalized, not skipped)
        assert result_df.shape[0] == 3
        
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


class TestDailyLogNormalization:
    """Test class for daily activity log normalization functionality."""

    def test_daily_log_structure(self):
        """
        Test that _normalize_daily_log produces exactly 11 columns.
        
        Create a fake daily_activity_log list and verify:
        - There is at least one "MARK_TO_MARKET" row per unique date
        - Every row has all 11 keys
        - _normalize_daily_log returns DataFrame with exactly 11 columns
        """
        # Create sample daily activity log with multiple days
        daily_log = [
            # Day 1: MARK_TO_MARKET row
            {
                'date': '2023-01-02',
                'ticker': 'PORTFOLIO',
                'action': 'MARK_TO_MARKET',
                'price': None,
                'quantity': None,
                'position_value': None,
                'cash_balance': 1000000.0,
                'total_portfolio_value': 1000000.0,
                'score': None,
                'signal': None,
                'notes': 'EOD valuation'
            },
            # Day 1: Signal evaluation
            {
                'date': '2023-01-02',
                'ticker': 'RELIANCE.NS',
                'action': 'HOLD',
                'price': 2500.0,
                'quantity': None,
                'position_value': None,
                'cash_balance': 1000000.0,
                'total_portfolio_value': 1000000.0,
                'score': 0.5,
                'signal': 'BUY',
                'notes': 'Signal evaluated: BUY'
            },
            # Day 2: MARK_TO_MARKET row
            {
                'date': '2023-01-03',
                'ticker': 'PORTFOLIO',
                'action': 'MARK_TO_MARKET',
                'price': None,
                'quantity': None,
                'position_value': None,
                'cash_balance': 950000.0,
                'total_portfolio_value': 950000.0,
                'score': None,
                'signal': None,
                'notes': 'EOD valuation'
            },
            # Day 2: BUY execution
            {
                'date': '2023-01-03',
                'ticker': 'RELIANCE.NS',
                'action': 'BUY',
                'price': 2500.0,
                'quantity': 20,
                'position_value': 50000.0,
                'cash_balance': 950000.0,
                'total_portfolio_value': 1000000.0,
                'score': None,
                'signal': 'Trend',
                'notes': 'Order executed at 2500.0'
            },
            # Day 3: STOP_LOSS_HIT
            {
                'date': '2023-01-04',
                'ticker': 'RELIANCE.NS',
                'action': 'STOP_LOSS_HIT',
                'price': 2375.0,
                'quantity': 20,
                'position_value': 0.0,
                'cash_balance': 997500.0,
                'total_portfolio_value': 997500.0,
                'score': None,
                'signal': None,
                'notes': 'Stop loss triggered'
            },
            # Day 3: MARK_TO_MARKET row
            {
                'date': '2023-01-04',
                'ticker': 'PORTFOLIO',
                'action': 'MARK_TO_MARKET',
                'price': None,
                'quantity': None,
                'position_value': None,
                'cash_balance': 997500.0,
                'total_portfolio_value': 997500.0,
                'score': None,
                'signal': None,
                'notes': 'EOD valuation'
            }
        ]
        
        # Call _normalize_daily_log
        result_df = _normalize_daily_log(daily_log)
        
        # Assert the resulting DataFrame has exactly 11 columns
        assert result_df.shape[1] == 11, \
            f"Expected 11 columns, got {result_df.shape[1]}"
        
        # Assert column names match EXPECTED_DAILY_COLUMNS
        assert list(result_df.columns) == EXPECTED_DAILY_COLUMNS, \
            f"Column names do not match EXPECTED_DAILY_COLUMNS.\nGot: {list(result_df.columns)}\nExpected: {EXPECTED_DAILY_COLUMNS}"
        
        # Verify we have 6 rows
        assert result_df.shape[0] == 6, f"Expected 6 rows, got {result_df.shape[0]}"
        
        # Assert there is at least one "MARK_TO_MARKET" row per unique date
        unique_dates = set(d['date'] for d in daily_log)
        mtm_rows = result_df[result_df['action'] == 'MARK_TO_MARKET']
        mtm_dates = set(mtm_rows['date'].tolist())
        
        assert unique_dates == mtm_dates, \
            f"Not all dates have MARK_TO_MARKET rows. Missing: {unique_dates - mtm_dates}"
        
        # Assert every row has all 11 keys
        for idx, row in result_df.iterrows():
            non_null_count = row.notna().sum()
            # At minimum, date, ticker, action, cash_balance, total_portfolio_value should be present
            assert pd.notna(row['date']), f"Row {idx} missing 'date'"
            assert pd.notna(row['ticker']), f"Row {idx} missing 'ticker'"
            assert pd.notna(row['action']), f"Row {idx} missing 'action'"
            assert pd.notna(row['cash_balance']), f"Row {idx} missing 'cash_balance'"
            assert pd.notna(row['total_portfolio_value']), f"Row {idx} missing 'total_portfolio_value'"

    def test_normalize_empty_daily_log(self):
        """Test handling of empty daily log."""
        result_df = _normalize_daily_log([])
        
        assert result_df.shape[0] == 0
        assert result_df.shape[1] == 11
        assert list(result_df.columns) == EXPECTED_DAILY_COLUMNS

    def test_normalize_none_daily_log(self):
        """Test handling of None daily log."""
        result_df = _normalize_daily_log(None)
        
        assert result_df.shape[0] == 0
        assert result_df.shape[1] == 11
        assert list(result_df.columns) == EXPECTED_DAILY_COLUMNS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
