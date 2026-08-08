"""
Tests for Universe module.
"""

import pytest
from pathlib import Path
import pandas as pd

from src.universe import discover_available_tickers


def test_discover_available_tickers(tmp_path):
    """Test discovering available tickers from parquet files."""
    # Create two dummy parquet files
    file_a = tmp_path / "AAA.NS.parquet"
    file_b = tmp_path / "BBB.NS.parquet"
    
    # Create minimal valid parquet files
    df_a = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "close": [100.0]})
    df_b = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "close": [200.0]})
    
    df_a.to_parquet(file_a)
    df_b.to_parquet(file_b)
    
    # Test discovery
    result = discover_available_tickers(str(tmp_path))
    assert result == ["AAA.NS", "BBB.NS"]


def test_discover_available_tickers_empty_dir(tmp_path):
    """Test discovering tickers from an empty directory returns empty list."""
    result = discover_available_tickers(str(tmp_path))
    assert result == []
