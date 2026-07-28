"""
Unit tests for data loader functionality.
"""

import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data_loader import process_and_cache_data, load_data

@pytest.fixture
def mock_csv_data(tmp_path):
    """Creates a temporary mocked CSV data file representing raw Dukascopy data."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    
    file_path = raw_dir / "xauusd-mock.csv"
    
    # Create simple 1-minute data in UTC for 2 hours
    dates = pd.date_range(start="2023-01-02 12:00:00", periods=120, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates.view("int64") // 10**6, # convert to ms timestamp
        "open": [1900.0] * 120,
        "high": [1905.0] * 120,
        "low": [1895.0] * 120,
        "close": [1902.0] * 120,
        "volume": [100] * 120
    })
    
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_process_and_cache_data(mock_csv_data, tmp_path, caplog):
    """Test if processing correctly resamples, sets timezone, and saves to parquet."""
    processed_dir = tmp_path / "data" / "processed"
    
    df = process_and_cache_data(
        csv_path=mock_csv_data,
        instrument="XAUUSD",
        timeframe="1H",
        processed_dir=str(processed_dir)
    )
    
    # 120 minutes = 2 hours, so resampled 1H data should have 2 rows
    assert len(df) == 2
    
    # Check if timezone is America/New_York
    assert str(df.index.tz) == "America/New_York"
    
    # Check if parquet file was created
    expected_cache = processed_dir / "XAUUSD_1H.parquet"
    assert expected_cache.exists()
    
    # Check values
    assert df.iloc[0]["open"] == 1900.0
    assert df.iloc[0]["volume"] == 6000 # 60 * 100
    
    # Verify no gap warnings were emitted for complete data
    assert "missing/gap candles" not in caplog.text

@patch("src.data_loader.download_dukascopy_data")
def test_load_data_with_missing_cache(mock_download, mock_csv_data, tmp_path):
    """Test loading data when cache is missing, triggering a download."""
    mock_download.return_value = mock_csv_data
    processed_dir = tmp_path / "data" / "processed"
    raw_dir = tmp_path / "data" / "raw"
    
    df = load_data(
        instrument="XAUUSD",
        timeframe="1H",
        start_date="2023-01-01",
        end_date="2023-01-05",
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir)
    )
    
    assert mock_download.called
    assert len(df) == 2
    assert (processed_dir / "XAUUSD_1H.parquet").exists()

def test_load_data_with_existing_cache(tmp_path):
    """Test loading data when cache exists, avoiding download."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    
    # Create fake parquet cache
    dates = pd.date_range(start="2023-01-02 08:00:00", periods=5, freq="1h", tz="America/New_York")
    df = pd.DataFrame({
        "open": [1900.0] * 5,
        "high": [1905.0] * 5,
        "low": [1895.0] * 5,
        "close": [1902.0] * 5,
        "volume": [6000] * 5
    }, index=dates)
    
    cache_path = processed_dir / "XAUUSD_1H.parquet"
    df.to_parquet(cache_path)
    
    with patch("src.data_loader.download_dukascopy_data") as mock_download:
        loaded_df = load_data(
            instrument="XAUUSD",
            timeframe="1H",
            start_date="2023-01-01",
            end_date="2023-01-05",
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(processed_dir)
        )
        
        assert not mock_download.called
        assert len(loaded_df) == 5
        assert loaded_df.iloc[0]["open"] == 1900.0

def test_gap_detection(tmp_path, caplog):
    """Test gap validation explicitly triggers logging on missing candles."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    file_path = raw_dir / "xauusd-gap.csv"
    
    # Create data with an intentional 3-hour gap
    dates_part1 = pd.date_range(start="2023-01-02 10:00:00", periods=60, freq="1min", tz="UTC")
    dates_part2 = pd.date_range(start="2023-01-02 14:00:00", periods=60, freq="1min", tz="UTC")
    dates = dates_part1.union(dates_part2)
    
    df = pd.DataFrame({
        "timestamp": dates.view("int64") // 10**6,
        "open": [1900.0] * 120,
        "high": [1905.0] * 120,
        "low": [1895.0] * 120,
        "close": [1902.0] * 120,
        "volume": [100] * 120
    })
    
    df.to_csv(file_path, index=False)
    
    processed_dir = tmp_path / "data" / "processed"
    process_and_cache_data(
        csv_path=str(file_path),
        instrument="XAUUSD",
        timeframe="1H",
        processed_dir=str(processed_dir)
    )
    
    # Verify the warning was logged for the missing data gap
    assert "missing/gap candles" in caplog.text
    assert "Gap detected ending at" in caplog.text
