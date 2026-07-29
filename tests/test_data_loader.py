import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch
from src.data_loader import DataLoader, DataQualityError

@pytest.fixture
def synthetic_1h_data():
    # Create 48 hours of 1H data starting Monday
    dates = pd.date_range(start="2024-01-01 00:00:00", periods=48, freq="1h", tz="America/New_York")
    df = pd.DataFrame({
        'open': [100.0] * 48,
        'high': [105.0] * 48,
        'low': [95.0] * 48,
        'close': [102.0] * 48,
        'volume': [1000] * 48
    }, index=dates)
    df.index.name = 'datetime'
    return df

@pytest.fixture
def data_loader(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    return DataLoader(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

def test_session_tagging(data_loader, synthetic_1h_data):
    tagged_df = data_loader._tag_sessions(synthetic_1h_data)
    
    # Check boundaries (times are in America/New_York)
    # Asian: 19:00 - 03:00
    assert tagged_df.loc[tagged_df.index.hour == 20, 'session'].iloc[0] == 'Asian'
    assert tagged_df.loc[tagged_df.index.hour == 2, 'session'].iloc[0] == 'Asian'
    
    # London: 03:00 - 08:00 (since 08:00 starts overlap)
    assert tagged_df.loc[tagged_df.index.hour == 5, 'session'].iloc[0] == 'London'
    
    # Overlap: 08:00 - 11:00
    assert tagged_df.loc[tagged_df.index.hour == 9, 'session'].iloc[0] == 'Overlap'
    
    # NY: 11:00 - 17:00
    assert tagged_df.loc[tagged_df.index.hour == 14, 'session'].iloc[0] == 'NY'

def test_gap_detection_raises_error(data_loader, synthetic_1h_data):
    # Introduce a 3-hour gap (which is > 1.5x expected 1H interval)
    df_with_gap = synthetic_1h_data.drop(synthetic_1h_data.index[5:8])
    
    with pytest.raises(DataQualityError) as exc_info:
        data_loader._validate_data(df_with_gap, "1H")
        
    assert "missing/gap candles" in str(exc_info.value)

@patch('src.data_loader.yf.download')
def test_full_load_and_resample(mock_yf_download, data_loader, synthetic_1h_data):
    # Mock yfinance to return synthetic 1H data
    mock_yf_download.return_value = synthetic_1h_data
    
    # Test resampling to 4H
    df_4h = data_loader.get_data("XAUUSD", "4H", "2024-01-01", "2024-01-02", force_download=True)
    
    # 48 hours of 1H data -> 12 periods of 4H data
    assert len(df_4h) == 12
    # Volume should be sum of 4 hours (1000 * 4)
    assert df_4h['volume'].iloc[0] == 4000
    # Check caching
    assert len(list(data_loader.processed_dir.glob("*.parquet"))) == 1
