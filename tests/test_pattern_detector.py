"""
Unit tests for the pattern detector module (FVG, BOS, CHoCH).
"""

import pandas as pd
import pytest
from src.pattern_detector import (
    Direction, detect_fvgs, find_swing_points, detect_structure_events
)

def create_synthetic_data(candles: list) -> pd.DataFrame:
    """
    Creates a DataFrame from a list of dicts containing OHLC data.
    """
    df = pd.DataFrame(candles)
    # Start at arbitrary time
    df['datetime'] = pd.date_range(start="2023-01-01", periods=len(df), freq="1h")
    df.set_index('datetime', inplace=True)
    return df

def test_detect_bullish_fvg():
    """Test detection of a Bullish Fair Value Gap."""
    # c1 high is 10, c3 low is 12 -> Gap of 2
    data = [
        {"open": 8, "high": 10, "low": 7, "close": 9},    # C1
        {"open": 9, "high": 15, "low": 9, "close": 14},   # C2
        {"open": 14, "high": 18, "low": 12, "close": 17}, # C3
    ]
    df = create_synthetic_data(data)
    fvgs = detect_fvgs(df)
    
    assert len(fvgs) == 1
    assert fvgs[0].direction == Direction.BULLISH
    assert fvgs[0].top == 12
    assert fvgs[0].bottom == 10
    assert fvgs[0].timestamp == df.index[1]

def test_detect_bearish_fvg():
    """Test detection of a Bearish Fair Value Gap."""
    # c1 low is 12, c3 high is 10 -> Gap of 2
    data = [
        {"open": 14, "high": 15, "low": 12, "close": 13}, # C1
        {"open": 13, "high": 13, "low": 7, "close": 8},   # C2
        {"open": 8, "high": 10, "low": 5, "close": 6},    # C3
    ]
    df = create_synthetic_data(data)
    fvgs = detect_fvgs(df)
    
    assert len(fvgs) == 1
    assert fvgs[0].direction == Direction.BEARISH
    assert fvgs[0].top == 12
    assert fvgs[0].bottom == 10
    assert fvgs[0].timestamp == df.index[1]

def test_swing_points():
    """Test finding swing highs and lows."""
    # We need left=2, right=2 so 5 candles to form a swing
    data = [
        {"high": 10, "low": 5, "close": 8},
        {"high": 12, "low": 6, "close": 10},
        {"high": 15, "low": 8, "close": 14}, # Swing High
        {"high": 14, "low": 7, "close": 12},
        {"high": 13, "low": 4, "close": 6},  # Swing Low candidate if followed by 2 higher lows
        {"high": 10, "low": 5, "close": 8},
        {"high": 12, "low": 6, "close": 10},
    ]
    df = create_synthetic_data(data)
    swings = find_swing_points(df, left=2, right=2)
    
    assert len(swings) == 2
    assert swings[0].is_high == True
    assert swings[0].price == 15
    assert swings[0].timestamp == df.index[2]
    
    assert swings[1].is_high == False
    assert swings[1].price == 4
    assert swings[1].timestamp == df.index[4]

def test_bullish_bos():
    """Test detection of a Bullish Break of Structure."""
    # Need a swing high, then a break above it
    data = [
        # Form swing high at idx 2
        {"high": 10, "low": 5, "close": 8},
        {"high": 12, "low": 6, "close": 10},
        {"high": 15, "low": 8, "close": 14}, # Swing High (15)
        {"high": 14, "low": 7, "close": 12},
        {"high": 13, "low": 4, "close": 6},  # Confirm swing high (right=2)
        
        # Break above 15
        {"high": 14, "low": 5, "close": 10},
        {"high": 17, "low": 12, "close": 16}, # Break!
    ]
    df = create_synthetic_data(data)
    events = detect_structure_events(df, left=2, right=2)
    
    assert len(events) == 1
    assert events[0].event_type == "BOS"
    assert events[0].direction == Direction.BULLISH
    assert events[0].price_level == 15
    assert events[0].timestamp == df.index[6]

def test_choch():
    """Test detection of Change of Character (Bearish -> Bullish)."""
    data = [
        # Form swing low at idx 2 (price=2)
        {"high": 10, "low": 5, "close": 8},
        {"high": 8, "low": 4, "close": 6},
        {"high": 6, "low": 2, "close": 4},   # Swing Low (2)
        {"high": 4, "low": 3, "close": 4},
        {"high": 4, "low": 4, "close": 4},   # Confirm swing low
        
        # Bearish break to establish bearish trend
        {"high": 6, "low": 1, "close": 1},   # BOS Bearish (breaks 2) -> Now trend is Bearish
        
        # Form new swing high at idx 7 (price=12)
        {"high": 8, "low": 2, "close": 5},
        {"high": 12, "low": 4, "close": 10}, # Swing High (12)
        {"high": 10, "low": 3, "close": 6},
        {"high": 9, "low": 2, "close": 5},   # Confirm swing high
        
        # Break above swing high -> CHOCH Bullish
        {"high": 15, "low": 8, "close": 14}, # Break!
    ]
    df = create_synthetic_data(data)
    events = detect_structure_events(df, left=2, right=2)
    
    assert len(events) == 2
    assert events[0].event_type == "BOS"
    assert events[0].direction == Direction.BEARISH
    
    assert events[1].event_type == "CHOCH"
    assert events[1].direction == Direction.BULLISH
    assert events[1].price_level == 12
    assert events[1].timestamp == df.index[10]
