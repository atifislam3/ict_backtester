"""
Pattern detector module for finding ICT patterns (FVG, BOS, CHoCH).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from enum import Enum

class Direction(Enum):
    """Enumeration for market direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class FVGEvent:
    """Represents a Fair Value Gap event."""
    timestamp: pd.Timestamp
    direction: Direction
    top: float
    bottom: float

@dataclass
class SwingPoint:
    """Represents a structural swing high or low."""
    timestamp: pd.Timestamp
    is_high: bool
    price: float

@dataclass
class StructureEvent:
    """Represents a BOS or CHoCH event."""
    timestamp: pd.Timestamp
    event_type: str  # "BOS" or "CHOCH"
    direction: Direction
    price_level: float

def detect_fvgs(df: pd.DataFrame) -> List[FVGEvent]:
    """Detects Fair Value Gaps (FVG) from 3 consecutive candles."""
    fvgs = []
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i-2], df.iloc[i]
        
        # Bullish FVG: Gap between C1 high and C3 low
        if c3['low'] > c1['high']:
            fvgs.append(FVGEvent(df.index[i-1], Direction.BULLISH, c3['low'], c1['high']))
            
        # Bearish FVG: Gap between C1 low and C3 high
        elif c3['high'] < c1['low']:
            fvgs.append(FVGEvent(df.index[i-1], Direction.BEARISH, c1['low'], c3['high']))
            
    return fvgs

def find_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[SwingPoint]:
    """Finds swing highs and lows using a local window."""
    swings = []
    for i in range(left, len(df) - right):
        window = df.iloc[i-left:i+right+1]
        c_high, c_low = df['high'].iloc[i], df['low'].iloc[i]
        
        if c_high == window['high'].max() and sum(window['high'] == c_high) == 1:
            swings.append(SwingPoint(df.index[i], True, c_high))
            
        elif c_low == window['low'].min() and sum(window['low'] == c_low) == 1:
            swings.append(SwingPoint(df.index[i], False, c_low))
            
    return swings

def _update_active_swings(
    swings: List[SwingPoint], 
    confirm_time: pd.Timestamp, 
    idx: int,
    last_high: Optional[SwingPoint],
    last_low: Optional[SwingPoint]
) -> Tuple[int, Optional[SwingPoint], Optional[SwingPoint]]:
    """Updates the most recent confirmed swing points based on current confirmation time."""
    while idx < len(swings) and swings[idx].timestamp <= confirm_time:
        if swings[idx].is_high:
            last_high = swings[idx]
        else:
            last_low = swings[idx]
        idx += 1
    return idx, last_high, last_low

def _evaluate_break(
    close: float, 
    timestamp: pd.Timestamp, 
    last_high: Optional[SwingPoint],
    last_low: Optional[SwingPoint],
    current_trend: Optional[Direction]
) -> Tuple[Optional[StructureEvent], Optional[Direction], Optional[SwingPoint], Optional[SwingPoint]]:
    """Evaluates if the current close breaks structure to form BOS or CHoCH."""
    if last_high and close > last_high.price:
        event_type = "CHOCH" if current_trend == Direction.BEARISH else "BOS"
        event = StructureEvent(timestamp, event_type, Direction.BULLISH, last_high.price)
        return event, Direction.BULLISH, None, last_low
        
    if last_low and close < last_low.price:
        event_type = "CHOCH" if current_trend == Direction.BULLISH else "BOS"
        event = StructureEvent(timestamp, event_type, Direction.BEARISH, last_low.price)
        return event, Direction.BEARISH, last_high, None
        
    return None, current_trend, last_high, last_low

def detect_structure_events(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[StructureEvent]:
    """Detects Break of Structure (BOS) and Change of Character (CHOCH)."""
    swings = find_swing_points(df, left, right)
    events = []
    last_high, last_low, current_trend = None, None, None
    swing_idx = 0
    
    for i in range(len(df)):
        timestamp, close = df.index[i], df['close'].iloc[i]
        
        # A swing is confirmed 'right' candles after it forms.
        confirm_time = df.index[i - right] if i >= right else df.index[0]
        
        swing_idx, last_high, last_low = _update_active_swings(
            swings, confirm_time, swing_idx, last_high, last_low
        )
        
        event, current_trend, last_high, last_low = _evaluate_break(
            close, timestamp, last_high, last_low, current_trend
        )
        
        if event:
            events.append(event)
            
    return events
