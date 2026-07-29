import pytest
from datetime import datetime, timedelta
from src.models import Candle, PatternParams
from src.events import EventBus, CandleEvent, PatternEvent
from src.pattern_detector import PatternDetector, DetailedPatternEvent, SwingPoint

def create_candle(idx, o, h, l, c):
    dt = datetime(2024, 1, 1) + timedelta(hours=idx)
    return Candle(timestamp=dt, open=o, high=h, low=l, close=c)

bus = EventBus()
det = PatternDetector(bus, PatternParams(), pivot_lookback=1)
emitted = []
bus.subscribe(PatternEvent, lambda e: emitted.append(e))

candles_data = [
    (0, 10, 20, 5, 10),
    (1, 10, 25, 5, 10),
    (2, 10, 20, 5, 10),
    (3, 10, 20, 5, 10),
    (4, 10, 20, 10, 15),
    (5, 10, 22, 10, 15),
    (6, 10, 20, 10, 15),
    (7, 10, 20, 8, 15),
    (8, 10, 20, 12, 15),
    (9, 15, 30, 15, 23)
]
for row in candles_data:
    det.on_candle(CandleEvent(candle=create_candle(*row)))
    
print(f"Swings: {det.swings}")
print(f"Emitted: {emitted}")
