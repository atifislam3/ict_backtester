import pytest
from datetime import datetime, timedelta
from src.models import Candle, PatternParams
from src.events import EventBus, CandleEvent, PatternEvent
from src.pattern_detector import PatternDetector, DetailedPatternEvent, SwingPoint

def create_candle(idx, o, h, l, c):
    dt = datetime(2024, 1, 1) + timedelta(hours=idx)
    return Candle(timestamp=dt, open=o, high=h, low=l, close=c)

@pytest.fixture
def detector():
    bus = EventBus()
    return PatternDetector(bus, PatternParams(), pivot_lookback=2)

def test_no_pattern_without_lookback(detector):
    emitted = []
    detector.event_bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    # Send 4 candles, pivot_lookback is 2 -> requires 5 candles for one swing
    for i in range(4):
        detector.on_candle(CandleEvent(candle=create_candle(i, 10, 20, 5, 15)))
        
    assert len(detector.swings) == 0
    assert len(emitted) == 0

def test_bullish_fvg_and_mitigation(detector):
    emitted = []
    detector.event_bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    c0 = create_candle(0, 10, 12, 8, 11)
    c1 = create_candle(1, 11, 15, 10, 14)
    c2 = create_candle(2, 14, 18, 13, 17) # Gap: c0.high(12) < c2.low(13)
    c3 = create_candle(3, 17, 19, 11, 11) # Mitigation: closes at 11, which is <= 13 (top of gap)
    
    detector.on_candle(CandleEvent(candle=c0))
    detector.on_candle(CandleEvent(candle=c1))
    detector.on_candle(CandleEvent(candle=c2))
    
    assert len(emitted) == 1
    fvg = emitted[0]
    assert fvg.pattern_type == 'FVG'
    assert fvg.direction == 'BULLISH'
    assert fvg.metadata['status'] == 'unmitigated'
    
    # Mitigation candle
    detector.on_candle(CandleEvent(candle=c3))
    
    assert len(emitted) == 2
    mit = emitted[1]
    assert mit.pattern_type == 'FVG'
    assert mit.metadata['status'] == 'mitigated'

def test_zero_lookahead_emission(detector):
    emitted = []
    detector.event_bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    # Create a scenario where FVG happens at idx 2
    c0 = create_candle(0, 10, 12, 8, 11)
    c1 = create_candle(1, 11, 15, 10, 14)
    
    detector.on_candle(CandleEvent(candle=c0))
    assert len(emitted) == 0
    
    detector.on_candle(CandleEvent(candle=c1))
    assert len(emitted) == 0
    
    c2 = create_candle(2, 14, 18, 13, 17) # FVG here
    detector.on_candle(CandleEvent(candle=c2))
    assert len(emitted) == 1 # Emits EXACTLY when c2 is passed

def test_bullish_bos():
    bus = EventBus()
    det = PatternDetector(bus, PatternParams(), pivot_lookback=1)
    emitted = []
    bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    # H1, L1, H2 (lower high), L2 (higher low), Breakout
    # H1
    det.on_candle(CandleEvent(candle=create_candle(0, 10, 20, 6, 10))) # c0
    det.on_candle(CandleEvent(candle=create_candle(1, 10, 25, 6, 10))) # H1 @ 25
    det.on_candle(CandleEvent(candle=create_candle(2, 10, 20, 6, 10))) # confirmed H1
    
    # L1
    det.on_candle(CandleEvent(candle=create_candle(3, 10, 20, 5, 10))) # L1 @ 5
    det.on_candle(CandleEvent(candle=create_candle(4, 10, 20, 10, 15))) # confirmed L1
    
    # H2 (Lower High)
    det.on_candle(CandleEvent(candle=create_candle(5, 10, 22, 10, 15))) # H2 @ 22
    det.on_candle(CandleEvent(candle=create_candle(6, 10, 20, 10, 15))) # confirmed H2
    
    # L2 (Higher Low)
    det.on_candle(CandleEvent(candle=create_candle(7, 10, 20, 8, 15))) # L2 @ 8
    det.on_candle(CandleEvent(candle=create_candle(8, 10, 20, 12, 15))) # confirmed L2
    
    # Breakout above H2 (22)
    det.on_candle(CandleEvent(candle=create_candle(9, 15, 30, 15, 23))) 
    
    bos_events = [e for e in emitted if e.pattern_type == 'BOS']
    assert len(bos_events) == 1
    assert bos_events[0].direction == 'BULLISH'

def test_bearish_choch():
    bus = EventBus()
    det = PatternDetector(bus, PatternParams(), pivot_lookback=1)
    emitted = []
    bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    # Uptrend: L0, H0, L1, H1, L2, H2 -> break below L2
    # L0
    det.on_candle(CandleEvent(candle=create_candle(0, 10, 15, 5, 10)))
    det.on_candle(CandleEvent(candle=create_candle(1, 10, 15, 2, 10))) # L0 @ 2
    det.on_candle(CandleEvent(candle=create_candle(2, 10, 15, 5, 10)))
    
    # H0
    det.on_candle(CandleEvent(candle=create_candle(3, 10, 20, 5, 10))) # H0 @ 20
    det.on_candle(CandleEvent(candle=create_candle(4, 10, 15, 6, 10)))
    
    # L1
    det.on_candle(CandleEvent(candle=create_candle(5, 10, 15, 3, 10))) # L1 @ 3 (wait: L1 should be > L0. 3 > 2 is good)
    det.on_candle(CandleEvent(candle=create_candle(6, 10, 15, 10, 10)))
    
    # H1
    det.on_candle(CandleEvent(candle=create_candle(7, 10, 25, 10, 10))) # H1 @ 25
    det.on_candle(CandleEvent(candle=create_candle(8, 10, 15, 11, 10)))
    
    # L2
    det.on_candle(CandleEvent(candle=create_candle(9, 10, 15, 4, 10))) # L2 @ 4 (wait: L2 should be > L1. 4 > 3 is good)
    det.on_candle(CandleEvent(candle=create_candle(10, 10, 15, 15, 10)))
    
    # H2
    det.on_candle(CandleEvent(candle=create_candle(11, 10, 30, 15, 10))) # H2 @ 30
    det.on_candle(CandleEvent(candle=create_candle(12, 10, 15, 15, 10)))
    
    # Break below L2 (4) and it made a higher high (30 > 25)
    det.on_candle(CandleEvent(candle=create_candle(13, 15, 15, 2, 2)))
    
    choch_events = [e for e in emitted if e.pattern_type == 'CHoCH']
    assert len(choch_events) == 1
    assert choch_events[0].direction == 'BEARISH'
    assert choch_events[0].metadata['prior_trend'] == 'UPTREND'

def test_conflicting_patterns_same_candle():
    bus = EventBus()
    det = PatternDetector(bus, PatternParams(), pivot_lookback=1)
    emitted = []
    bus.subscribe(PatternEvent, lambda e: emitted.append(e))
    
    # Create BOS setup
    det.on_candle(CandleEvent(candle=create_candle(0, 10, 20, 6, 10))) 
    det.on_candle(CandleEvent(candle=create_candle(1, 10, 25, 6, 10))) # H1 @ 25
    det.on_candle(CandleEvent(candle=create_candle(2, 10, 20, 6, 10))) 
    det.on_candle(CandleEvent(candle=create_candle(3, 10, 20, 5, 10))) # L1 @ 5
    det.on_candle(CandleEvent(candle=create_candle(4, 10, 20, 10, 15))) 
    det.on_candle(CandleEvent(candle=create_candle(5, 10, 22, 10, 15))) # H2 @ 22
    det.on_candle(CandleEvent(candle=create_candle(6, 10, 20, 10, 15))) 
    det.on_candle(CandleEvent(candle=create_candle(7, 10, 20, 8, 15))) # L2 @ 8
    det.on_candle(CandleEvent(candle=create_candle(8, 15, 20, 12, 15))) # previous candle for FVG (c0)
    det.on_candle(CandleEvent(candle=create_candle(9, 15, 20, 10, 15))) # c1
    
    # Breakout above H2 (22) AND FVG (c0.high 20 < c2.low 21)
    det.on_candle(CandleEvent(candle=create_candle(10, 21, 30, 21, 23))) 
    
    types = [e.pattern_type for e in emitted]
    assert 'BOS' in types
    assert 'FVG' in types
    assert types.count('BOS') == 1
    assert types.count('FVG') == 1
