from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from src.events import EventBus, CandleEvent, PatternEvent
from src.models import Candle, PatternParams

@dataclass(kw_only=True)
class SwingPoint:
    timestamp: datetime
    price: float
    type: str  # 'HIGH' or 'LOW'

@dataclass
class DetailedPatternEvent(PatternEvent):
    price_levels: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

def detect_swing(candles: List[Candle], index: int, lookback: int = 5) -> List[SwingPoint]:
    """Pure function to detect swings with zero lookahead."""
    if index < lookback * 2:
        return []
        
    candidate_idx = index - lookback
    candidate = candles[candidate_idx]
    swings = []
    
    # Check High
    is_high = True
    for j in range(1, lookback + 1):
        if candles[candidate_idx - j].high >= candidate.high or candles[candidate_idx + j].high >= candidate.high:
            is_high = False
            break
    if is_high:
        swings.append(SwingPoint(timestamp=candidate.timestamp, price=candidate.high, type='HIGH'))
        
    # Check Low
    is_low = True
    for j in range(1, lookback + 1):
        if candles[candidate_idx - j].low <= candidate.low or candles[candidate_idx + j].low <= candidate.low:
            is_low = False
            break
    if is_low:
        swings.append(SwingPoint(timestamp=candidate.timestamp, price=candidate.low, type='LOW'))
        
    return swings

def detect_fvg(candles: List[Candle], index: int) -> Optional[DetailedPatternEvent]:
    """Pure function to detect FVG on current index."""
    if index < 2:
        return None
        
    c0 = candles[index-2]
    c2 = candles[index]
    
    # Bullish FVG
    if c0.high < c2.low:
        gap = c2.low - c0.high
        fill = c0.high + (gap / 2)
        return DetailedPatternEvent(
            timestamp=c2.timestamp, pattern_type='FVG', direction='BULLISH', price_level=c2.low,
            price_levels={'top': c2.low, 'bottom': c0.high, 'fill_price': fill},
            metadata={'status': 'unmitigated', 'gap_size': gap}
        )
        
    # Bearish FVG
    if c0.low > c2.high:
        gap = c0.low - c2.high
        fill = c2.high + (gap / 2)
        return DetailedPatternEvent(
            timestamp=c2.timestamp, pattern_type='FVG', direction='BEARISH', price_level=c2.high,
            price_levels={'top': c0.low, 'bottom': c2.high, 'fill_price': fill},
            metadata={'status': 'unmitigated', 'gap_size': gap}
        )
    return None

def get_alternating_swings(swings: List[SwingPoint]) -> List[SwingPoint]:
    if not swings:
        return []
    alt = [swings[0]]
    for s in swings[1:]:
        if s.type != alt[-1].type:
            alt.append(s)
        else:
            if s.type == 'HIGH' and s.price > alt[-1].price:
                alt[-1] = s
            elif s.type == 'LOW' and s.price < alt[-1].price:
                alt[-1] = s
    return alt

def detect_bos(candles: List[Candle], swings: List[SwingPoint], index: int) -> Optional[DetailedPatternEvent]:
    if len(candles) < 2 or not swings:
        return None
        
    c = candles[index]
    prev_c = candles[index-1]
    
    highs = [s for s in swings if s.type == 'HIGH']
    lows = [s for s in swings if s.type == 'LOW']
    if not highs or not lows:
        return None
        
    last_high = highs[-1]
    last_low = lows[-1]
    
    # Bullish BOS: closes above prior bearish swing high after making higher low
    if c.close > last_high.price and prev_c.close <= last_high.price:
        high_idx = next((i for i, can in enumerate(candles) if can.timestamp == last_high.timestamp), -1)
        if high_idx != -1 and high_idx < index:
            actual_low = min((can.low for can in candles[high_idx+1:index+1]), default=float('inf'))
            prev_highs = [s for s in highs if s.timestamp < last_high.timestamp]
            prev_lows = [s for s in lows if s.timestamp < last_high.timestamp]
            
            if prev_highs and prev_lows:
                if last_high.price < prev_highs[-1].price and actual_low > prev_lows[-1].price:
                    return DetailedPatternEvent(
                        timestamp=c.timestamp, pattern_type='BOS', direction='BULLISH',
                        price_level=last_high.price, metadata={'actual_low': actual_low}
                    )
                    
    # Bearish BOS: closes below prior bullish swing low after making lower high
    if c.close < last_low.price and prev_c.close >= last_low.price:
        low_idx = next((i for i, can in enumerate(candles) if can.timestamp == last_low.timestamp), -1)
        if low_idx != -1 and low_idx < index:
            actual_high = max((can.high for can in candles[low_idx+1:index+1]), default=float('-inf'))
            prev_lows = [s for s in lows if s.timestamp < last_low.timestamp]
            prev_highs = [s for s in highs if s.timestamp < last_low.timestamp]
            
            if prev_lows and prev_highs:
                if last_low.price > prev_lows[-1].price and actual_high < prev_highs[-1].price:
                    return DetailedPatternEvent(
                        timestamp=c.timestamp, pattern_type='BOS', direction='BEARISH',
                        price_level=last_low.price, metadata={'actual_high': actual_high}
                    )
    return None

def detect_choch(candles: List[Candle], swings: List[SwingPoint], index: int) -> Optional[DetailedPatternEvent]:
    if len(candles) < 2 or len(swings) < 4:
        return None
        
    c = candles[index]
    prev_c = candles[index-1]
    alt = get_alternating_swings(swings)
    
    if len(alt) < 4:
        return None
        
    # Bullish CHoCH: break above most recent lower high after downtrend (H1, L1, H2, L2)
    last_high_idx = next((i for i in range(len(alt)-1, -1, -1) if alt[i].type == 'HIGH'), -1)
    if last_high_idx >= 3:
        h2 = alt[last_high_idx]
        l1 = alt[last_high_idx-1]
        h1 = alt[last_high_idx-2]
        l0 = alt[last_high_idx-3]
        
        if h2.price < h1.price and l1.price < l0.price:  # Downtrend
            if c.close > h2.price and prev_c.close <= h2.price:
                high_idx = next((i for i, can in enumerate(candles) if can.timestamp == h2.timestamp), -1)
                if high_idx != -1 and high_idx < index:
                    actual_low = min((can.low for can in candles[high_idx+1:index+1]), default=float('inf'))
                    if actual_low < l1.price: # Confirmed lower low before break
                        return DetailedPatternEvent(
                            timestamp=c.timestamp, pattern_type='CHoCH', direction='BULLISH',
                            price_level=h2.price, metadata={'prior_trend': 'DOWNTREND'}
                        )
                        
    # Bearish CHoCH: break below most recent higher low after uptrend (L1, H1, L2, H2)
    last_low_idx = next((i for i in range(len(alt)-1, -1, -1) if alt[i].type == 'LOW'), -1)
    if last_low_idx >= 3:
        l2 = alt[last_low_idx]
        h1 = alt[last_low_idx-1]
        l1 = alt[last_low_idx-2]
        h0 = alt[last_low_idx-3]
        
        if l2.price > l1.price and h1.price > h0.price: # Uptrend
            if c.close < l2.price and prev_c.close >= l2.price:
                low_idx = next((i for i, can in enumerate(candles) if can.timestamp == l2.timestamp), -1)
                if low_idx != -1 and low_idx < index:
                    actual_high = max((can.high for can in candles[low_idx+1:index+1]), default=float('-inf'))
                    if actual_high > h1.price: # Confirmed higher high
                        return DetailedPatternEvent(
                            timestamp=c.timestamp, pattern_type='CHoCH', direction='BEARISH',
                            price_level=l2.price, metadata={'prior_trend': 'UPTREND'}
                        )
    return None

class PatternDetector:
    def __init__(self, event_bus: EventBus, params: PatternParams, pivot_lookback: int = 5):
        self.event_bus = event_bus
        self.params = params
        self.pivot_lookback = pivot_lookback
        self.candles: List[Candle] = []
        self.swings: List[SwingPoint] = []
        self.unmitigated_fvgs: List[DetailedPatternEvent] = []
        
        self.event_bus.subscribe(CandleEvent, self.on_candle)
        
    def on_candle(self, event: CandleEvent):
        self.candles.append(event.candle)
        i = len(self.candles) - 1
        
        new_swings = detect_swing(self.candles, i, self.pivot_lookback)
        self.swings.extend(new_swings)
        
        bos = detect_bos(self.candles, self.swings, i)
        if bos:
            self.event_bus.emit(bos)
            
        choch = detect_choch(self.candles, self.swings, i)
        if choch:
            self.event_bus.emit(choch)
            
        fvg = detect_fvg(self.candles, i)
        if fvg:
            self.unmitigated_fvgs.append(fvg)
            self.event_bus.emit(fvg)
            
        self._check_fvg_mitigations(i)

    def _check_fvg_mitigations(self, i: int):
        c = self.candles[i]
        still_unmitigated = []
        for fvg in self.unmitigated_fvgs:
            mitigated = False
            if fvg.direction == 'BULLISH' and c.close <= fvg.price_levels['top']:
                mitigated = True
            elif fvg.direction == 'BEARISH' and c.close >= fvg.price_levels['bottom']:
                mitigated = True
                
            if mitigated:
                mitigated_event = DetailedPatternEvent(
                    timestamp=c.timestamp, pattern_type='FVG', direction=fvg.direction,
                    price_level=fvg.price_level, price_levels=fvg.price_levels,
                    metadata={'status': 'mitigated', 'original_time': fvg.timestamp}
                )
                self.event_bus.emit(mitigated_event)
            else:
                still_unmitigated.append(fvg)
        self.unmitigated_fvgs = still_unmitigated
