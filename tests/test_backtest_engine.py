import pytest
from datetime import datetime, timedelta
import pandas as pd
from src.events import EventBus, CandleEvent, PatternEvent
from src.models import Candle, ConfigSchema, DateRange, RiskSchema, SessionsSchema, PatternsSchema, InstrumentSchema, ExecutionSchema
from src.pattern_detector import DetailedPatternEvent
from src.backtest_engine import EventDrivenBacktester

def create_candle(idx, o, h, l, c):
    dt = datetime(2024, 1, 1, 0, 0) + timedelta(hours=idx)
    return Candle(timestamp=dt, open=o, high=h, low=l, close=c)

@pytest.fixture
def config():
    return ConfigSchema(
        instrument=InstrumentSchema(symbol='EURUSD=X', display_name='EURUSD'),
        timeframe='1H',
        date_range=DateRange(start='2024-01-01', end='2024-01-31'),
        risk=RiskSchema(risk_per_trade=0.01, rr_ratio=2.0, atr_multiplier=1.5, max_daily_risk=0.03),
        execution=ExecutionSchema(spread_pips=2.0),
        sessions=SessionsSchema(),
        patterns=PatternsSchema()
    )

def test_no_lookahead_bias(config):
    bus = EventBus()
    engine = EventDrivenBacktester(bus, config)
    
    # Send 50 candles
    for i in range(51):
        bus.emit(CandleEvent(candle=create_candle(i, 1.1000, 1.1020, 1.0980, 1.1010)))
        
    assert engine.current_index == 50
    assert engine._get_atr() > 0 # ATR should be calculated on 0-50
    
    # Mock a pattern at index 50
    bus.emit(PatternEvent(pattern_type='BOS', direction='BULLISH', price_level=1.1015, timestamp=engine.candles[50].timestamp))
    
    # Should place stop order at 1.1015
    assert len(engine.pending_orders) == 1
    assert engine.pending_orders[0].requested_price == 1.1015
    
    # Test lookahead explicitly
    with pytest.raises(IndexError):
        engine.get_candle_by_index(51)

def test_limit_order_fill_fvg(config):
    bus = EventBus()
    engine = EventDrivenBacktester(bus, config)
    
    # Candle 0
    bus.emit(CandleEvent(candle=create_candle(0, 1.1000, 1.1020, 1.0980, 1.1010)))
    
    # Emit FVG
    bus.emit(DetailedPatternEvent(
        pattern_type='FVG', direction='BULLISH', price_level=1.1015,
        price_levels={'fill_price': 1.1010}, metadata={'status': 'unmitigated'},
        timestamp=engine.candles[0].timestamp
    ))
    
    # Should have a pending limit order at 1.1010
    assert len(engine.pending_orders) == 1
    assert engine.pending_orders[0].order_type == 'LIMIT'
    
    # Next candle doesn't reach 1.1010 (low is 1.1012)
    bus.emit(CandleEvent(candle=create_candle(1, 1.1015, 1.1030, 1.1012, 1.1025)))
    assert len(engine.open_trades) == 0
    
    # Candle 2 drops to 1.1005, which is <= 1.1010. Should fill.
    bus.emit(CandleEvent(candle=create_candle(2, 1.1020, 1.1025, 1.1005, 1.1015)))
    
    assert len(engine.open_trades) == 1
    assert engine.open_trades[0]['entry_price'] > 1.1010 # Includes spread

def test_circuit_breaker(config):
    bus = EventBus()
    engine = EventDrivenBacktester(bus, config, initial_capital=10000.0)
    engine.max_daily_risk_pct = 2.0 # Force circuit breaker at 2% daily loss
    
    # Send initial candle
    bus.emit(CandleEvent(candle=create_candle(0, 1.1000, 1.1000, 1.1000, 1.1000)))
    
    # Directly manipulate daily_pnl to simulate losing trades
    engine.daily_pnl = -250.0 # -2.5% loss
    
    # Emit next candle to trigger circuit breaker check
    bus.emit(CandleEvent(candle=create_candle(1, 1.1000, 1.1000, 1.1000, 1.1000)))
    
    assert engine.circuit_breaker_active == True
    
    # Now try to emit a pattern
    bus.emit(PatternEvent(pattern_type='BOS', direction='BULLISH', price_level=1.1015, timestamp=engine.candles[1].timestamp))
    
    # Should ignore pattern
    assert len(engine.pending_orders) == 0

def test_spread_commission_impact(config):
    bus = EventBus()
    # Force 0 slippage by mocking random, set spread to 2 pips
    config.execution.spread_pips = 2.0  
    engine = EventDrivenBacktester(bus, config, initial_capital=10000.0)
    
    bus.emit(CandleEvent(candle=create_candle(0, 1.1000, 1.1020, 1.0980, 1.1010)))
    
    # Market order
    bus.emit(PatternEvent(pattern_type='UNKNOWN', direction='BULLISH', price_level=1.1010, timestamp=engine.candles[0].timestamp))
    
    # Fill on next open (1.1010)
    import random
    original_random = random.uniform
    random.uniform = lambda a, b: 0.0 # No slippage
    
    bus.emit(CandleEvent(candle=create_candle(1, 1.1010, 1.1050, 1.1000, 1.1040)))
    random.uniform = original_random
    
    # Entry price should be 1.1010 + half spread (0.0001) = 1.1011
    assert engine.open_trades[0]['entry_price'] == 1.1011
    
    # Size = ...
    size = engine.open_trades[0]['size']
    
    # Close trade on next candle 1.1040 open
    # We'll just trigger TP by making high = 1.1080
    trade = engine.open_trades[0]
    tp = trade['take_profit']
    bus.emit(CandleEvent(candle=create_candle(2, 1.1040, tp + 0.0010, 1.1030, tp)))
    
    assert len(engine.closed_trades) == 1
    res = engine.closed_trades[0]
    
    # Exit price should be TP - half spread (0.0001)
    expected_exit = tp - 0.0001
    assert res.exit_price == pytest.approx(expected_exit)
    
    # Commission = 7.0 per lot
    expected_commission = 7.0 * size
    expected_pnl = (expected_exit - 1.1011) * (size * 100000) - expected_commission
    assert res.pnl == pytest.approx(expected_pnl)
