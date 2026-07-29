import pytest
import pandas as pd
from src.backtest_engine import BacktestEngine, Trade
from src.pattern_detector import Direction

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'open':  [10, 15, 25, 30, 35, 25, 20, 20, 20, 20],
        'high':  [12, 20, 30, 35, 40, 30, 25, 25, 25, 25],
        'low':   [ 5, 10, 25, 28, 24, 15, 10,  5,  0,  0],
        'close': [11, 19, 29, 34, 25, 18, 15, 10,  5,  5]
    }, index=dates)
    return df

@pytest.fixture
def default_config():
    return {
        'instrument': 'XAUUSD', 
        'rr_ratio': 2.0, 
        'spread_assumption': 1.5, 
        'stop_loss_pips': 50.0 # 5 points for XAU
    }

def test_engine_initialization(sample_data, default_config):
    engine = BacktestEngine(sample_data, default_config)
    
    assert engine.spread_pips == 1.5
    assert engine.rr_ratio == 2.0
    assert engine.pip_size == 0.1 # XAUUSD
    assert engine.spread == pytest.approx(0.15)
    assert engine.sl_dist == 5.0 # 50 pips * 0.1

def test_trade_execution_and_management(sample_data, default_config):
    default_config['spread_assumption'] = 0.0
    engine = BacktestEngine(sample_data, default_config)
    results = engine.run()
    
    assert not results.empty
    assert len(results) >= 1
    
    first_trade = results.iloc[0]
    assert first_trade['direction'] == Direction.BULLISH.value
    assert first_trade['entry_price'] == 25.0
    
    # Trade entry at idx 4 (L=24). SL=20.0, TP=35.0
    # idx 5: L=15 -> Hits SL
    assert first_trade['exit_price'] == 20.0
    assert first_trade['result'] == "LOSS"
    assert first_trade['r_multiple'] == -1.0

def test_no_lookahead_bias(sample_data, default_config):
    """
    Test that an FVG formed at candle idx 2 (c3) is NOT traded on candle 2 itself,
    even though the low of c3 touches the entry price. It should only be available
    from candle 3 onwards, and in this data, triggered on candle 4.
    """
    default_config['spread_assumption'] = 0.0
    engine = BacktestEngine(sample_data, default_config)
    engine.run()
    
    assert len(engine.trades) > 0
    first_trade = engine.trades[0]
    
    entry_time = first_trade.entry_time
    entry_idx = sample_data.index.get_loc(entry_time)
    
    # If there was lookahead bias, it would trigger at idx 2 because 
    # the FVG's assigned timestamp is idx 1, and idx 2's low hits 25.
    # However, it must correctly trigger at idx 4.
    assert entry_idx > 2
    assert entry_idx == 4
    
def test_forex_pip_size(sample_data):
    config = {
        'instrument': 'EURUSD', 
        'rr_ratio': 2.0, 
        'spread_assumption': 1.0, 
        'stop_loss_pips': 10.0
    }
    engine = BacktestEngine(sample_data, config)
    assert engine.pip_size == 0.0001
    assert engine.spread == 0.0001
    assert engine.sl_dist == 0.001
