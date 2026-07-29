import pytest
import pandas as pd
import numpy as np
from src.statistics import calculate_statistics

def test_statistics_empty_trades():
    """Test statistics with empty trade list to ensure no division by zero."""
    empty_df = pd.DataFrame(columns=[
        'entry_time', 'exit_time', 'direction', 'entry_price',
        'exit_price', 'stop_loss', 'take_profit', 'pattern_type',
        'result', 'r_multiple'
    ])
    
    stats = calculate_statistics(empty_df)
    
    assert stats['total_trades'] == 0
    assert stats['win_rate_pct'] == 0.0
    assert stats['loss_rate_pct'] == 0.0
    assert stats['avg_win_dollar'] == 0.0
    assert stats['avg_loss_dollar'] == 0.0
    assert stats['avg_rr'] == 0.0
    assert stats['expectancy_r'] == 0.0
    assert stats['expectancy_dollar'] == 0.0
    assert stats['max_drawdown_dollar'] == 0.0
    assert stats['max_drawdown_pct'] == 0.0
    assert stats['profit_factor'] == 0.0
    assert stats['annualized_sharpe_ratio'] == 0.0
    assert stats['session_performance'] == {}
    assert stats['pattern_performance'] == {}

def test_statistics_all_winners():
    """Test statistics with all winning trades, checking profit factor infinity and zero drawdown."""
    trades = [
        {
            'entry_time': pd.Timestamp('2023-01-01 09:00:00'),
            'exit_time': pd.Timestamp('2023-01-01 10:00:00'),
            'direction': 'BULLISH',
            'entry_price': 100.0,
            'exit_price': 110.0,
            'stop_loss': 90.0,
            'take_profit': 110.0,
            'pattern_type': 'FVG',
            'result': 'WIN',
            'r_multiple': 1.0
        },
        {
            'entry_time': pd.Timestamp('2023-01-02 14:00:00'),
            'exit_time': pd.Timestamp('2023-01-02 15:00:00'),
            'direction': 'BEARISH',
            'entry_price': 100.0,
            'exit_price': 90.0,
            'stop_loss': 110.0,
            'take_profit': 90.0,
            'pattern_type': 'STRUCT',
            'result': 'WIN',
            'r_multiple': 2.0
        }
    ]
    df = pd.DataFrame(trades)
    
    # 1% risk on 100,000 = 1,000 risk per trade
    # Win 1: 1R = 1,000 PnL
    # Win 2: 2R = 2,000 PnL
    # Total PnL = 3,000. Peak balance = 103,000, max drawdown = 0.
    
    stats = calculate_statistics(df, initial_capital=100000.0, risk_percent=1.0)
    
    assert stats['total_trades'] == 2
    assert stats['win_rate_pct'] == 100.0
    assert stats['loss_rate_pct'] == 0.0
    assert stats['avg_win_r'] == 1.5  # (1 + 2) / 2
    assert stats['avg_loss_r'] == 0.0
    assert stats['avg_win_dollar'] == 1500.0
    assert stats['avg_loss_dollar'] == 0.0
    assert stats['avg_rr'] == float('inf')
    assert stats['expectancy_r'] == 1.5
    assert stats['expectancy_dollar'] == 1500.0
    assert stats['max_drawdown_dollar'] == 0.0
    assert stats['max_drawdown_pct'] == 0.0
    assert stats['profit_factor'] == float('inf')
    
    # Check session breakdown
    assert 'London' in stats['session_performance']
    assert 'NY' in stats['session_performance']
    
    london_stats = stats['session_performance']['London']
    assert london_stats['total_trades'] == 1
    assert london_stats['win_rate_pct'] == 100.0
    
    # Check pattern breakdown
    assert 'FVG' in stats['pattern_performance']
    assert 'STRUCT' in stats['pattern_performance']
    
    fvg_stats = stats['pattern_performance']['FVG']
    assert fvg_stats['total_trades'] == 1
    assert fvg_stats['avg_win_r'] == 1.0

def test_statistics_mixed_trades():
    """Test statistics with a mix of winning and losing trades."""
    trades = [
        {
            'entry_time': pd.Timestamp('2023-01-01 09:00:00'),
            'exit_time': pd.Timestamp('2023-01-01 10:00:00'),
            'direction': 'BULLISH',
            'pattern_type': 'FVG',
            'result': 'WIN',
            'r_multiple': 2.0
        },
        {
            'entry_time': pd.Timestamp('2023-01-02 10:00:00'),
            'exit_time': pd.Timestamp('2023-01-02 11:00:00'),
            'direction': 'BEARISH',
            'pattern_type': 'FVG',
            'result': 'LOSS',
            'r_multiple': -1.0
        },
        {
            'entry_time': pd.Timestamp('2023-01-03 15:00:00'),
            'exit_time': pd.Timestamp('2023-01-03 16:00:00'),
            'direction': 'BULLISH',
            'pattern_type': 'STRUCT',
            'result': 'WIN',
            'r_multiple': 3.0
        }
    ]
    df = pd.DataFrame(trades)
    
    # Risk 1000 per trade
    # T1: WIN, +2000 PnL (Balance 102000, Peak 102000)
    # T2: LOSS, -1000 PnL (Balance 101000, Drawdown 1000, pct ~ 0.98%)
    # T3: WIN, +3000 PnL (Balance 104000, Peak 104000)
    
    stats = calculate_statistics(df, initial_capital=100000.0, risk_percent=1.0)
    
    assert stats['total_trades'] == 3
    assert stats['win_rate_pct'] == pytest.approx(66.666, rel=1e-3)
    assert stats['loss_rate_pct'] == pytest.approx(33.333, rel=1e-3)
    
    assert stats['avg_win_r'] == 2.5 # (2+3)/2
    assert stats['avg_loss_r'] == -1.0
    
    assert stats['avg_win_dollar'] == 2500.0
    assert stats['avg_loss_dollar'] == -1000.0
    
    assert stats['avg_rr'] == 2.5
    
    # Expectancy = (0.6666 * 2.5) - (0.3333 * 1.0) = 1.6666 - 0.3333 = 1.3333
    assert stats['expectancy_r'] == pytest.approx(1.3333, rel=1e-3)
    assert stats['expectancy_dollar'] == pytest.approx(1333.33, rel=1e-3)
    
    assert stats['max_drawdown_dollar'] == 1000.0
    assert stats['max_drawdown_pct'] == pytest.approx(1000 / 102000 * 100, rel=1e-3)
    
    # Profit factor = Gross Profit / Gross Loss = 5000 / 1000 = 5.0
    assert stats['profit_factor'] == 5.0
