import pytest
import numpy as np
from datetime import datetime, timedelta
from src.models import TradeResult
from src.statistics import calculate_statistics, PerformanceReport

def test_zero_trades_edge_case():
    report = calculate_statistics([])
    assert isinstance(report, PerformanceReport)
    assert report.total_trades == 0
    assert report.win_rate == 0.0
    assert report.profit_factor == 0.0
    assert report.sharpe_ratio == 0.0
    assert report.max_drawdown_pct == 0.0

def test_all_winning_trades_edge_case():
    trades = [
        TradeResult(
            entry_time=datetime(2024, 1, 1, 10, 0), exit_time=datetime(2024, 1, 1, 11, 0),
            entry_price=1.1000, exit_price=1.1050, direction='LONG', size=1.0,
            pnl=500.0, r_multiple=2.0, pattern_type='BOS', session='London'
        ),
        TradeResult(
            entry_time=datetime(2024, 1, 2, 10, 0), exit_time=datetime(2024, 1, 2, 11, 0),
            entry_price=1.1000, exit_price=1.1050, direction='LONG', size=1.0,
            pnl=500.0, r_multiple=2.0, pattern_type='FVG', session='London'
        )
    ]
    
    report = calculate_statistics(trades)
    
    assert report.total_trades == 2
    assert report.win_rate == 1.0
    assert report.loss_rate == 0.0
    assert report.profit_factor == float('inf') # Core requirement
    assert report.consecutive_wins == 2
    assert report.consecutive_losses == 0

def test_monte_carlo_drawdown_stats():
    # Generate 20 trades: 10 wins of $100, 10 losses of $50
    trades = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(10):
        # Wins
        trades.append(TradeResult(
            entry_time=base_time + timedelta(days=i, hours=1), exit_time=base_time + timedelta(days=i, hours=2),
            entry_price=1.0, exit_price=1.1, direction='LONG', size=1.0,
            pnl=100.0, r_multiple=2.0, pattern_type='BOS', session='NY'
        ))
        # Losses
        trades.append(TradeResult(
            entry_time=base_time + timedelta(days=i+10, hours=1), exit_time=base_time + timedelta(days=i+10, hours=2),
            entry_price=1.0, exit_price=0.9, direction='LONG', size=1.0,
            pnl=-50.0, r_multiple=-1.0, pattern_type='FVG', session='London'
        ))
        
    report = calculate_statistics(trades)
    
    assert report.total_trades == 20
    assert report.expectancy == pytest.approx(25.0) # (0.5 * 100) + (0.5 * -50) = 50 - 25 = 25
    
    # Check Monte Carlo (triggers for > 5 trades)
    mc = report.monte_carlo
    assert mc.p95_drawdown_pct > mc.median_drawdown_pct
    assert mc.median_drawdown_pct >= mc.p05_drawdown_pct
    # Since total pnl is +500 and max possible sequential loss is -500 (all losses in a row)
    # on a $10,000 account, max DD is around 5%. Probability of ruin (50% DD) should be 0.
    assert mc.probability_of_ruin_pct == 0.0

def test_breakdown_metrics_populated():
    trades = [
        TradeResult(
            entry_time=datetime(2024, 1, 1, 9, 0), exit_time=datetime(2024, 1, 1, 10, 0), # 9am is NY usually but using string
            entry_price=1.0, exit_price=1.1, direction='LONG', size=1.0,
            pnl=100.0, r_multiple=2.0, pattern_type='BOS', session='London'
        ),
        TradeResult(
            entry_time=datetime(2024, 1, 2, 15, 0), exit_time=datetime(2024, 1, 2, 16, 0),
            entry_price=1.0, exit_price=0.9, direction='LONG', size=1.0,
            pnl=-50.0, r_multiple=-1.0, pattern_type='FVG', session='NY'
        )
    ]
    
    report = calculate_statistics(trades)
    
    assert 'London' in report.breakdown.by_session
    assert 'NY' in report.breakdown.by_session
    assert report.breakdown.by_session['London']['win_rate'] == 1.0
    assert report.breakdown.by_session['NY']['win_rate'] == 0.0
    
    assert 'BOS' in report.breakdown.by_pattern
    assert 'FVG' in report.breakdown.by_pattern
