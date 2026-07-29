import logging
import math
import random
from typing import List, Dict, Any, Optional
from datetime import timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from src.models import TradeResult

logger = logging.getLogger(__name__)

class MonteCarloReport(BaseModel):
    median_drawdown_pct: float = 0.0
    p95_drawdown_pct: float = 0.0
    p05_drawdown_pct: float = 0.0
    probability_of_ruin_pct: float = 0.0

class BreakdownReport(BaseModel):
    by_session: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_pattern: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_day_of_week: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_hour_of_day: Dict[str, Dict[str, float]] = Field(default_factory=dict)

class PerformanceReport(BaseModel):
    total_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    breakeven_rate: float = 0.0
    
    average_win: float = 0.0
    average_loss: float = 0.0
    avg_rr: float = 0.0
    expectancy: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_duration: int = 0
    
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    avg_time_in_trade_minutes: float = 0.0
    median_time_in_trade_minutes: float = 0.0
    
    breakdown: BreakdownReport = Field(default_factory=BreakdownReport)
    monte_carlo: MonteCarloReport = Field(default_factory=MonteCarloReport)


def calculate_statistics(trades: List[TradeResult], initial_capital: float = 10000.0, risk_free_rate: float = 0.0) -> PerformanceReport:
    report = PerformanceReport()
    
    if not trades:
        logger.warning("Zero trades provided for statistics calculation. Returning N/A report.")
        return report

    df = pd.DataFrame([vars(t) for t in trades])
    
    # Core Metrics
    report.total_trades = len(df)
    
    # Categorize trades
    df['is_win'] = df['pnl'] > 0
    df['is_loss'] = df['pnl'] < 0
    df['is_be'] = df['pnl'] == 0
    
    wins = df[df['is_win']]
    losses = df[df['is_loss']]
    
    report.win_rate = len(wins) / report.total_trades
    report.loss_rate = len(losses) / report.total_trades
    report.breakeven_rate = len(df[df['is_be']]) / report.total_trades
    
    report.average_win = wins['pnl'].mean() if not wins.empty else 0.0
    report.average_loss = losses['pnl'].mean() if not losses.empty else 0.0
    
    # Weighted average R:R (assuming r_multiple exists)
    if 'r_multiple' in df.columns:
        report.avg_rr = df['r_multiple'].mean()
        
    report.expectancy = (report.win_rate * report.average_win) + (report.loss_rate * report.average_loss)
    
    gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
    
    if gross_loss == 0:
        if gross_profit > 0:
            report.profit_factor = float('inf')
            logger.warning("All winning trades! Profit factor is infinity.")
        else:
            report.profit_factor = 0.0
    else:
        report.profit_factor = gross_profit / gross_loss
        
    # Drawdown and Equity Curve
    df['equity'] = initial_capital + df['pnl'].cumsum()
    df['peak'] = df['equity'].cummax()
    df['drawdown_usd'] = df['peak'] - df['equity']
    df['drawdown_pct'] = (df['drawdown_usd'] / df['peak']) * 100

    report.max_drawdown_pct = df['drawdown_pct'].max()
    report.max_drawdown_usd = df['drawdown_usd'].max()
    
    # Drawdown duration
    in_dd = df['drawdown_usd'] > 0
    dd_blocks = (~in_dd).cumsum()
    dd_durations = in_dd.groupby(dd_blocks).sum()
    report.max_drawdown_duration = int(dd_durations.max()) if not dd_durations.empty else 0
    
    # Time in trade
    if 'exit_time' in df.columns and 'entry_time' in df.columns:
        valid_times = df.dropna(subset=['exit_time', 'entry_time'])
        if not valid_times.empty:
            duration = (valid_times['exit_time'] - valid_times['entry_time']).dt.total_seconds() / 60.0
            report.avg_time_in_trade_minutes = duration.mean()
            report.median_time_in_trade_minutes = duration.median()

    # Consecutive wins/losses
    df['win_block'] = (~df['is_win']).cumsum()
    df['loss_block'] = (~df['is_loss']).cumsum()
    win_streaks = df['is_win'].groupby(df['win_block']).sum()
    loss_streaks = df['is_loss'].groupby(df['loss_block']).sum()
    
    report.consecutive_wins = int(win_streaks.max()) if not win_streaks.empty else 0
    report.consecutive_losses = int(loss_streaks.max()) if not loss_streaks.empty else 0
    
    # Ratios
    returns = df['pnl'] / initial_capital
    if len(returns) > 1 and returns.std() > 0:
        annual_factor = math.sqrt(252 * 24) # Assuming hourly trades approx, standard approach is via duration, but we'll use a fixed arbitrary scaling or just standard dev per trade multiplied by sqrt of N
        # standard sharpe = mean / std * sqrt(N)
        mean_return = returns.mean()
        std_return = returns.std()
        
        report.sharpe_ratio = (mean_return - risk_free_rate) / std_return * math.sqrt(252) # Simplified annualization
        
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0
        if downside_std > 0:
            report.sortino_ratio = (mean_return - risk_free_rate) / downside_std * math.sqrt(252)
            
        if report.max_drawdown_pct > 0:
            total_return = (df['equity'].iloc[-1] - initial_capital) / initial_capital
            # Approximate annualized return
            if not valid_times.empty:
                days = (valid_times['exit_time'].max() - valid_times['entry_time'].min()).days
                days = max(days, 1)
                annual_return = (1 + total_return) ** (365 / days) - 1
                report.calmar_ratio = annual_return / (report.max_drawdown_pct / 100)
    
    # Breakdown Analysis
    bd = BreakdownReport()
    
    def _agg(group):
        c = len(group)
        w = len(group[group['pnl'] > 0])
        return {
            'win_rate': w / c if c > 0 else 0.0,
            'expectancy': group['pnl'].mean() if c > 0 else 0.0,
            'count': c
        }

    if 'session' in df.columns:
        for sess, grp in df.groupby('session'):
            bd.by_session[str(sess)] = _agg(grp)
            
    if 'pattern_type' in df.columns:
        for pat, grp in df.groupby('pattern_type'):
            bd.by_pattern[str(pat)] = _agg(grp)
            
    if 'entry_time' in df.columns:
        df['dow'] = df['entry_time'].dt.day_name()
        df['hour'] = df['entry_time'].dt.hour
        
        for dow, grp in df.groupby('dow'):
            bd.by_day_of_week[str(dow)] = _agg(grp)
            
        for hr, grp in df.groupby('hour'):
            bd.by_hour_of_day[str(hr)] = _agg(grp)
            
    report.breakdown = bd
    
    # Monte Carlo Simulation
    if len(trades) > 5:
        mc = MonteCarloReport()
        pnls = df['pnl'].values
        mc_dds = []
        ruin_count = 0
        ruin_threshold = initial_capital * 0.5
        
        for _ in range(10000):
            np.random.shuffle(pnls)
            eq = initial_capital + np.cumsum(pnls)
            
            if np.any(eq < ruin_threshold):
                ruin_count += 1
                
            peaks = np.maximum.accumulate(eq)
            dds = (peaks - eq) / peaks * 100
            mc_dds.append(np.max(dds))
            
        mc.p05_drawdown_pct = float(np.percentile(mc_dds, 5))
        mc.median_drawdown_pct = float(np.median(mc_dds))
        mc.p95_drawdown_pct = float(np.percentile(mc_dds, 95))
        mc.probability_of_ruin_pct = (ruin_count / 10000.0) * 100
        
        report.monte_carlo = mc
        
    return report
