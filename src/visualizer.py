import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import Candle, TradeResult
from src.pattern_detector import SwingPoint, DetailedPatternEvent
from src.statistics import PerformanceReport

def generate_html_report(
    candles: List[Candle],
    swings: List[SwingPoint],
    patterns: List[DetailedPatternEvent],
    trades: List[TradeResult],
    equity_df: pd.DataFrame,
    stats: PerformanceReport,
    output_dir: str = 'reports/'
) -> str:
    
    # 1. Setup Grid
    fig = make_subplots(
        rows=8, cols=4,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "xy", "colspan": 4, "rowspan": 2}, None, None, None],
            [None, None, None, None],
            [{"type": "xy", "colspan": 4}, None, None, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
            [{"type": "table", "colspan": 4}, None, None, None]
        ],
        subplot_titles=(
            "", "", "", "", 
            "Price Chart & Patterns", 
            "Volume", 
            "Equity Curve", "Underwater Chart (Drawdown)",
            "Session Win Rate", "Pattern Avg PnL",
            "Monthly Returns (%)", "R-Multiple Distribution",
            "Monte Carlo Summary"
        ),
        vertical_spacing=0.04,
        row_heights=[0.05, 0.25, 0.1, 0.05, 0.15, 0.15, 0.15, 0.1]
    )
    
    df_c = pd.DataFrame([vars(c) for c in candles])
    
    # 1. Indicators (Row 1)
    net_profit = sum(t.pnl for t in trades)
    fig.add_trace(go.Indicator(mode="number", value=net_profit, title={"text": "Net Profit ($)"}, number={'prefix': '$'}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=stats.win_rate*100, title={"text": "Win Rate (%)"}, number={'suffix': '%'}), row=1, col=2)
    pf = stats.profit_factor if stats.profit_factor != float('inf') else 999.99
    fig.add_trace(go.Indicator(mode="number", value=pf, title={"text": "Profit Factor"}), row=1, col=3)
    fig.add_trace(go.Indicator(mode="number", value=stats.max_drawdown_pct, title={"text": "Max Drawdown (%)"}, number={'suffix': '%'}), row=1, col=4)
    
    # 2. Candlestick (Row 2-3)
    fig.add_trace(go.Candlestick(
        x=df_c['timestamp'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'],
        increasing_line_color='#00C851', decreasing_line_color='#FF4444', name='Price'
    ), row=2, col=1)
    
    # Overlay Swings
    sh_ts = [s.timestamp for s in swings if s.type == 'HIGH']
    sh_p = [s.price for s in swings if s.type == 'HIGH']
    sl_ts = [s.timestamp for s in swings if s.type == 'LOW']
    sl_p = [s.price for s in swings if s.type == 'LOW']
    fig.add_trace(go.Scatter(x=sh_ts, y=sh_p, mode='markers', marker=dict(symbol='diamond', color='#FF4444', size=8), name='Swing High', legendgroup='Swings'), row=2, col=1)
    fig.add_trace(go.Scatter(x=sl_ts, y=sl_p, mode='markers', marker=dict(symbol='diamond', color='#00C851', size=8), name='Swing Low', legendgroup='Swings'), row=2, col=1)
    
    # Overlay BOS & CHoCH
    for ptype, marker_b, marker_s in [('BOS', 'triangle-up', 'triangle-down'), ('CHoCH', 'star', 'star')]:
        b_ts = [p.timestamp for p in patterns if p.pattern_type == ptype and p.direction == 'BULLISH']
        b_p = [p.price_level for p in patterns if p.pattern_type == ptype and p.direction == 'BULLISH']
        s_ts = [p.timestamp for p in patterns if p.pattern_type == ptype and p.direction == 'BEARISH']
        s_p = [p.price_level for p in patterns if p.pattern_type == ptype and p.direction == 'BEARISH']
        fig.add_trace(go.Scatter(x=b_ts, y=b_p, mode='markers', marker=dict(symbol=marker_b, color='#00C851', size=12), name=f'{ptype} Bullish', legendgroup=ptype), row=2, col=1)
        fig.add_trace(go.Scatter(x=s_ts, y=s_p, mode='markers', marker=dict(symbol=marker_s, color='#FF4444', size=12), name=f'{ptype} Bearish', legendgroup=ptype), row=2, col=1)
        
    # Unmitigated FVGs using polygon patches
    for direction, color, name in [('BULLISH', 'rgba(0,200,81,0.2)', 'Bullish FVG'), ('BEARISH', 'rgba(255,68,68,0.2)', 'Bearish FVG')]:
        px, py = [], []
        for p in patterns:
            if p.pattern_type == 'FVG' and p.direction == direction and p.metadata.get('status') == 'unmitigated':
                t_start = p.timestamp
                t_end = df_c['timestamp'].iloc[-1]
                top = p.price_levels.get('top', p.price_level)
                bot = p.price_levels.get('bottom', p.price_level)
                px.extend([t_start, t_end, t_end, t_start, t_start, None])
                py.extend([top, top, bot, bot, top, None])
        if px:
            fig.add_trace(go.Scatter(x=px, y=py, fill='toself', fillcolor=color, line=dict(color='rgba(255,255,255,0)'), name=name, legendgroup='FVG', hoverinfo='skip'), row=2, col=1)
            
    # Trade entries/exits
    el_ts = [t.entry_time for t in trades if t.direction == 'LONG']
    el_p = [t.entry_price for t in trades if t.direction == 'LONG']
    es_ts = [t.entry_time for t in trades if t.direction == 'SHORT']
    es_p = [t.entry_price for t in trades if t.direction == 'SHORT']
    ex_ts = [t.exit_time for t in trades]
    ex_p = [t.exit_price for t in trades]
    fig.add_trace(go.Scatter(x=el_ts, y=el_p, mode='markers', marker=dict(symbol='triangle-up', color='#33B5E5', size=14, line=dict(width=1, color='white')), name='Long Entry', legendgroup='Trades'), row=2, col=1)
    fig.add_trace(go.Scatter(x=es_ts, y=es_p, mode='markers', marker=dict(symbol='triangle-down', color='#FF4444', size=14, line=dict(width=1, color='white')), name='Short Entry', legendgroup='Trades'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ex_ts, y=ex_p, mode='markers', marker=dict(symbol='x', color='white', size=10), name='Exit', legendgroup='Trades'), row=2, col=1)
    
    # 3. Volume (Row 4)
    if 'volume' in df_c.columns:
        v_colors = ['#00C851' if o <= c else '#FF4444' for o, c in zip(df_c['open'], df_c['close'])]
        fig.add_trace(go.Bar(x=df_c['timestamp'], y=df_c['volume'], marker_color=v_colors, name='Volume'), row=4, col=1)
        
    # 4. Equity & Underwater (Row 5)
    if not equity_df.empty:
        fig.add_trace(go.Scatter(x=equity_df['timestamp'], y=equity_df['peak'], name='Peak', line=dict(color='#00C851', dash='dash')), row=5, col=1)
        fig.add_trace(go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'], name='Equity', line=dict(color='#33B5E5'), fill='tonexty', fillcolor='rgba(255,68,68,0.2)'), row=5, col=1)
        
        # Underwater (-drawdown so it points down)
        fig.add_trace(go.Scatter(x=equity_df['timestamp'], y=-equity_df['drawdown'], name='Drawdown (%)', fill='tozeroy', fillcolor='rgba(255,68,68,0.5)', line=dict(color='#FF4444')), row=5, col=3)
        
    # 5. Session & Pattern (Row 6)
    sessions = list(stats.breakdown.by_session.keys())
    win_rates = [stats.breakdown.by_session[s]['win_rate']*100 for s in sessions]
    fig.add_trace(go.Bar(x=sessions, y=win_rates, marker_color='#33B5E5', name='Win Rate'), row=6, col=1)
    
    patterns_lbl = list(stats.breakdown.by_pattern.keys())
    pat_pnl = [stats.breakdown.by_pattern[p].get('expectancy', 0) for p in patterns_lbl]
    fig.add_trace(go.Bar(x=patterns_lbl, y=pat_pnl, marker_color='#00C851', name='Avg PnL'), row=6, col=3)
    
    # 6. Heatmap & Histogram (Row 7)
    if not equity_df.empty:
        df_eq = equity_df.drop_duplicates(subset=['timestamp'], keep='last').set_index('timestamp')
        monthly = df_eq['equity'].resample('ME').last()
        if len(monthly) > 0:
            initial = equity_df['equity'].iloc[0]
            monthly_ret = monthly.pct_change() * 100
            monthly_ret.iloc[0] = ((monthly.iloc[0] / initial) - 1) * 100
            
            ret_df = pd.DataFrame({'ret': monthly_ret})
            dt_index = pd.DatetimeIndex(ret_df.index)
            ret_df['year'] = dt_index.year
            ret_df['month'] = dt_index.strftime('%b')
            pivot = ret_df.pivot(index='year', columns='month', values='ret')
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            cols = [m for m in months_order if m in pivot.columns]
            pivot = pivot[cols]
            
            # Need to format text properly, substituting NaN with empty string
            text_vals = []
            for row in pivot.values:
                text_vals.append([f"{v:.1f}%" if not np.isnan(v) else "" for v in row])
                
            fig.add_trace(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index, 
                colorscale='RdYlGn', zmid=0, showscale=False,
                text=text_vals, texttemplate="%{text}", hoverinfo="z"
            ), row=7, col=1)
            
    r_mults = [t.r_multiple for t in trades]
    if r_mults:
        fig.add_trace(go.Histogram(x=r_mults, nbinsx=20, marker_color='#33B5E5', name='R-Multiples'), row=7, col=3)
        
    # 7. Table (Row 8)
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'], fill_color='#2c2c2c', font=dict(color='white')),
        cells=dict(values=[
            ['Median Drawdown', '95% Drawdown', '5% Drawdown', 'Prob of Ruin'],
            [f"{stats.monte_carlo.median_drawdown_pct:.2f}%", f"{stats.monte_carlo.p95_drawdown_pct:.2f}%", 
             f"{stats.monte_carlo.p05_drawdown_pct:.2f}%", f"{stats.monte_carlo.probability_of_ruin_pct:.2f}%"]
        ], fill_color='#1c1c1c', font=dict(color='white'))
    ), row=8, col=1)
    
    # Layout and Buttons
    vis_all = [True] * len(fig.data)
    vis_no_fvg = [False if t.name and 'FVG' in t.name else True for t in fig.data]
    vis_no_struct = [False if t.name and ('BOS' in t.name or 'CHoCH' in t.name) else True for t in fig.data]
    
    fig.update_layout(
        template='plotly_dark',
        height=1800,
        title="ICT Backtester Pro - Interactive Report",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.02, showactive=True,
            buttons=[
                dict(label="All Layers", method="update", args=[{"visible": vis_all}]),
                dict(label="Hide FVGs", method="update", args=[{"visible": vis_no_fvg}]),
                dict(label="Hide Structure", method="update", args=[{"visible": vis_no_struct}])
            ]
        )]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"ict_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    fig.write_html(filename)
    return filename
