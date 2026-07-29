import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any

class ReportVisualizer:
    def __init__(self, trades_df: pd.DataFrame, stats: Dict[str, Any], output_dir: str = 'reports'):
        self.trades_df = trades_df.copy()
        self.stats = stats
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        if not self.trades_df.empty:
            if 'r_multiple' in self.trades_df.columns:
                self.trades_df['cumulative_r'] = self.trades_df['r_multiple'].cumsum()
                self.trades_df['peak_r'] = self.trades_df['cumulative_r'].cummax()
                self.trades_df['drawdown_r'] = self.trades_df['peak_r'] - self.trades_df['cumulative_r']
                
                # Determine time for plotting
                plot_time = self.trades_df.get('exit_time', self.trades_df.get('entry_time', pd.Series(index=self.trades_df.index)))
                if plot_time.isnull().any() and 'entry_time' in self.trades_df.columns:
                    plot_time = plot_time.fillna(self.trades_df['entry_time'])
                self.trades_df['plot_time'] = pd.to_datetime(plot_time)

    def generate_report(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        if self.trades_df.empty:
            html_content = "<html><body><h1>No trades to visualize</h1></body></html>"
            with open(filepath, 'w') as f:
                f.write(html_content)
            return filepath
            
        # 1. Equity Curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=self.trades_df['plot_time'],
            y=self.trades_df['cumulative_r'],
            mode='lines',
            name='Cumulative R',
            line=dict(color='blue', width=2)
        ))
        fig_equity.update_layout(title="Equity Curve (Cumulative R)", xaxis_title="Time", yaxis_title="Cumulative R")
        
        # 2. Underwater Chart (Drawdown)
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=self.trades_df['plot_time'],
            y=-self.trades_df['drawdown_r'], # Negative for underwater visual
            fill='tozeroy',
            mode='lines',
            name='Drawdown (R)',
            line=dict(color='red', width=1)
        ))
        fig_drawdown.update_layout(title="Underwater Chart (Drawdown in R)", xaxis_title="Time", yaxis_title="Drawdown (R)")
        
        # 3. Win rate by session (Bar chart)
        session_stats = self.stats.get('session_performance', {})
        sessions = list(session_stats.keys())
        win_rates = [session_stats[s].get('win_rate_pct', 0) for s in sessions]
        
        fig_session = go.Figure(data=[
            go.Bar(name='Win Rate', x=sessions, y=win_rates, marker_color='lightgreen')
        ])
        fig_session.update_layout(title="Win Rate by Session (%)", xaxis_title="Session", yaxis_title="Win Rate (%)", yaxis=dict(range=[0, 100]))
        
        # 4. Trade distribution histogram (R-multiples)
        fig_dist = px.histogram(
            self.trades_df, x="r_multiple", 
            title="Trade Distribution (R-Multiples)", 
            labels={"r_multiple": "R-Multiple"},
            nbins=20,
            color_discrete_sequence=['purple']
        )
        
        # 5. Monthly returns heatmap
        fig_heatmap = self._create_heatmap()
        
        # Combine into HTML
        html_content = self._build_html(
            fig_equity, fig_drawdown, fig_session, fig_dist, fig_heatmap, timestamp
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filepath
        
    def _create_heatmap(self) -> go.Figure:
        df = self.trades_df.copy()
        if 'plot_time' not in df.columns or df['plot_time'].isnull().all():
            return go.Figure().update_layout(title="Monthly Returns (No Time Data)")
            
        df['year'] = df['plot_time'].dt.year
        df['month'] = df['plot_time'].dt.month
        
        monthly_returns = df.groupby(['year', 'month'])['r_multiple'].sum().reset_index()
        monthly_pivot = monthly_returns.pivot(index='year', columns='month', values='r_multiple').fillna(0)
        
        # Ensure all 12 months exist as columns
        for m in range(1, 13):
            if m not in monthly_pivot.columns:
                monthly_pivot[m] = 0.0
                
        # Sort columns to ensure Jan-Dec order
        monthly_pivot = monthly_pivot[sorted(monthly_pivot.columns)]
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_pivot.values,
            x=month_names,
            y=monthly_pivot.index,
            colorscale='RdYlGn',
            zmid=0
        ))
        fig.update_layout(
            title="Monthly Returns Heatmap (R)", 
            xaxis_title="Month", 
            yaxis_title="Year", 
            yaxis=dict(autorange="reversed", type='category')
        )
        return fig
        
    def _build_html(self, fig_equity, fig_drawdown, fig_session, fig_dist, fig_heatmap, timestamp: str) -> str:
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
                h1 {{ color: #333; }}
                .stats-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
                .stat-box {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 200px; text-align: center; }}
                .stat-box h3 {{ margin-top: 0; color: #666; font-size: 14px; text-transform: uppercase; }}
                .stat-box p {{ margin: 0; font-size: 24px; font-weight: bold; color: #333; }}
                .chart-container {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
                .row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .col {{ flex: 1; min-width: 400px; }}
            </style>
        </head>
        <body>
            <h1>ICT Backtest Report</h1>
            
            <div class="stats-container">
                <div class="stat-box"><h3>Total Trades</h3><p>{self.stats.get('total_trades', 0)}</p></div>
                <div class="stat-box"><h3>Win Rate</h3><p>{self.stats.get('win_rate_pct', 0):.2f}%</p></div>
                <div class="stat-box"><h3>Expectancy (R)</h3><p>{self.stats.get('expectancy_r', 0):.2f}</p></div>
                <div class="stat-box"><h3>Profit Factor</h3><p>{self.stats.get('profit_factor', 0):.2f}</p></div>
                <div class="stat-box"><h3>Max Drawdown (R)</h3><p>{self.stats.get('max_drawdown_dollar', 0) / self.stats.get('avg_win_dollar', 1) * self.stats.get('avg_win_r', 1) if self.stats.get('avg_win_dollar') else 0:.2f}</p></div>
            </div>
            
            <div class="chart-container">
                {fig_equity.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="chart-container">
                {fig_drawdown.to_html(full_html=False, include_plotlyjs=False)}
            </div>
            
            <div class="row">
                <div class="col chart-container">
                    {fig_dist.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                <div class="col chart-container">
                    {fig_session.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>
            
            <div class="chart-container">
                {fig_heatmap.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </body>
        </html>
        '''
        return html

def generate_visual_report(trades_df: pd.DataFrame, stats: Dict[str, Any], output_dir: str = 'reports') -> str:
    """Helper function to instantiate and run the visualizer."""
    visualizer = ReportVisualizer(trades_df, stats, output_dir)
    return visualizer.generate_report()
