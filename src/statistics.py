import pandas as pd
import numpy as np
from typing import Dict, Any

class BacktestStatistics:
    def __init__(self, trades_df: pd.DataFrame, initial_capital: float = 100000.0, risk_percent: float = 1.0):
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.risk_percent = risk_percent
        self.risk_amount = initial_capital * (risk_percent / 100.0)
        
        # Prepare data if not empty
        if not self.trades_df.empty:
            self.trades_df['pnl'] = self.trades_df['r_multiple'] * self.risk_amount
            self.trades_df['cumulative_pnl'] = self.trades_df['pnl'].cumsum()
            self.trades_df['balance'] = self.initial_capital + self.trades_df['cumulative_pnl']
            
            # Determine session based on entry_time (UTC assumed)
            # Asian: 00:00 - 08:00, London: 08:00 - 13:00, NY: 13:00 - 21:00, Other: 21:00 - 00:00
            def get_session(dt):
                if pd.isna(dt):
                    return "Unknown"
                hour = dt.hour
                if 0 <= hour < 8:
                    return "Asian"
                elif 8 <= hour < 13:
                    return "London"
                elif 13 <= hour < 21:
                    return "NY"
                else:
                    return "Other"
            
            if 'entry_time' in self.trades_df.columns:
                self.trades_df['session'] = self.trades_df['entry_time'].apply(get_session)

    def calculate_all(self) -> Dict[str, Any]:
        """Calculates and returns all requested statistics."""
        if self.trades_df.empty:
            return self._empty_stats()
            
        stats = self._calculate_basic_stats(self.trades_df)
        stats['session_performance'] = self._calculate_breakdown('session')
        stats['pattern_performance'] = self._calculate_breakdown('pattern_type')
        
        return stats

    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return self._empty_stats()

        total_trades = len(df)
        wins = df[df['result'] == 'WIN']
        losses = df[df['result'] == 'LOSS']
        
        num_wins = len(wins)
        num_losses = len(losses)
        
        win_rate = (num_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        loss_rate = (num_losses / total_trades * 100.0) if total_trades > 0 else 0.0
        
        avg_win_r = wins['r_multiple'].mean() if num_wins > 0 else 0.0
        avg_loss_r = losses['r_multiple'].mean() if num_losses > 0 else 0.0
        
        avg_win_dollar = wins['pnl'].mean() if num_wins > 0 else 0.0
        avg_loss_dollar = losses['pnl'].mean() if num_losses > 0 else 0.0
        
        # Average R:R (Average Win R / absolute Average Loss R)
        if avg_loss_r < 0:
            avg_rr = avg_win_r / abs(avg_loss_r)
        elif num_wins > 0 and num_losses == 0:
            avg_rr = float('inf')
        else:
            avg_rr = 0.0
            
        # Expectancy in R: (Win Rate * Avg Win) - (Loss Rate * absolute Avg Loss)
        # Using decimals for rates
        expectancy_r = ((win_rate / 100.0) * avg_win_r) - ((loss_rate / 100.0) * abs(avg_loss_r))
        expectancy_dollar = expectancy_r * self.risk_amount
        
        gross_profit = wins['pnl'].sum() if num_wins > 0 else 0.0
        gross_loss = abs(losses['pnl'].sum()) if num_losses > 0 else 0.0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        # Drawdown calculation
        df['peak_balance'] = df['balance'].cummax()
        df['drawdown_dollar'] = df['peak_balance'] - df['balance']
        df['drawdown_pct'] = (df['drawdown_dollar'] / df['peak_balance']) * 100.0
        
        max_drawdown_dollar = df['drawdown_dollar'].max() if not df['drawdown_dollar'].empty else 0.0
        max_drawdown_pct = df['drawdown_pct'].max() if not df['drawdown_pct'].empty else 0.0
        
        # Annualized Sharpe Ratio
        # Convert to daily returns to compute standard daily Sharpe
        sharpe_ratio = 0.0
        if 'exit_time' in df.columns:
            daily_returns = df.set_index('exit_time')['pnl'].resample('D').sum()
            daily_pct_returns = daily_returns / self.initial_capital
            if len(daily_pct_returns) > 1 and daily_pct_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * (daily_pct_returns.mean() / daily_pct_returns.std())
        
        return {
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "loss_rate_pct": loss_rate,
            "avg_win_dollar": avg_win_dollar,
            "avg_loss_dollar": avg_loss_dollar,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "avg_rr": avg_rr,
            "expectancy_r": expectancy_r,
            "expectancy_dollar": expectancy_dollar,
            "max_drawdown_dollar": max_drawdown_dollar,
            "max_drawdown_pct": max_drawdown_pct,
            "profit_factor": profit_factor,
            "annualized_sharpe_ratio": sharpe_ratio
        }

    def _calculate_breakdown(self, column_name: str) -> Dict[str, Any]:
        breakdown = {}
        if column_name not in self.trades_df.columns:
            return breakdown
            
        unique_vals = self.trades_df[column_name].dropna().unique()
        for val in unique_vals:
            subset = self.trades_df[self.trades_df[column_name] == val].copy()
            breakdown[str(val)] = self._calculate_basic_stats(subset)
            
        return breakdown

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "avg_win_dollar": 0.0,
            "avg_loss_dollar": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "avg_rr": 0.0,
            "expectancy_r": 0.0,
            "expectancy_dollar": 0.0,
            "max_drawdown_dollar": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "annualized_sharpe_ratio": 0.0,
            "session_performance": {},
            "pattern_performance": {}
        }

def calculate_statistics(trades_df: pd.DataFrame, initial_capital: float = 100000.0, risk_percent: float = 1.0) -> Dict[str, Any]:
    stats_engine = BacktestStatistics(trades_df, initial_capital, risk_percent)
    return stats_engine.calculate_all()
