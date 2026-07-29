import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from src.pattern_detector import Direction, FVGEvent, StructureEvent, detect_fvgs, detect_structure_events
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: Direction
    stop_loss: float
    take_profit: float
    pattern_type: str = "UNKNOWN"
    
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None

class BacktestEngine:
    """
    Event-driven, walk-forward simulation engine.
    Processes candles chronologically to prevent lookahead bias.
    """
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self.trades: List[Trade] = []
        self.pending_setups: List[Dict[str, Any]] = []
        self.active_trade: Optional[Trade] = None
        
        self.rr_ratio = config.get('rr_ratio', 2.0)
        self.spread_pips = config.get('spread_assumption', 1.5)
        self.stop_loss_pips = config.get('stop_loss_pips', 10.0)
        self.instrument = config.get('instrument', 'XAUUSD').upper()
        
        if 'XAU' in self.instrument:
            self.pip_size = 0.1
        elif 'JPY' in self.instrument:
            self.pip_size = 0.01
        else:
            self.pip_size = 0.0001
            
        self.spread = self.spread_pips * self.pip_size
        self.sl_dist = self.stop_loss_pips * self.pip_size

    def run(self) -> pd.DataFrame:
        logger.info("Starting backtest...")
        
        # Pre-calculate events but map them strictly by their confirmation time.
        # This ensures they are only introduced into the simulation when they
        # would actually be known in real-time.
        fvgs = detect_fvgs(self.df)
        events = detect_structure_events(self.df)
        
        fvg_by_time = {}
        for fvg in fvgs:
            loc = self.df.index.get_loc(fvg.timestamp)
            if isinstance(loc, slice):
                loc = loc.start
            # FVG is confirmed on the candle AFTER its timestamp
            if loc + 1 < len(self.df):
                conf_time = self.df.index[loc + 1]
                fvg_by_time.setdefault(conf_time, []).append(fvg)
                
        struct_by_time = {}
        for ev in events:
            # Structure breaks are confirmed on the very candle they occur
            struct_by_time.setdefault(ev.timestamp, []).append(ev)
            
        for i in tqdm(range(len(self.df)), desc="Running Backtest"):
            curr_time = self.df.index[i]
            candle = self.df.iloc[i]
            
            # 1. Manage active trade using current candle's price action
            if self.active_trade:
                trade = self.active_trade
                if trade.direction == Direction.BULLISH:
                    # Check SL first, realistically if it gap down
                    if candle['low'] <= trade.stop_loss:
                        trade.exit_time = curr_time
                        trade.exit_price = trade.stop_loss
                        trade.result = "LOSS"
                        trade.r_multiple = -1.0
                        self.trades.append(trade)
                        self.active_trade = None
                    elif candle['high'] >= trade.take_profit:
                        trade.exit_time = curr_time
                        trade.exit_price = trade.take_profit
                        trade.result = "WIN"
                        trade.r_multiple = self.rr_ratio
                        self.trades.append(trade)
                        self.active_trade = None
                else:
                    if candle['high'] + self.spread >= trade.stop_loss:
                        trade.exit_time = curr_time
                        trade.exit_price = trade.stop_loss
                        trade.result = "LOSS"
                        trade.r_multiple = -1.0
                        self.trades.append(trade)
                        self.active_trade = None
                    elif candle['low'] + self.spread <= trade.take_profit:
                        trade.exit_time = curr_time
                        trade.exit_price = trade.take_profit
                        trade.result = "WIN"
                        trade.r_multiple = self.rr_ratio
                        self.trades.append(trade)
                        self.active_trade = None

            # 2. Check for entry using current candle
            # This happens before adding new setups from the current candle close,
            # ensuring we don't enter on the same candle a setup is formed.
            if not self.active_trade and self.pending_setups:
                triggered = None
                for setup in self.pending_setups:
                    if candle['low'] <= setup['entry_price'] <= candle['high']:
                        triggered = setup
                        break
                            
                if triggered:
                    self.pending_setups.remove(triggered)
                    if triggered['direction'] == Direction.BULLISH:
                        entry = triggered['entry_price'] + self.spread
                        sl = entry - self.sl_dist
                        tp = entry + (self.sl_dist * self.rr_ratio)
                    else:
                        entry = triggered['entry_price']
                        sl = entry + self.sl_dist
                        tp = entry - (self.sl_dist * self.rr_ratio)
                        
                    self.active_trade = Trade(
                        entry_time=curr_time,
                        entry_price=entry,
                        direction=triggered['direction'],
                        stop_loss=sl,
                        take_profit=tp,
                        pattern_type=triggered['type']
                    )
                    # Clear other pending setups when in a trade
                    self.pending_setups.clear()

            # 3. Add new setups formed at the CLOSE of the current candle
            if curr_time in fvg_by_time:
                for fvg in fvg_by_time[curr_time]:
                    entry_price = fvg.top if fvg.direction == Direction.BULLISH else fvg.bottom
                    self.pending_setups.append({
                        'type': 'FVG',
                        'direction': fvg.direction,
                        'entry_price': entry_price,
                        'time': curr_time
                    })
            if curr_time in struct_by_time:
                for ev in struct_by_time[curr_time]:
                    self.pending_setups.append({
                        'type': 'STRUCT',
                        'direction': ev.direction,
                        'entry_price': ev.price_level,
                        'time': curr_time
                    })

        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        return self.get_trades_df()

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction.value,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'pattern_type': t.pattern_type,
                'result': t.result,
                'r_multiple': t.r_multiple
            })
        return pd.DataFrame(records)
