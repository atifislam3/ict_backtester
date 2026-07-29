import logging
import random
import math
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from src.events import EventBus, CandleEvent, PatternEvent, FillEvent
from src.models import Candle, TradeResult, ConfigSchema
from src.pattern_detector import DetailedPatternEvent

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class EquitySnapshot:
    timestamp: datetime
    equity: float
    drawdown: float
    peak: float

@dataclass(kw_only=True)
class Order:
    order_type: str # 'MARKET', 'LIMIT', 'STOP'
    direction: str # 'LONG', 'SHORT'
    requested_price: float
    size: float
    stop_loss: float
    take_profit: float
    pattern_type: str
    timestamp: datetime
    fill_price: Optional[float] = None
    slippage: float = 0.0
    status: str = 'PENDING' # 'PENDING', 'FILLED', 'CANCELLED'

class PositionSizer:
    def __init__(self, risk_pct: float, contract_size: int = 100000):
        self.risk_pct = risk_pct
        self.contract_size = contract_size

    def calculate_size(self, equity: float, entry_price: float, stop_loss: float, pair: str = 'EURUSD') -> float:
        risk_amount = equity * (self.risk_pct / 100.0)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        
        # for forex, 1 pip = 0.0001
        # size is in lots (1 lot = 100,000 units)
        units = risk_amount / risk_per_unit
        lots = units / self.contract_size
        return round(lots, 2)

class EventDrivenBacktester:
    def __init__(self, event_bus: EventBus, config: ConfigSchema, initial_capital: float = 10000.0):
        self.event_bus = event_bus
        self.config = config
        
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        
        self.open_trades: List[Dict[str, Any]] = []
        self.closed_trades: List[TradeResult] = []
        self.equity_curve: List[EquitySnapshot] = []
        self.pending_orders: List[Order] = []
        self.order_log: List[Order] = []
        
        self.candles: List[Candle] = []
        self.current_index = -1
        
        self.sizer = PositionSizer(risk_pct=config.risk_settings.risk_per_trade_pct)
        
        self.max_daily_risk_pct = 5.0 # configurable circuit breaker
        self.daily_pnl = 0.0
        self.current_day: Optional[datetime.date] = None
        self.circuit_breaker_active = False
        
        # Subscribe to events
        self.event_bus.subscribe(CandleEvent, self.on_candle)
        self.event_bus.subscribe(PatternEvent, self.on_pattern)
        
        # For ATR calculation
        self.atr_period = 14
        self.tr_history: List[float] = []

    def get_candle_by_index(self, idx: int) -> Candle:
        if idx > self.current_index:
            logger.warning(f"Anti-Lookahead Violation: Attempted to access future candle at index {idx} (current={self.current_index})")
            raise IndexError("Lookahead bias detected.")
        return self.candles[idx]

    def _update_atr(self, candle: Candle):
        if self.current_index == 0:
            tr = candle.high - candle.low
        else:
            prev = self.candles[self.current_index - 1]
            tr = max(candle.high - candle.low, abs(candle.high - prev.close), abs(candle.low - prev.close))
        self.tr_history.append(tr)
        if len(self.tr_history) > self.atr_period:
            self.tr_history.pop(0)

    def _get_atr(self) -> float:
        if not self.tr_history:
            return 0.0010 # Default fallback
        return sum(self.tr_history) / len(self.tr_history)

    def _get_session(self, dt: datetime) -> str:
        h = dt.hour
        if 8 <= h < 11: return 'Overlap'
        if 3 <= h < 8: return 'London'
        if 11 <= h < 17: return 'NY'
        return 'Asian'

    def on_pattern(self, event: PatternEvent):
        # Apply circuit breaker
        if self.circuit_breaker_active:
            return
            
        c = self.candles[self.current_index]
        
        # Calculate stops/TPs
        atr = self._get_atr()
        is_detailed = isinstance(event, DetailedPatternEvent)
        
        if event.pattern_type == 'FVG':
            if is_detailed and event.metadata.get('status') == 'unmitigated':
                # Limit order at 50% gap fill
                fill_price = event.price_levels.get('fill_price', event.price_level)
                sl = fill_price - (atr * 1.5) if event.direction == 'BULLISH' else fill_price + (atr * 1.5)
                tp = fill_price + (abs(fill_price - sl) * self.config.risk_settings.rr_ratio) if event.direction == 'BULLISH' else fill_price - (abs(fill_price - sl) * self.config.risk_settings.rr_ratio)
                
                size = self.sizer.calculate_size(self.equity, fill_price, sl)
                if size > 0:
                    order = Order(
                        order_type='LIMIT', direction='LONG' if event.direction == 'BULLISH' else 'SHORT',
                        requested_price=fill_price, size=size, stop_loss=sl, take_profit=tp,
                        pattern_type=event.pattern_type, timestamp=c.timestamp
                    )
                    self.pending_orders.append(order)
                    self.order_log.append(order)
                    
        elif event.pattern_type in ('BOS', 'CHoCH'):
            # Stop order at the break level
            sl = event.price_level - (atr * 1.5) if event.direction == 'BULLISH' else event.price_level + (atr * 1.5)
            tp = event.price_level + (abs(event.price_level - sl) * self.config.risk_settings.rr_ratio) if event.direction == 'BULLISH' else event.price_level - (abs(event.price_level - sl) * self.config.risk_settings.rr_ratio)
            
            size = self.sizer.calculate_size(self.equity, event.price_level, sl)
            if size > 0:
                order = Order(
                    order_type='STOP', direction='LONG' if event.direction == 'BULLISH' else 'SHORT',
                    requested_price=event.price_level, size=size, stop_loss=sl, take_profit=tp,
                    pattern_type=event.pattern_type, timestamp=c.timestamp
                )
                self.pending_orders.append(order)
                self.order_log.append(order)
        else:
            # Default market order
            sl = c.close - (atr * 1.5) if event.direction == 'BULLISH' else c.close + (atr * 1.5)
            tp = c.close + (abs(c.close - sl) * self.config.risk_settings.rr_ratio) if event.direction == 'BULLISH' else c.close - (abs(c.close - sl) * self.config.risk_settings.rr_ratio)
            size = self.sizer.calculate_size(self.equity, c.close, sl)
            if size > 0:
                order = Order(
                    order_type='MARKET', direction='LONG' if event.direction == 'BULLISH' else 'SHORT',
                    requested_price=c.close, size=size, stop_loss=sl, take_profit=tp,
                    pattern_type=event.pattern_type, timestamp=c.timestamp
                )
                self.pending_orders.append(order)
                self.order_log.append(order)

    def on_candle(self, event: CandleEvent):
        c = event.candle
        self.candles.append(c)
        self.current_index += 1
        
        self._update_atr(c)
        
        # Reset daily circuit breaker
        c_date = c.timestamp.date()
        if self.current_day != c_date:
            self.current_day = c_date
            self.daily_pnl = 0.0
            self.circuit_breaker_active = False
            
        # Process pending orders
        still_pending = []
        for order in self.pending_orders:
            if order.status != 'PENDING':
                continue
                
            filled = False
            fill_price = 0.0
            slippage = 0.0
            
            pip_size = 0.01 if self.config.instrument == 'XAUUSD' else 0.0001
            half_spread = (self.config.risk_settings.spread_pips / 2.0) * pip_size
            
            if order.order_type == 'MARKET':
                fill_price = c.open
                slippage = random.uniform(0, 1) * pip_size
                filled = True
            elif order.order_type == 'LIMIT':
                if order.direction == 'LONG' and c.low <= order.requested_price:
                    fill_price = order.requested_price
                    filled = True
                elif order.direction == 'SHORT' and c.high >= order.requested_price:
                    fill_price = order.requested_price
                    filled = True
            elif order.order_type == 'STOP':
                if order.direction == 'LONG' and c.high >= order.requested_price:
                    fill_price = max(c.open, order.requested_price)
                    slippage = random.uniform(0, 1) * pip_size
                    filled = True
                elif order.direction == 'SHORT' and c.low <= order.requested_price:
                    fill_price = min(c.open, order.requested_price)
                    slippage = random.uniform(0, 1) * pip_size
                    filled = True
                    
            if filled:
                # Apply spread and slippage
                actual_fill = fill_price + half_spread + slippage if order.direction == 'LONG' else fill_price - half_spread - slippage
                order.fill_price = actual_fill
                order.slippage = slippage
                order.status = 'FILLED'
                
                self.open_trades.append({
                    'direction': order.direction,
                    'entry_price': actual_fill,
                    'size': order.size,
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                    'pattern_type': order.pattern_type,
                    'entry_time': c.timestamp,
                    'session': self._get_session(c.timestamp)
                })
                
                self.event_bus.emit(FillEvent(
                    fill_time=c.timestamp, fill_price=actual_fill, direction=order.direction,
                    size=order.size, action='ENTRY'
                ))
            else:
                # Time expiration for limit/stop orders? E.g. expire after 10 bars
                still_pending.append(order)
                
        self.pending_orders = still_pending
        
        # Process open trades
        still_open = []
        for trade in self.open_trades:
            close_price = None
            
            # Check TP/SL
            if trade['direction'] == 'LONG':
                if c.low <= trade['stop_loss']:
                    close_price = trade['stop_loss']
                elif c.high >= trade['take_profit']:
                    close_price = trade['take_profit']
            else:
                if c.high >= trade['stop_loss']:
                    close_price = trade['stop_loss']
                elif c.low <= trade['take_profit']:
                    close_price = trade['take_profit']
                    
            if close_price is not None:
                # Calculate PnL
                pip_size = 0.01 if self.config.instrument == 'XAUUSD' else 0.0001
                half_spread = (self.config.risk_settings.spread_pips / 2.0) * pip_size
                actual_exit = close_price - half_spread if trade['direction'] == 'LONG' else close_price + half_spread
                
                commission = 7.0 * trade['size']
                
                if trade['direction'] == 'LONG':
                    pnl = (actual_exit - trade['entry_price']) * (trade['size'] * 100000) - commission
                else:
                    pnl = (trade['entry_price'] - actual_exit) * (trade['size'] * 100000) - commission
                    
                self.equity += pnl
                self.daily_pnl += pnl
                
                risk_amount = abs(trade['entry_price'] - trade['stop_loss']) * (trade['size'] * 100000)
                r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
                
                res = TradeResult(
                    entry_time=trade['entry_time'], exit_time=c.timestamp,
                    entry_price=trade['entry_price'], exit_price=actual_exit,
                    direction=trade['direction'], size=trade['size'], pnl=pnl,
                    r_multiple=r_multiple, pattern_type=trade['pattern_type'], session=trade['session']
                )
                self.closed_trades.append(res)
                
                self.event_bus.emit(FillEvent(
                    fill_time=c.timestamp, fill_price=actual_exit, direction=trade['direction'],
                    size=trade['size'], action='EXIT', realized_pnl=pnl
                ))
            else:
                still_open.append(trade)
                
        self.open_trades = still_open
        
        # Check circuit breaker
        if not self.circuit_breaker_active:
            daily_pct = (self.daily_pnl / self.equity) * 100
            if daily_pct <= -self.max_daily_risk_pct:
                self.circuit_breaker_active = True
                logger.warning(f"Circuit breaker activated at {c.timestamp}. Daily PnL: {self.daily_pnl}")
                self.pending_orders.clear() # Cancel pending orders
        
        # Update equity curve
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = ((self.peak_equity - self.equity) / self.peak_equity) * 100 if self.peak_equity > 0 else 0.0
        
        self.equity_curve.append(EquitySnapshot(
            timestamp=c.timestamp, equity=self.equity, drawdown=drawdown, peak=self.peak_equity
        ))

    def get_results(self):
        eq_df = pd.DataFrame([vars(s) for s in self.equity_curve])
        order_df = pd.DataFrame([vars(o) for o in self.order_log])
        return self.closed_trades, eq_df, order_df
