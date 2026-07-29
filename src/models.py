from datetime import datetime
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class InstrumentSchema(BaseModel):
    symbol: str = "XAUUSD=X"
    display_name: str = "XAUUSD"

class DateRange(BaseModel):
    start: str
    end: str

class RiskSchema(BaseModel):
    risk_per_trade: float = 0.01
    rr_ratio: float = 2.0
    atr_multiplier: float = 1.5
    max_daily_risk: float = 0.03

class ExecutionSchema(BaseModel):
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0

class SessionsSchema(BaseModel):
    asian: str = "20:00-00:00"
    london: str = "03:00-11:00"
    ny: str = "08:00-17:00"
    overlap: str = "08:00-11:00"

class PatternsSchema(BaseModel):
    pivot_lookback: int = 5
    min_swing_count: int = 3
    fvg_min_size_pips: float = 0.5

class ReportingSchema(BaseModel):
    output_dir: str = "reports"
    save_json: bool = True

class ConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    instrument: InstrumentSchema = Field(default_factory=InstrumentSchema)
    timeframe: str = "1H"
    date_range: DateRange
    risk: RiskSchema = Field(default_factory=RiskSchema)
    execution: ExecutionSchema = Field(default_factory=ExecutionSchema)
    sessions: SessionsSchema = Field(default_factory=SessionsSchema)
    patterns: PatternsSchema = Field(default_factory=PatternsSchema)
    reporting: ReportingSchema = Field(default_factory=ReportingSchema)

class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class TradeResult(BaseModel):
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    direction: Literal['LONG', 'SHORT']
    size: float
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    pattern_type: str
    session: str
