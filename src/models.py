from datetime import datetime
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class DateRange(BaseModel):
    start: str
    end: str

class RiskSettings(BaseModel):
    risk_per_trade_pct: float = Field(default=1.0, ge=0.0)
    rr_ratio: float = Field(default=2.0, gt=0.0)
    stop_loss_pips: float = Field(default=10.0, gt=0.0)
    spread_pips: float = Field(default=1.5, ge=0.0)

class SessionTime(BaseModel):
    start: str
    end: str

class SessionTimes(BaseModel):
    london: SessionTime
    ny: SessionTime
    asian: SessionTime

class PatternParams(BaseModel):
    fvg_min_size: float = 0.0
    bos_min_size: float = 0.0

class ConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    instrument: str = "XAUUSD"
    timeframe: str = "1H"
    date_range: DateRange
    risk_settings: RiskSettings = Field(default_factory=RiskSettings)
    session_times: SessionTimes
    pattern_params: PatternParams = Field(default_factory=PatternParams)

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
