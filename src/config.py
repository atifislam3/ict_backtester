"""
Configuration module to load settings from config.yaml.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SessionTimes:
    """Represents start and end times for a trading session."""
    start: str
    end: str

@dataclass
class Config:
    """Configuration class for the ICT Backtester."""
    instrument: str
    timeframe: str
    start_date: str
    end_date: str
    risk_per_trade: float
    rr_ratio: float
    spread_assumption: float
    london_session: SessionTimes
    ny_session: SessionTimes
    asian_session: SessionTimes

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Loads configuration from a YAML file.
    
    Args:
        config_path (str): The path to the config.yaml file.
        
    Returns:
        Config: A dataclass containing the typed configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    session_data = data.get("session_times", {})
    
    return Config(
        instrument=data.get("instrument", "XAUUSD"),
        timeframe=data.get("timeframe", "1H"),
        start_date=data.get("date_range", {}).get("start", "2023-01-01"),
        end_date=data.get("date_range", {}).get("end", "2023-12-31"),
        risk_per_trade=float(data.get("risk_per_trade", 0.01)),
        rr_ratio=float(data.get("rr_ratio", 2.0)),
        spread_assumption=float(data.get("spread_assumption", 1.5)),
        london_session=SessionTimes(**session_data.get("london", {"start": "03:00", "end": "11:00"})),
        ny_session=SessionTimes(**session_data.get("ny", {"start": "08:00", "end": "17:00"})),
        asian_session=SessionTimes(**session_data.get("asian", {"start": "19:00", "end": "03:00"}))
    )
