from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Type, Any
from src.models import Candle

@dataclass
class Event:
    """Base class for all events in the system."""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CandleEvent(Event):
    candle: Candle

@dataclass
class PatternEvent(Event):
    pattern_type: str  # 'BOS', 'CHoCH', 'FVG'
    direction: str     # 'BULLISH' or 'BEARISH'
    price_level: float

@dataclass
class SignalEvent(Event):
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern_type: str
    session: str

@dataclass
class FillEvent(Event):
    fill_time: datetime
    fill_price: float
    direction: str
    size: float
    action: str  # 'ENTRY' or 'EXIT'
    realized_pnl: float = 0.0

class EventBus:
    """Lightweight EventBus for decoupled pub/sub communication."""
    def __init__(self) -> None:
        self._subscribers: Dict[Type[Event], List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: Type[Event], handler: Callable[[Any], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        handlers = self._subscribers.get(type(event), [])
        for handler in handlers:
            handler(event)
