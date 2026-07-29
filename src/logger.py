import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Define custom TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = trace  # type: ignore

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_logger(
    name: str = "ict_backtester",
    log_dir: str = "logs",
    level: int = logging.INFO,
    json_format: bool = False,
    console_output: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
        
    Path(log_dir).mkdir(exist_ok=True)
    
    # File handler with rotation (10 MB max, keep 5 backups)
    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    
    if json_format:
        file_handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
    logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger
