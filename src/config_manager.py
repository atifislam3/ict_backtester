import os
import yaml
from pathlib import Path
from typing import Dict, Any
from src.models import ConfigSchema

class ConfigManager:
    """Manages loading, validating, and overriding configuration."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path

    def load_config(self) -> ConfigSchema:
        """Loads config from YAML, applies ENV overrides, and validates."""
        raw_config = self._load_yaml()
        raw_config = self._apply_env_overrides(raw_config)
        
        # Validate and return
        return ConfigSchema(**raw_config)

    def _load_yaml(self) -> Dict[str, Any]:
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        return data or {}

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies environment variables starting with ICT_.
        Example: ICT_INSTRUMENT=EURUSD overrides instrument.
        Example: ICT_RISK_PER_TRADE=0.02 overrides risk_settings.risk_per_trade_pct.
        """
        prefix = "ICT_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
                
            env_key = key[len(prefix):].lower()
            
            # Type cast based on value
            parsed_value = self._parse_env_value(value)
            
            # Special case mapping as requested
            if env_key == "risk_per_trade":
                if "risk_settings" not in config:
                    config["risk_settings"] = {}
                config["risk_settings"]["risk_per_trade_pct"] = parsed_value
                continue
                
            # Generic nested mapping (e.g. ICT_RISK_SETTINGS__RR_RATIO)
            if "__" in env_key:
                parts = env_key.split("__")
                current = config
                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = parsed_value
            else:
                config[env_key] = parsed_value
                
        return config

    def _parse_env_value(self, value: str) -> Any:
        # Simple heuristic to cast numeric/bool values from env vars
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
