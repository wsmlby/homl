import os
import json
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path.home() / ".homl"
CONFIG_FILE = CONFIG_DIR / "config.json"

def ensure_config_exists():
    """Ensures the config directory and file exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.is_file():
        CONFIG_FILE.touch()
        CONFIG_FILE.write_text("{}")

def load_config() -> Dict[str, Any]:
    """Loads the configuration from the JSON file."""
    ensure_config_exists()
    with open(CONFIG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_config(config: Dict[str, Any]):
    """Saves the configuration to the JSON file."""
    ensure_config_exists()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def set_config_value(key: str, value: Any):
    """Sets a specific key-value pair in the config."""
    config = load_config()
    config[key] = value
    save_config(config)

def get_config_value(key: str, default: Any = None) -> Any:
    """Gets a specific value from the config."""
    config = load_config()
    return config.get(key, default)
