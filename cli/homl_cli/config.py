import os
import json
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path.home() / ".homl"
CONFIG_FILE = CONFIG_DIR / "config.json"
# Define the default socket path within the user's homl config directory
DEFAULT_SOCKET_PATH = CONFIG_DIR / "run" / "homl.sock"

# Config keys and descriptions
CONFIG_KEYS = {
    "port": "Port number for the OpenAI-compatible API server (default: 7456)",
    "model_home": "Model location on the host (default: ~/.homl/models)",
    "model_load_timeout": "Timeout (seconds) for model loading (default: 180)",
    "model_unload_idle_time": "Idle time (seconds) before unloading a model (default: 600)",
    "socket_path": "Path to the server socket file"
}

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

def get_socket_path() -> str:
    """Gets the socket path from config, or returns the default."""
    # In the future, a `homl config` command could change this value
    path = get_config_value("socket_path", str(DEFAULT_SOCKET_PATH))
    return f"unix://{path}"
