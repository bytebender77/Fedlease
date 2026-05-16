import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigDict(dict):
    """Dictionary subclass with attribute-style access and deep merge."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return ConfigDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        obj = self
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj


def load_config(config_path: str) -> ConfigDict:
    """Load YAML config file and return as ConfigDict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigDict(_nested_to_configdict(raw or {}))


def _nested_to_configdict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _nested_to_configdict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_nested_to_configdict(i) for i in d]
    return d


def merge_configs(base: ConfigDict, override: Dict) -> ConfigDict:
    """Deep merge override into base config."""
    result = ConfigDict(base.copy())
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(ConfigDict(result[key]), value)
        else:
            result[key] = value
    return result


def save_config(config: ConfigDict, path: str) -> None:
    """Save ConfigDict to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False, indent=2)
