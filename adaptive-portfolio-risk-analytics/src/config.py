"""
Configuration management for adaptive portfolio optimization platform.

This module provides centralized configuration loading from YAML files,
environment variables, and Python dictionaries.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

# Load environment variables when python-dotenv is available.
if load_dotenv is not None:
    load_dotenv()


class ConfigManager:
    """
    Centralized configuration manager for the platform.

    Loads configuration from:
    - YAML config files
    - Environment variables
    - Python dictionaries (runtime overrides)
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager.

        Parameters
        ----------
        config_dir : str, optional
            Path to configuration directory. Defaults to ./config
        """
        self.config_dir = Path(config_dir or "./config")
        self.config: Dict[str, Any] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all YAML configuration files from config directory."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return

        for config_file in sorted(self.config_dir.glob("*.yaml")):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                    self.config.update(file_config)
            except Exception as e:
                from src.logging_config import get_logger

                logger = get_logger(__name__)
                logger.warning("Failed to load config file %s: %s", config_file, e,
                               exc_info=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation, e.g., "portfolio.rebalance_freq")
        default : Any, optional
            Default value if key not found

        Returns
        -------
        Any
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : Any
            Value to set
        """
        keys = key.split(".")
        d = self.config

        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def merge(self, other: Dict[str, Any]) -> None:
        """
        Merge additional configuration.

        Parameters
        ----------
        other : dict
            Configuration dictionary to merge
        """
        self._deep_update(self.config, other)

    @staticmethod
    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                existing = d.get(k, {})
                if not isinstance(existing, dict):
                    existing = {}
                d[k] = ConfigManager._deep_update(existing, v)
            else:
                d[k] = v
        return d


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get global configuration manager instance (singleton).

    Returns
    -------
    ConfigManager
        Global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
