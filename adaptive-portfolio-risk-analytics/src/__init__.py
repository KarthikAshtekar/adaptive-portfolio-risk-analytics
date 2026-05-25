"""
Adaptive Portfolio Allocation and Risk Analytics Platform

A professional-grade quantitative portfolio optimization and risk analytics system
for institutional portfolio management and research.
"""

__version__ = "0.1.0"
__author__ = "Quantitative Finance Team"
__email__ = "team@example.com"

from src.config import get_config, ConfigManager
from src.logging_config import get_logger, LoggerSetup

__all__ = [
    "get_config",
    "ConfigManager",
    "get_logger",
    "LoggerSetup",
]
