"""
Adaptive Portfolio Allocation and Risk Analytics Platform

An evidence-gated portfolio construction and risk analytics research platform.
"""

__version__ = "1.3.0"
__author__ = "Karthik Ashtekar"

from src.config import get_config, ConfigManager
from src.logging_config import get_logger, LoggerSetup

__all__ = [
    "get_config",
    "ConfigManager",
    "get_logger",
    "LoggerSetup",
]
