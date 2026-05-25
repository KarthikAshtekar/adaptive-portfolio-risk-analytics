"""
Logging configuration and utilities for the platform.

Provides centralized, structured logging with support for file and console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger

# Remove default handler
loguru_logger.remove()


class LoggerSetup:
    """Configure and manage logging for the platform."""

    _initialized = False

    @classmethod
    def setup(
        cls,
        name: str = "adaptive_portfolio",
        level: str = "INFO",
        log_dir: Optional[str] = None,
        format_str: Optional[str] = None,
    ) -> logging.Logger:
        """
        Configure logging for the application.

        Parameters
        ----------
        name : str
            Logger name
        level : str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir : str, optional
            Directory for log files. Defaults to ./logs
        format_str : str, optional
            Log format string. Uses default if None

        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        if cls._initialized:
            return logging.getLogger(name)

        # Create logs directory
        log_dir = Path(log_dir or "./logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Default format
        if format_str is None:
            format_str = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )

        # Configure standard logging
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_str)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (rotating)
        log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5  # 10MB, keep 5 files
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Configure loguru with same settings
        loguru_logger.add(
            sys.stderr,
            level=level.upper(),
            format=format_str,
        )
        loguru_logger.add(
            str(log_file),
            rotation="500 MB",
            retention="7 days",
            level=level.upper(),
            format=format_str,
        )

        cls._initialized = True
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


# Initialize default logger
_default_logger = LoggerSetup.setup()
