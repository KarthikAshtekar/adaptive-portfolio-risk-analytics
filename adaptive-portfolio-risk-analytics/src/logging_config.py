"""
Logging configuration and utilities for the platform.

Provides centralized, structured logging with support for file and console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

loguru_logger: Any | None
try:
    from loguru import logger as _imported_loguru_logger
except ImportError:  # pragma: no cover - optional dependency fallback
    loguru_logger = None
else:
    loguru_logger = _imported_loguru_logger
    # Remove default handler
    loguru_logger.remove()


class LoggerSetup:
    """Configure and manage logging for the platform."""

    _initialized = False
    _configured_level = logging.INFO

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
            logger = logging.getLogger(name)
            logger.setLevel(cls._configured_level)
            return logger

        # Create logs directory
        log_dir_path = Path(log_dir or "./logs")
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Default format
        if format_str is None:
            format_str = (
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )

        # Configure standard logging
        resolved_level = getattr(logging, level.upper(), logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(resolved_level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_formatter = logging.Formatter(format_str)
        console_handler.setFormatter(console_formatter)
        logger.handlers.clear()
        logger.addHandler(console_handler)

        # File handler (rotating)
        log_file = log_dir_path / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,
            backupCount=5,  # 10MB, keep 5 files
        )
        file_handler.setLevel(resolved_level)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Configure loguru when available.
        if loguru_logger is not None:
            loguru_format = (
                "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {file}:{line} - {message}"
            )
            loguru_logger.add(
                sys.stderr,
                level=level.upper(),
                format=loguru_format,
            )
            loguru_logger.add(
                str(log_file),
                rotation="500 MB",
                retention="7 days",
                level=level.upper(),
                format=loguru_format,
            )

        cls._configured_level = resolved_level
        cls._initialized = True
        configured_logger = logging.getLogger(name)
        configured_logger.setLevel(resolved_level)
        return configured_logger


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
    if not LoggerSetup._initialized:
        LoggerSetup.setup()
    logger = logging.getLogger(name)
    logger.setLevel(LoggerSetup._configured_level)
    return logger


# Initialize default logger
_default_logger = LoggerSetup.setup()
