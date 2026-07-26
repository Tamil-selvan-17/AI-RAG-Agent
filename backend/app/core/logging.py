"""
Centralized logging configuration using loguru.

Import `logger` from this module anywhere in the app for consistent,
structured logging output.
"""

import sys

from loguru import logger

from app.core.config import get_settings

_settings = get_settings()

logger.remove()
logger.add(
    sys.stdout,
    level=_settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=True,
    diagnose=False,
)

logger.add(
    "logs/app.log",
    level=_settings.log_level,
    rotation="10 MB",
    retention="14 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=False,
)

__all__ = ["logger"]
