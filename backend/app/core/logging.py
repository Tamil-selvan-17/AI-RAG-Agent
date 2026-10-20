"""
Centralized logging configuration using loguru.

Import `logger` from this module anywhere in the app for consistent,
structured logging output. Every line carries the current request's ID
(see app.core.request_id), or "-" outside a request, so all log lines for
one request can be found together.
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings

_settings = get_settings()

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[request_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
logger.configure(extra={"request_id": "-"})
logger.add(
    sys.stdout,
    level=_settings.log_level,
    format=_FORMAT,
    colorize=True,
    backtrace=True,
    diagnose=False,
)

if _settings.log_to_file:
    # Hosts like Render keep stdout logs already and have an ephemeral disk,
    # so file logging can be turned off there with LOG_TO_FILE=false.
    logger.add(
        Path(_settings.log_dir) / "app.log",
        level=_settings.log_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[request_id]} | "
        "{name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=False,
    )

__all__ = ["logger"]
