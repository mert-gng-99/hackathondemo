"""Shared structured logger for NeuroBridge pipelines.

All modules in `src/` must obtain their logger via `get_logger(__name__)`
instead of using `print()`. This guarantees consistent format and INFO-level
traceability across pipelines (per AGENTS.md §4).
"""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide singleton logger for the given name.

    Idempotent: repeated calls with the same name return the same Logger
    instance and never stack duplicate handlers.

    Args:
        name: Dotted logger name, conventionally `__name__`.
        level: Logging level (default `logging.INFO`).

    Returns:
        Configured `logging.Logger` writing to stdout.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
