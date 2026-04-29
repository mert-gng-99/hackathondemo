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

    Idempotent on handler attachment: repeated calls with the same name
    return the same Logger instance and never stack duplicate stdout
    StreamHandlers. The most recent call wins on `level`, so callers can
    raise/lower verbosity at runtime without rebuilding the logger.

    Note on `propagate=False`: records do NOT bubble up to the root logger.
    If a framework (FastAPI, Uvicorn, MLflow) needs to capture records via
    a root handler in week-2 work, this default will need to be revisited.

    Args:
        name: Dotted logger name, conventionally `__name__`.
        level: Logging level (default `logging.INFO`).

    Returns:
        Configured `logging.Logger` writing to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        for h in logger.handlers
    ):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    return logger
