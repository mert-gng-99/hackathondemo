"""Unit tests for the shared structured logger."""
from __future__ import annotations

import logging

from src.core.logger import get_logger


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger("neurobridge.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "neurobridge.test"


def test_get_logger_attaches_single_handler() -> None:
    """Repeated calls must not duplicate handlers (idempotence)."""
    name = "neurobridge.idempotent"
    first = get_logger(name)
    second = get_logger(name)
    assert first is second
    assert len(first.handlers) == 1


def test_get_logger_default_level_is_info() -> None:
    logger = get_logger("neurobridge.level_check")
    assert logger.level == logging.INFO


def test_get_logger_emits_formatted_record(caplog) -> None:
    logger = get_logger("neurobridge.emit")
    with caplog.at_level(logging.INFO, logger="neurobridge.emit"):
        logger.info("hello-world")
    assert any("hello-world" in record.message for record in caplog.records)
