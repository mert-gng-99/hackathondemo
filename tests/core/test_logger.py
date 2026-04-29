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


def test_get_logger_emits_record_to_handler_stream() -> None:
    """The logger writes records through its StreamHandler."""
    import io

    logger = get_logger("neurobridge.emit_capture")
    handler = logger.handlers[0]
    buf = io.StringIO()
    original_stream = handler.stream
    handler.stream = buf
    try:
        logger.info("hello-world")
    finally:
        handler.stream = original_stream

    output = buf.getvalue()
    assert "hello-world" in output


def test_get_logger_format_includes_level_and_name() -> None:
    """Format string must include level and logger name (per AGENTS.md §3 traceability)."""
    import io

    logger = get_logger("neurobridge.format_check")
    handler = logger.handlers[0]
    buf = io.StringIO()
    original_stream = handler.stream
    handler.stream = buf
    try:
        logger.info("payload")
    finally:
        handler.stream = original_stream

    output = buf.getvalue()
    assert "INFO" in output
    assert "neurobridge.format_check" in output
    assert "payload" in output
