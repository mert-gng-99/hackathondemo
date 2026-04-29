"""Root conftest: ensure caplog captures from non-propagating loggers."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def _patch_caplog_for_no_propagate(caplog):
    """Attach the caplog handler directly to every named logger that has
    propagate=False so that ``caplog`` can still capture their records."""
    _orig_at_level = caplog.at_level.__func__  # type: ignore[attr-defined]

    @contextmanager
    def patched_at_level(
        self,
        level: int,
        logger: str | None = None,
    ) -> Generator[None, None, None]:
        logger_obj = logging.getLogger(logger) if logger else logging.getLogger()
        added = False
        if not logger_obj.propagate:
            logger_obj.addHandler(self.handler)
            added = True
        try:
            with _orig_at_level(self, level, logger):
                yield
        finally:
            if added:
                logger_obj.removeHandler(self.handler)

    import types
    caplog.at_level = types.MethodType(patched_at_level, caplog)
    yield
