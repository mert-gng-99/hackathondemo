"""Repo-wide pytest fixtures.

Pins MLflow's tracking URI to a per-session tmp directory so pipeline tests
don't litter `./mlruns/` in the working tree, and so test runs are isolated
from production MLflow state.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_mlflow_tracking_uri() -> Iterator[None]:
    tmp_root = Path(tempfile.mkdtemp(prefix="mlflow_test_"))
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_root}"
    yield
    # Don't rmtree — pytest tmpdir cleanup or OS handles it; rmtree
    # races with mlflow background writes on slow CI.
