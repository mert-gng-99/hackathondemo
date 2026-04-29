"""MLflow tracking helper used by all three pipelines.

Wraps `mlflow.start_run` so each pipeline can log params, metrics, and an
output artifact in one block. Honors `NEUROBRIDGE_DISABLE_MLFLOW=1` for
environments where the tracking server is not reachable (offline demos, CI
without mlflow service). When disabled, yields `None` and does no I/O.

Tracking URI source of truth: the standard `MLFLOW_TRACKING_URI` env var.
Tests pin this via the repo-wide conftest.py autouse fixture.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import mlflow

from src.core.logger import get_logger

logger = get_logger(__name__)

_DISABLE_FLAG = "NEUROBRIDGE_DISABLE_MLFLOW"


@contextlib.contextmanager
def track_pipeline_run(
    experiment_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    artifact_path: Path,
) -> Iterator[str | None]:
    """Context manager that creates an MLflow run for one pipeline invocation.

    On enter: creates/loads `experiment_name`, starts a run, logs params + metrics.
    On exit: logs `artifact_path` as an artifact and ends the run.

    Yields the active `run_id` (str), or `None` if MLflow is disabled.

    Args:
        experiment_name: e.g. "bbb_pipeline" / "eeg_pipeline" / "mri_pipeline".
        params: Run parameters (input path, hyper-params, etc.). Stringified by MLflow.
        metrics: Numeric metrics (row counts, durations).
        artifact_path: Path to the produced Parquet — logged as a run artifact.
    """
    if os.environ.get(_DISABLE_FLAG) == "1":
        logger.info("MLflow disabled via %s=1; skipping run tracking", _DISABLE_FLAG)
        yield None
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        try:
            yield run.info.run_id
        finally:
            if Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))
            else:
                logger.warning(
                    "artifact_path %s does not exist; skipping artifact log",
                    artifact_path,
                )
