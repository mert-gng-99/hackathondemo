"""Tests for src.core.tracking."""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.core import tracking


class TestTrackPipelineRun:
    def test_creates_run_with_experiment_name(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="bbb_pipeline",
            params={"input_path": "x.csv"},
            metrics={"rows_in": 6.0, "rows_out": 4.0},
            artifact_path=out,
        ) as run_id:
            assert run_id is not None
        runs = mlflow.search_runs(experiment_names=["bbb_pipeline"])
        assert len(runs) >= 1

    def test_logs_params(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="bbb_pipeline_params",
            params={"n_bits": 2048, "radius": 2},
            metrics={},
            artifact_path=out,
        ):
            pass
        runs = mlflow.search_runs(experiment_names=["bbb_pipeline_params"])
        assert "params.n_bits" in runs.columns
        assert runs.iloc[0]["params.n_bits"] == "2048"

    def test_logs_metrics(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="eeg_pipeline_metrics",
            params={},
            metrics={"duration_sec": 1.234, "rows_out": 100.0},
            artifact_path=out,
        ):
            pass
        runs = mlflow.search_runs(experiment_names=["eeg_pipeline_metrics"])
        assert runs.iloc[0]["metrics.duration_sec"] == 1.234
        assert runs.iloc[0]["metrics.rows_out"] == 100.0

    def test_logs_artifact(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="mri_pipeline_artifact",
            params={},
            metrics={},
            artifact_path=out,
        ) as run_id:
            pass
        artifacts = mlflow.MlflowClient().list_artifacts(run_id)
        assert any(a.path.endswith("out.parquet") for a in artifacts)

    def test_disabled_via_env_returns_no_op(self, monkeypatch, tmp_path: Path):
        """Setting NEUROBRIDGE_DISABLE_MLFLOW=1 must skip MLflow entirely
        (used by live demo when the tracking server is down)."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_MLFLOW", "1")
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="should_not_appear",
            params={"x": 1},
            metrics={"y": 2.0},
            artifact_path=out,
        ) as run_id:
            assert run_id is None
        # No "should_not_appear" experiment was created
        names = [e.name for e in mlflow.search_experiments()]
        assert "should_not_appear" not in names
