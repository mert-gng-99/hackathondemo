"""Tests for /pipeline/{bbb,eeg,mri} POST endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestBBBRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/bbb",
            json={
                "input_path": str(_FIXTURES / "bbbp_sample.csv"),
                "output_path": str(out),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["rows"] > 0
        assert out.exists()

    def test_returns_404_when_input_missing(self, tmp_path: Path):
        resp = client.post(
            "/pipeline/bbb",
            json={
                "input_path": str(tmp_path / "does_not_exist.csv"),
                "output_path": str(tmp_path / "out.parquet"),
            },
        )
        assert resp.status_code == 404

    def test_returns_422_on_malformed_body(self):
        resp = client.post("/pipeline/bbb", json={"banana": 1})
        assert resp.status_code == 422  # pydantic validation


class TestEEGRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        fif = _FIXTURES / "eeg_sample.fif"
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/eeg",
            json={"input_path": str(fif), "output_path": str(out)},
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] > 0


class TestMRIRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        from tests.fixtures.build_mri_fixture import build as build_mri
        fixture_dir = build_mri(out_dir=tmp_path / "mri_fixture")
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/mri",
            json={
                "input_dir": str(fixture_dir),
                "sites_csv": str(fixture_dir / "sites.csv"),
                "output_path": str(out),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] > 0


class TestBBBPredictRoute:
    def _setup_model_artifact(self, tmp_path: Path) -> Path:
        """Build features + train + save a tiny model. Returns artifact path."""
        from src.pipelines import bbb_pipeline
        from src.models import bbb_model
        import pandas as pd
        features_path = tmp_path / "features.parquet"
        bbb_pipeline.run_pipeline(
            input_path=_FIXTURES / "bbbp_sample.csv",
            output_path=features_path,
        )
        df = pd.read_parquet(features_path)
        model = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        artifact = tmp_path / "bbb_model.joblib"
        bbb_model.save(model, artifact)
        return artifact

    def test_returns_200_with_prediction_and_attributions(self, tmp_path: Path, monkeypatch):
        import pytest
        artifact = self._setup_model_artifact(tmp_path)
        monkeypatch.setenv("BBB_MODEL_PATH", str(artifact))

        resp = client.post(
            "/predict/bbb",
            json={"smiles": "CCO", "top_k": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] in (0, 1)
        assert body["label_text"] in ("permeable", "non-permeable")
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["top_features"]) == 5
        for f in body["top_features"]:
            assert f["feature"].startswith("fp_")
            assert isinstance(f["shap_value"], float)
        # Day-6 calibration assertions: trained test fixture model has
        # _neurobridge_calibration metadata, so calibration must be populated.
        assert body["calibration"] is not None
        cal = body["calibration"]
        valid_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
        assert any(
            cal["threshold"] == pytest.approx(t) for t in valid_thresholds
        ), f"threshold {cal['threshold']} not in {valid_thresholds}"
        assert cal["threshold"] <= body["confidence"]
        assert 0.0 <= cal["precision"] <= 1.0
        assert isinstance(cal["support"], int)
        assert cal["support"] >= 0

    def test_returns_400_on_invalid_smiles(self, tmp_path: Path, monkeypatch):
        artifact = self._setup_model_artifact(tmp_path)
        monkeypatch.setenv("BBB_MODEL_PATH", str(artifact))

        resp = client.post(
            "/predict/bbb",
            json={"smiles": "this_is_not_a_smiles", "top_k": 5},
        )
        assert resp.status_code == 400

    def test_returns_503_when_artifact_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("BBB_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
        resp = client.post(
            "/predict/bbb",
            json={"smiles": "CCO", "top_k": 5},
        )
        assert resp.status_code == 503
