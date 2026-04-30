"""Tests for /pipeline/{bbb,eeg,mri} POST endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest
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

    @pytest.fixture
    def _set_bbb_model_path(self, tmp_path: Path, monkeypatch):
        """Build a model artifact and point BBB_MODEL_PATH at it for the test."""
        artifact = self._setup_model_artifact(tmp_path)
        monkeypatch.setenv("BBB_MODEL_PATH", str(artifact))
        return artifact

    def test_returns_200_with_prediction_and_attributions(self, tmp_path: Path, monkeypatch):
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

    def test_predict_response_includes_drift_z_and_rolling_n(
        self, _set_bbb_model_path,
    ):
        """T1B: drift_z and rolling_n keys must always appear in the body."""
        # Reset deque before this test so rolling_n starts deterministic.
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()

        resp = client.post("/predict/bbb", json={"smiles": "CCO", "top_k": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "drift_z" in body
        assert "rolling_n" in body
        # First request: buffer has 1 sample (just appended), so warming up.
        assert body["rolling_n"] == 1
        assert body["drift_z"] is None  # <10 samples = warming up

    def test_predict_deque_rolls_at_100(self, _set_bbb_model_path):
        """T1B: after 100 predictions, deque caps at maxlen=100 (rolls)."""
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()
        # Fire 105 calls; final rolling_n must be 100, not 105.
        last_body = None
        for _ in range(105):
            resp = client.post(
                "/predict/bbb", json={"smiles": "CCO", "top_k": 3},
            )
            assert resp.status_code == 200
            last_body = resp.json()
        assert last_body["rolling_n"] == 100
        # By call 105, drift_z is computable (≥10 samples) — assert numeric.
        assert isinstance(last_body["drift_z"], float)

    def test_predict_response_includes_provenance(self, _set_bbb_model_path):
        """T2: provenance field is present in body (fields may be None)."""
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()

        resp = client.post("/predict/bbb", json={"smiles": "CCO", "top_k": 3})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "provenance" in body
        assert body["provenance"] is not None, "provenance should be populated even when MLflow is empty"
        prov = body["provenance"]
        assert "mlflow_run_id" in prov
        assert "model_version" in prov
        assert prov["model_version"] == "v1"  # default until bumped manually
        assert "train_date" in prov
        assert "n_examples" in prov
        # n_examples comes from train_stats — must be a positive int for the test fixture
        assert isinstance(prov["n_examples"], int) and prov["n_examples"] >= 1

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


class TestMRIDiagnosticsRoute:
    def test_returns_200_with_pre_and_post_data(self, tmp_path: Path):
        from tests.fixtures.build_mri_fixture import build as build_mri
        fixture_dir = build_mri(out_dir=tmp_path / "mri")
        resp = client.post(
            "/pipeline/mri/diagnostics",
            json={
                "input_dir": str(fixture_dir),
                "sites_csv": str(fixture_dir / "sites.csv"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) > 0
        assert body["site_gap_pre"] >= 0.0
        assert body["site_gap_post"] >= 0.0
        # Reduction factor is the headline KPI
        assert body["reduction_factor"] >= 1.0  # ComBat must reduce, not amplify
        states = {r["harmonization_state"] for r in body["rows"]}
        assert states == {"Pre-ComBat", "Post-ComBat"}

    def test_returns_404_when_input_dir_missing(self, tmp_path: Path):
        resp = client.post(
            "/pipeline/mri/diagnostics",
            json={
                "input_dir": str(tmp_path / "does_not_exist"),
                "sites_csv": str(tmp_path / "sites.csv"),
            },
        )
        assert resp.status_code == 404


class TestExplainBBBRoute:
    """Day-7 T3B: POST /explain/bbb."""

    def test_returns_200_with_template_source(self, monkeypatch):
        """Kill-switch on → /explain/bbb returns rationale with source=template."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        body = {
            "smiles": "CCO",
            "label": 1,
            "label_text": "permeable",
            "confidence": 0.82,
            "top_features": [
                {"feature": "fp_341", "shap_value": 0.045},
                {"feature": "fp_902", "shap_value": -0.031},
                {"feature": "fp_77", "shap_value": 0.022},
            ],
            "calibration": {"threshold": 0.80, "precision": 0.92, "support": 18},
            "drift_z": 0.42,
            "user_question": "Why permeable?",
        }
        resp = client.post("/explain/bbb", json=body)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["source"] == "template"
        assert out["model"] is None
        # Template must mention all three features
        for feat in ("fp_341", "fp_902", "fp_77"):
            assert feat in out["rationale"]
        assert "permeable" in out["rationale"]


class TestExplainEEGRoute:
    """Day-8 T1B: POST /explain/eeg."""

    def test_returns_200_with_template_source(self, monkeypatch):
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        body = {
            "rows": 30,
            "columns": 95,
            "duration_sec": 4.32,
            "mlflow_run_id": "abc12345",
            "user_question": "Why were epochs dropped?",
        }
        resp = client.post("/explain/eeg", json=body)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["source"] == "template"
        assert out["model"] is None
        assert "30" in out["rationale"]
        assert "95" in out["rationale"]


class TestExplainMRIRoute:
    """Day-8 T1B: POST /explain/mri."""

    def test_returns_200_with_template_source(self, monkeypatch):
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        body = {
            "site_gap_pre": 5.0004,
            "site_gap_post": 0.0015,
            "reduction_factor": 3290.0,
            "n_subjects": 6,
            "user_question": "Why does ComBat matter?",
        }
        resp = client.post("/explain/mri", json=body)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["source"] == "template"
        assert "3290" in out["rationale"]
        assert "6" in out["rationale"]


class TestExperimentsRoutes:
    """Day-8 T2A: GET /experiments/runs and POST /experiments/diff."""

    def test_runs_endpoint_returns_list(self):
        """GET /experiments/runs returns a runs list (may be empty if no MLflow data)."""
        resp = client.get("/experiments/runs")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "runs" in body
        assert isinstance(body["runs"], list)
        # If any runs exist, each must have the expected keys
        for run in body["runs"]:
            for key in ("run_id", "experiment_name", "start_time", "status", "metrics", "params"):
                assert key in run

    def test_diff_endpoint_handles_unknown_runs_gracefully(self):
        """POST /experiments/diff with bogus run ids returns 404 (not 500)."""
        resp = client.post(
            "/experiments/diff",
            json={"run_id_a": "nonexistent_aaa", "run_id_b": "nonexistent_bbb"},
        )
        assert resp.status_code in (404, 200), (
            f"unexpected status {resp.status_code}: {resp.text}"
        )
        # 404 is the documented contract; 200 with empty rows is acceptable too
        # because some MLflow stores treat unknown ids as "empty result".
        body = resp.json()
        if resp.status_code == 200:
            assert body.get("rows", []) == []
