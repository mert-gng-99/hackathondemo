"""Integration test for POST /fusion/predict."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestFusionRoute:
    def test_happy_path_mri_only(self) -> None:
        body = {
            "mri": {
                "label_text": "alzheimers",
                "label": 1,
                "confidence": 0.88,
                "probabilities": [
                    {"label_text": "control", "probability": 0.12},
                    {"label_text": "alzheimers", "probability": 0.88},
                ],
            },
        }
        r = client.post("/fusion/predict", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "diseases" in data
        assert any(d["disease"] == "alzheimers" for d in data["diseases"])
        assert data["top_disease"] in {"alzheimers", "parkinsons", "other"}

    def test_empty_input_returns_baseline(self) -> None:
        r = client.post("/fusion/predict", json={})
        assert r.status_code == 200
        data = r.json()
        for d in data["diseases"]:
            assert abs(d["probability"] - 0.5) < 1e-6
        assert "mri" in data["missing_inputs"]

    def test_invalid_probability_returns_422(self) -> None:
        body = {
            "mri": {
                "label_text": "x",
                "label": 0,
                "confidence": 1.5,
                "probabilities": [{"label_text": "x", "probability": 1.5}],
            },
        }
        r = client.post("/fusion/predict", json=body)
        assert r.status_code == 422
