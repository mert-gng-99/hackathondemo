"""Integration tests for /predict/bbb_permeability_map and /research/drug_dose_adjustment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


def _png(path: Path) -> Path:
    arr = (np.random.RandomState(0).rand(170, 170, 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))
    return path


@pytest.fixture()
def client_proxy(monkeypatch, tmp_path):
    ckpt = build_dummy_2d(tmp_path / "best.pt")
    monkeypatch.setenv("MRI_MODEL_PATH_2D", str(ckpt))
    return TestClient(app), tmp_path


class TestBBBPermeabilityMapRoute:
    def test_heuristic_proxy_happy_path(self, client_proxy) -> None:
        client, tmp_path = client_proxy
        img = _png(tmp_path / "scan.png")
        r = client.post(
            "/predict/bbb_permeability_map",
            json={"input_path": str(img), "mode": "heuristic_proxy"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert 0.0 <= data["permeability_score"] <= 1.0
        assert data["interpretation"] in {
            "BBB intact", "mild leakage", "moderate leakage", "severe leakage",
        }
        assert data["method"] == "heuristic_proxy"
        assert data["voxel_map_available"] is False

    def test_unknown_mode_returns_400(self, client_proxy) -> None:
        client, tmp_path = client_proxy
        img = _png(tmp_path / "scan.png")
        r = client.post(
            "/predict/bbb_permeability_map",
            json={"input_path": str(img), "mode": "bogus_mode"},
        )
        assert r.status_code == 400

    def test_missing_input_returns_404(self, client_proxy) -> None:
        client, tmp_path = client_proxy
        r = client.post(
            "/predict/bbb_permeability_map",
            json={"input_path": str(tmp_path / "missing.png"), "mode": "heuristic_proxy"},
        )
        assert r.status_code == 404


class TestDrugDoseAdjustmentRoute:
    def test_intact_bbb_returns_baseline(self) -> None:
        client = TestClient(app)
        r = client.post("/research/drug_dose_adjustment", json={
            "baseline_dose_mg": 100.0,
            "bbb_permeability_score": 0.05,
            "drug_bbb_permeable": True,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["recommended_dose_mg"] == pytest.approx(100.0)
        assert data["risk_level"] == "low"

    def test_leaky_bbb_permeable_drug_reduced(self) -> None:
        client = TestClient(app)
        r = client.post("/research/drug_dose_adjustment", json={
            "baseline_dose_mg": 100.0,
            "bbb_permeability_score": 0.5,
            "drug_bbb_permeable": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["recommended_dose_mg"] == pytest.approx(65.0)
        assert data["risk_level"] == "moderate"
        assert "research suggestion" in data["rationale"].lower()

    def test_severe_leakage_high_risk(self) -> None:
        client = TestClient(app)
        r = client.post("/research/drug_dose_adjustment", json={
            "baseline_dose_mg": 100.0,
            "bbb_permeability_score": 0.85,
            "drug_bbb_permeable": True,
        })
        data = r.json()
        assert data["risk_level"] == "high"

    def test_negative_baseline_returns_422(self) -> None:
        client = TestClient(app)
        r = client.post("/research/drug_dose_adjustment", json={
            "baseline_dose_mg": -1.0,
            "bbb_permeability_score": 0.5,
        })
        assert r.status_code == 422  # pydantic gt=0.0 validation

    def test_score_out_of_range_returns_422(self) -> None:
        client = TestClient(app)
        r = client.post("/research/drug_dose_adjustment", json={
            "baseline_dose_mg": 100.0,
            "bbb_permeability_score": 1.5,
        })
        assert r.status_code == 422
