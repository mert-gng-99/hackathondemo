"""Integration: POST /predict/mri with MRI_MODEL_KIND=resnet18_2d."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


@pytest.fixture()
def client_2d(monkeypatch, tmp_path):
    monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
    ckpt = build_dummy_2d(tmp_path / "best.pt")
    monkeypatch.setenv("MRI_MODEL_PATH_2D", str(ckpt))
    return TestClient(app)


def test_predict_mri_2d_happy_path(client_2d, tmp_path):
    img_path = tmp_path / "scan.png"
    Image.fromarray((np.random.RandomState(0).rand(170, 170, 3) * 255).astype("uint8")).save(str(img_path))

    r = client_2d.post("/predict/mri", json={"input_path": str(img_path)})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] in {
        "MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented",
    }
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4


def test_predict_mri_2d_missing_artifact_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
    monkeypatch.setenv("MRI_MODEL_PATH_2D", str(tmp_path / "missing.pt"))
    img_path = tmp_path / "scan.png"
    Image.fromarray((np.random.RandomState(0).rand(170, 170, 3) * 255).astype("uint8")).save(str(img_path))
    client = TestClient(app)
    r = client.post("/predict/mri", json={"input_path": str(img_path)})
    assert r.status_code == 503
    assert "kind=resnet18_2d" in r.text
