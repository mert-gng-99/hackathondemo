"""Integration: POST /predict/eeg."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


@pytest.fixture()
def client(monkeypatch, tmp_path):
    artifact = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
    monkeypatch.setenv("EEG_CLF_ARTIFACT", str(artifact))
    return TestClient(app)


def test_predict_eeg_happy_path(client):
    body = {"features": [0.0] * 16}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] in {"control", "alzheimers"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 2


def test_predict_eeg_alzheimers_profile(client):
    body = {"features": [2.0] * 16}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] == "alzheimers"


def test_predict_eeg_feature_mismatch_returns_400(client):
    body = {"features": [0.0] * 8}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code == 400


def test_predict_eeg_missing_artifact_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("EEG_CLF_ARTIFACT", str(tmp_path / "missing.joblib"))
    client = TestClient(app)
    r = client.post("/predict/eeg", json={"features": [0.0] * 16})
    assert r.status_code == 503
