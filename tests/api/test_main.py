"""Tests for the FastAPI app surface (health + schema imports)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_get_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_get_health_returns_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_get_health_returns_pipeline_list(self):
        resp = client.get("/health")
        body = resp.json()
        assert set(body["pipelines"]) == {"bbb", "eeg", "mri"}
