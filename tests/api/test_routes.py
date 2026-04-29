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
