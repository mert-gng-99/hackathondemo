"""Tests for src.models.eeg_model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import eeg_model
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


class TestEEGModel:
    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="EEG classifier artifact not found"):
            eeg_model.load(tmp_path / "nope.joblib")

    def test_predict_returns_full_dict(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        features = np.zeros((16,), dtype=np.float32)

        out = eeg_model.predict_features(clf, features)

        assert set(out) == {"label", "label_text", "confidence", "probabilities"}
        assert out["label"] in {0, 1}
        assert out["label_text"] in eeg_model.DEFAULT_LABELS
        assert 0.0 <= out["confidence"] <= 1.0
        probs = out["probabilities"]
        assert len(probs) == 2
        assert abs(sum(p["probability"] for p in probs) - 1.0) < 1e-5

    def test_alzheimers_separation_with_synthetic_features(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        alz_features = np.full((16,), 2.0, dtype=np.float32)
        ctrl_features = np.zeros((16,), dtype=np.float32)

        alz_pred = eeg_model.predict_features(clf, alz_features)
        ctrl_pred = eeg_model.predict_features(clf, ctrl_features)

        assert alz_pred["label_text"] == "alzheimers"
        assert ctrl_pred["label_text"] == "control"

    def test_label_override_via_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("EEG_CLF_LABELS", "no_disease,alzheimers")
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        out = eeg_model.predict_features(clf, np.zeros((16,), dtype=np.float32))
        assert out["label_text"] in {"no_disease", "alzheimers"}

    def test_feature_count_mismatch_raises(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        with pytest.raises(ValueError, match="feature count"):
            eeg_model.predict_features(clf, np.zeros((8,), dtype=np.float32))
