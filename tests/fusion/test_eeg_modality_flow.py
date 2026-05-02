"""End-to-end: EEG classifier output flows into fusion as the `eeg` modality."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fusion import engine
from src.fusion.types import (
    FusionInput,
    ModalityClassProb,
    ModalityPrediction,
)
from src.models import eeg_model
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


def _eeg_pred_from_features(model, features: np.ndarray) -> ModalityPrediction:
    raw = eeg_model.predict_features(model, features)
    return ModalityPrediction(
        label_text=raw["label_text"],
        label=raw["label"],
        confidence=raw["confidence"],
        probabilities=[
            ModalityClassProb(label_text=p["label_text"], probability=p["probability"])
            for p in raw["probabilities"]
        ],
    )


class TestEEGFusionFlow:
    def test_alzheimers_eeg_lifts_alzheimers_disease_score(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        model = eeg_model.load(ckpt)
        eeg_pred = _eeg_pred_from_features(model, np.full((16,), 2.0, dtype=np.float32))

        out = engine.fuse(FusionInput(eeg=eeg_pred))

        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability > 0.5
        assert any(c.modality == "eeg" for c in alz.contributions)
        assert "mri" in out.missing_inputs

    def test_control_eeg_does_not_inflate_alzheimers(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        model = eeg_model.load(ckpt)
        eeg_pred = _eeg_pred_from_features(model, np.zeros((16,), dtype=np.float32))

        out = engine.fuse(FusionInput(eeg=eeg_pred))

        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability < 0.5
