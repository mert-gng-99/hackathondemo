"""Tests for src.models.mri_model — image-based MRI DL inference surface."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import mri_model
from tests.fixtures.build_dummy_mri_onnx import build as build_dummy_mri_onnx


_FIXTURE_MRI = Path(__file__).resolve().parents[1] / "fixtures" / "mri_sample" / "subject_0.nii.gz"


class TestMRIDLModel:
    def test_preprocess_volume_returns_batch_channel_tensor(self) -> None:
        volume = np.ones((4, 5, 6), dtype=np.float32)
        volume[1:3, 1:4, 2:5] = 5.0

        out = mri_model.preprocess_volume(volume, target_shape=(8, 8, 8))

        assert out.shape == (1, 1, 8, 8, 8)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_preprocess_rejects_nan_volume(self) -> None:
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        volume[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="finite numeric 3-D"):
            mri_model.preprocess_volume(volume, target_shape=(8, 8, 8))

    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MRI model artifact not found"):
            mri_model.load(tmp_path / "missing.onnx")

    def test_predict_nifti_with_dummy_onnx(self, tmp_path: Path) -> None:
        artifact = build_dummy_mri_onnx(tmp_path / "mri_model.onnx")
        model = mri_model.load(artifact)

        result = mri_model.predict_nifti(
            model,
            _FIXTURE_MRI,
            target_shape=(8, 8, 8),
            label_names=("control", "abnormal"),
        )

        assert result["label"] == 1
        assert result["label_text"] == "abnormal"
        assert result["confidence"] > 0.5
        probs = result["probabilities"]
        assert len(probs) == 2
        assert sum(p["probability"] for p in probs) == pytest.approx(1.0, abs=1e-6)

    def test_predict_warns_on_label_count_mismatch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        artifact = build_dummy_mri_onnx(tmp_path / "mri_model.onnx")
        model = mri_model.load(artifact)

        # mri_model.logger has propagate=False (src/core/logger.py), so pytest's
        # caplog root handler never sees its records. Attach caplog.handler directly.
        mri_model.logger.addHandler(caplog.handler)
        try:
            result = mri_model.predict_nifti(
                model,
                _FIXTURE_MRI,
                target_shape=(8, 8, 8),
                label_names=("control", "abnormal", "extra"),
            )
        finally:
            mri_model.logger.removeHandler(caplog.handler)

        assert result["label_text"] in {"class_0", "class_1"}
        assert any(
            "label_names length" in rec.message and "overriding" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
