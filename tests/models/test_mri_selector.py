"""Tests for src.models.mri_selector — env-var-driven 2D / 3D dispatch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import mri_selector
from tests.fixtures.build_dummy_mri_onnx import build as build_dummy_3d
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


_FIXTURE_MRI = Path(__file__).resolve().parents[1] / "fixtures" / "mri_sample" / "subject_0.nii.gz"


class TestSelector:
    def test_default_kind_is_volumetric(self, monkeypatch) -> None:
        monkeypatch.delenv("MRI_MODEL_KIND", raising=False)
        assert mri_selector.current_kind() == "volumetric_onnx"

    def test_explicit_2d_selection(self, monkeypatch) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
        assert mri_selector.current_kind() == "resnet18_2d"

    def test_unknown_kind_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "neural_net_supreme")
        with pytest.raises(ValueError, match="unknown MRI_MODEL_KIND"):
            mri_selector.current_kind()

    def test_predict_routes_to_volumetric(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "volumetric_onnx")
        artifact = build_dummy_3d(tmp_path / "vol.onnx")
        result = mri_selector.predict(
            input_path=_FIXTURE_MRI,
            checkpoint_path=artifact,
            target_shape=(8, 8, 8),
            label_names=("control", "abnormal"),
        )
        assert result["label_text"] in {"control", "abnormal"}

    def test_predict_routes_to_2d(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
        artifact = build_dummy_2d(tmp_path / "best.pt")
        img_path = tmp_path / "scan.png"
        Image.fromarray((np.random.RandomState(0).rand(160, 160, 3) * 255).astype("uint8")).save(str(img_path))
        result = mri_selector.predict(
            input_path=img_path,
            checkpoint_path=artifact,
        )
        assert result["label_text"] in mri_selector.label_names_for_kind("resnet18_2d")
