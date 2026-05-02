"""Tests for src.models.mri_dl_2d — pretrained 4-class Alzheimer's resnet18."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import mri_dl_2d
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


def _png(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))
    return path


class TestMRIDL2D:
    def test_class_to_idx_matches_trainer(self) -> None:
        assert mri_dl_2d.CLASS_TO_IDX == {
            "MildDemented": 0,
            "ModerateDemented": 1,
            "NonDemented": 2,
            "VeryMildDemented": 3,
        }

    def test_idx_to_class_is_consistent(self) -> None:
        for name, idx in mri_dl_2d.CLASS_TO_IDX.items():
            assert mri_dl_2d.IDX_TO_CLASS[idx] == name

    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MRI 2D checkpoint not found"):
            mri_dl_2d.load(tmp_path / "nope.pt")

    def test_predict_image_returns_full_probs(self, tmp_path: Path) -> None:
        ckpt = build_dummy_2d(tmp_path / "best.pt")
        model = mri_dl_2d.load(ckpt)
        img = _png(tmp_path / "scan.png")

        result = mri_dl_2d.predict_image(model, img)

        assert set(result) == {"label", "label_text", "confidence", "probabilities"}
        assert result["label"] in {0, 1, 2, 3}
        assert result["label_text"] in mri_dl_2d.CLASS_TO_IDX
        assert 0.0 <= result["confidence"] <= 1.0
        probs = result["probabilities"]
        assert len(probs) == 4
        assert abs(sum(p["probability"] for p in probs) - 1.0) < 1e-5
        assert {p["label_text"] for p in probs} == set(mri_dl_2d.CLASS_TO_IDX)

    def test_predict_works_for_grayscale_input(self, tmp_path: Path) -> None:
        ckpt = build_dummy_2d(tmp_path / "best.pt")
        model = mri_dl_2d.load(ckpt)
        gray = (np.random.RandomState(1).rand(180, 180) * 255).astype(np.uint8)
        path = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(str(path))

        result = mri_dl_2d.predict_image(model, path)
        assert 0.0 <= result["confidence"] <= 1.0
