"""Tests for src.models.bbb_permeability_map — researcher BBB scorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import bbb_permeability_map as bbb_perm
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


def _png(path: Path, size: tuple[int, int] = (170, 170)) -> Path:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))
    return path


class TestComputeFromClassifierProbs:
    def test_full_nondemented_yields_zero_score(self) -> None:
        probs = [
            {"label_text": "MildDemented", "probability": 0.0},
            {"label_text": "ModerateDemented", "probability": 0.0},
            {"label_text": "NonDemented", "probability": 1.0},
            {"label_text": "VeryMildDemented", "probability": 0.0},
        ]
        assert bbb_perm.compute_from_classifier_probs(probs) == pytest.approx(0.0)

    def test_no_nondemented_yields_one(self) -> None:
        probs = [
            {"label_text": "MildDemented", "probability": 0.5},
            {"label_text": "ModerateDemented", "probability": 0.5},
        ]
        assert bbb_perm.compute_from_classifier_probs(probs) == pytest.approx(1.0)

    def test_partial_yields_complement(self) -> None:
        probs = [
            {"label_text": "NonDemented", "probability": 0.7},
            {"label_text": "MildDemented", "probability": 0.3},
        ]
        assert bbb_perm.compute_from_classifier_probs(probs) == pytest.approx(0.3)

    def test_case_insensitive_label_match(self) -> None:
        probs = [{"label_text": "nondemented", "probability": 0.6}]
        assert bbb_perm.compute_from_classifier_probs(probs) == pytest.approx(0.4)


class TestComputePermeability:
    def test_heuristic_proxy_with_dummy_2d(self, tmp_path: Path) -> None:
        ckpt = build_dummy_2d(tmp_path / "best.pt")
        img = _png(tmp_path / "scan.png")
        result = bbb_perm.compute_permeability(
            input_path=img, mode="heuristic_proxy", checkpoint_path=ckpt,
        )
        assert 0.0 <= result["permeability_score"] <= 1.0
        assert result["interpretation"] in {
            "BBB intact", "mild leakage", "moderate leakage", "severe leakage",
        }
        assert result["method"] == "heuristic_proxy"
        assert result["voxel_map_available"] is False

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        img = _png(tmp_path / "scan.png")
        with pytest.raises(ValueError, match="unknown BBB permeability mode"):
            bbb_perm.compute_permeability(input_path=img, mode="bogus")  # type: ignore[arg-type]

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MRI input not found"):
            bbb_perm.compute_permeability(input_path=tmp_path / "nope.png")

    def test_dce_mode_without_artifact_raises(self, tmp_path: Path) -> None:
        # Existence check on the input is the first gate; placeholder file is fine.
        img = tmp_path / "fake_dce.nii.gz"
        img.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="DCE-MRI BBB permeability artifact"):
            bbb_perm.compute_permeability(
                input_path=img, mode="dce_onnx",
                checkpoint_path=tmp_path / "missing.onnx",
            )

    def test_interpretation_thresholds(self) -> None:
        assert bbb_perm._interpret(0.10) == "BBB intact"
        assert bbb_perm._interpret(0.30) == "mild leakage"
        assert bbb_perm._interpret(0.55) == "moderate leakage"
        assert bbb_perm._interpret(0.80) == "severe leakage"
