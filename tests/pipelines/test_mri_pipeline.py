"""Unit + integration tests for the MRI ComBat pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipelines.mri_pipeline import is_valid_volume


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mri_sample"


class TestIsValidVolume:
    def test_accepts_3d_finite_array(self) -> None:
        vol = np.zeros((8, 8, 8), dtype=np.float64)
        assert is_valid_volume(vol) is True

    def test_rejects_wrong_dimension(self) -> None:
        assert is_valid_volume(np.zeros((8, 8))) is False
        assert is_valid_volume(np.zeros((8, 8, 8, 2))) is False

    def test_rejects_nan(self) -> None:
        vol = np.zeros((8, 8, 8))
        vol[0, 0, 0] = np.nan
        assert is_valid_volume(vol) is False

    def test_rejects_inf(self) -> None:
        vol = np.zeros((8, 8, 8))
        vol[1, 1, 1] = np.inf
        assert is_valid_volume(vol) is False
        vol[1, 1, 1] = -np.inf
        assert is_valid_volume(vol) is False

    def test_rejects_empty(self) -> None:
        assert is_valid_volume(np.zeros((0, 8, 8))) is False
        assert is_valid_volume(np.zeros((8, 0, 8))) is False
        assert is_valid_volume(np.zeros((8, 8, 0))) is False

    def test_rejects_non_numeric_dtype(self) -> None:
        vol = np.array([[["a", "b"], ["c", "d"]]])
        assert is_valid_volume(vol) is False

    def test_rejects_non_array(self) -> None:
        assert is_valid_volume([[[1, 2]], [[3, 4]]]) is False
        assert is_valid_volume(None) is False
