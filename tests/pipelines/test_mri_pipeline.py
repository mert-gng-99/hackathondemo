"""Unit + integration tests for the MRI ComBat pipeline."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.pipelines.mri_pipeline import (
    is_valid_volume,
    mask_brain,
)


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


class TestMaskBrain:
    def _load_subject(self, sid: str) -> np.ndarray:
        return nib.load(FIXTURE_DIR / f"{sid}.nii.gz").get_fdata()

    def test_returns_bool_mask_of_same_shape(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == vol.shape

    def test_mask_separates_brain_from_background(self) -> None:
        """Default threshold should keep the spherical-brain center voxels in."""
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        # The fixture's brain region (radius 3 around center) intensity is ~10;
        # background is ~0.1. Some brain voxels MUST survive the mask.
        assert mask.sum() > 0
        # The center voxel (always brain) MUST be in the mask.
        center = tuple(s // 2 for s in vol.shape)
        assert mask[center]

    def test_mask_drops_low_intensity_background(self) -> None:
        """Voxels with intensity well below the brain core must be excluded."""
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol, intensity_threshold=5.0)
        # Background voxels (intensity ~0.1) must NOT be in the mask.
        bg_voxel = (0, 0, 0)
        assert mask[bg_voxel] == False  # noqa: E712

    def test_explicit_threshold_overrides_default(self) -> None:
        vol = self._load_subject("subject_0")
        # A very high threshold should produce far fewer mask voxels.
        mask_default = mask_brain(vol)
        mask_strict = mask_brain(vol, intensity_threshold=100.0)
        assert mask_strict.sum() < mask_default.sum()

    def test_does_not_mutate_input(self) -> None:
        vol = self._load_subject("subject_0")
        original = vol.copy()
        _ = mask_brain(vol)
        np.testing.assert_array_equal(vol, original)

    def test_morphological_cleanup_removes_isolated_voxels(self) -> None:
        """A single bright voxel surrounded by background must be removed by the
        opening-style morphological cleanup."""
        vol = np.zeros((8, 8, 8), dtype=np.float64)
        vol[4, 4, 4] = 100.0
        mask = mask_brain(vol, intensity_threshold=50.0)
        # Without cleanup, the single voxel would survive. With morphological
        # opening, it must be removed.
        assert mask.sum() == 0
