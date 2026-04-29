"""Unit + integration tests for the MRI ComBat pipeline."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.pipelines.mri_pipeline import (
    DEFAULT_N_ROI_AXES,
    ROI_STATS,
    extract_features_from_volume,
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

    def test_constant_volume_returns_all_false_mask_with_warning(self) -> None:
        """A constant-valued volume produces an empty mask AND logs a WARNING."""
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import mri_pipeline as mod

        vol = np.full((8, 8, 8), 5.0, dtype=np.float64)

        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            mask = mask_brain(vol)
        finally:
            handler.stream = original_stream

        assert mask.sum() == 0
        log_output = buf.getvalue()
        assert "all-False mask" in log_output
        assert "downstream features for this volume will be all-zero" in log_output


class TestExtractFeaturesFromVolume:
    def _load_subject(self, sid: str) -> np.ndarray:
        return nib.load(FIXTURE_DIR / f"{sid}.nii.gz").get_fdata()

    def test_returns_dict_with_correct_keys(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        feats = extract_features_from_volume(vol, mask)
        n_roi = int(np.prod(DEFAULT_N_ROI_AXES))
        expected = {
            f"feat_roi{i}_{stat}"
            for i in range(n_roi)
            for stat in ROI_STATS
        }
        assert set(feats.keys()) == expected

    def test_feature_count_matches_contract(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        feats = extract_features_from_volume(vol, mask)
        n_roi = int(np.prod(DEFAULT_N_ROI_AXES))
        assert len(feats) == n_roi * len(ROI_STATS)

    def test_all_features_finite_float(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        feats = extract_features_from_volume(vol, mask)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k}: {type(v).__name__}"
            assert np.isfinite(v), f"{k}: {v}"

    def test_voxel_count_is_integer_valued(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        feats = extract_features_from_volume(vol, mask)
        for k, v in feats.items():
            if k.endswith("_voxel_count"):
                # voxel_count stored as float for column-uniformity, but must be
                # a whole number.
                assert v == float(int(v))

    def test_empty_mask_yields_zero_features(self) -> None:
        """If a volume has zero brain voxels (mask all False), every stat
        must default to 0.0 — not NaN — to preserve the no-NaN Parquet contract."""
        vol = self._load_subject("subject_0")
        empty_mask = np.zeros_like(vol, dtype=bool)
        feats = extract_features_from_volume(vol, empty_mask)
        for k, v in feats.items():
            assert v == 0.0, f"{k}: {v}"

    def test_deterministic_for_same_input(self) -> None:
        vol = self._load_subject("subject_0")
        mask = mask_brain(vol)
        a = extract_features_from_volume(vol, mask)
        b = extract_features_from_volume(vol, mask)
        assert a == b

    def test_roi_stats_labels_and_funcs_stay_in_sync(self) -> None:
        """ROI_STATS labels must equal the names in _ROI_STATS_FUNCS — single source of truth."""
        from src.pipelines.mri_pipeline import _ROI_STATS_FUNCS

        derived_names = tuple(name for name, _ in _ROI_STATS_FUNCS)
        assert derived_names == ROI_STATS

    def test_raises_on_shape_mismatch(self) -> None:
        """volume.shape and mask.shape must agree — the contract is enforced."""
        vol = np.zeros((8, 8, 8), dtype=np.float64)
        bad_mask = np.zeros((4, 4, 4), dtype=bool)
        with pytest.raises(ValueError, match=r"volume\.shape .* != mask\.shape"):
            extract_features_from_volume(vol, bad_mask)
