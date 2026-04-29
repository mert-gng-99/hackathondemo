"""Unit + integration tests for the MRI ComBat pipeline."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.mri_pipeline import (
    DEFAULT_N_ROI_AXES,
    ROI_STATS,
    extract_features_from_volume,
    harmonize_combat,
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


class TestHarmonizeCombat:
    def _build_two_site_features(self) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Synthesize a 6-row × 4-feature table with a clear site bias."""
        rng = np.random.default_rng(seed=42)
        feature_cols = ["feat_roi0_mean", "feat_roi1_mean", "feat_roi2_mean", "feat_roi3_mean"]
        # Site A baseline: mean ~0; Site B baseline: mean ~5 (the bias to remove).
        site_a = rng.normal(loc=0.0, scale=1.0, size=(3, 4))
        site_b = rng.normal(loc=5.0, scale=1.0, size=(3, 4))
        df = pd.DataFrame(
            np.vstack([site_a, site_b]),
            columns=feature_cols,
        )
        sites = pd.Series(["A", "A", "A", "B", "B", "B"], name="site")
        return df, sites, feature_cols

    def test_returns_dataframe_same_shape_and_columns(self) -> None:
        df, sites, feature_cols = self._build_two_site_features()
        out = harmonize_combat(df, sites, feature_cols)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == df.shape
        assert list(out.columns) == feature_cols

    def test_reduces_site_mean_difference(self) -> None:
        """ComBat must shrink the per-site mean gap on every harmonized column."""
        df, sites, feature_cols = self._build_two_site_features()
        gap_before = (
            df.loc[sites == "B", feature_cols].mean()
            - df.loc[sites == "A", feature_cols].mean()
        ).abs()

        out = harmonize_combat(df, sites, feature_cols)
        gap_after = (
            out.loc[sites == "B", feature_cols].mean()
            - out.loc[sites == "A", feature_cols].mean()
        ).abs()

        # Every column's site gap must shrink (ComBat aligns site means).
        assert (gap_after < gap_before).all(), (
            f"gap_before={gap_before.tolist()} gap_after={gap_after.tolist()}"
        )

    def test_output_dtype_float64(self) -> None:
        df, sites, feature_cols = self._build_two_site_features()
        out = harmonize_combat(df, sites, feature_cols)
        for c in feature_cols:
            assert out[c].dtype == np.float64, f"{c} → {out[c].dtype}"

    def test_no_nan_in_output(self) -> None:
        df, sites, feature_cols = self._build_two_site_features()
        out = harmonize_combat(df, sites, feature_cols)
        assert out[feature_cols].notna().all().all()
        assert np.isfinite(out[feature_cols].to_numpy()).all()

    def test_deterministic(self) -> None:
        df, sites, feature_cols = self._build_two_site_features()
        a = harmonize_combat(df, sites, feature_cols)
        b = harmonize_combat(df.copy(), sites.copy(), list(feature_cols))
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_raises_on_single_site(self) -> None:
        """ComBat needs at least 2 sites; a single-site dataset is malformed."""
        df, _, feature_cols = self._build_two_site_features()
        sites_one = pd.Series(["A"] * len(df), name="site")
        with pytest.raises(ValueError, match="at least 2 sites"):
            harmonize_combat(df, sites_one, feature_cols)

    def test_raises_on_empty_feature_cols(self) -> None:
        df, sites, _ = self._build_two_site_features()
        with pytest.raises(ValueError, match="feature_cols must be a non-empty list"):
            harmonize_combat(df, sites, [])

    def test_raises_on_length_mismatch(self) -> None:
        df, sites, feature_cols = self._build_two_site_features()
        # sites has 6 entries; truncate to 5 to force a mismatch.
        bad_sites = sites.iloc[:5]
        with pytest.raises(ValueError, match=r"features has 6 rows but sites has 5 elements"):
            harmonize_combat(df, bad_sites, feature_cols)
