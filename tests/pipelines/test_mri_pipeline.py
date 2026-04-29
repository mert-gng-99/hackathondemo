"""Unit + integration tests for the MRI ComBat pipeline."""
from __future__ import annotations

import shutil
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
    run_pipeline,
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


class TestRunPipeline:
    def _stage_inputs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Copy the committed MRI fixture into a tmp_path layout."""
        raw_dir = tmp_path / "data" / "raw" / "mri"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        for src in FIXTURE_DIR.iterdir():
            shutil.copy(src, raw_dir / src.name)
        sites_csv = raw_dir / "sites.csv"
        output_path = proc_dir / "mri_features.parquet"
        return raw_dir, sites_csv, output_path

    def test_end_to_end_writes_processed_parquet(self, tmp_path: Path) -> None:
        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 6
        assert "subject_id" in df.columns
        assert "site" in df.columns
        assert any(c.startswith("feat_roi") for c in df.columns)

    def test_run_pipeline_preserves_float64_for_features(self, tmp_path: Path) -> None:
        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        df = pd.read_parquet(output_path)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        for c in feat_cols:
            assert df[c].dtype == np.float64, f"{c} widened to {df[c].dtype}"

    def test_run_pipeline_is_idempotent(self, tmp_path: Path) -> None:
        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        first = output_path.read_bytes()
        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        second = output_path.read_bytes()
        assert first == second, "MRI pipeline output must be byte-deterministic"

    def test_run_pipeline_reduces_site_gap(self, tmp_path: Path) -> None:
        """End-to-end: ComBat must shrink the per-site mean gap in feat_roi0_mean."""
        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        df = pd.read_parquet(output_path)
        site_means = df.groupby("site")["feat_roi0_mean"].mean()
        gap = abs(site_means["B"] - site_means["A"])
        assert gap < 1.0, f"site gap after ComBat: {gap}"

    def test_run_pipeline_raises_when_input_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MRI input directory not found"):
            run_pipeline(
                input_dir=tmp_path / "nope",
                sites_csv=tmp_path / "sites.csv",
                output_path=tmp_path / "out.parquet",
            )

    def test_run_pipeline_rejects_directory_as_output(self, tmp_path: Path) -> None:
        raw_dir, sites_csv, _ = self._stage_inputs(tmp_path)
        bad_output = tmp_path / "out_dir"
        bad_output.mkdir()
        with pytest.raises(IsADirectoryError, match="must be a file"):
            run_pipeline(
                input_dir=raw_dir, sites_csv=sites_csv, output_path=bad_output,
            )

    def test_run_pipeline_drops_invalid_volumes(self, tmp_path: Path) -> None:
        """A NaN-containing volume must be logged + dropped, not silently included."""
        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        # Corrupt subject_5 to contain NaN. Re-save in place.
        bad = nib.load(raw_dir / "subject_5.nii.gz").get_fdata()
        bad[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(bad, affine=np.eye(4)), raw_dir / "subject_5.nii.gz")

        run_pipeline(
            input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
        )
        df = pd.read_parquet(output_path)
        # 5 surviving valid subjects (subject_5 dropped).
        assert len(df) == 5
        assert "subject_5" not in df["subject_id"].tolist()

    def test_run_pipeline_handles_all_constant_features(self, tmp_path: Path) -> None:
        """Degenerate dataset: every feature column is constant — ComBat must be
        skipped gracefully with a WARNING, not crash with ValueError."""
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import mri_pipeline as mod

        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)
        # Overwrite all volumes with the same constant intensity so every
        # feature column is identical across subjects.
        affine = np.eye(4)
        for nii in sorted(raw_dir.glob("*.nii.gz")):
            const_vol = np.full((8, 8, 8), 7.0, dtype=np.float64)
            nib.save(nib.Nifti1Image(const_vol, affine=affine), nii)

        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            run_pipeline(
                input_dir=raw_dir, sites_csv=sites_csv,
                output_path=output_path, intensity_threshold=1.0,
            )
        finally:
            handler.stream = original_stream

        df = pd.read_parquet(output_path)
        assert len(df) == 6
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        # All-zero-variance fallback: features pass through unchanged.
        assert df[feat_cols].notna().all().all()
        log_output = buf.getvalue()
        assert "ComBat skipped" in log_output

    def test_run_pipeline_extraction_log_precedes_write(self, tmp_path: Path) -> None:
        """The 'Feature extraction complete' INFO must fire BEFORE the
        'Wrote processed features' INFO so that operators get a summary
        even if to_parquet raises."""
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import mri_pipeline as mod

        raw_dir, sites_csv, output_path = self._stage_inputs(tmp_path)

        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            run_pipeline(
                input_dir=raw_dir, sites_csv=sites_csv, output_path=output_path,
            )
        finally:
            handler.stream = original_stream

        log_output = buf.getvalue()
        extract_idx = log_output.index("Feature extraction complete:")
        wrote_idx = log_output.index("Wrote processed features to")
        assert extract_idx < wrote_idx, "extraction summary must precede write log"


import mlflow
from src.pipelines import mri_pipeline as _mri_for_mlflow_test
from tests.fixtures import build_mri_fixture as _build_mri_for_mlflow_test


class TestMRIPipelineMLflow:
    def test_run_pipeline_creates_mlflow_run(self, tmp_path):
        fixture_dir = _build_mri_for_mlflow_test.build(out_dir=tmp_path / "mri_fixture")
        out = tmp_path / "out.parquet"
        _mri_for_mlflow_test.run_pipeline(
            input_dir=fixture_dir, output_path=out,
        )
        runs = mlflow.search_runs(
            experiment_names=["mri_pipeline"],
            order_by=["start_time DESC"],
        )
        assert len(runs) >= 1
        assert "metrics.subjects_out" in runs.columns
        assert runs.iloc[0]["metrics.subjects_out"] > 0
