"""Unit + integration tests for the EEG pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from src.pipelines.eeg_pipeline import (
    bandpass_filter,
    compute_features_from_epoch,
    extract_features_from_recording,
    is_valid_epoch,
    remove_artifacts_with_ica,
    run_pipeline,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "eeg_sample.fif"


EEG_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
STATS = ("mean", "std", "var", "skew", "kurtosis")


class TestIsValidEpoch:
    def test_accepts_2d_finite_array(self) -> None:
        epoch = np.zeros((4, 256), dtype=np.float64)
        assert is_valid_epoch(epoch) is True

    def test_rejects_wrong_dimension(self) -> None:
        assert is_valid_epoch(np.zeros((4,))) is False
        assert is_valid_epoch(np.zeros((4, 256, 2))) is False

    def test_rejects_nan(self) -> None:
        epoch = np.zeros((4, 256))
        epoch[0, 10] = np.nan
        assert is_valid_epoch(epoch) is False

    def test_rejects_inf(self) -> None:
        epoch = np.zeros((4, 256))
        epoch[1, 5] = np.inf
        assert is_valid_epoch(epoch) is False
        epoch[1, 5] = -np.inf
        assert is_valid_epoch(epoch) is False

    def test_rejects_empty(self) -> None:
        assert is_valid_epoch(np.zeros((0, 256))) is False
        assert is_valid_epoch(np.zeros((4, 0))) is False

    def test_rejects_non_array(self) -> None:
        assert is_valid_epoch([[1, 2, 3]]) is False
        assert is_valid_epoch(None) is False

    def test_rejects_non_numeric_dtype(self) -> None:
        """String / object dtype arrays must be rejected without raising."""
        epoch = np.array([["a", "b"], ["c", "d"]])
        assert is_valid_epoch(epoch) is False


class TestBandpassFilter:
    def _load(self) -> mne.io.BaseRaw:
        return mne.io.read_raw_fif(FIXTURE, preload=True, verbose="ERROR")

    def test_returns_raw_instance(self) -> None:
        raw = self._load()
        out = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert isinstance(out, mne.io.BaseRaw)

    def test_preserves_shape(self) -> None:
        raw = self._load()
        n_ch_before, n_t_before = raw.get_data().shape
        out = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert out.get_data().shape == (n_ch_before, n_t_before)

    def test_attenuates_dc_component(self) -> None:
        """A bandpass with l_freq=1.0 must remove a DC offset."""
        raw = self._load()
        # Inject a large DC offset on every channel.
        data = raw.get_data() + 1e-3
        raw_dc = mne.io.RawArray(data, raw.info, verbose="ERROR")
        out = bandpass_filter(raw_dc, l_freq=1.0, h_freq=40.0)
        # Mean on each channel should be near zero (much smaller than 1e-3).
        assert np.all(np.abs(out.get_data().mean(axis=1)) < 1e-4)

    def test_does_not_mutate_input(self) -> None:
        raw = self._load()
        original_mean = raw.get_data().mean()
        _ = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert raw.get_data().mean() == pytest.approx(original_mean, rel=1e-12)

    def test_rejects_inverted_frequency_range(self) -> None:
        """l_freq must be strictly < h_freq; otherwise raise instead of silently corrupting data."""
        raw = self._load()
        with pytest.raises(ValueError, match="must be strictly less than"):
            bandpass_filter(raw, l_freq=40.0, h_freq=1.0)
        with pytest.raises(ValueError, match="must be strictly less than"):
            bandpass_filter(raw, l_freq=10.0, h_freq=10.0)


class TestRemoveArtifactsWithIca:
    def _load(self) -> mne.io.BaseRaw:
        return mne.io.read_raw_fif(FIXTURE, preload=True, verbose="ERROR")

    def test_returns_raw_instance(self) -> None:
        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        out = remove_artifacts_with_ica(
            raw, eog_ch_name="EOG061", n_components=4, random_state=97,
        )
        assert isinstance(out, mne.io.BaseRaw)

    def test_preserves_shape(self) -> None:
        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        before = raw.get_data().shape
        out = remove_artifacts_with_ica(
            raw, eog_ch_name="EOG061", n_components=4, random_state=97,
        )
        assert out.get_data().shape == before

    def test_reduces_eog_correlation_on_frontal_channel(self) -> None:
        """ICA must reduce correlation between EOG and Cz (the bleed channel)."""
        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        before = raw.get_data()
        cz_idx = raw.ch_names.index("Cz")
        eog_idx = raw.ch_names.index("EOG061")
        corr_before = abs(np.corrcoef(before[cz_idx], before[eog_idx])[0, 1])

        out = remove_artifacts_with_ica(
            raw, eog_ch_name="EOG061", n_components=4, random_state=97,
        )
        after = out.get_data()
        corr_after = abs(np.corrcoef(after[cz_idx], after[eog_idx])[0, 1])
        # Allow for noise — but the dominant EOG bleed must be reduced.
        assert corr_after < corr_before

    def test_no_eog_channel_is_a_noop(self) -> None:
        """Without an EOG reference, ICA can't auto-reject — should pass through."""
        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        out = remove_artifacts_with_ica(
            raw, eog_ch_name=None, n_components=4, random_state=97,
        )
        # Identical shape; data approximately equal (no rejection happened).
        assert out.get_data().shape == raw.get_data().shape
        np.testing.assert_allclose(
            out.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12
        )

    def test_is_deterministic_with_seed(self) -> None:
        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        a = remove_artifacts_with_ica(
            raw, eog_ch_name="EOG061", n_components=4, random_state=97,
        )
        b = remove_artifacts_with_ica(
            raw, eog_ch_name="EOG061", n_components=4, random_state=97,
        )
        np.testing.assert_allclose(a.get_data(), b.get_data(), rtol=1e-12, atol=1e-15)

    def test_unknown_eog_channel_logs_warning_and_is_a_noop(self) -> None:
        """A misconfigured eog_ch_name (typo) must not silently behave like None."""
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import eeg_pipeline as mod

        raw = bandpass_filter(self._load(), l_freq=1.0, h_freq=40.0)
        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            out = remove_artifacts_with_ica(
                raw, eog_ch_name="EOG_DOES_NOT_EXIST",
                n_components=4, random_state=97,
            )
        finally:
            handler.stream = original_stream

        # Behavior: ICA was skipped (no-op) but the log differentiates it from None.
        np.testing.assert_allclose(out.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)
        log_output = buf.getvalue()
        assert "ICA skipped: eog_ch_name='EOG_DOES_NOT_EXIST' not found" in log_output


class TestComputeFeaturesFromEpoch:
    def test_returns_1d_float_array(self) -> None:
        epoch = np.random.default_rng(0).standard_normal((4, 256))
        out = compute_features_from_epoch(epoch, sfreq=256.0)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 1
        assert out.dtype == np.float64

    def test_feature_count_matches_contract(self) -> None:
        """Each channel contributes len(EEG_BANDS) PSD features + len(STATS) stats."""
        n_channels = 4
        epoch = np.random.default_rng(0).standard_normal((n_channels, 256))
        out = compute_features_from_epoch(epoch, sfreq=256.0)
        expected = n_channels * (len(EEG_BANDS) + len(STATS))
        assert out.shape == (expected,)

    def test_alpha_band_dominates_for_alpha_signal(self) -> None:
        """Pure 10 Hz sine on 1 channel should put most PSD power in alpha (8-13 Hz)."""
        sfreq = 256.0
        t = np.arange(int(sfreq * 2.0)) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * t)[None, :]  # (1, n_samples)
        out = compute_features_from_epoch(signal, sfreq=sfreq)
        # Layout for n_channels=1: [psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma, mean, std, var, skew, kurtosis]
        psd_block = out[: len(EEG_BANDS)]
        alpha_idx = EEG_BANDS.index("alpha")
        assert psd_block[alpha_idx] == psd_block.max()

    def test_finite_output(self) -> None:
        epoch = np.random.default_rng(0).standard_normal((4, 256))
        out = compute_features_from_epoch(epoch, sfreq=256.0)
        assert np.all(np.isfinite(out))

    def test_deterministic_for_same_input(self) -> None:
        epoch = np.random.default_rng(0).standard_normal((4, 256))
        a = compute_features_from_epoch(epoch, sfreq=256.0)
        b = compute_features_from_epoch(epoch, sfreq=256.0)
        np.testing.assert_array_equal(a, b)

    def test_stats_labels_and_funcs_stay_in_sync(self) -> None:
        """STATS labels must equal the names in _STATS_FUNCS — single source of truth."""
        from src.pipelines.eeg_pipeline import _STATS_FUNCS

        derived_names = tuple(name for name, _ in _STATS_FUNCS)
        assert derived_names == STATS

    def test_constant_channel_yields_finite_features(self) -> None:
        """A flat-line channel must not produce NaN features (skew/kurtosis are undefined for zero-variance)."""
        epoch = np.zeros((4, 512), dtype=np.float64)
        out = compute_features_from_epoch(epoch, sfreq=256.0)
        assert np.all(np.isfinite(out))


class TestExtractFeaturesFromRecording:
    def _load(self) -> mne.io.BaseRaw:
        return mne.io.read_raw_fif(FIXTURE, preload=True, verbose="ERROR")

    def test_returns_dataframe(self) -> None:
        raw = self._load()
        df = extract_features_from_recording(
            raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_epochs(self) -> None:
        """10 s recording / 2 s epoch = 5 epochs."""
        raw = self._load()
        df = extract_features_from_recording(
            raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        assert len(df) == 5

    def test_column_naming_is_deterministic_and_explicit(self) -> None:
        raw = self._load()
        df = extract_features_from_recording(
            raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        # 4 EEG channels: Cz, Pz, O1, O2 (EOG channel is excluded from features).
        for ch in ("Cz", "Pz", "O1", "O2"):
            for band in EEG_BANDS:
                assert f"feat_{ch}_psd_{band}" in df.columns
            for stat in STATS:
                assert f"feat_{ch}_{stat}" in df.columns

    def test_no_feat_for_eog_channel(self) -> None:
        raw = self._load()
        df = extract_features_from_recording(
            raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        assert not any("EOG061" in c for c in df.columns)

    def test_all_features_finite_float64(self) -> None:
        raw = self._load()
        df = extract_features_from_recording(
            raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert all(df[c].dtype == np.float64 for c in feat_cols)
        assert df[feat_cols].notna().all().all()
        assert np.isfinite(df[feat_cols].to_numpy()).all()

    def test_drops_invalid_epochs_with_warning(self) -> None:
        """A NaN in the recording: at least one epoch dropped, no NaN survives, WARNING is logged.

        The bandpass filter is a long FIR convolution, so a single NaN sample
        spreads across many samples. The principled behavior is therefore:
        (a) drop every contaminated epoch, not just the source epoch, and
        (b) guarantee no NaN in the output. The exact drop count depends on
        the filter's FIR length, so we assert range + cleanliness instead of
        an exact number. The WARNING line is part of the AGENTS.md §4
        traceability contract and must always fire when drops happen.
        """
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import eeg_pipeline as mod

        raw = self._load()
        data = raw.get_data().copy()
        data[0, -10] = np.nan
        bad_raw = mne.io.RawArray(data, raw.info, verbose="ERROR")

        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            df = extract_features_from_recording(
                bad_raw, epoch_duration_s=2.0, eog_ch_name="EOG061",
                n_components=4, random_state=97,
            )
        finally:
            handler.stream = original_stream

        # At least one epoch dropped (vs the clean 5-row baseline).
        assert len(df) < 5
        # No NaN/inf must survive into the feature table.
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert df[feat_cols].notna().all().all()
        assert np.isfinite(df[feat_cols].to_numpy()).all()
        # AGENTS.md §4: the WARNING line was actually emitted.
        log_output = buf.getvalue()
        assert "Dropping" in log_output and "epochs with invalid samples" in log_output

    def test_raises_when_epoch_duration_too_small(self) -> None:
        raw = self._load()
        with pytest.raises(ValueError, match="must be >= 1"):
            extract_features_from_recording(
                raw, epoch_duration_s=1e-6, eog_ch_name="EOG061",
                n_components=4, random_state=97,
            )


class TestRunPipeline:
    def test_end_to_end_writes_processed_parquet(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "rec.fif"
        output_path = proc_dir / "eeg_features.parquet"
        shutil.copy(FIXTURE, input_path)

        run_pipeline(
            input_path=input_path, output_path=output_path,
            epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )

        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 5
        assert all(c.startswith("feat_") for c in df.columns)

    def test_run_pipeline_preserves_float64_dtype(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "rec.fif"
        output_path = proc_dir / "eeg_features.parquet"
        shutil.copy(FIXTURE, input_path)

        run_pipeline(
            input_path=input_path, output_path=output_path,
            epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        df = pd.read_parquet(output_path)
        for col in df.columns:
            assert df[col].dtype == np.float64, f"{col} widened to {df[col].dtype}"

    def test_run_pipeline_is_idempotent(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "rec.fif"
        output_path = proc_dir / "eeg_features.parquet"
        shutil.copy(FIXTURE, input_path)

        run_pipeline(
            input_path=input_path, output_path=output_path,
            epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        first = output_path.read_bytes()
        run_pipeline(
            input_path=input_path, output_path=output_path,
            epoch_duration_s=2.0, eog_ch_name="EOG061",
            n_components=4, random_state=97,
        )
        second = output_path.read_bytes()
        assert first == second, "EEG pipeline output must be byte-deterministic"

    def test_run_pipeline_raises_when_input_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_pipeline(
                input_path=tmp_path / "nope.fif",
                output_path=tmp_path / "out.parquet",
            )

    def test_run_pipeline_rejects_directory_as_output(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        input_path = raw_dir / "rec.fif"
        shutil.copy(FIXTURE, input_path)
        bad_output = tmp_path / "out_dir"
        bad_output.mkdir()
        with pytest.raises(IsADirectoryError, match="must be a file"):
            run_pipeline(
                input_path=input_path, output_path=bad_output,
                epoch_duration_s=2.0, eog_ch_name="EOG061",
                n_components=4, random_state=97,
            )
