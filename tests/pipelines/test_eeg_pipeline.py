"""Unit + integration tests for the EEG pipeline."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from src.pipelines.eeg_pipeline import (
    bandpass_filter,
    is_valid_epoch,
    remove_artifacts_with_ica,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "eeg_sample.fif"


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
