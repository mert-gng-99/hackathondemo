"""Generate a deterministic synthetic MNE Raw fixture for EEG pipeline tests.

The fixture is committed to the repo alongside this script so test runs are
reproducible without re-running the script. Re-run only if the contract changes.

Channels: 4 EEG (Cz, Pz, O1, O2) + 1 EOG (EOG061).
Sampling rate: 256 Hz. Duration: 10 s.
Synthetic content: a 10 Hz alpha sine on each EEG channel, plus a 1.5 Hz EOG
"blink" injected on EOG061 and bleed-through on the frontal-most EEG channel
(Cz) so ICA has something to detect.

NOTE: byte-determinism of the .fif output is coupled to ``mne==1.7.1`` (pinned
in requirements.txt). If that pin is upgraded, re-run this script and commit
the rebuilt artifact alongside the dependency bump.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np


def build() -> Path:
    rng = np.random.default_rng(seed=42)
    sfreq = 256.0
    duration_s = 10.0
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq

    # Base alpha (10 Hz) + small white noise on every EEG channel.
    eeg_alpha = np.sin(2 * np.pi * 10.0 * t)
    eeg_noise = rng.standard_normal((4, n_samples)) * 1e-6
    eeg = (eeg_alpha[None, :] * 1e-5) + eeg_noise

    # EOG blink: low-frequency square-ish pulse train at ~1.5 Hz.
    eog_pulse = (np.sin(2 * np.pi * 1.5 * t) > 0.95).astype(float) * 1e-4

    # Bleed EOG into Cz (channel 0) so ICA finds an EOG-correlated component.
    eeg[0] += 0.3 * eog_pulse

    data = np.vstack([eeg, eog_pulse[None, :]])  # shape: (5, n_samples)

    info = mne.create_info(
        ch_names=["Cz", "Pz", "O1", "O2", "EOG061"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "eeg", "eeg", "eog"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    out = Path(__file__).parent / "eeg_sample.fif"
    raw.save(out, overwrite=True, verbose="ERROR")
    return out


if __name__ == "__main__":
    p = build()
    print(f"Wrote {p}")
