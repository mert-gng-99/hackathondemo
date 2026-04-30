"""Defaults shown in the Streamlit UI must point at files that actually exist
in the deployed image. The HF container does not seed `data/raw/eeg.fif`
unless we explicitly do so in the Dockerfile, so the EEG tab default must
either be a fixture file (always present) or be seeded server-side.
"""
from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PY = REPO_ROOT / "src" / "frontend" / "app.py"


def _eeg_default_path() -> str:
    """Extract the default value passed to the EEG `input_path` text_input.

    The line looks like:
        eeg_in = st.text_input("Input FIF/EDF path", "tests/fixtures/eeg_sample.fif", key="eeg_in")
    We pull the second positional arg (the default value).
    """
    text = APP_PY.read_text()
    match = re.search(
        r'st\.text_input\(\s*"Input FIF/EDF path"\s*,\s*"([^"]+)"',
        text,
    )
    assert match is not None, "could not locate EEG text_input default in app.py"
    return match.group(1)


def test_eeg_default_path_points_at_existing_file() -> None:
    default = _eeg_default_path()
    abs_path = REPO_ROOT / default
    assert abs_path.exists(), (
        f"EEG default path {default!r} resolves to {abs_path} which does "
        f"not exist on disk. The HF container does not seed this path "
        f"either, so users get an HTTP 404 the moment they click Run."
    )


def test_eeg_default_path_is_eeg_sample_fixture() -> None:
    """Pin the exact fix: default must be the canonical fixture so both
    local dev and the deployed image agree."""
    assert _eeg_default_path() == "tests/fixtures/eeg_sample.fif"
