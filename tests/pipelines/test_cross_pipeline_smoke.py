"""End-to-end smoke test exercising all three pipelines back-to-back.

Asserts each pipeline produces a non-empty Parquet at its expected schema —
the hackathon-judge "does the whole stack still work?" check. Each pipeline
uses its own fixture (no cross-modality data sharing).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def test_bbb_pipeline_smoke(tmp_path: Path) -> None:
    """Run the BBB pipeline on the committed CSV fixture; validate fp_ column count."""
    out = tmp_path / "bbb.parquet"
    bbb_pipeline.run_pipeline(
        input_path=_FIXTURES / "bbbp_sample.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    assert len(df) == 4
    assert sum(c.startswith("fp_") for c in df.columns) == 2048


def test_eeg_pipeline_smoke(tmp_path: Path) -> None:
    """Use the committed EEG fixture; build_eeg_fixture.build() takes no args."""
    fif = _FIXTURES / "eeg_sample.fif"
    if not fif.exists():
        pytest.skip(f"Committed EEG fixture missing: {fif}")
    out = tmp_path / "eeg.parquet"
    eeg_pipeline.run_pipeline(input_path=fif, output_path=out)
    df = pd.read_parquet(out)
    assert len(df) == 5
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert len(feat_cols) > 0


def test_mri_pipeline_smoke(tmp_path: Path) -> None:
    """Use the MRI fixture builder to materialize NIfTI inputs + sites.csv."""
    from tests.fixtures.build_mri_fixture import build as build_mri
    fixture_dir = build_mri(out_dir=tmp_path / "mri_fixture")
    out = tmp_path / "mri.parquet"
    mri_pipeline.run_pipeline(
        input_dir=fixture_dir,
        sites_csv=fixture_dir / "sites.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    assert len(df) == 6
    assert "subject_id" in df.columns
    assert "site" in df.columns


def test_all_three_pipelines_run_in_one_process(tmp_path: Path) -> None:
    """Sanity: nothing in pipeline A leaks state that breaks pipeline B."""
    test_bbb_pipeline_smoke(tmp_path / "bbb")
    test_eeg_pipeline_smoke(tmp_path / "eeg")
    test_mri_pipeline_smoke(tmp_path / "mri")
