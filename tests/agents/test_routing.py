"""Tests for src.agents.routing — deterministic workflow-guard fallbacks."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.routing import route_pipeline_input


class TestRoutePipelineInput:
    def test_smiles_routes_to_bbb(self) -> None:
        assert route_pipeline_input("CCO") == (
            "run_bbb_pipeline",
            {"smiles": "CCO", "top_k": 5},
        )

    def test_eeg_path_routes_to_eeg(self) -> None:
        name, args = route_pipeline_input("data/raw/sample.fif")
        assert name == "run_eeg_pipeline"
        assert args == {"input_path": "data/raw/sample.fif"}

    def test_nifti_path_routes_to_mri_with_parent_dir(self) -> None:
        name, args = route_pipeline_input("data/raw/subjects/subject_0.nii.gz")
        assert name == "run_mri_pipeline"
        assert args["input_dir"] == "data/raw/subjects"
        assert "sites_csv" in args

    def test_existing_local_dir_without_slash_routes_to_mri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "subject_dir").mkdir()
        name, args = route_pipeline_input("subject_dir")
        assert name == "run_mri_pipeline"
        assert args["input_dir"] == "subject_dir"

    def test_bare_string_with_no_matching_dir_still_routes_to_bbb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Nothing on disk named "Aspirin" — should be treated as a SMILES-like token
        name, args = route_pipeline_input("Aspirin")
        assert name == "run_bbb_pipeline"
        assert args == {"smiles": "Aspirin", "top_k": 5}
