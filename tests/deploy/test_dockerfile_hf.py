"""Smoke test: Dockerfile.hf is well-formed and contains expected stages.

We don't actually build the image (too slow for unit tests). We just verify
the file exists, is non-empty, and has the load-bearing instructions.
"""
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile.hf"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    if not DOCKERFILE.exists():
        pytest.skip(f"{DOCKERFILE} does not exist yet (Day-8 T3 RED phase)")
    return DOCKERFILE.read_text()


class TestDockerfileHF:
    """Day-8 T3: Hugging Face Spaces Dockerfile smoke."""

    def test_dockerfile_exists_and_nonempty(self):
        assert DOCKERFILE.exists(), f"missing {DOCKERFILE}"
        assert DOCKERFILE.stat().st_size > 0, f"{DOCKERFILE} is empty"

    def test_dockerfile_contains_required_stages(self, dockerfile_text):
        """The HF Dockerfile must:
        - Start FROM a Python base
        - Install requirements.txt
        - Seed data/raw/bbbp.csv AND data/raw/eeg.fif from fixtures
        - Build the BBB model artifact at build time
        - Run all three pipelines (BBB / EEG / MRI) so mlruns/ has one
          run per modality available to /experiments/runs at startup
        - Expose port 7860 (HF Spaces convention)
        - Launch via supervisord
        """
        text = dockerfile_text.lower()
        assert "from python" in text, "must FROM a Python base image"
        assert "requirements.txt" in text, "must reference requirements.txt"
        assert "src.models.bbb_model" in dockerfile_text, (
            "must build the BBB model artifact at image-build time"
        )
        assert "src.pipelines.bbb_pipeline" in dockerfile_text, (
            "must run BBB pipeline at build so mlruns/ has a BBB run"
        )
        assert "src.pipelines.eeg_pipeline" in dockerfile_text, (
            "must run EEG pipeline at build so mlruns/ has an EEG run"
        )
        assert "src.pipelines.mri_pipeline" in dockerfile_text, (
            "must run MRI pipeline at build so mlruns/ has an MRI run"
        )
        assert "tests/fixtures/eeg_sample.fif" in dockerfile_text, (
            "must seed data/raw/eeg.fif from the bundled fixture so the "
            "Signal tab works without user file upload"
        )
        assert "7860" in text, "must expose port 7860 (HF Spaces convention)"
        assert "supervisord" in text, (
            "must launch FastAPI + Streamlit via supervisord"
        )

    def test_dockerfile_does_not_disable_mlflow(self, dockerfile_text):
        """The MLflow kill-switch must be absent from the Dockerfile —
        file-store mlruns/ is built into the image and is safe to expose
        on the read-only demo. Re-introducing the kill-switch would
        silently kill the Experiments tab and the BBB provenance strip.

        Scan ENV lines only, not comment lines, so a future docstring or
        comment that cites the env name does not false-positive.
        """
        import re

        env_lines = [
            line
            for line in dockerfile_text.splitlines()
            if re.match(
                r"\s*(ENV\s+)?NEUROBRIDGE_DISABLE_MLFLOW\s*=\s*1",
                line,
                re.IGNORECASE,
            )
        ]
        assert not env_lines, (
            "Dockerfile must NOT set NEUROBRIDGE_DISABLE_MLFLOW=1 — that "
            "empties the Experiments tab and blanks the BBB provenance "
            "strip. If you need to disable MLflow at runtime, set the env "
            "manually on the Space, do not bake it into the image. "
            f"Offending lines: {env_lines}"
        )

    def test_dockerfile_byte_identical_to_hf_alias(self):
        """`Dockerfile` (HF auto-discovers this name) and `Dockerfile.hf`
        (canonical/readable name) must stay byte-identical. Drift between
        the two means HF builds a different image than the one the test
        suite verifies — silent deploy regression. Catch it in CI."""
        canonical = REPO_ROOT / "Dockerfile.hf"
        alias = REPO_ROOT / "Dockerfile"
        assert canonical.exists(), f"missing {canonical}"
        assert alias.exists(), f"missing {alias}"
        assert canonical.read_bytes() == alias.read_bytes(), (
            "Dockerfile and Dockerfile.hf have diverged. Re-sync them: "
            "`cp Dockerfile.hf Dockerfile` (or vice versa)."
        )
