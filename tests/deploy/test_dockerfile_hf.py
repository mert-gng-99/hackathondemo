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
        - Build the BBB model artifact at build time
        - Set NEUROBRIDGE_DISABLE_MLFLOW=1 by default
        - Expose port 7860 (HF Spaces convention)
        - Launch via supervisord
        """
        text = dockerfile_text.lower()
        assert "from python" in text, "must FROM a Python base image"
        assert "requirements.txt" in text, "must reference requirements.txt"
        assert "src.models.bbb_model" in dockerfile_text, (
            "must build the BBB model artifact at image-build time"
        )
        assert "neurobridge_disable_mlflow" in text, (
            "must set NEUROBRIDGE_DISABLE_MLFLOW for HF deploy"
        )
        assert "7860" in text, "must expose port 7860 (HF Spaces convention)"
        assert "supervisord" in text, (
            "must launch FastAPI + Streamlit via supervisord"
        )
