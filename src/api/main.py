"""NeuroBridge FastAPI entrypoint.

Exposes /health for liveness. Pipeline routes are mounted in Task 8.
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import HealthResponse

app = FastAPI(
    title="NeuroBridge Enterprise",
    description="Three-modality clinical-ML pipeline surface (BBB / EEG / MRI).",
    version="0.4.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — used by docker-compose health checks and Streamlit."""
    return HealthResponse(status="ok", pipelines=["bbb", "eeg", "mri"])
