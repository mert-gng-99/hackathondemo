# NeuroBridge Enterprise — Hugging Face Spaces deployment image
# Single container running FastAPI (port 8000) + Streamlit (port 7860).
# HF Spaces routes :7860 to the public URL automatically.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPLOY_ENV=hf_spaces \
    NEUROBRIDGE_DISABLE_MLFLOW=1 \
    NEUROBRIDGE_DISABLE_LLM=1

# --- system deps for RDKit, nibabel, MNE ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- project source ---
COPY src/ ./src/
COPY tests/fixtures/ ./tests/fixtures/
COPY data/raw/ ./data/raw/
COPY supervisord.conf ./supervisord.conf

# --- build BBB model artifact at image-build time ---
# This makes the first /predict/bbb call instant on cold start.
RUN python -m src.models.bbb_model

# --- HF Spaces convention ---
EXPOSE 7860

# --- launch FastAPI + Streamlit under supervisord ---
CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
