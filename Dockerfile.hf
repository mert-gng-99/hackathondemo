# NeuroBridge Enterprise — Hugging Face Spaces deployment image
# Single container running FastAPI (port 8000) + Streamlit (port 7860).
# HF Spaces routes :7860 to the public URL automatically.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPLOY_ENV=hf_spaces

# --- system deps for RDKit, nibabel, MNE ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
# Install CPU-only torch first to avoid pulling ~2GB of NVIDIA CUDA wheels
# (cublas/cudnn/nccl/...) that we never use on a CPU-only HF Space and which
# blow past the build-time disk budget. Subsequent pip install -r sees torch
# already at the pinned version and skips it.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 \
    && pip install -r requirements.txt

# --- project source ---
COPY src/ ./src/
COPY tests/fixtures/ ./tests/fixtures/
COPY scripts/ ./scripts/
COPY supervisord.conf ./supervisord.conf
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Seed demo artifacts FIRST so even if a heavier pipeline step fails, the
# core showcase paths (MRI 2D, MRI volumetric ONNX, EEG joblib, clinical
# RAG, axial PNG) still work. seed_demo_artifacts.py is idempotent.
RUN python scripts/seed_demo_artifacts.py

# Seed raw data from fixtures so the deployed Signal/Image/Molecule tabs
# work on first click. Then run all three pipelines so mlruns/ contains
# one run per modality — feeds /experiments/runs and the BBB provenance
# strip. data/raw/* is gitignored locally so we cannot COPY it.
#
# NEUROBRIDGE_DISABLE_MLFLOW=1 during build avoids MLflow run-tagging
# fragility in the slim image (no real .git tree to tag against). The
# entrypoint can re-run with MLflow on if desired.
RUN mkdir -p data/raw data/processed && \
    cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv && \
    cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif && \
    NEUROBRIDGE_DISABLE_MLFLOW=1 python -m src.pipelines.bbb_pipeline && \
    NEUROBRIDGE_DISABLE_MLFLOW=1 python -m src.models.bbb_model && \
    NEUROBRIDGE_DISABLE_MLFLOW=1 python -c "from pathlib import Path; from src.pipelines.eeg_pipeline import run_pipeline; run_pipeline(input_path=Path('tests/fixtures/eeg_sample.fif'), output_path=Path('data/processed/eeg_features.parquet'))" && \
    NEUROBRIDGE_DISABLE_MLFLOW=1 python -c "from pathlib import Path; from src.pipelines.mri_pipeline import run_pipeline; run_pipeline(input_dir=Path('tests/fixtures/mri_sample'), sites_csv=Path('tests/fixtures/mri_sample/sites.csv'), output_path=Path('data/processed/mri_features.parquet'))"

# --- RAG knowledge base ingest ---
# Build the FAISS index from any seed docs in tests/fixtures/kb_sample/
# (always present) plus data/knowledge_base/ (optional, user-supplied via
# additional COPY layer or volume mount). Empty KB → empty index, agent
# still functions, retrieve_context just returns no chunks.
COPY tests/fixtures/kb_sample/ ./data/knowledge_base/seed/
RUN python -m src.rag.ingest data/knowledge_base data/processed/faiss_index

# --- Re-run demo-artifact seeding after RAG ingest in case any step above
#     altered what's on disk. Idempotent — only fills missing artifacts.
RUN python scripts/seed_demo_artifacts.py

# --- HF Spaces convention ---
EXPOSE 7860

# --- launch FastAPI + Streamlit under supervisord ---
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
