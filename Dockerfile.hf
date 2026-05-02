# NeuroBridge Enterprise — Hugging Face Spaces deployment image
# Single container running FastAPI (port 8000) + Streamlit (port 7860).
# HF Spaces routes :7860 to the public URL automatically.
#
# Build philosophy: install deps + copy code + seed lightweight stub
# artifacts. Heavy pipeline runs (BBB train, EEG/MRI feature extraction,
# RAG ingest) live in docker-entrypoint.sh so they happen on first
# container start — the build can't fail because of pipeline logic, and
# the runtime is idempotent (no re-train if artifacts are present).

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

# --- Demo-time stub artifacts (MRI 2D / MRI volumetric ONNX / EEG joblib /
#     clinical TF-IDF RAG / axial PNG fixture). Idempotent script — also
#     re-run by the entrypoint at container start to fill any missing slots.
RUN python scripts/seed_demo_artifacts.py

# Seed kb_sample docs into the knowledge_base directory; entrypoint will
# build the FAISS index from these on first start.
COPY tests/fixtures/kb_sample/ ./data/knowledge_base/seed/

# --- HF Spaces convention ---
EXPOSE 7860

# --- launch FastAPI + Streamlit under supervisord ---
# docker-entrypoint.sh handles all the heavy lifting on first start:
# - copy raw fixtures into data/raw if missing
# - run BBB pipeline + train BBB classifier if artifacts missing
# - run EEG pipeline if features parquet missing
# - run MRI pipeline if features parquet missing
# - build FAISS index if missing
# - re-seed demo stub artifacts if missing
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
