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
COPY supervisord.conf ./supervisord.conf
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Seed raw data from fixtures so the deployed Signal/Image/Molecule tabs
# work on first click. Then run all three pipelines so mlruns/ contains
# one run per modality — feeds /experiments/runs and the BBB provenance
# strip. data/raw/* is gitignored locally so we cannot COPY it.
RUN mkdir -p data/raw data/processed && \
    cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv && \
    cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif && \
    python -m src.pipelines.bbb_pipeline && \
    python -m src.models.bbb_model && \
    python -c "from pathlib import Path; from src.pipelines.eeg_pipeline import run_pipeline; run_pipeline(input_path=Path('tests/fixtures/eeg_sample.fif'), output_path=Path('data/processed/eeg_features.parquet'))" && \
    python -c "from pathlib import Path; from src.pipelines.mri_pipeline import run_pipeline; run_pipeline(input_dir=Path('tests/fixtures/mri_sample'), sites_csv=Path('tests/fixtures/mri_sample/sites.csv'), output_path=Path('data/processed/mri_features.parquet'))"

# --- RAG knowledge base ingest ---
# Build the FAISS index from any seed docs in tests/fixtures/kb_sample/
# (always present) plus data/knowledge_base/ (optional, user-supplied via
# additional COPY layer or volume mount). Empty KB → empty index, agent
# still functions, retrieve_context just returns no chunks.
COPY tests/fixtures/kb_sample/ ./data/knowledge_base/seed/
RUN python -m src.rag.ingest data/knowledge_base data/processed/faiss_index

# --- HF Spaces convention ---
EXPOSE 7860

# --- launch FastAPI + Streamlit under supervisord ---
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
