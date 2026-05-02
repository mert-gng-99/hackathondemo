#!/bin/sh
set -eu

mkdir -p data/raw data/processed data/knowledge_base/seed

# Seed raw fixtures so the Signal/Image/Molecule tabs work on first click.
if [ -f tests/fixtures/bbbp_sample.csv ] && [ ! -f data/raw/bbbp.csv ]; then
  cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv
fi

if [ -f tests/fixtures/eeg_sample.fif ] && [ ! -f data/raw/eeg.fif ]; then
  cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif
fi

if [ -d tests/fixtures/kb_sample ] && [ ! -f data/knowledge_base/seed/lipinski_rule_of_five.md ]; then
  cp tests/fixtures/kb_sample/* data/knowledge_base/seed/
fi

# Demo-time stub artifacts (MRI 2D / volumetric ONNX / EEG joblib / clinical
# RAG / axial PNG). Idempotent — only fills missing.
python scripts/seed_demo_artifacts.py || true

# BBB feature extraction + classifier training (idempotent).
if [ ! -f data/processed/bbbp_features.parquet ]; then
  NEUROBRIDGE_DISABLE_MLFLOW=1 python -m src.pipelines.bbb_pipeline || true
fi

if [ ! -f data/processed/bbb_model.joblib ]; then
  NEUROBRIDGE_DISABLE_MLFLOW=1 python -m src.models.bbb_model || true
fi

# EEG feature extraction (idempotent).
if [ ! -f data/processed/eeg_features.parquet ]; then
  NEUROBRIDGE_DISABLE_MLFLOW=1 python -c "
from pathlib import Path
from src.pipelines.eeg_pipeline import run_pipeline
run_pipeline(
    input_path=Path('tests/fixtures/eeg_sample.fif'),
    output_path=Path('data/processed/eeg_features.parquet'),
)
" || true
fi

# MRI feature extraction + ComBat harmonization (idempotent).
if [ ! -f data/processed/mri_features.parquet ]; then
  NEUROBRIDGE_DISABLE_MLFLOW=1 python -c "
from pathlib import Path
from src.pipelines.mri_pipeline import run_pipeline
run_pipeline(
    input_dir=Path('tests/fixtures/mri_sample'),
    sites_csv=Path('tests/fixtures/mri_sample/sites.csv'),
    output_path=Path('data/processed/mri_features.parquet'),
)
" || true
fi

# RAG FAISS index build (idempotent).
if [ ! -f data/processed/faiss_index/index.bin ]; then
  python -m src.rag.ingest data/knowledge_base data/processed/faiss_index || true
fi

exec "$@"
