#!/bin/sh
set -eu

mkdir -p data/raw data/processed data/knowledge_base/seed

if [ -f tests/fixtures/bbbp_sample.csv ] && [ ! -f data/raw/bbbp.csv ]; then
  cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv
fi

if [ -f tests/fixtures/eeg_sample.fif ] && [ ! -f data/raw/eeg.fif ]; then
  cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif
fi

if [ -d tests/fixtures/kb_sample ] && [ ! -f data/knowledge_base/seed/lipinski_rule_of_five.md ]; then
  cp tests/fixtures/kb_sample/* data/knowledge_base/seed/
fi

if [ ! -f data/processed/bbbp_features.parquet ]; then
  NEUROBRIDGE_DISABLE_MLFLOW=1 python -m src.pipelines.bbb_pipeline
fi

if [ ! -f data/processed/bbb_model.joblib ]; then
  python -m src.models.bbb_model
fi

if [ ! -f data/processed/faiss_index/index.bin ]; then
  python -m src.rag.ingest data/knowledge_base data/processed/faiss_index
fi

exec "$@"
