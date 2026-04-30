# AGENTS.md — NeuroBridge Enterprise Pipeline

> Read this file at the start of every session. It is the contract every agent
> (human or LLM) operates under in this repository.

## 1. Project Vision

**NeuroBridge Enterprise** is a B2B SaaS platform that solves three structural
problems in real-world clinical/biomedical ML pipelines:

1. **Data Drift** between hospitals and acquisition sites (multi-center MRI).
2. **Missing Modalities** (a patient may have MRI but no EEG, or vice versa).
3. **Artifacts** in raw biosignals (eye blinks, line noise, motion in EEG).

The platform exposes three production pipelines behind a single FastAPI surface:

| Modality | Pipeline | Core Technique |
|---|---|---|
| Image (MRI / fMRI) | `src/pipelines/mri_pipeline.py` | ComBat Harmonization for site-level domain shift |
| Signal (EEG) | `src/pipelines/eeg_pipeline.py` | MNE-Python + ICA for artifact removal |
| Tabular (BBB / molecules) | `src/pipelines/bbb_pipeline.py` | RDKit Morgan fingerprints from SMILES |

All experiment runs are tracked in **MLflow**. All services ship as **Docker** images.

## 2. Directory Layout (load-bearing — do not violate)

```
.
├── AGENTS.md                 # This file
├── README.md
├── requirements.txt
├── pytest.ini
├── conftest.py               # Repo-wide pytest fixtures (autouse: pins MLFLOW_TRACKING_URI to tmp dir for test isolation)
├── Dockerfile                # Production image (FastAPI + pipelines)
├── docker-compose.yml        # api + mlflow services for local stack
├── .dockerignore
├── .streamlit/
│   └── config.toml           # Streamlit theme tokens
├── data/
│   ├── raw/                  # Untouched source data. NEVER train on this directly.
│   └── processed/            # Pipeline output as Parquet (preserves dtypes; overwritten each run; see §4).
├── src/
│   ├── api/                  # FastAPI surface
│   │   ├── main.py           # App factory + /health
│   │   ├── routes.py         # POST /pipeline/{bbb,eeg,mri} dispatch
│   │   └── schemas.py        # Shared Pydantic request/response models
│   ├── core/                 # Cross-cutting utilities
│   │   ├── logger.py         # Structured logger (mandatory in every pipeline)
│   │   ├── determinism.py    # Thread-pin env vars (OMP/OPENBLAS/MKL/pyarrow)
│   │   ├── storage.py        # Parquet read/write helpers (snappy, single-threaded, deterministic)
│   │   └── tracking.py       # MLflow `track_pipeline_run` context manager (see §7)
│   ├── pipelines/            # One file per modality. Pure functions + a `run_pipeline()` entry.
│   ├── models/               # Downstream decision-layer models (consume processed features)
│   │   └── bbb_model.py      # BBB-permeability classifier + SHAP explainer + trainer CLI
│   └── frontend/
│       └── app.py            # Streamlit dashboard (3 tabs, one per modality)
└── tests/
    ├── core/
    ├── api/
    ├── frontend/
    ├── pipelines/            # incl. test_cross_pipeline_smoke.py for integration coverage
    └── fixtures/             # Tiny synthetic data files used by tests (NOT a Python package — no __init__.py)
```

**Rules:**
- New modality → new file under `src/pipelines/`. No mixing modalities in one file.
- Anything imported by 2+ pipelines → `src/core/`.
- Pipeline code (`src/pipelines/`, `src/core/`) must not read from or write to any path outside `data/`. Test code may read `tests/fixtures/`. The `data/` boundary is the storage contract for production data.
- `tests/fixtures/` holds CSV / numpy / DICOM blobs — do **not** add an `__init__.py` there.

## 3. Coding Standards

- **Python 3.10–3.12** (the pinned native-extension dependencies do not yet ship cp313+ wheels). Use `from __future__ import annotations` when needed for forward refs.
- **Type hints are mandatory** on every public function/method (parameters and return).
- **Modular structure.** One responsibility per function. If a function exceeds ~40 lines or 3 levels of nesting, split it.
- **TDD is the default workflow.** Write the failing test first, watch it fail, then implement. Tests live in `tests/` mirroring `src/`.
- **Logging is mandatory** for every pipeline. Use `src.core.logger.get_logger(__name__)`. No `print()` in `src/`.
- **Docstrings** on every public function — one-line summary + Args/Returns when non-trivial.
- **No hard-coded paths in business logic.** Pass paths as arguments to `run_pipeline(input_path, output_path)`.
- **Format & lint:** keep imports sorted; prefer `pathlib.Path` over `os.path`.
- **Commits are small and frequent.** Each green test → commit.

## 4. Data Readiness Principles

> **The Golden Rule: never train a model directly on raw data. Raw data must always pass through a pipeline first.**

Every modality pipeline MUST guarantee, before writing to `data/processed/`:

1. **Schema validity** — required columns present, expected dtypes.
2. **Domain validity** — invalid records (e.g. unparseable SMILES, NaN-only EEG epochs, corrupted DICOMs) are **logged with their identifier and dropped**, never silently coerced.
3. **Determinism** — given the same `data/raw/` input, the pipeline produces byte-identical `data/processed/` output. No wall-clock, no random seeds without explicit seeding.
4. **Traceability** — log row count in, row count out, and percentage dropped at INFO level.
5. **Idempotence** — re-running the pipeline overwrites `data/processed/` cleanly; no append, no partial writes.

**Determinism environment**: byte-identical output requires deterministic
floating-point reductions. Each pipeline module sets `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and pins pyarrow to
single-threaded mode at import time. CI runners and developer machines do
not need to set these manually — the pipeline modules handle it — but
overriding them in the environment will break Determinism rule 3.

**ComBat determinism boundary**: the MRI pipeline's `harmonize_combat` wraps
`neuroHarmonize.harmonizationLearn` and applies `np.round(14)` to its output.
This is a defensive measure: with the thread-pinning above, harmonization is
already bit-identical, but the rounding guarantees byte-identity even when
the env-pin discipline is bypassed (e.g. a sub-process that re-exports a
thread count). It discards ~5 trailing-mantissa bits of float64 — well below
ComBat's biological effect-size precision floor.

A model training script is allowed to import from `data/processed/` only. If a
training script references `data/raw/` directly, that is a bug and must be
refactored into a pipeline.

## 5. How to Add a New Pipeline (checklist)

1. Add `tests/pipelines/test_<name>_pipeline.py` with the failing tests first.
2. Create `src/pipelines/<name>_pipeline.py` exposing `run_pipeline(input_path: Path, output_path: Path) -> None`.
3. Use `get_logger(__name__)` for all status output (per §3).
4. Apply the §4 Data Readiness contract: validate + drop invalid records with a logged WARNING (identifier + count), log row count in/out/dropped at INFO, write deterministically, and overwrite (do not append) on re-run.
5. Write deterministic output to `output_path`.
6. Document any new dependency in `requirements.txt` (pinned).
7. Add a one-line entry to this file's pipeline table.

## 6. Storage Format Convention

All `data/processed/` outputs MUST be **Parquet** (`pyarrow` engine, `compression="snappy"`):
- Preserves dtypes (uint8 fingerprints stay uint8; float64 EEG features stay float64) — CSV silently widens numeric columns and is unsuitable for the high-dimensional float arrays produced by the EEG and MRI pipelines.
- Byte-deterministic with fixed compression and single-threaded writes (satisfies §4 Determinism).
- Read with `pd.read_parquet(path)`; no dtype hints required.

The raw `data/raw/` inputs may be in any vendor-supplied format (CSV for BBBP, EDF/FIF for EEG, NIfTI for MRI).

## 7. Experiment Tracking

Every `run_pipeline()` invocation logs to MLflow via `src.core.tracking.track_pipeline_run`:

- **Experiment names** match the pipeline module: `bbb_pipeline`, `eeg_pipeline`, `mri_pipeline`.
- **Params**: input/output paths and pipeline hyperparameters (e.g. BBB `n_bits` / `radius`, EEG `epoch_duration_s` / `random_state`, MRI `intensity_threshold` / `n_roi_axes`).
- **Metrics**: row counts (`rows_in`, `rows_out`, `rows_dropped` — or modality equivalent like `subjects_in/out/dropped`) and `duration_sec`.
- **Artifact**: the produced Parquet at `data/processed/<modality>_features.parquet`.

The tracking URI is read from `MLFLOW_TRACKING_URI` (defaults to `./mlruns/` when unset).

**Live-demo lifeline**: set `NEUROBRIDGE_DISABLE_MLFLOW=1` to skip tracking entirely — the helper yields `None` and emits no MLflow calls. Use this when the tracking server is unreachable (offline demo, network outage, or CI without an MLflow service). Pipelines complete normally; only the run metadata is lost.

The repo-wide `conftest.py` autouse fixture pins `MLFLOW_TRACKING_URI` to a tmp directory for the test session, so the production `mlruns/` directory is never written by the test suite. Tests that interact with MLflow (in `tests/core/test_tracking.py` and the per-pipeline `Test<Modality>PipelineMLflow` classes) all share this isolated store.

## 8. Decision Layer (Downstream Models)

Pipelines produce features (`data/processed/<modality>_features.parquet`).
Downstream models live in `src/models/` and consume those features:

| Model | File | Output | Endpoint |
|---|---|---|---|
| BBB permeability | `src/models/bbb_model.py` | `data/processed/bbb_model.joblib` | `POST /predict/bbb` |

Each downstream model module exposes a uniform surface:
- `train(df, label_col, ...)` → fitted classifier
- `save(model, path)` / `load(path)` → joblib artifact I/O
- `predict_with_proba(model, smiles)` → `{label, confidence}` (confidence is the max-class probability)
- `explain_prediction(model, smiles, top_k)` → SHAP top-k attributions sorted by `|shap_value|` descending

The API loads the joblib artifact at request time. If the artifact is
missing, the endpoint returns **HTTP 503** with a remediation hint pointing
at the trainer CLI (`python -m src.models.<name>`). This keeps the API
process startup fast and lets operators retrain without redeploying — the
Day-5 analog of Day-4's `NEUROBRIDGE_DISABLE_MLFLOW` lifeline.

**Determinism**: all classifiers are seeded (`random_state=42` default),
`n_jobs=1` (no tree-parallelism races). Re-running the trainer on the same
Parquet produces identical predictions.

**Override `BBB_MODEL_PATH`** env var to point the API at a non-default
artifact location (used by tests for tmp_path isolation).
