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

**Calibration metadata** (Day 6): `train()` does an 80/20 stratified split,
computes precision-at-confidence-threshold bins on the held-out test set,
and stashes them on `model._neurobridge_calibration: list[dict]` (sorted
ascending by threshold). The API includes the bin matching each
prediction's confidence in `BBBPredictResponse.calibration`. UI uses this
to render an honest trust caption ("≥75% confident → 92% precision, n=18").
For tiny test fixtures where stratified split fails, calibration falls
back to zero-support bins so the API contract is always populated.

## 9. Demo Features (Day 6)

The frontend includes three jury-day demo amplifiers that don't change
the core contract:

- **Edge-case dropdown** (BBB tab): a curated catalog of 5 robustness
  probes — invalid SMILES, empty input, OOD macrocycle (cyclosporine-like),
  heavy halogenated aromatic. Each has a stated expectation; the UI
  visualizes graceful failure (HTTP 400 → recoverable warning, never
  a crash).
- **Calibration trust caption** (BBB decision card): renders the
  precision-at-confidence-threshold from `BBBPredictResponse.calibration`.
  Demonstrates that the system knows what it doesn't know.
- **MRI ComBat diagnostics** (MRI tab): `POST /pipeline/mri/diagnostics`
  runs the pipeline twice (pre + post ComBat) and returns long-format
  data + site-gap KPIs (Pre, Post, Reduction factor). The UI renders
  a faceted altair density plot — visual proof that ComBat removes
  site-driven domain shift.

## 10. Drift Surface (Day 7)

Each predict route maintains a per-worker rolling window of recent
prediction confidences (`collections.deque(maxlen=100)`). Train-time
median + std are stashed on `model._neurobridge_train_stats` (joblib
roundtrip-safe). The drift z-score is `(rolling_median − train_median) /
max(train_std, 1e-9)`, computed only when the buffer holds ≥10 samples
AND the model has the train-stats attribute. The `/predict/bbb`
response carries `drift_z: float | None` and `rolling_n: int`. The UI
renders a one-line caption with a magnitude tag (in-band, mild,
significant). Worker restart clears the deque; this is acceptable for
demo and removes the audit-trail concern.

## 11. LLM Explainer Surface (Day 7 + 9)

`src/llm/explainer.py` is the single entry point for natural-language
rationales. `explain(payload)` always returns `{rationale, source,
model}`. The deterministic template path is the source of truth for
tests; the LLM path is OpenRouter via the `openai==1.51.0` SDK and
walks a **smartest → smallest free-tier fallback chain**
(`_DEFAULT_FREE_MODEL_CHAIN`, 10 ids — head: `inclusionai/ling-2.6-1t:free`).
The chain is overridable at runtime via `OPENROUTER_FREE_MODELS`
(comma-separated). Status-code classification:

- `401` → key is bad → bail to template + actionable WARNING (rotate at
  https://openrouter.ai/keys, enable free-model data-sharing at
  https://openrouter.ai/settings/privacy).
- `400` → prompt-shape mismatch on this model → advance to next.
- `402 / 403 / 404 / 429 / 5xx` → advance to next.
- Network/timeout → bail to template (switching models won't help).

Two env knobs control the gate:

- `OPENROUTER_API_KEY` — when absent, fallback to template.
- `NEUROBRIDGE_DISABLE_LLM=1` — hard kill-switch; force template even
  if a key is set. Use this for demo days when you want fully
  deterministic, reproducible rationales.

**Prompt design** (`_build_llm_prompt`): two intent modes. When the
caller supplies `user_question`, the model is instructed to
language-match (Turkish question → Turkish answer), answer the
question directly (not a canned paper-style summary), and respond
conversationally to off-topic / greeting questions. When no
`user_question` is supplied, falls back to the original 2-4 sentence
paper-style rationale.

The `POST /explain/bbb` endpoint mirrors this contract. Pydantic
enforces a non-empty `top_features` list (422 on empty); every other
failure mode degrades to template + WARNING log + `source="template"`.

**Diagnostics**: `GET /diag/openrouter` (`src/api/main.py`) returns
key-presence (length + 12-char prefix only), kill-switch state, chain
length, first model id, and the result of an 8-token probe call
against that model. Surfaced in Streamlit as the sidebar "🔧 Diagnose
LLM" button. Use it when the deployed Space shows `source="template"`
unexpectedly — the most common causes are a missing/misnamed
`OPENROUTER_API_KEY` Space secret or a revoked key.

## 12. Multi-Modal Explainer (Day 8)

`src/llm/explainer.py` exposes `explain(payload, modality)` where
`modality ∈ {"bbb", "eeg", "mri"}`. Each modality has its own
deterministic template (`_template_explain_bbb / _eeg / _mri`) and
its own LLM prompt header. Unknown modality strings degrade to the
BBB template with a warning log; the function never raises. The
hybrid OpenRouter fallback contract from §11 applies uniformly.

The API exposes three matching endpoints — `POST /explain/{bbb,eeg,mri}` —
each on the `explain_router` (`/explain` prefix). Streamlit surfaces
the BBB version in the AI Assistant tab and the EEG/MRI versions as
inline expanders inside their respective pipeline tabs.

## 13. Experiments Surface (Day 8)

`GET /experiments/runs` returns up to 50 most recent MLflow runs
across the bbb/eeg/mri experiments, flattened into a list of
`MLflowRunSummary` (run_id, experiment_name, start_time, status,
metrics, params). `POST /experiments/diff {run_id_a, run_id_b}`
returns a side-by-side metric+param diff (`RunDiffRow`).

When `NEUROBRIDGE_DISABLE_MLFLOW=1`, both endpoints return empty
responses without raising — required for the HF Spaces deployment
where there is no writable mlruns/ tree. Unknown run ids → 404.

The Streamlit "Experiments" tab is the user-facing surface. Cached
in session state with an explicit Refresh button.

## 14. Deploy Surface (Day 8)

`Dockerfile.hf` is the Hugging Face Spaces image. Single container,
two processes (FastAPI :8000 + Streamlit :7860) launched via
`supervisord.conf`. Build-time `RUN python -m src.models.bbb_model`
bakes the model artifact into the image so the first `/predict/bbb`
call is instant on cold start.

Default environment: `DEPLOY_ENV=hf_spaces`,
`NEUROBRIDGE_DISABLE_MLFLOW=1`. The LLM kill-switch is **not** set —
deployed Spaces use the real OpenRouter free-tier chain (§11) when
`OPENROUTER_API_KEY` is configured in the Space's Secrets panel. Set
`NEUROBRIDGE_DISABLE_LLM=1` only when you want to force the
deterministic template path for a fully-reproducible demo.

The README's YAML front-matter declares the Space metadata
(SDK=docker, port=7860, app_file=src/frontend/app.py).

## 15. Orchestrator Agent Surface

`src/agents/orchestrator.py` exposes a single-agent function-calling
loop over the openai SDK (no LangChain / framework dep). The agent
holds 4 tools, defined in `src/agents/tools.py`:

- `run_bbb_pipeline(smiles, top_k)` — wraps `POST /predict/bbb`
- `run_eeg_pipeline(input_path)` — wraps `POST /pipeline/eeg`
- `run_mri_pipeline(input_dir, sites_csv)` — wraps `POST /pipeline/mri`
- `retrieve_context(query, k)` — wraps `src/rag/retrieve.py`

The system prompt (`src/agents/prompts.py:ORCHESTRATOR_SYSTEM_PROMPT`)
locks the workflow: pick exactly one pipeline → run it → formulate a
focused retrieval query → call retrieve_context → synthesize a
3-5 sentence response that cites at least one chunk. Language of the
final response is mirrored from the user's question.

`POST /agent/run` is the public surface. Default model is
`google/gemini-2.0-flash-exp:free` on OpenRouter (function-calling
support verified). Override via `NEUROBRIDGE_AGENT_MODEL` env var.
Returns 503 when `OPENROUTER_API_KEY` is unset.

Diagnostics: `GET /diag/agent` returns key presence, configured model,
RAG index status (chunk count), and the registered tool names.

## 16. RAG Surface

`src/rag/` is the retrieval layer. Stack: `fastembed`
(`BAAI/bge-small-en-v1.5`, 384-dim, ONNX, no torch dep) for
embeddings + `faiss-cpu` (`IndexFlatIP` after L2-norm = cosine) for
vector search.

Knowledge base lives at `data/knowledge_base/` (gitignored;
user-supplied `.md` / `.txt` / `.pdf`). Build the FAISS index with:

    python -m src.rag.ingest [<input_dir> [<output_dir>]]

Defaults: input=`data/knowledge_base/`, output=`data/processed/faiss_index/`.
The Dockerfile runs this at build time so deployed Spaces start with
a populated index. Empty KB → empty index → `retrieve_context`
returns 0 chunks; the agent surfaces this and answers from the
pipeline result alone.

`tests/fixtures/kb_sample/` ships 3 seed markdown files (Lipinski,
ComBat, MNE+ICA) — these double as test fixtures and as the demo
seed if no user-supplied PDFs are added.
