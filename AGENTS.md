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
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/                  # Untouched source data. NEVER train on this directly.
│   └── processed/            # Pipeline output. Model-ready. Versioned outputs.
├── src/
│   ├── api/                  # FastAPI routers, request/response schemas
│   ├── pipelines/            # One file per modality. Pure functions + a `run_pipeline()` entry.
│   └── core/                 # Cross-cutting utilities: logging, config, MLflow helpers
└── tests/
    ├── core/
    ├── pipelines/
    └── fixtures/             # Tiny synthetic data files used by tests (NOT a Python package — no __init__.py)
```

**Rules:**
- New modality → new file under `src/pipelines/`. No mixing modalities in one file.
- Anything imported by 2+ pipelines → `src/core/`.
- Never read from or write to paths outside `data/`. The `data/` boundary is the storage contract.
- `tests/fixtures/` holds CSV / numpy / DICOM blobs — do **not** add an `__init__.py` there.

## 3. Coding Standards

- **Python 3.10+.** Use `from __future__ import annotations` when needed for forward refs.
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

A model training script is allowed to import from `data/processed/` only. If a
training script references `data/raw/` directly, that is a bug and must be
refactored into a pipeline.

## 5. How to Add a New Pipeline (checklist)

1. Add `tests/pipelines/test_<name>_pipeline.py` with the failing tests first.
2. Create `src/pipelines/<name>_pipeline.py` exposing `run_pipeline(input_path: Path, output_path: Path) -> None`.
3. Use `get_logger(__name__)` for all status output.
4. Validate inputs and drop invalid rows with a logged warning.
5. Write deterministic output to `output_path`.
6. Document any new dependency in `requirements.txt` (pinned).
7. Add a one-line entry to this file's pipeline table.
