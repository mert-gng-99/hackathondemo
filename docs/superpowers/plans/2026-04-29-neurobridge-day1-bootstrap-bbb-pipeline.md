# NeuroBridge Day 1 — Bootstrap & BBB Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insider One Hackathon Day 1 — bootstrap the NeuroBridge Enterprise repo (governance + dependencies) and ship the first working pipeline (BBBP / RDKit Morgan fingerprints) end-to-end with TDD.

**Architecture:** Modular `src/` layout with three sibling pipeline packages (image / signal / tabular). Day 1 lands the **tabular (BBB)** pipeline only. A shared `src/core/logger.py` standardizes structured logging across pipelines. RDKit is used for SMILES parsing and Morgan fingerprint generation; invalid SMILES are logged and dropped at the validation layer (Data Readiness gate). The pipeline reads `data/raw/bbbp.csv` and writes a model-ready `data/processed/bbbp_features.csv`.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, RDKit, MNE-Python, MLflow, Pytest, Docker.

---

## File Structure

Files created in this plan:

| Path | Responsibility |
|---|---|
| `AGENTS.md` | Agent-facing rulebook: vision, dir layout, coding standards, Data Readiness principles. |
| `requirements.txt` | Pinned Python deps for all 3 pipelines + API + tracking. |
| `.gitignore` | Standard Python + data/processed + MLflow artifacts ignore. |
| `pytest.ini` | Pytest config (rootdir, testpaths). |
| `src/__init__.py` | Mark `src` as a package. |
| `src/core/__init__.py` | Core/shared utilities package. |
| `src/core/logger.py` | `get_logger(name)` — structured stdout logger reused by all pipelines. |
| `src/pipelines/__init__.py` | Pipelines package. |
| `src/pipelines/bbb_pipeline.py` | BBBP SMILES → Morgan FP feature extractor + I/O orchestrator. |
| `src/api/__init__.py` | FastAPI package placeholder (filled later in week). |
| `tests/__init__.py` | Tests root. |
| `tests/core/__init__.py` | Core tests package. |
| `tests/core/test_logger.py` | Logger unit tests. |
| `tests/pipelines/__init__.py` | Pipeline tests package. |
| `tests/pipelines/test_bbb_pipeline.py` | BBB pipeline unit + integration tests. |
| `tests/fixtures/bbbp_sample.csv` | Tiny BBBP fixture (mix of valid + invalid SMILES). |
| `data/raw/.gitkeep` | Keep raw data folder under git, real CSVs ignored. |
| `data/processed/.gitkeep` | Keep processed folder under git. |

---

## Task 1: Project Skeleton & Git Bootstrap

**Files:**
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/pipelines/__init__.py`
- Create: `src/api/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/pipelines/__init__.py`
- Create: `tests/fixtures/` (folder)

- [ ] **Step 1: Create directory skeleton**

Run:
```bash
cd /Users/mertgungor/Desktop/hackathon
mkdir -p data/raw data/processed \
         src/core src/pipelines src/api \
         tests/core tests/pipelines tests/fixtures
```

- [ ] **Step 2: Create empty package markers**

Run:
```bash
touch src/__init__.py src/core/__init__.py src/pipelines/__init__.py src/api/__init__.py \
      tests/__init__.py tests/core/__init__.py tests/pipelines/__init__.py \
      data/raw/.gitkeep data/processed/.gitkeep
```

- [ ] **Step 3: Write `.gitignore`**

Create `.gitignore`:
```gitignore
# Byte-compiled / cache
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual envs
.venv/
venv/
env/

# Data — only keep folder structure, never raw payloads
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# MLflow / experiment tracking
mlruns/
mlartifacts/

# IDE
.idea/
.vscode/
.DS_Store
```

- [ ] **Step 4: Write `pytest.ini`**

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -v --tb=short
```

- [ ] **Step 5: Initialize git and commit skeleton**

Run:
```bash
cd /Users/mertgungor/Desktop/hackathon
git init -b main
git add .gitignore pytest.ini data/ src/ tests/
git commit -m "chore: bootstrap NeuroBridge project skeleton"
```

Expected: a single commit with the skeleton tree.

---

## Task 2: AGENTS.md — Project Rulebook

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write `AGENTS.md`**

Create `AGENTS.md`:
````markdown
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
    └── fixtures/             # Tiny synthetic data files used by tests
```

**Rules:**
- New modality → new file under `src/pipelines/`. No mixing modalities in one file.
- Anything imported by 2+ pipelines → `src/core/`.
- Never read from or write to paths outside `data/`. The `data/` boundary is the storage contract.

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
````

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add AGENTS.md with vision, layout, standards, data readiness rules"
```

---

## Task 3: requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write `requirements.txt`**

Create `requirements.txt`:
```text
# --- Web / API layer ---
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2

# --- Core data stack ---
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.1

# --- Modality: tabular / molecules (BBB pipeline) ---
rdkit==2024.3.5

# --- Modality: signal (EEG pipeline) ---
mne==1.7.1

# --- Modality: image (MRI pipeline) ---
nibabel==5.2.1
neuroharmonize==2.4.5  # ComBat harmonization wrapper

# --- Experiment tracking ---
mlflow==2.16.0

# --- Tooling / tests ---
pytest==8.3.3
pytest-cov==5.0.0
httpx==0.27.2  # FastAPI test client
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin runtime + dev dependencies for all three modalities"
```

> **Note for engineer:** dependency installation (creating a venv, `pip install -r requirements.txt`) is delegated to the human / CI. The plan does not assume a venv is active. Subsequent tasks rely on `rdkit`, `pytest`, etc. being importable; if the environment is not yet set up, set it up before Task 4.

---

## Task 4: Shared Logger (`src/core/logger.py`) — TDD

**Files:**
- Create: `tests/core/test_logger.py`
- Create: `src/core/logger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_logger.py`:
```python
"""Unit tests for the shared structured logger."""
from __future__ import annotations

import logging

from src.core.logger import get_logger


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger("neurobridge.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "neurobridge.test"


def test_get_logger_attaches_single_handler() -> None:
    """Repeated calls must not duplicate handlers (idempotence)."""
    name = "neurobridge.idempotent"
    first = get_logger(name)
    second = get_logger(name)
    assert first is second
    assert len(first.handlers) == 1


def test_get_logger_default_level_is_info() -> None:
    logger = get_logger("neurobridge.level_check")
    assert logger.level == logging.INFO


def test_get_logger_emits_formatted_record(caplog) -> None:
    logger = get_logger("neurobridge.emit")
    with caplog.at_level(logging.INFO, logger="neurobridge.emit"):
        logger.info("hello-world")
    assert any("hello-world" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_logger.py -v`

Expected: 4 FAILS with `ModuleNotFoundError: No module named 'src.core.logger'`.

- [ ] **Step 3: Implement the logger**

Create `src/core/logger.py`:
```python
"""Shared structured logger for NeuroBridge pipelines.

All modules in `src/` must obtain their logger via `get_logger(__name__)`
instead of using `print()`. This guarantees consistent format and INFO-level
traceability across pipelines (per AGENTS.md §4).
"""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide singleton logger for the given name.

    Idempotent: repeated calls with the same name return the same Logger
    instance and never stack duplicate handlers.

    Args:
        name: Dotted logger name, conventionally `__name__`.
        level: Logging level (default `logging.INFO`).

    Returns:
        Configured `logging.Logger` writing to stdout.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_logger.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/logger.py tests/core/test_logger.py
git commit -m "feat(core): add shared structured logger with idempotent handler attach"
```

---

## Task 5: BBB Pipeline — Test Fixture & SMILES Validation (TDD)

**Files:**
- Create: `tests/fixtures/bbbp_sample.csv`
- Create: `tests/pipelines/test_bbb_pipeline.py`
- Create: `src/pipelines/bbb_pipeline.py`

- [ ] **Step 1: Create the test fixture CSV**

Create `tests/fixtures/bbbp_sample.csv` (matches Kaggle BBBP schema: `num,name,p_np,smiles`):
```csv
num,name,p_np,smiles
1,Propanol,1,CCCO
2,Benzene,1,c1ccccc1
3,Aspirin,1,CC(=O)OC1=CC=CC=C1C(=O)O
4,InvalidMol,0,this_is_not_a_smiles
5,Caffeine,1,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
6,EmptyMol,0,
```

Two rows are invalid by design: row 4 (garbage string) and row 6 (empty). Both must be filtered out by the pipeline.

- [ ] **Step 2: Write the failing test for `is_valid_smiles`**

Create `tests/pipelines/test_bbb_pipeline.py`:
```python
"""Unit + integration tests for the BBB (SMILES → Morgan FP) pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.bbb_pipeline import is_valid_smiles


FIXTURE = Path(__file__).parent.parent / "fixtures" / "bbbp_sample.csv"


class TestIsValidSmiles:
    def test_accepts_simple_alcohol(self) -> None:
        assert is_valid_smiles("CCCO") is True

    def test_accepts_aromatic_ring(self) -> None:
        assert is_valid_smiles("c1ccccc1") is True

    def test_rejects_garbage_string(self) -> None:
        assert is_valid_smiles("this_is_not_a_smiles") is False

    def test_rejects_empty_string(self) -> None:
        assert is_valid_smiles("") is False

    def test_rejects_none(self) -> None:
        assert is_valid_smiles(None) is False  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        import math
        assert is_valid_smiles(math.nan) is False  # type: ignore[arg-type]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/pipelines/test_bbb_pipeline.py -v`

Expected: FAILS with `ModuleNotFoundError: No module named 'src.pipelines.bbb_pipeline'`.

- [ ] **Step 4: Implement `is_valid_smiles`**

Create `src/pipelines/bbb_pipeline.py`:
```python
"""BBB (Blood-Brain Barrier) molecule pipeline.

Reads the Kaggle BBBP dataset (SMILES strings + binary penetration label),
filters chemically invalid SMILES, computes Morgan circular fingerprints with
RDKit, and writes a model-ready feature table to `data/processed/`.

This module follows the Data Readiness contract in AGENTS.md §4:
schema validity, domain validity (drop invalid SMILES), determinism,
traceability (row count in / out / dropped), and idempotent output.
"""
from __future__ import annotations

import math
from typing import Any

from rdkit import Chem, RDLogger

from src.core.logger import get_logger

logger = get_logger(__name__)

# Suppress RDKit's noisy C++-level warning stream; we surface our own
# structured warnings via the project logger when a SMILES fails to parse.
RDLogger.DisableLog("rdApp.*")


def is_valid_smiles(smiles: Any) -> bool:
    """Return True iff `smiles` is a non-empty string parseable by RDKit.

    Handles the full set of garbage we expect from real CSVs:
    None, NaN floats, empty strings, and unparseable text.
    """
    if smiles is None:
        return False
    if isinstance(smiles, float) and math.isnan(smiles):
        return False
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/pipelines/test_bbb_pipeline.py -v`

Expected: 6 PASS in `TestIsValidSmiles`.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/bbbp_sample.csv tests/pipelines/test_bbb_pipeline.py src/pipelines/bbb_pipeline.py
git commit -m "feat(bbb): add SMILES validity guard with RDKit + test fixture"
```

---

## Task 6: BBB Pipeline — Morgan Fingerprint Extraction (TDD)

**Files:**
- Modify: `tests/pipelines/test_bbb_pipeline.py`
- Modify: `src/pipelines/bbb_pipeline.py`

- [ ] **Step 1: Write the failing test for `compute_morgan_fingerprint`**

Append to `tests/pipelines/test_bbb_pipeline.py`:
```python
import numpy as np

from src.pipelines.bbb_pipeline import compute_morgan_fingerprint


class TestComputeMorganFingerprint:
    def test_returns_numpy_array_of_correct_length(self) -> None:
        fp = compute_morgan_fingerprint("CCCO", n_bits=2048, radius=2)
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8

    def test_only_zero_or_one(self) -> None:
        fp = compute_morgan_fingerprint("c1ccccc1", n_bits=1024, radius=2)
        assert set(np.unique(fp).tolist()).issubset({0, 1})

    def test_different_molecules_yield_different_fingerprints(self) -> None:
        fp_a = compute_morgan_fingerprint("CCCO", n_bits=2048, radius=2)
        fp_b = compute_morgan_fingerprint("c1ccccc1", n_bits=2048, radius=2)
        assert not np.array_equal(fp_a, fp_b)

    def test_invalid_smiles_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid SMILES"):
            compute_morgan_fingerprint("not_a_smiles", n_bits=2048, radius=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipelines/test_bbb_pipeline.py::TestComputeMorganFingerprint -v`

Expected: 4 FAILS with `ImportError: cannot import name 'compute_morgan_fingerprint'`.

- [ ] **Step 3: Implement `compute_morgan_fingerprint`**

Append to `src/pipelines/bbb_pipeline.py`:
```python
import numpy as np
from rdkit.Chem import AllChem


def compute_morgan_fingerprint(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """Compute the Morgan (ECFP-like) circular fingerprint for a SMILES.

    Args:
        smiles: A SMILES string already known to be valid. Pass through
            `is_valid_smiles` first if the source is untrusted.
        n_bits: Length of the bit vector. 2048 is the de-facto default
            for downstream scikit-learn classifiers.
        radius: Morgan radius (2 ≈ ECFP4).

    Returns:
        A 1-D `np.ndarray` of length `n_bits` and dtype `uint8`, where
        each element is 0 or 1.

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    # RDKit ships a fast C++ writer into a preallocated numpy buffer.
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bit_vect, arr)
    return arr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipelines/test_bbb_pipeline.py -v`

Expected: all tests so far PASS (6 from Task 5 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add tests/pipelines/test_bbb_pipeline.py src/pipelines/bbb_pipeline.py
git commit -m "feat(bbb): add Morgan fingerprint extraction with shape/dtype guarantees"
```

---

## Task 7: BBB Pipeline — DataFrame Feature Extraction (TDD)

**Files:**
- Modify: `tests/pipelines/test_bbb_pipeline.py`
- Modify: `src/pipelines/bbb_pipeline.py`

- [ ] **Step 1: Write the failing test for `extract_features_from_dataframe`**

Append to `tests/pipelines/test_bbb_pipeline.py`:
```python
from src.pipelines.bbb_pipeline import extract_features_from_dataframe


class TestExtractFeaturesFromDataFrame:
    def test_filters_invalid_smiles(self) -> None:
        raw = pd.read_csv(FIXTURE)
        # Sanity: fixture contains 6 rows total, 2 are invalid by construction.
        assert len(raw) == 6

        features = extract_features_from_dataframe(raw, smiles_col="smiles", n_bits=128, radius=2)

        # Only the 4 chemically valid rows should remain.
        assert len(features) == 4

    def test_preserves_label_column(self) -> None:
        raw = pd.read_csv(FIXTURE)
        features = extract_features_from_dataframe(raw, smiles_col="smiles", n_bits=128, radius=2)
        assert "p_np" in features.columns

    def test_expands_fingerprint_into_named_columns(self) -> None:
        raw = pd.read_csv(FIXTURE)
        features = extract_features_from_dataframe(raw, smiles_col="smiles", n_bits=128, radius=2)
        fp_cols = [c for c in features.columns if c.startswith("fp_")]
        assert len(fp_cols) == 128
        # All FP columns must be 0/1 integers.
        assert features[fp_cols].isin([0, 1]).all().all()

    def test_drops_smiles_string_after_expansion(self) -> None:
        """Once expanded to bits, the original SMILES string adds no signal."""
        raw = pd.read_csv(FIXTURE)
        features = extract_features_from_dataframe(raw, smiles_col="smiles", n_bits=128, radius=2)
        assert "smiles" not in features.columns

    def test_resets_index(self) -> None:
        raw = pd.read_csv(FIXTURE)
        features = extract_features_from_dataframe(raw, smiles_col="smiles", n_bits=128, radius=2)
        assert list(features.index) == list(range(len(features)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipelines/test_bbb_pipeline.py::TestExtractFeaturesFromDataFrame -v`

Expected: 5 FAILS with `ImportError: cannot import name 'extract_features_from_dataframe'`.

- [ ] **Step 3: Implement `extract_features_from_dataframe`**

Append to `src/pipelines/bbb_pipeline.py`:
```python
import pandas as pd


def extract_features_from_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    n_bits: int = 2048,
    radius: int = 2,
) -> pd.DataFrame:
    """Convert a DataFrame of (SMILES + metadata) into model-ready features.

    Steps:
      1. Validate every SMILES with `is_valid_smiles`. Invalid rows are
         logged at WARNING with their original index and dropped.
      2. Compute the Morgan fingerprint for each remaining SMILES.
      3. Expand the bit vector into `n_bits` integer columns named
         `fp_0 ... fp_{n_bits - 1}` and concatenate with the surviving
         non-SMILES metadata.

    Args:
        df: Raw DataFrame; must contain `smiles_col`.
        smiles_col: Name of the SMILES column (default `"smiles"`).
        n_bits: Fingerprint length.
        radius: Morgan radius.

    Returns:
        A new DataFrame with the SMILES column dropped and `n_bits` new
        `fp_*` columns appended. Index is reset to 0..N-1.

    Raises:
        KeyError: if `smiles_col` is missing from `df`.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"DataFrame is missing required column {smiles_col!r}")

    n_total = len(df)
    valid_mask = df[smiles_col].apply(is_valid_smiles)
    n_invalid = int((~valid_mask).sum())

    if n_invalid:
        invalid_indices = df.index[~valid_mask].tolist()
        logger.warning(
            "Dropping %d/%d rows with invalid SMILES (indices=%s)",
            n_invalid, n_total, invalid_indices,
        )

    valid_df = df.loc[valid_mask].reset_index(drop=True)

    fingerprints = np.stack(
        [
            compute_morgan_fingerprint(s, n_bits=n_bits, radius=radius)
            for s in valid_df[smiles_col].tolist()
        ],
        axis=0,
    )
    fp_columns = [f"fp_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_columns, dtype=np.uint8)

    metadata = valid_df.drop(columns=[smiles_col]).reset_index(drop=True)
    out = pd.concat([metadata, fp_df], axis=1)

    logger.info(
        "Feature extraction complete: in=%d, out=%d, dropped=%d (%.2f%%)",
        n_total, len(out), n_invalid, 100.0 * n_invalid / max(n_total, 1),
    )
    return out
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/pipelines/test_bbb_pipeline.py -v`

Expected: all tests so far PASS (6 + 4 + 5 = 15).

- [ ] **Step 5: Commit**

```bash
git add tests/pipelines/test_bbb_pipeline.py src/pipelines/bbb_pipeline.py
git commit -m "feat(bbb): expand SMILES → Morgan FP into model-ready DataFrame with drift logging"
```

---

## Task 8: BBB Pipeline — `run_pipeline` Orchestrator + CLI (TDD)

**Files:**
- Modify: `tests/pipelines/test_bbb_pipeline.py`
- Modify: `src/pipelines/bbb_pipeline.py`

- [ ] **Step 1: Write the failing integration test for `run_pipeline`**

Append to `tests/pipelines/test_bbb_pipeline.py`:
```python
import shutil

from src.pipelines.bbb_pipeline import run_pipeline


class TestRunPipeline:
    def test_end_to_end_writes_processed_csv(self, tmp_path: Path) -> None:
        # Arrange: copy fixture into a synthetic raw layout.
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        output_path = proc_dir / "bbbp_features.csv"
        shutil.copy(FIXTURE, input_path)

        # Act
        run_pipeline(input_path=input_path, output_path=output_path, n_bits=128, radius=2)

        # Assert: file exists
        assert output_path.exists(), "pipeline must write processed CSV"

        # Assert: content is correct
        out = pd.read_csv(output_path)
        assert len(out) == 4  # 6 raw - 2 invalid
        assert "p_np" in out.columns
        assert sum(c.startswith("fp_") for c in out.columns) == 128
        assert "smiles" not in out.columns

    def test_run_pipeline_is_idempotent(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        output_path = proc_dir / "bbbp_features.csv"
        shutil.copy(FIXTURE, input_path)

        run_pipeline(input_path=input_path, output_path=output_path, n_bits=64, radius=2)
        first_bytes = output_path.read_bytes()
        run_pipeline(input_path=input_path, output_path=output_path, n_bits=64, radius=2)
        second_bytes = output_path.read_bytes()

        assert first_bytes == second_bytes, "pipeline output must be byte-deterministic"

    def test_run_pipeline_raises_when_input_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_pipeline(
                input_path=tmp_path / "nope.csv",
                output_path=tmp_path / "out.csv",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipelines/test_bbb_pipeline.py::TestRunPipeline -v`

Expected: 3 FAILS with `ImportError: cannot import name 'run_pipeline'`.

- [ ] **Step 3: Implement `run_pipeline` and CLI entrypoint**

Append to `src/pipelines/bbb_pipeline.py`:
```python
from pathlib import Path

DEFAULT_INPUT = Path("data/raw/bbbp.csv")
DEFAULT_OUTPUT = Path("data/processed/bbbp_features.csv")


def run_pipeline(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    smiles_col: str = "smiles",
    n_bits: int = 2048,
    radius: int = 2,
) -> None:
    """Run the BBB pipeline end-to-end: raw CSV → processed feature CSV.

    Reads the Kaggle BBBP CSV at `input_path`, validates and converts
    SMILES into Morgan fingerprints, and writes the model-ready table
    to `output_path`. Output is overwritten on every run (idempotent).

    Args:
        input_path: Path to the raw BBBP CSV (must include `smiles_col`).
        output_path: Where to write the processed feature CSV. Parent
            directory is created if missing.
        smiles_col: SMILES column name in the raw CSV.
        n_bits: Morgan fingerprint length.
        radius: Morgan radius.

    Raises:
        FileNotFoundError: if `input_path` does not exist.
        KeyError: if `smiles_col` is missing from the CSV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw BBBP file not found: {input_path}")

    logger.info("Reading raw BBBP from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, columns=%s", len(df), list(df.columns))

    features = extract_features_from_dataframe(
        df, smiles_col=smiles_col, n_bits=n_bits, radius=radius,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info("Wrote processed features to %s (rows=%d, cols=%d)",
                output_path, len(features), features.shape[1])


if __name__ == "__main__":
    # Production-ready CLI entrypoint:
    #   python -m src.pipelines.bbb_pipeline
    run_pipeline()
```

- [ ] **Step 4: Run the full test suite to verify everything passes**

Run: `pytest -v`

Expected: 22 PASS (4 logger + 18 BBB: 6 SMILES validity + 4 Morgan FP + 5 DataFrame + 3 run_pipeline).

- [ ] **Step 5: Commit**

```bash
git add tests/pipelines/test_bbb_pipeline.py src/pipelines/bbb_pipeline.py
git commit -m "feat(bbb): add run_pipeline orchestrator + CLI entrypoint with idempotent writes"
```

---

## Task 9: Final Wiring & Day-1 Acceptance Check

**Files:** none modified (verification + docs only)

- [ ] **Step 1: Run the full suite one last time**

Run: `pytest -v --tb=short`

Expected: **22 passed**, no warnings other than RDKit deprecation notices (already silenced via `RDLogger.DisableLog`).

- [ ] **Step 2: Confirm the CLI works against a real (or sample) BBBP file**

If a real Kaggle BBBP dump is available, place it at `data/raw/bbbp.csv`. Otherwise copy the fixture for a smoke run:
```bash
cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv
python -m src.pipelines.bbb_pipeline
```

Expected stdout (timestamps will differ):
```
... | INFO    | src.pipelines.bbb_pipeline | Reading raw BBBP from data/raw/bbbp.csv
... | INFO    | src.pipelines.bbb_pipeline | Loaded 6 rows, columns=['num', 'name', 'p_np', 'smiles']
... | WARNING | src.pipelines.bbb_pipeline | Dropping 2/6 rows with invalid SMILES (indices=[3, 5])
... | INFO    | src.pipelines.bbb_pipeline | Feature extraction complete: in=6, out=4, dropped=2 (33.33%)
... | INFO    | src.pipelines.bbb_pipeline | Wrote processed features to data/processed/bbbp_features.csv (rows=4, cols=2050)
```

And confirm the output:
```bash
ls -lh data/processed/bbbp_features.csv
head -1 data/processed/bbbp_features.csv | tr ',' '\n' | head -5
```

Expected: file exists, header begins with `num,name,p_np,fp_0,fp_1,...`.

- [ ] **Step 3: Final commit (sample raw seeded for next agent's smoke test)**

If you copied the fixture into `data/raw/bbbp.csv`, **do not commit it** (gitignored by design). Just leave it on disk for local runs. Confirm git is clean:

```bash
git status
```

Expected: `nothing to commit, working tree clean` (data files ignored).

---

## Day-1 Definition of Done

- [ ] `AGENTS.md` lives at the repo root and documents vision, layout, standards, and the Data Readiness contract.
- [ ] `requirements.txt` pins all deps for the three modalities + FastAPI + MLflow + tests.
- [ ] `src/core/logger.py` exposes `get_logger()` with idempotent handler attachment.
- [ ] `src/pipelines/bbb_pipeline.py` exposes `is_valid_smiles`, `compute_morgan_fingerprint`, `extract_features_from_dataframe`, and `run_pipeline`.
- [ ] Invalid SMILES are **logged with their indices** and dropped (Data Readiness §2).
- [ ] `pytest -v` is green with **22 tests** (4 logger + 18 BBB).
- [ ] Running `python -m src.pipelines.bbb_pipeline` against `data/raw/bbbp.csv` produces a deterministic `data/processed/bbbp_features.csv`.
- [ ] Each task above ended in its own commit; `git log --oneline` shows ≥ 8 atomic commits for the day.
