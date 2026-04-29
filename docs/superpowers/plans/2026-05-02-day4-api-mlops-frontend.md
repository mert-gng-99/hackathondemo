# Day 4 — API, Orchestration & Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the three Day-1/2/3 pipelines (BBB, EEG, MRI) in a productionized, demo-ready stack: shared core utilities, MLflow tracking, FastAPI surface, Docker compose orchestration, and a Streamlit B2B dashboard — without breaking the 106 existing green tests.

**Architecture:** Three concentric rings around the pipelines. Inner ring (`src/core/`) deduplicates threading-determinism + Parquet write + MLflow tracking helpers used by all three pipelines. Middle ring (`src/api/`) exposes each pipeline as a FastAPI POST endpoint with shared Pydantic request/response schemas. Outer ring (`src/frontend/`) is a Streamlit dashboard that calls the FastAPI surface (NOT the pipeline modules) and surfaces MLflow run links. `Dockerfile` + `docker-compose.yml` boot FastAPI + an MLflow tracking server side-by-side.

**Tech Stack:** FastAPI 0.115, Pydantic 2.9, MLflow 2.16, Streamlit (new dependency, pinned in this plan), Docker Compose v2. All existing pins (numpy/pandas/scipy/scikit-learn/rdkit/mne/nibabel/neuroharmonize/pyarrow) untouched.

---

## File Structure

```
src/
├── core/
│   ├── logger.py               # (existing)
│   ├── determinism.py          # NEW — Task 1: pin_threads()
│   ├── storage.py              # NEW — Task 2: write_parquet()
│   └── tracking.py             # NEW — Task 5: track_pipeline_run()
├── pipelines/
│   ├── bbb_pipeline.py         # MODIFY (Tasks 3, 6)
│   ├── eeg_pipeline.py         # MODIFY (Tasks 3, 6)
│   └── mri_pipeline.py         # MODIFY (Tasks 3, 6)
├── api/
│   ├── __init__.py             # (existing, empty)
│   ├── schemas.py              # NEW — Task 7
│   ├── routes.py               # NEW — Task 8
│   └── main.py                 # NEW — Task 7
└── frontend/
    ├── __init__.py             # NEW — Task 10
    └── app.py                  # NEW — Task 10

tests/
├── core/
│   ├── test_logger.py          # (existing)
│   ├── test_determinism.py     # NEW — Task 1
│   ├── test_storage.py         # NEW — Task 2
│   └── test_tracking.py        # NEW — Task 5
├── pipelines/
│   ├── test_bbb_pipeline.py    # (existing)
│   ├── test_eeg_pipeline.py    # (existing)
│   ├── test_mri_pipeline.py    # (existing)
│   └── test_cross_pipeline_smoke.py  # NEW — Task 4
├── api/
│   ├── __init__.py             # NEW — Task 7
│   ├── test_main.py            # NEW — Task 7
│   └── test_routes.py          # NEW — Task 8
└── frontend/
    ├── __init__.py             # NEW — Task 10
    └── test_app_import.py      # NEW — Task 10

Dockerfile                       # NEW — Task 9
docker-compose.yml               # NEW — Task 9
.dockerignore                    # NEW — Task 9
requirements.txt                 # MODIFY (Task 10: add streamlit)
AGENTS.md                        # MODIFY (Task 11: §2 layout, §6 add tracking note)
README.md                        # MODIFY (Task 11)
```

**Test count target:** 106 (existing) + ~30 (new) = **~136 tests green at end of Day 4**.

---

## Task 1: `src/core/determinism.py` — extract thread-pinning helper

**Why this task:** All three pipelines copy-paste the same six lines pinning OMP/OPENBLAS/MKL/pyarrow to single-thread mode. Drift risk is real (Day-2 review caught it). Extract into one helper, add tests, rewire pipelines in Task 3.

**Files:**
- Create: `src/core/determinism.py`
- Create: `tests/core/test_determinism.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_determinism.py`:

```python
"""Tests for src.core.determinism."""
from __future__ import annotations

import os

import pyarrow as pa

from src.core import determinism


class TestPinThreads:
    def test_sets_omp_env_var(self):
        os.environ.pop("OMP_NUM_THREADS", None)
        determinism.pin_threads()
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_sets_openblas_env_var(self):
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
        determinism.pin_threads()
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"

    def test_sets_mkl_env_var(self):
        os.environ.pop("MKL_NUM_THREADS", None)
        determinism.pin_threads()
        assert os.environ["MKL_NUM_THREADS"] == "1"

    def test_pins_pyarrow_cpu_count_to_1(self):
        pa.set_cpu_count(4)
        determinism.pin_threads()
        assert pa.cpu_count() == 1

    def test_pins_pyarrow_io_thread_count_to_1(self):
        pa.set_io_thread_count(4)
        determinism.pin_threads()
        assert pa.io_thread_count() == 1

    def test_does_not_override_existing_env(self):
        """User explicitly setting OMP_NUM_THREADS=2 must win — pin_threads()
        uses os.environ.setdefault so an upstream override is preserved."""
        os.environ["OMP_NUM_THREADS"] = "2"
        try:
            determinism.pin_threads()
            assert os.environ["OMP_NUM_THREADS"] == "2"
        finally:
            os.environ["OMP_NUM_THREADS"] = "1"

    def test_idempotent(self):
        determinism.pin_threads()
        determinism.pin_threads()
        assert pa.cpu_count() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/core/test_determinism.py -v
```
Expected: 7 errors / fails — module `src.core.determinism` does not exist.

- [ ] **Step 3: Implement `src/core/determinism.py`**

```python
"""Threading determinism: pin BLAS / OpenMP / pyarrow to single-threaded mode.

Multi-threaded floating-point reductions reorder operands non-deterministically
on each call, breaking the byte-identity guarantee in AGENTS.md §4 rule 3. Each
pipeline calls `pin_threads()` at import time to lock the process to a single
thread before any numerical work runs.

Honors pre-set env vars: if the caller exported `OMP_NUM_THREADS=4` upstream,
that value is preserved (we use `setdefault`, not `setitem`). The user is
responsible for the determinism trade-off in that case.
"""
from __future__ import annotations

import os

import pyarrow as pa

_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def pin_threads() -> None:
    """Pin BLAS / OpenMP / pyarrow to single-threaded mode (idempotent)."""
    for var in _ENV_VARS:
        os.environ.setdefault(var, "1")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/core/test_determinism.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/core/determinism.py tests/core/test_determinism.py
git commit -m "feat(core): extract pin_threads() helper for determinism"
```

---

## Task 2: `src/core/storage.py` — extract Parquet write helper

**Why this task:** All three pipelines repeat the same `output_path.parent.mkdir(...) / IsADirectoryError check / to_parquet(engine="pyarrow", compression="snappy", index=False)` pattern. Extract once.

**Files:**
- Create: `src/core/storage.py`
- Create: `tests/core/test_storage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_storage.py`:

```python
"""Tests for src.core.storage."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.core import storage


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class TestWriteParquet:
    def test_writes_parquet_at_path(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = tmp_path / "out.parquet"
        storage.write_parquet(df, out)
        round_trip = pd.read_parquet(out)
        pd.testing.assert_frame_equal(round_trip, df)

    def test_creates_parent_directories(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "deep" / "nested" / "out.parquet"
        storage.write_parquet(df, out)
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        storage.write_parquet(pd.DataFrame({"a": [1]}), out)
        storage.write_parquet(pd.DataFrame({"a": [2]}), out)
        assert pd.read_parquet(out)["a"].tolist() == [2]

    def test_raises_if_path_is_directory(self, tmp_path: Path):
        (tmp_path / "out.parquet").mkdir()
        with pytest.raises(IsADirectoryError):
            storage.write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "out.parquet")

    def test_byte_deterministic_on_repeat(self, tmp_path: Path):
        df = pd.DataFrame({"a": list(range(100)), "b": list(range(100, 200))})
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        storage.write_parquet(df, a)
        storage.write_parquet(df, b)
        assert _md5(a) == _md5(b)

    def test_preserves_uint8_dtype(self, tmp_path: Path):
        """BBB fingerprints are uint8; writing must not silently widen."""
        df = pd.DataFrame({"fp_0": pd.Series([0, 1], dtype="uint8")})
        out = tmp_path / "out.parquet"
        storage.write_parquet(df, out)
        assert pd.read_parquet(out)["fp_0"].dtype == "uint8"

    def test_index_not_persisted(self, tmp_path: Path):
        """index=False must be the default — round-trip should reset to RangeIndex."""
        df = pd.DataFrame({"a": [1, 2]}, index=["foo", "bar"])
        out = tmp_path / "out.parquet"
        storage.write_parquet(df, out)
        assert list(pd.read_parquet(out).index) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/core/test_storage.py -v
```
Expected: 7 errors — module not found.

- [ ] **Step 3: Implement `src/core/storage.py`**

```python
"""Deterministic Parquet I/O for `data/processed/` outputs.

Implements AGENTS.md §6 storage convention: pyarrow engine, snappy compression,
index suppressed. Combined with `src.core.determinism.pin_threads`, this writes
byte-identical Parquet files across runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write `df` to `output_path` as deterministic, snappy-compressed Parquet.

    Creates parent directories as needed. Overwrites any existing file at
    `output_path`. Raises `IsADirectoryError` if `output_path` resolves to an
    existing directory (caller passed a directory by mistake).

    Args:
        df: DataFrame to persist. Dtypes preserved (uint8 stays uint8, etc.).
        output_path: Destination file path (parent directories auto-created).

    Raises:
        IsADirectoryError: if `output_path` is an existing directory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"output_path must be a file, got a directory: {output_path}"
        )
    df.to_parquet(
        output_path, index=False, engine="pyarrow", compression="snappy",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/core/test_storage.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/core/storage.py tests/core/test_storage.py
git commit -m "feat(core): extract write_parquet() helper for §6 storage contract"
```

---

## Task 3: Refactor BBB / EEG / MRI pipelines to use core helpers

**Why this task:** Replace three duplicate copies of the env-pinning block + the `to_parquet(...)` call with the new helpers. Existing tests must stay green (this is pure refactor — zero behavior change).

**Files:**
- Modify: `src/pipelines/bbb_pipeline.py` (replace env block + to_parquet)
- Modify: `src/pipelines/eeg_pipeline.py` (same)
- Modify: `src/pipelines/mri_pipeline.py` (same)

- [ ] **Step 1: Refactor `bbb_pipeline.py`**

In `src/pipelines/bbb_pipeline.py`:

Replace the env-pinning block (currently lines ~28-35, the `os.environ.setdefault(...)` lines + `pa.set_cpu_count(1)` + `pa.set_io_thread_count(1)`):

```python
# Old:
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
```

With:

```python
# New:
from src.core.determinism import pin_threads

pin_threads()
```

Remove the now-unused `import os` and `import pyarrow as pa` lines if they have no other call sites in this file (they don't — verify). Keep the comment block above explaining why determinism matters.

In `run_pipeline()`, replace the trailing block:

```python
# Old:
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.is_dir():
    raise IsADirectoryError(...)
features.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
```

With:

```python
# New:
from src.core.storage import write_parquet  # at top of module
...
write_parquet(features, output_path)
```

Keep the `logger.info("Wrote processed features to %s ...")` line immediately after — it remains the user-visible trace.

- [ ] **Step 2: Run BBB tests**

```
pytest tests/pipelines/test_bbb_pipeline.py -v
```
Expected: 23 passed (unchanged from Day 1).

- [ ] **Step 3: Repeat refactor for `eeg_pipeline.py` and `mri_pipeline.py`**

Apply identical replacements. Same imports added (`from src.core.determinism import pin_threads`, `from src.core.storage import write_parquet`). Same env-block deletion. Same `to_parquet → write_parquet` swap.

- [ ] **Step 4: Run full pipeline test suite**

```
pytest tests/pipelines/ -v
```
Expected: 23 (BBB) + 37 (EEG) + 39 (MRI) = 99 passed. Plus 7 logger + 7 determinism + 7 storage = 113 tests green at this point. Verify count.

- [ ] **Step 5: Commit each pipeline refactor as its own commit**

```bash
git add src/pipelines/bbb_pipeline.py
git commit -m "refactor(bbb): use core.determinism + core.storage helpers"

git add src/pipelines/eeg_pipeline.py
git commit -m "refactor(eeg): use core.determinism + core.storage helpers"

git add src/pipelines/mri_pipeline.py
git commit -m "refactor(mri): use core.determinism + core.storage helpers"
```

---

## Task 4: Cross-pipeline smoke test

**Why this task:** A single test runs all three pipelines back-to-back against their fixtures and asserts each produces a non-empty Parquet with expected schema. This is the hackathon-judge "does the whole thing work?" test.

**Files:**
- Create: `tests/pipelines/test_cross_pipeline_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/pipelines/test_cross_pipeline_smoke.py`:

```python
"""End-to-end smoke test exercising all three pipelines back-to-back.

Asserts each pipeline produces a non-empty Parquet at its expected schema —
the hackathon-judge "does the whole stack still work?" check. Each pipeline
uses its own fixture (no cross-modality data sharing).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def test_bbb_pipeline_smoke(tmp_path: Path):
    out = tmp_path / "bbb.parquet"
    bbb_pipeline.run_pipeline(
        input_path=_FIXTURES / "bbbp_sample.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert sum(c.startswith("fp_") for c in df.columns) == 2048


def test_eeg_pipeline_smoke(tmp_path: Path):
    """Use the EEG fixture builder to materialize the FIF input."""
    from tests.fixtures.build_eeg_fixture import build as build_eeg
    fif = build_eeg(out_dir=tmp_path / "eeg_fixture")
    out = tmp_path / "eeg.parquet"
    eeg_pipeline.run_pipeline(input_path=fif, output_path=out)
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert "epoch_id" in df.columns


def test_mri_pipeline_smoke(tmp_path: Path):
    """Use the MRI fixture builder to materialize NIfTI inputs + sites.csv."""
    from tests.fixtures.build_mri_fixture import build as build_mri
    fixture_dir = build_mri(out_dir=tmp_path / "mri_fixture")
    out = tmp_path / "mri.parquet"
    mri_pipeline.run_pipeline(
        input_dir=fixture_dir,
        sites_csv=fixture_dir / "sites.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert "subject_id" in df.columns
    assert "site" in df.columns


def test_all_three_pipelines_run_in_one_process(tmp_path: Path):
    """Sanity: nothing in pipeline A leaks state that breaks pipeline B."""
    test_bbb_pipeline_smoke(tmp_path / "bbb")
    test_eeg_pipeline_smoke(tmp_path / "eeg")
    test_mri_pipeline_smoke(tmp_path / "mri")
```

> **Verify before writing:** confirm `tests/fixtures/build_eeg_fixture.py` and `build_mri_fixture.py` both expose a `build(out_dir: Path)` function. If `build_eeg_fixture.py` doesn't exist or has a different signature, adapt the test to use whatever loader the existing EEG tests use — read `tests/pipelines/test_eeg_pipeline.py` first and mirror its fixture-loading pattern. **Do not invent file paths.**

- [ ] **Step 2: Run tests**

```
pytest tests/pipelines/test_cross_pipeline_smoke.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/pipelines/test_cross_pipeline_smoke.py
git commit -m "test: cross-pipeline smoke run for all three modalities"
```

---

## Task 5: `src/core/tracking.py` — MLflow helper

**Why this task:** Each pipeline needs to log `params` (input path, configuration), `metrics` (row counts, runtime), and the output Parquet as an artifact. Writing four `mlflow.start_run / mlflow.log_param / mlflow.log_metric / mlflow.log_artifact` calls inline in each pipeline is duplication and breaks the existing tests (MLflow writes to a real `mlruns/` dir by default). Wrap in one helper.

**Files:**
- Create: `src/core/tracking.py`
- Create: `tests/core/test_tracking.py`
- Create: `conftest.py` at repo root (autouse fixture pinning `MLFLOW_TRACKING_URI` to a tmp path during tests)

- [ ] **Step 1: Write `conftest.py` to isolate MLflow during tests**

Create `/Users/mertgungor/Desktop/hackathon/conftest.py`:

```python
"""Repo-wide pytest fixtures.

Pins MLflow's tracking URI to a per-session tmp directory so pipeline tests
don't litter `./mlruns/` in the working tree, and so test runs are isolated
from production MLflow state.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_mlflow_tracking_uri() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="mlflow_test_"))
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_root}"
    yield
    # Don't rmtree — pytest tmpdir cleanup or OS handles it; rmtree
    # races with mlflow background writes on slow CI.
```

- [ ] **Step 2: Write failing tests for tracking helper**

Create `tests/core/test_tracking.py`:

```python
"""Tests for src.core.tracking."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd

from src.core import tracking


class TestTrackPipelineRun:
    def test_creates_run_with_experiment_name(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="bbb_pipeline",
            params={"input_path": "x.csv"},
            metrics={"rows_in": 6.0, "rows_out": 4.0},
            artifact_path=out,
        ) as run_id:
            assert run_id is not None
        runs = mlflow.search_runs(experiment_names=["bbb_pipeline"])
        assert len(runs) >= 1

    def test_logs_params(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="bbb_pipeline_params",
            params={"n_bits": 2048, "radius": 2},
            metrics={},
            artifact_path=out,
        ):
            pass
        runs = mlflow.search_runs(experiment_names=["bbb_pipeline_params"])
        assert "params.n_bits" in runs.columns
        assert runs.iloc[0]["params.n_bits"] == "2048"

    def test_logs_metrics(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="eeg_pipeline_metrics",
            params={},
            metrics={"duration_sec": 1.234, "rows_out": 100.0},
            artifact_path=out,
        ):
            pass
        runs = mlflow.search_runs(experiment_names=["eeg_pipeline_metrics"])
        assert runs.iloc[0]["metrics.duration_sec"] == 1.234
        assert runs.iloc[0]["metrics.rows_out"] == 100.0

    def test_logs_artifact(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="mri_pipeline_artifact",
            params={},
            metrics={},
            artifact_path=out,
        ) as run_id:
            pass
        artifacts = mlflow.MlflowClient().list_artifacts(run_id)
        assert any(a.path.endswith("out.parquet") for a in artifacts)

    def test_disabled_via_env_returns_no_op(self, monkeypatch, tmp_path: Path):
        """Setting NEUROBRIDGE_DISABLE_MLFLOW=1 must skip MLflow entirely
        (used by Docker compose dev mode where the tracking server is down)."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_MLFLOW", "1")
        out = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(out)
        with tracking.track_pipeline_run(
            experiment_name="should_not_appear",
            params={"x": 1},
            metrics={"y": 2.0},
            artifact_path=out,
        ) as run_id:
            assert run_id is None
        # No "should_not_appear" experiment was created
        names = [e.name for e in mlflow.search_experiments()]
        assert "should_not_appear" not in names
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/core/test_tracking.py -v
```
Expected: errors — module not found.

- [ ] **Step 4: Implement `src/core/tracking.py`**

```python
"""MLflow tracking helper used by all three pipelines.

Wraps `mlflow.start_run` so each pipeline can log params, metrics, and an
output artifact in one block. Honors `NEUROBRIDGE_DISABLE_MLFLOW=1` for
environments where the tracking server is not reachable (offline demos, CI
without mlflow service). When disabled, yields `None` and does no I/O.

Tracking URI source of truth: the standard `MLFLOW_TRACKING_URI` env var.
Tests pin this via the repo-wide conftest.py autouse fixture.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import mlflow

from src.core.logger import get_logger

logger = get_logger(__name__)

_DISABLE_FLAG = "NEUROBRIDGE_DISABLE_MLFLOW"


@contextlib.contextmanager
def track_pipeline_run(
    experiment_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    artifact_path: Path,
) -> Iterator[str | None]:
    """Context manager that creates an MLflow run for one pipeline invocation.

    On enter: creates/loads `experiment_name`, starts a run, logs params + metrics.
    On exit: logs `artifact_path` as an artifact and ends the run.

    Yields the active `run_id` (str), or `None` if MLflow is disabled.

    Args:
        experiment_name: e.g. "bbb_pipeline" / "eeg_pipeline" / "mri_pipeline".
        params: Run parameters (input path, hyper-params, etc.). Stringified by MLflow.
        metrics: Numeric metrics (row counts, durations).
        artifact_path: Path to the produced Parquet — logged as a run artifact.
    """
    if os.environ.get(_DISABLE_FLAG) == "1":
        logger.info("MLflow disabled via %s=1; skipping run tracking", _DISABLE_FLAG)
        yield None
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        try:
            yield run.info.run_id
        finally:
            if Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/core/test_tracking.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add conftest.py src/core/tracking.py tests/core/test_tracking.py
git commit -m "feat(core): add MLflow tracking helper with disable env-flag"
```

---

## Task 6: Wire MLflow tracking into all three pipelines

**Why this task:** Each `run_pipeline()` should log params (input/output paths + hyperparams), metrics (rows_in / rows_out / duration_sec), and the output Parquet as an artifact.

**Files:**
- Modify: `src/pipelines/bbb_pipeline.py` (`run_pipeline` function)
- Modify: `src/pipelines/eeg_pipeline.py` (same)
- Modify: `src/pipelines/mri_pipeline.py` (same)
- Modify: `tests/pipelines/test_bbb_pipeline.py` (add 1 test that asserts an MLflow run is created)
- Modify: `tests/pipelines/test_eeg_pipeline.py` (same)
- Modify: `tests/pipelines/test_mri_pipeline.py` (same)

- [ ] **Step 1: Add MLflow assertion test to BBB**

Append to `tests/pipelines/test_bbb_pipeline.py`:

```python
import mlflow
from src.pipelines import bbb_pipeline as _bbb_for_mlflow_test


class TestBBBPipelineMLflow:
    def test_run_pipeline_creates_mlflow_run(self, tmp_path):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "bbbp_sample.csv"
        out = tmp_path / "out.parquet"
        _bbb_for_mlflow_test.run_pipeline(input_path=fixture, output_path=out)
        runs = mlflow.search_runs(experiment_names=["bbb_pipeline"])
        assert len(runs) >= 1
        assert "metrics.rows_out" in runs.columns
        assert runs.iloc[0]["metrics.rows_out"] > 0
```

- [ ] **Step 2: Run failing test**

```
pytest tests/pipelines/test_bbb_pipeline.py::TestBBBPipelineMLflow -v
```
Expected: FAIL — no `bbb_pipeline` experiment.

- [ ] **Step 3: Wire MLflow into `bbb_pipeline.run_pipeline`**

In `src/pipelines/bbb_pipeline.py`, modify `run_pipeline`:

```python
import time

from src.core.tracking import track_pipeline_run

def run_pipeline(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    smiles_col: str = "smiles",
    n_bits: int = 2048,
    radius: int = 2,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw BBBP file not found: {input_path}")

    started = time.perf_counter()
    logger.info("Reading raw BBBP from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    features = extract_features_from_dataframe(
        df, smiles_col=smiles_col, n_bits=n_bits, radius=radius,
    )
    write_parquet(features, output_path)
    duration_sec = time.perf_counter() - started

    logger.info(
        "Wrote processed features to %s (rows=%d, cols=%d)",
        output_path, len(features), features.shape[1],
    )

    with track_pipeline_run(
        experiment_name="bbb_pipeline",
        params={
            "input_path": str(input_path),
            "output_path": str(output_path),
            "n_bits": n_bits,
            "radius": radius,
        },
        metrics={
            "rows_in": float(len(df)),
            "rows_out": float(len(features)),
            "rows_dropped": float(len(df) - len(features)),
            "duration_sec": duration_sec,
        },
        artifact_path=output_path,
    ):
        pass
```

- [ ] **Step 4: Run BBB test suite**

```
pytest tests/pipelines/test_bbb_pipeline.py -v
```
Expected: 24 passed (was 23 + 1 new MLflow test).

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/bbb_pipeline.py tests/pipelines/test_bbb_pipeline.py
git commit -m "feat(bbb): log run params, metrics, and parquet artifact to MLflow"
```

- [ ] **Step 6: Repeat for EEG**

Add a TestEEGPipelineMLflow class to `tests/pipelines/test_eeg_pipeline.py` mirroring the BBB pattern (using the EEG fixture). In `src/pipelines/eeg_pipeline.py`, wire MLflow into `run_pipeline` with experiment_name="eeg_pipeline" and EEG-relevant params (input_path, l_freq, h_freq, epoch_duration, etc.) and metrics (epochs_in, epochs_out, channels, duration_sec).

Run: `pytest tests/pipelines/test_eeg_pipeline.py -v` → 38 passed.
Commit: `feat(eeg): log run params, metrics, and parquet artifact to MLflow`.

- [ ] **Step 7: Repeat for MRI**

Add a TestMRIPipelineMLflow class. Wire MLflow into MRI `run_pipeline` with experiment_name="mri_pipeline" and MRI-relevant params (input_dir, sites_csv, n_roi_axes) and metrics (subjects_in, subjects_out, sites_count, duration_sec).

Run: `pytest tests/pipelines/test_mri_pipeline.py -v` → 40 passed.
Commit: `feat(mri): log run params, metrics, and parquet artifact to MLflow`.

- [ ] **Step 8: Run full test suite**

```
pytest -v
```
Expected total: ~119 tests passed (113 prior + 3 MLflow tests + 3 in-pipeline rewires; verify exact count, fix any reds).

---

## Task 7: FastAPI scaffolding — `schemas.py` + `main.py` + /health

**Why this task:** Stand up the FastAPI app with shared Pydantic models before adding pipeline routes. /health returns 200 OK so docker-compose health checks have something to poll.

**Files:**
- Create: `src/api/schemas.py`
- Create: `src/api/main.py`
- Create: `tests/api/__init__.py` (empty)
- Create: `tests/api/test_main.py`

- [ ] **Step 1: Write failing tests for /health**

Create `tests/api/__init__.py` (empty file).

Create `tests/api/test_main.py`:

```python
"""Tests for the FastAPI app surface (health + schema imports)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_get_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_get_health_returns_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_get_health_returns_pipeline_list(self):
        resp = client.get("/health")
        body = resp.json()
        assert set(body["pipelines"]) == {"bbb", "eeg", "mri"}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/api/test_main.py -v
```
Expected: ImportError — module not found.

- [ ] **Step 3: Implement `src/api/schemas.py`**

```python
"""Pydantic request / response models for the NeuroBridge FastAPI surface.

Each pipeline accepts its own request schema (BBBRequest / EEGRequest /
MRIRequest) but they all return a unified PipelineResponse — the dashboard
can render a single result card regardless of modality.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BBBRequest(BaseModel):
    input_path: str = Field(..., description="CSV path with a 'smiles' column")
    output_path: str = Field(..., description="Parquet output path")
    smiles_col: str = "smiles"
    n_bits: int = 2048
    radius: int = 2


class EEGRequest(BaseModel):
    input_path: str = Field(..., description="FIF or EDF file")
    output_path: str = Field(..., description="Parquet output path")
    l_freq: float = 1.0
    h_freq: float = 40.0
    epoch_duration_sec: float = 2.0


class MRIRequest(BaseModel):
    input_dir: str = Field(..., description="Directory of .nii.gz files")
    sites_csv: str = Field(..., description="CSV mapping subject_id → site")
    output_path: str = Field(..., description="Parquet output path")


class PipelineResponse(BaseModel):
    """Uniform response for every pipeline route."""
    status: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float
    mlflow_run_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    pipelines: list[str]
```

- [ ] **Step 4: Implement `src/api/main.py`**

```python
"""NeuroBridge FastAPI entrypoint.

Exposes /health for liveness and mounts pipeline routes from src.api.routes.
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import HealthResponse

app = FastAPI(
    title="NeuroBridge Enterprise",
    description="Three-modality clinical-ML pipeline surface (BBB / EEG / MRI).",
    version="0.4.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", pipelines=["bbb", "eeg", "mri"])
```

- [ ] **Step 5: Run tests to verify pass**

```
pytest tests/api/test_main.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas.py src/api/main.py tests/api/__init__.py tests/api/test_main.py
git commit -m "feat(api): scaffold FastAPI app + /health + shared Pydantic schemas"
```

---

## Task 8: FastAPI pipeline routes

**Why this task:** Three POST endpoints — one per modality — each invokes the corresponding `run_pipeline()` and returns the unified `PipelineResponse`. Errors mapped to HTTP codes: missing input → 404, bad path → 400, pipeline crash → 500.

**Files:**
- Create: `src/api/routes.py`
- Create: `tests/api/test_routes.py`
- Modify: `src/api/main.py` (mount the router)

- [ ] **Step 1: Write failing route tests**

Create `tests/api/test_routes.py`:

```python
"""Tests for /pipeline/{bbb,eeg,mri} POST endpoints."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestBBBRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/bbb",
            json={
                "input_path": str(_FIXTURES / "bbbp_sample.csv"),
                "output_path": str(out),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["rows"] > 0
        assert out.exists()

    def test_returns_404_when_input_missing(self, tmp_path: Path):
        resp = client.post(
            "/pipeline/bbb",
            json={
                "input_path": str(tmp_path / "does_not_exist.csv"),
                "output_path": str(tmp_path / "out.parquet"),
            },
        )
        assert resp.status_code == 404

    def test_returns_422_on_malformed_body(self):
        resp = client.post("/pipeline/bbb", json={"banana": 1})
        assert resp.status_code == 422  # pydantic validation


class TestEEGRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        from tests.fixtures.build_eeg_fixture import build as build_eeg
        fif = build_eeg(out_dir=tmp_path / "eeg_fixture")
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/eeg",
            json={"input_path": str(fif), "output_path": str(out)},
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] > 0


class TestMRIRoute:
    def test_returns_200_with_valid_input(self, tmp_path: Path):
        from tests.fixtures.build_mri_fixture import build as build_mri
        fixture_dir = build_mri(out_dir=tmp_path / "mri_fixture")
        out = tmp_path / "out.parquet"
        resp = client.post(
            "/pipeline/mri",
            json={
                "input_dir": str(fixture_dir),
                "sites_csv": str(fixture_dir / "sites.csv"),
                "output_path": str(out),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["rows"] > 0
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/api/test_routes.py -v
```
Expected: 5 errors — endpoints return 404 (route not mounted).

- [ ] **Step 3: Implement `src/api/routes.py`**

```python
"""POST /pipeline/{bbb,eeg,mri} routes — thin dispatchers over the pipelines.

Each route validates its request body via Pydantic, invokes the pipeline,
reads back the produced Parquet to populate row/column counts, and returns
a uniform PipelineResponse. Pipeline-domain errors map to standard HTTP
codes: FileNotFoundError → 404, ValueError → 400, anything else → 500.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BBBRequest, EEGRequest, MRIRequest, PipelineResponse,
)
from src.core.logger import get_logger
from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline")


def _wrap(experiment_name: str, output_path: Path, fn) -> PipelineResponse:
    """Run `fn()` (the pipeline call), gather metrics, return PipelineResponse."""
    started = time.perf_counter()
    try:
        fn()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    duration_sec = time.perf_counter() - started

    df = pd.read_parquet(output_path)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        max_results=1,
        order_by=["start_time DESC"],
    )
    run_id = runs.iloc[0]["run_id"] if len(runs) else None

    return PipelineResponse(
        status="ok",
        output_path=str(output_path),
        rows=len(df),
        columns=df.shape[1],
        duration_sec=duration_sec,
        mlflow_run_id=run_id,
    )


@router.post("/bbb", response_model=PipelineResponse)
def run_bbb(req: BBBRequest) -> PipelineResponse:
    return _wrap(
        "bbb_pipeline",
        Path(req.output_path),
        lambda: bbb_pipeline.run_pipeline(
            input_path=Path(req.input_path),
            output_path=Path(req.output_path),
            smiles_col=req.smiles_col,
            n_bits=req.n_bits,
            radius=req.radius,
        ),
    )


@router.post("/eeg", response_model=PipelineResponse)
def run_eeg(req: EEGRequest) -> PipelineResponse:
    return _wrap(
        "eeg_pipeline",
        Path(req.output_path),
        lambda: eeg_pipeline.run_pipeline(
            input_path=Path(req.input_path),
            output_path=Path(req.output_path),
            l_freq=req.l_freq,
            h_freq=req.h_freq,
            epoch_duration_sec=req.epoch_duration_sec,
        ),
    )


@router.post("/mri", response_model=PipelineResponse)
def run_mri(req: MRIRequest) -> PipelineResponse:
    return _wrap(
        "mri_pipeline",
        Path(req.output_path),
        lambda: mri_pipeline.run_pipeline(
            input_dir=Path(req.input_dir),
            sites_csv=Path(req.sites_csv),
            output_path=Path(req.output_path),
        ),
    )
```

> **Verify before writing:** confirm `eeg_pipeline.run_pipeline` and `mri_pipeline.run_pipeline` accept the parameter names used in the lambdas (`l_freq`, `h_freq`, `epoch_duration_sec` for EEG; `input_dir`, `sites_csv`, `output_path` for MRI). Read the actual function signatures first; if names differ, adjust the request schema in `src/api/schemas.py` to match. **Do not invent parameter names.**

- [ ] **Step 4: Mount router in `src/api/main.py`**

Edit `src/api/main.py`, after the `app = FastAPI(...)` line:

```python
from src.api.routes import router as pipeline_router

app.include_router(pipeline_router)
```

- [ ] **Step 5: Run tests**

```
pytest tests/api/ -v
```
Expected: 8 passed (3 main + 5 routes).

- [ ] **Step 6: Commit**

```bash
git add src/api/routes.py src/api/main.py tests/api/test_routes.py
git commit -m "feat(api): POST /pipeline/{bbb,eeg,mri} dispatch routes"
```

---

## Task 9: Dockerfile + docker-compose.yml

**Why this task:** Single-command boot for FastAPI + MLflow tracking server. Judges run `docker compose up`, browse to localhost:8000 / localhost:5000, see the system live.

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```
.venv/
.venv312/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
data/raw/*
data/processed/*
mlruns/
.git/
docs/
tests/
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
# NeuroBridge Enterprise — multi-stage build, FastAPI + pipeline runtime image.
# Python 3.12 because RDKit / scikit-learn / numpy pins ship cp310-cp312 wheels only.
FROM python:3.12-slim AS runtime

# System deps required by RDKit (libxrender, libxext) and nibabel/MNE
# (libgomp). Slim base lacks them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer is cached when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY AGENTS.md README.md ./

# Determinism env vars baked in (the pipelines re-pin defensively but
# baking them avoids a brief race on container start).
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.0
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri /mlflow/mlruns
      --default-artifact-root /mlflow/mlruns
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow/mlruns

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow
    volumes:
      - ./data:/app/data

volumes:
  mlflow-data:
```

- [ ] **Step 4: Validate compose syntax**

```
docker compose config
```
Expected: prints the resolved YAML with no errors. (Skip this step if Docker is not installed locally; the file syntax is straightforward.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat(deploy): Dockerfile + compose for api + mlflow server"
```

---

## Task 10: Streamlit B2B dashboard

**Why this task:** The hackathon judges' first impression. Three tabs (Molecule / Signal / Image), each fires a POST to the FastAPI surface and shows the result + an MLflow link.

**Files:**
- Modify: `requirements.txt` (add `streamlit==1.39.0`)
- Create: `src/frontend/__init__.py`
- Create: `src/frontend/app.py`
- Create: `tests/frontend/__init__.py`
- Create: `tests/frontend/test_app_import.py`

- [ ] **Step 1: Add streamlit to requirements**

In `requirements.txt`, after the `# --- Tooling / tests ---` block (or under a new `# --- Frontend ---` block):

```
# --- Frontend (B2B dashboard) ---
streamlit==1.39.0
```

Run: `pip install -r requirements.txt` to install it locally.

- [ ] **Step 2: Write smoke import test**

Create `tests/frontend/__init__.py` (empty).

Create `tests/frontend/test_app_import.py`:

```python
"""Smoke-test that the Streamlit app module imports cleanly.

Streamlit UIs are hard to unit-test without `streamlit.testing` (which
spawns a headless app); for hackathon scope we settle for a clean import
+ presence of the page-config call. Manual UX testing via `streamlit run`.
"""
from __future__ import annotations


def test_app_module_imports():
    from src.frontend import app  # noqa: F401


def test_app_module_defines_main():
    from src.frontend import app
    assert hasattr(app, "main")
    assert callable(app.main)
```

- [ ] **Step 3: Run failing test**

```
pytest tests/frontend/ -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `src/frontend/__init__.py`** (empty file).

- [ ] **Step 5: Implement `src/frontend/app.py`**

```python
"""NeuroBridge Enterprise — Streamlit B2B dashboard.

Three tabs (Molecule / Signal / Image), each fires a POST request against the
sibling FastAPI service and renders a result card with row counts, runtime,
and a link to the corresponding MLflow run.

Launch: `streamlit run src/frontend/app.py`
"""
from __future__ import annotations

import os

import httpx
import streamlit as st


_API_URL = os.environ.get("NEUROBRIDGE_API_URL", "http://localhost:8000")
_MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _post(endpoint: str, payload: dict) -> dict:
    resp = httpx.post(f"{_API_URL}{endpoint}", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def _render_result(body: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Rows", body["rows"])
    cols[1].metric("Columns", body["columns"])
    cols[2].metric("Runtime (sec)", f"{body['duration_sec']:.2f}")
    st.success(f"Wrote: `{body['output_path']}`")
    if body.get("mlflow_run_id"):
        st.markdown(
            f"**MLflow run:** [{body['mlflow_run_id']}]"
            f"({_MLFLOW_URL}/#/experiments/0/runs/{body['mlflow_run_id']})"
        )


def main() -> None:
    st.set_page_config(
        page_title="NeuroBridge Enterprise",
        page_icon="🧠",
        layout="wide",
    )
    st.title("NeuroBridge Enterprise")
    st.caption(
        "Three-modality clinical ML platform — solving Data Drift, "
        "Missing Modalities, and Artifacts."
    )

    bbb_tab, eeg_tab, mri_tab = st.tabs([
        "🧪 Molecule (BBB)",
        "🧠 Signal (EEG)",
        "📷 Image (MRI)",
    ])

    with bbb_tab:
        st.subheader("Blood-Brain-Barrier penetration — Morgan fingerprint")
        bbb_in = st.text_input("Input CSV path", "data/raw/bbbp.csv")
        bbb_out = st.text_input("Output Parquet path", "data/processed/bbbp_features.parquet")
        if st.button("Run BBB Pipeline"):
            with st.spinner("Computing fingerprints..."):
                _render_result(_post("/pipeline/bbb", {
                    "input_path": bbb_in, "output_path": bbb_out,
                }))

    with eeg_tab:
        st.subheader("EEG — bandpass + ICA artifact removal")
        eeg_in = st.text_input("Input FIF/EDF path", "data/raw/eeg.fif")
        eeg_out = st.text_input("Output Parquet path", "data/processed/eeg_features.parquet")
        if st.button("Run EEG Pipeline"):
            with st.spinner("Filtering + ICA..."):
                _render_result(_post("/pipeline/eeg", {
                    "input_path": eeg_in, "output_path": eeg_out,
                }))

    with mri_tab:
        st.subheader("MRI — site-level ComBat harmonization")
        mri_dir = st.text_input("Input NIfTI dir", "data/raw/mri/")
        sites_csv = st.text_input("Sites CSV", "data/raw/mri/sites.csv")
        mri_out = st.text_input("Output Parquet path", "data/processed/mri_features.parquet")
        if st.button("Run MRI Pipeline"):
            with st.spinner("Masking + ComBat..."):
                _render_result(_post("/pipeline/mri", {
                    "input_dir": mri_dir,
                    "sites_csv": sites_csv,
                    "output_path": mri_out,
                }))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

```
pytest tests/frontend/ -v
```
Expected: 2 passed.

- [ ] **Step 7: Smoke-launch Streamlit (manual)**

```
streamlit run src/frontend/app.py
```
Open <http://localhost:8501>, click each tab. Expect: page loads, three tabs visible, Run buttons present (clicking will fail without the FastAPI service running — that's fine, this is a UI render check).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/frontend/__init__.py src/frontend/app.py \
        tests/frontend/__init__.py tests/frontend/test_app_import.py
git commit -m "feat(frontend): Streamlit dashboard with 3 modality tabs"
```

---

## Task 11: AGENTS.md + README.md updates + final DoD

**Why this task:** Document the new layers in the contract file and roadmap. Run the full smoke verification one more time. Tag the commit.

**Files:**
- Modify: `AGENTS.md` (§2 directory tree, new sub-section in §6 about MLflow tracking)
- Modify: `README.md` (status table + Quick Start + Day-4 in roadmap)

- [ ] **Step 1: Update `AGENTS.md`**

In §2 Directory Layout, add `src/frontend/` and the new `src/core/{determinism,storage,tracking}.py` files. Add an entry for `Dockerfile` and `docker-compose.yml`.

After §6 Storage Format Convention, add §7:

```markdown
## 7. Experiment Tracking

Every `run_pipeline()` invocation logs to MLflow via `src.core.tracking.track_pipeline_run`:
- **Experiment names** are the pipeline module name (`bbb_pipeline`, `eeg_pipeline`, `mri_pipeline`).
- **Params**: input/output paths and pipeline hyperparameters.
- **Metrics**: row counts (in/out/dropped) and `duration_sec`.
- **Artifact**: the output Parquet at `data/processed/<modality>_features.parquet`.

The tracking URI is read from `MLFLOW_TRACKING_URI` (defaults to `./mlruns/` when unset).
Set `NEUROBRIDGE_DISABLE_MLFLOW=1` to skip tracking entirely (offline / CI fallback).

The repo-wide `conftest.py` autouse fixture pins `MLFLOW_TRACKING_URI` to a tmp dir for tests
so the production `mlruns/` directory is never written from the test suite.
```

- [ ] **Step 2: Update `README.md`**

- Status table: replace Day-4 "(planned)" with "Shipped — N tests green" once final count is known.
- Add a Quick Start section: `docker compose up`, point browsers at `:8000/docs` (FastAPI Swagger) and `:8501` (Streamlit).
- Add to "Where to Look": `docs/superpowers/plans/2026-05-02-day4-api-mlops-frontend.md`, `src/core/{determinism,storage,tracking}.py`, `src/api/`, `src/frontend/`.
- Roadmap: mark Day 4 done.

- [ ] **Step 3: Run full test suite for DoD**

```
pytest -v
```
Expected: ~136 tests passed total. If any reds, debug before continuing.

- [ ] **Step 4: Verify three CLI smoke runs still produce identical Parquets**

```
md5 data/processed/bbbp_features.parquet
md5 data/processed/eeg_features.parquet
md5 data/processed/mri_features.parquet
python -m src.pipelines.bbb_pipeline
python -m src.pipelines.eeg_pipeline
python -m src.pipelines.mri_pipeline
md5 data/processed/bbbp_features.parquet
md5 data/processed/eeg_features.parquet
md5 data/processed/mri_features.parquet
```
Expected: each MD5 unchanged across runs (idempotent / byte-deterministic preserved).

- [ ] **Step 5: Verify FastAPI surface end-to-end (manual)**

```
uvicorn src.api.main:app --port 8000 &
curl -s http://localhost:8000/health | jq
curl -s -X POST http://localhost:8000/pipeline/bbb \
  -H 'Content-Type: application/json' \
  -d '{"input_path": "data/raw/bbbp.csv", "output_path": "/tmp/bbb.parquet"}' | jq
```
Expected: 200 with `{"status":"ok", "rows": >0, ...}`. Kill the uvicorn process when done.

- [ ] **Step 6: Final commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: Day-4 close-out — AGENTS §7 tracking, README MLOps surface"
```

---

## Definition of Done (Day 4)

| Check | Pass criterion |
|---|---|
| All tests green | `pytest -v` reports ~136 passed |
| `src/core/{determinism,storage,tracking}.py` exist with their own test files | yes |
| BBB / EEG / MRI pipelines all use `pin_threads()` + `write_parquet()` (no duplicate inline blocks) | grep verifies |
| BBB / EEG / MRI pipelines all log to MLflow under their `<modality>_pipeline` experiment | mlflow.search_runs returns ≥1 per pipeline |
| `POST /pipeline/{bbb,eeg,mri}` route works with FastAPI `TestClient` | tests/api/test_routes.py green |
| `streamlit run src/frontend/app.py` renders 3 tabs without crashing | manual smoke |
| `docker compose config` parses cleanly | yes |
| Existing 106 tests still green (no regressions from refactor) | yes |
| Output Parquets remain byte-identical across runs | md5 stable |
| AGENTS.md §7 documents the MLflow contract | yes |

When all rows are green, push: `git push origin main`. Day 5 (production hardening: rate limits, OpenAPI auth, tracing) becomes optional polish on top of an already shippable system.
