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
