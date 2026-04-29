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
