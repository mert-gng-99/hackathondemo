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
