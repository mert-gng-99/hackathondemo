"""Unit + integration tests for the BBB (SMILES → Morgan FP) pipeline."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.pipelines.bbb_pipeline import (
    compute_morgan_fingerprint,
    extract_features_from_dataframe,
    is_valid_smiles,
    run_pipeline,
)


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
        assert is_valid_smiles(None) is False

    def test_rejects_nan(self) -> None:
        import math
        assert is_valid_smiles(math.nan) is False


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

    def test_raises_key_error_on_missing_smiles_col(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(KeyError, match="missing required column 'smiles'"):
            extract_features_from_dataframe(df, smiles_col="smiles", n_bits=64)

    def test_returns_empty_dataframe_when_all_invalid(self) -> None:
        """All-invalid input must produce a typed empty result, not crash."""
        df = pd.DataFrame(
            {
                "p_np": [0, 0],
                "smiles": ["", "still_garbage"],
            }
        )
        out = extract_features_from_dataframe(df, smiles_col="smiles", n_bits=32)
        assert len(out) == 0
        assert "p_np" in out.columns
        assert sum(c.startswith("fp_") for c in out.columns) == 32
        assert "smiles" not in out.columns

    def test_emits_warning_and_info_logs(self) -> None:
        """AGENTS.md §4 traceability: log invalid drops + in/out/dropped counts."""
        import io
        import logging

        from src.core.logger import get_logger
        from src.pipelines import bbb_pipeline as mod

        # Swap the module logger's stream so we can capture output.
        logger = get_logger(mod.__name__, level=logging.INFO)
        handler = logger.handlers[0]
        buf = io.StringIO()
        original_stream = handler.stream
        handler.stream = buf
        try:
            df = pd.read_csv(FIXTURE)
            extract_features_from_dataframe(df, smiles_col="smiles", n_bits=32)
        finally:
            handler.stream = original_stream

        output = buf.getvalue()
        assert "Dropping 2/6 rows with invalid SMILES" in output
        assert "Feature extraction complete: in=6, out=4, dropped=2" in output


class TestRunPipeline:
    def test_end_to_end_writes_processed_parquet(self, tmp_path: Path) -> None:
        # Arrange: copy fixture into a synthetic raw layout.
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        output_path = proc_dir / "bbbp_features.parquet"
        shutil.copy(FIXTURE, input_path)

        # Act
        run_pipeline(input_path=input_path, output_path=output_path, n_bits=128, radius=2)

        # Assert: file exists
        assert output_path.exists(), "pipeline must write processed Parquet"

        # Assert: content is correct
        out = pd.read_parquet(output_path)
        assert len(out) == 4  # 6 raw - 2 invalid
        assert "p_np" in out.columns
        assert sum(c.startswith("fp_") for c in out.columns) == 128
        assert "smiles" not in out.columns

    def test_run_pipeline_preserves_uint8_dtype(self, tmp_path: Path) -> None:
        """The Parquet round-trip must keep fp_* columns as uint8 (not widen to int64)."""
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        output_path = proc_dir / "bbbp_features.parquet"
        shutil.copy(FIXTURE, input_path)

        run_pipeline(input_path=input_path, output_path=output_path, n_bits=64, radius=2)
        out = pd.read_parquet(output_path)
        fp_cols = [c for c in out.columns if c.startswith("fp_")]
        for col in fp_cols:
            assert out[col].dtype == np.uint8, f"{col} widened to {out[col].dtype}"

    def test_run_pipeline_is_idempotent(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        proc_dir = tmp_path / "data" / "processed"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        output_path = proc_dir / "bbbp_features.parquet"
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
                output_path=tmp_path / "out.parquet",
            )

    def test_run_pipeline_rejects_directory_as_output(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        input_path = raw_dir / "bbbp.csv"
        shutil.copy(FIXTURE, input_path)

        # output_path points at an existing directory, not a file
        bad_output = tmp_path / "out_dir"
        bad_output.mkdir()

        with pytest.raises(IsADirectoryError, match="must be a file"):
            run_pipeline(input_path=input_path, output_path=bad_output, n_bits=32)
