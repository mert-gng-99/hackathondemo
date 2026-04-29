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
