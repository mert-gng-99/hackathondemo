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
