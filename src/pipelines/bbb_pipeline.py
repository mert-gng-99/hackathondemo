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

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from src.core.logger import get_logger

logger = get_logger(__name__)

# Suppress RDKit's noisy C++-level warning stream; we surface our own
# structured warnings via the project logger when a SMILES fails to parse.
#
# IMPORTANT: this call is process-global and irreversible from this module's
# import. Any other code (other pipelines, the FastAPI surface, tests) that
# relies on RDKit warnings will be affected. If a future modality needs
# fine-grained RDKit log control, move this into an explicit
# `configure_rdkit_logging()` helper invoked from `run_pipeline()` instead.
RDLogger.DisableLog("rdApp.*")


def is_valid_smiles(smiles: str | float | None) -> bool:
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
    ConvertToNumpyArray(bit_vect, arr)
    return arr
