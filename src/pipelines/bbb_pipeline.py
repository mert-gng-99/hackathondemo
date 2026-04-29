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
import pandas as pd
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
        radius: Morgan radius (2 ≈ ECFP4). Passed to RDKit's modern
            MorganGenerator API.

    Returns:
        A 1-D `np.ndarray` of length `n_bits` and dtype `uint8`, where
        each element is 0 or 1.

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bit_vect = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(bit_vect, arr)
    return arr


def _compute_fingerprint_matrix(
    valid_smiles: list[str],
    n_bits: int,
    radius: int,
) -> np.ndarray:
    """Stack Morgan fingerprints into a (N, n_bits) uint8 matrix.

    Caller must guarantee `valid_smiles` is non-empty and every entry has
    already passed `is_valid_smiles`.
    """
    return np.stack(
        [
            compute_morgan_fingerprint(s, n_bits=n_bits, radius=radius)
            for s in valid_smiles
        ],
        axis=0,
    )


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

    On empty input or when every row is invalid, returns a DataFrame with
    the expected columns and zero rows (rather than raising), so callers
    downstream see a well-typed result instead of an exception.

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

    fp_columns = [f"fp_{i}" for i in range(n_bits)]
    metadata_columns = [c for c in df.columns if c != smiles_col]

    n_total = len(df)
    valid_mask = df[smiles_col].apply(is_valid_smiles)
    n_invalid = int((~valid_mask).sum())

    if n_invalid:
        invalid_indices = df.index[~valid_mask].tolist()
        display = invalid_indices[:10]
        suffix = (
            f"... (+{len(invalid_indices) - 10} more)"
            if len(invalid_indices) > 10
            else ""
        )
        logger.warning(
            "Dropping %d/%d rows with invalid SMILES (indices=%s%s)",
            n_invalid, n_total, display, suffix,
        )

    valid_df = df.loc[valid_mask].reset_index(drop=True)

    if len(valid_df) == 0:
        logger.info(
            "Feature extraction complete: in=%d, out=0, dropped=%d (%.2f%%)",
            n_total, n_invalid, 100.0 * n_invalid / max(n_total, 1),
        )
        return pd.DataFrame(columns=metadata_columns + fp_columns)

    fingerprints = _compute_fingerprint_matrix(
        valid_df[smiles_col].tolist(), n_bits=n_bits, radius=radius,
    )
    fp_df = pd.DataFrame(fingerprints, columns=fp_columns, dtype=np.uint8)
    metadata = valid_df.drop(columns=[smiles_col])
    out = pd.concat([metadata, fp_df], axis=1)

    logger.info(
        "Feature extraction complete: in=%d, out=%d, dropped=%d (%.2f%%)",
        n_total, len(out), n_invalid, 100.0 * n_invalid / max(n_total, 1),
    )
    return out
