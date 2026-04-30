"""BBB-permeability downstream classifier — train / save / load / predict.

Built on top of `data/processed/bbbp_features.parquet` produced by
`src.pipelines.bbb_pipeline`. Uses scikit-learn's `RandomForestClassifier`
(no XGBoost — saves a heavy dep without losing accuracy at this scale).

The model takes a 2,048-bit Morgan fingerprint as input. SHAP-based
explanation is added in Task 2 (`explain_prediction`).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.core.logger import get_logger
from src.pipelines.bbb_pipeline import (
    compute_morgan_fingerprint,
    is_valid_smiles,
)

logger = get_logger(__name__)


_FP_COL_PREFIX = "fp_"


def _split_features_and_label(
    df: pd.DataFrame, label_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pull out fp_* columns as X and `label_col` as y. Returns (X, y, fp_col_names)."""
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} not in DataFrame")
    fp_cols = [c for c in df.columns if c.startswith(_FP_COL_PREFIX)]
    if not fp_cols:
        raise KeyError(
            f"No {_FP_COL_PREFIX}* columns found — was this DataFrame produced "
            f"by bbb_pipeline.run_pipeline?"
        )
    X = df[fp_cols].to_numpy()
    y = df[label_col].to_numpy()
    return X, y, fp_cols


def train(
    df: pd.DataFrame,
    label_col: str = "p_np",
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on Morgan fingerprints.

    Args:
        df: Output of `bbb_pipeline.run_pipeline` — has `fp_0..fp_N-1` cols
            plus a binary `label_col`.
        label_col: Name of the binary target column. Defaults to "p_np".
        n_estimators: Number of trees. 100 is the sklearn default.
        random_state: Seed for split + tree construction (determinism).

    Returns:
        Fitted `RandomForestClassifier` with `feature_names_in_` set so
        downstream callers can map SHAP values back to fp_<bit> indices.
    """
    X, y, fp_cols = _split_features_and_label(df, label_col)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    # Stash the column names under a project-owned attribute so SHAP (Task 2)
    # can map values back to fp_<bit> indices. Sklearn's own feature_names_in_
    # is only set automatically when fit receives a DataFrame; setting it
    # manually fires UserWarning on every predict call.
    model._neurobridge_fp_cols = list(fp_cols)
    logger.info(
        "Trained BBB classifier: n=%d, n_features=%d, classes=%s",
        len(y), X.shape[1], model.classes_.tolist(),
    )
    return model


def save(model: RandomForestClassifier, path: Path) -> None:
    """Persist a fitted model to `path` (parent dirs auto-created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved BBB model to %s", path)


def load(path: Path) -> RandomForestClassifier:
    """Load a previously-saved model. Raises FileNotFoundError on missing artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BBB model artifact not found: {path}")
    return joblib.load(path)


def predict_with_proba(
    model: RandomForestClassifier,
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> dict[str, object]:
    """Predict BBB permeability for a single SMILES.

    Returns:
        `{"label": int, "confidence": float}` where confidence is the
        predicted class's probability (max class probability — model's
        self-rated certainty).

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    if not is_valid_smiles(smiles):
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = compute_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    proba = model.predict_proba(fp.reshape(1, -1))[0]
    label_idx = int(np.argmax(proba))
    label = int(model.classes_[label_idx])
    return {
        "label": label,
        "confidence": float(proba[label_idx]),
    }
