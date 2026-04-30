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


def explain_prediction(
    model: RandomForestClassifier,
    smiles: str,
    top_k: int = 5,
    n_bits: int = 2048,
    radius: int = 2,
) -> list[dict[str, object]]:
    """Return the top-`top_k` feature attributions (SHAP values) for `smiles`.

    Uses `shap.TreeExplainer` (exact for tree ensembles, no sampling). The
    explanation is for the *predicted* class — i.e. SHAP values that pushed
    the model toward whichever label was returned by `predict_with_proba`.

    Reads fingerprint column names from `model._neurobridge_fp_cols` (set by
    `train()`). Falls back to `fp_<index>` if the attribute is missing — useful
    for models loaded from a joblib without the project-owned attribute.

    Args:
        model: Fitted classifier from `train()` or `load()`.
        smiles: A SMILES string (validated via `is_valid_smiles`).
        top_k: How many top features to return. Default 5 — matches the
            jury-demo budget (more bars = noisier waterfall chart).
        n_bits / radius: Must match training-time fingerprint settings.

    Returns:
        A list of `{"feature": "fp_<bit_idx>", "shap_value": float}` dicts,
        sorted by `abs(shap_value)` descending.

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    import shap  # local import — heavy module, only loaded when needed

    if not is_valid_smiles(smiles):
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = compute_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    X = fp.reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    # uint8 fingerprints cause benign additivity violations in SHAP's
    # reconstruction (base + sum != model output within tolerance); the
    # default check produces false-positive errors on tree ensembles
    # over quantized inputs, so we skip it.
    shap_values = explainer.shap_values(X, check_additivity=False)
    # `shap_values` shape varies by sklearn / shap versions:
    #   - older: list of (1, n_features) arrays, one per class
    #   - newer: ndarray of shape (1, n_features, n_classes) for binary RF
    #   - or (1, n_features) when output already condensed
    if isinstance(shap_values, list):
        proba = model.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        per_feature = shap_values[label_idx][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (1, n_features, n_classes)
            proba = model.predict_proba(X)[0]
            label_idx = int(np.argmax(proba))
            per_feature = arr[0, :, label_idx]
        else:
            # (1, n_features)
            per_feature = arr[0]

    fp_cols = (
        list(model._neurobridge_fp_cols)
        if hasattr(model, "_neurobridge_fp_cols")
        else [f"fp_{i}" for i in range(len(per_feature))]
    )

    pairs = sorted(
        zip(fp_cols, per_feature, strict=True),
        key=lambda p: abs(p[1]),
        reverse=True,
    )
    return [
        {"feature": str(name), "shap_value": float(value)}
        for name, value in pairs[:top_k]
    ]
