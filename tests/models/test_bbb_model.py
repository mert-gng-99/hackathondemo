"""Tests for src.models.bbb_model — train, save/load, predict, uncertainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import bbb_model


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def trained_model_and_features():
    """Train one tiny model from the committed BBBP fixture; cache for the module."""
    from src.pipelines import bbb_pipeline
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bbb_model_test_"))
    out = tmp / "features.parquet"
    bbb_pipeline.run_pipeline(
        input_path=_FIXTURES / "bbbp_sample.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    # Tiny n_estimators for test speed; real training uses default 100.
    model = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
    return model, df


class TestTrain:
    def test_returns_fitted_classifier(self, trained_model_and_features):
        model, _ = trained_model_and_features
        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2

    def test_raises_on_missing_label_column(self, trained_model_and_features):
        _, df = trained_model_and_features
        with pytest.raises(KeyError):
            bbb_model.train(df.drop(columns=["p_np"]), label_col="p_np")

    def test_deterministic_with_random_state(self, trained_model_and_features):
        _, df = trained_model_and_features
        m1 = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        m2 = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        X = df[fp_cols].to_numpy()
        np.testing.assert_array_equal(m1.predict_proba(X), m2.predict_proba(X))


class TestSaveLoad:
    def test_save_then_load_roundtrip(self, trained_model_and_features, tmp_path: Path):
        model, df = trained_model_and_features
        artifact = tmp_path / "bbb_model.joblib"
        bbb_model.save(model, artifact)
        assert artifact.exists()

        reloaded = bbb_model.load(artifact)
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        X = df[fp_cols].to_numpy()
        np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))

    def test_load_raises_on_missing_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            bbb_model.load(tmp_path / "does_not_exist.joblib")


class TestPredictWithProba:
    def test_returns_label_and_confidence(self, trained_model_and_features):
        model, _ = trained_model_and_features
        result = bbb_model.predict_with_proba(model, "CCO")
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in (0, 1)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_raises_on_invalid_smiles(self, trained_model_and_features):
        model, _ = trained_model_and_features
        with pytest.raises(ValueError):
            bbb_model.predict_with_proba(model, "this_is_not_a_smiles_AT_ALL")

    def test_confidence_equals_max_class_probability(self, trained_model_and_features):
        """confidence is the max class probability — verifies against raw predict_proba."""
        model, _ = trained_model_and_features
        from src.pipelines.bbb_pipeline import compute_morgan_fingerprint
        fp = compute_morgan_fingerprint("CCO").reshape(1, -1)
        raw_proba = model.predict_proba(fp)[0]
        result = bbb_model.predict_with_proba(model, "CCO")
        assert abs(result["confidence"] - float(max(raw_proba))) < 1e-9
