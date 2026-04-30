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


class TestExplainPrediction:
    def test_returns_top_k_features(self, trained_model_and_features):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=5)
        assert len(attributions) == 5
        for a in attributions:
            assert "feature" in a
            assert "shap_value" in a
            assert isinstance(a["shap_value"], float)

    def test_features_sorted_by_absolute_shap_value_descending(
        self, trained_model_and_features,
    ):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=10)
        abs_vals = [abs(a["shap_value"]) for a in attributions]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_features_named_fp_INDEX(self, trained_model_and_features):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=3)
        for a in attributions:
            assert a["feature"].startswith("fp_")
            int(a["feature"].split("_")[1])  # parses cleanly

    def test_raises_on_invalid_smiles(self, trained_model_and_features):
        model, _ = trained_model_and_features
        with pytest.raises(ValueError):
            bbb_model.explain_prediction(model, "still_not_a_smiles", top_k=5)

    def test_deterministic_output(self, trained_model_and_features):
        """AGENTS.md §4 rule 3: identical input → identical SHAP attributions."""
        model, _ = trained_model_and_features
        r1 = bbb_model.explain_prediction(model, "CCO", top_k=5)
        r2 = bbb_model.explain_prediction(model, "CCO", top_k=5)
        assert r1 == r2


class TestCalibrationMetadata:
    def test_train_attaches_calibration_attribute(self, trained_model_and_features):
        model, _ = trained_model_and_features
        assert hasattr(model, "_neurobridge_calibration")
        bins = model._neurobridge_calibration
        assert isinstance(bins, list)
        # Always at least one bin (the lowest-threshold one)
        assert len(bins) >= 1
        for b in bins:
            assert "threshold" in b
            assert "precision" in b
            assert "support" in b
            assert 0.0 <= b["threshold"] <= 1.0
            assert 0.0 <= b["precision"] <= 1.0
            assert b["support"] >= 0

    def test_calibration_thresholds_are_sorted_ascending(
        self, trained_model_and_features,
    ):
        model, _ = trained_model_and_features
        thresholds = [b["threshold"] for b in model._neurobridge_calibration]
        assert thresholds == sorted(thresholds)

    def test_calibration_survives_save_load_roundtrip(
        self, trained_model_and_features, tmp_path: Path,
    ):
        model, _ = trained_model_and_features
        artifact = tmp_path / "calibrated.joblib"
        bbb_model.save(model, artifact)
        reloaded = bbb_model.load(artifact)
        assert hasattr(reloaded, "_neurobridge_calibration")
        assert reloaded._neurobridge_calibration == model._neurobridge_calibration


class TestTrainStatsMetadata:
    """Day 7 — T1A: train()-time confidence distribution stash."""

    def test_train_attaches_train_stats_attribute(self, trained_model_and_features):
        model, _ = trained_model_and_features
        assert hasattr(model, "_neurobridge_train_stats")
        stats = model._neurobridge_train_stats
        assert isinstance(stats, dict)
        for key in ("median", "std", "n_train"):
            assert key in stats, f"missing key {key!r} in train stats"
        assert 0.0 <= stats["median"] <= 1.0
        assert stats["std"] >= 0.0
        assert stats["n_train"] >= 1

    def test_train_stats_survives_save_load_roundtrip(
        self, trained_model_and_features, tmp_path: Path,
    ):
        from src.models import bbb_model
        model, _ = trained_model_and_features
        path = tmp_path / "m.joblib"
        bbb_model.save(model, path)
        reloaded = bbb_model.load(path)
        assert hasattr(reloaded, "_neurobridge_train_stats")
        assert reloaded._neurobridge_train_stats == model._neurobridge_train_stats
