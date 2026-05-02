"""Seed demo artifacts so every showcase path works without external data.

Idempotent — skips any artifact that already exists. Safe to call during
Docker build OR at container start.

Generates:
- data/processed/mri_dl_2d/best_model.pt        (random resnet18 4-class)
- data/processed/mri_model.onnx                  (dynamic-D/H/W ONNX, biased toward 'abnormal')
- data/processed/eeg_clf.joblib                  (synthetic-separable RandomForest)
- data/external_rag/index/rag_index.pkl          (4-chunk synthetic clinical TF-IDF)
- tests/fixtures/mri_sample/subject_0_axial.png  (axial slice from the bundled NIfTI)
"""
from __future__ import annotations

import sys
from pathlib import Path


def seed_mri_dl_2d() -> Path:
    out = Path("data/processed/mri_dl_2d/best_model.pt")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    torch.save(model.state_dict(), str(out))
    return out


def seed_mri_volumetric_onnx() -> Path:
    out = Path("data/processed/mri_model.onnx")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, "D", "H", "W"],
    )
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    value = helper.make_tensor("const_logits", TensorProto.FLOAT, [1, 2], [0.3, 2.1])
    node = helper.make_node("Constant", inputs=[], outputs=["logits"], value=value)
    graph = helper.make_graph([node], "demo_mri_classifier", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, str(out))
    return out


def seed_eeg_clf() -> Path:
    out = Path("data/processed/eeg_clf.joblib")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n_features = 16
    X_ctrl = rng.normal(0.0, 1.0, size=(100, n_features))
    X_alz = rng.normal(2.0, 1.0, size=(100, n_features))
    X = np.vstack([X_ctrl, X_alz])
    y = np.array([0] * 100 + [1] * 100)
    clf = RandomForestClassifier(n_estimators=12, max_depth=6, random_state=0)
    clf.fit(X, y)
    joblib.dump(clf, str(out))
    return out


def seed_clinical_rag_index() -> Path:
    """Tiny synthetic clinical TF-IDF index (4 chunks). Replace with the real
    pre-built pickle to upgrade quality without code changes."""
    out = Path("data/external_rag/index/rag_index.pkl")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    from datetime import datetime
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.rag.clinical.types import ClinicalChunk

    chunks = [
        ClinicalChunk(0, "alzheimers_lifestyle.pdf", 1, 1,
                      "Aerobic exercise and Mediterranean diet are associated with reduced cognitive decline in older adults at risk for Alzheimer's disease."),
        ClinicalChunk(1, "parkinsons_motor.pdf", 1, 1,
                      "Levodopa remains the most effective symptomatic treatment for motor symptoms of Parkinson's disease."),
        ClinicalChunk(2, "alzheimers_mci.pdf", 2, 2,
                      "Mild cognitive impairment may progress to dementia; MMSE and MoCA are standard screening tools."),
        ClinicalChunk(3, "parkinsons_nutrition.pdf", 1, 1,
                      "Dietary patterns rich in antioxidants and omega-3 fatty acids are linked to lower Parkinson's risk."),
    ]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, norm="l2")
    matrix = vectorizer.fit_transform([c.text for c in chunks])

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(out.parent),
        "chunk_words": 220,
        "overlap_words": 45,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    return out


def seed_axial_png() -> Path:
    """Axial mid-slice PNG from the bundled NIfTI fixture for the Researcher tab."""
    out = Path("tests/fixtures/mri_sample/subject_0_axial.png")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import nibabel as nib
    import numpy as np
    from PIL import Image

    src = Path("tests/fixtures/mri_sample/subject_0.nii.gz")
    vol = np.asarray(nib.load(str(src)).get_fdata(), dtype=np.float32)
    mid = vol.shape[2] // 2
    slc = vol[:, :, mid]
    norm = (slc - slc.min()) / max(slc.max() - slc.min(), 1e-6)
    Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(str(out))
    return out


def main() -> int:
    seeds = [
        ("MRI 2D resnet18 state_dict",   seed_mri_dl_2d),
        ("MRI volumetric ONNX",          seed_mri_volumetric_onnx),
        ("EEG sklearn classifier",       seed_eeg_clf),
        ("Clinical TF-IDF RAG index",    seed_clinical_rag_index),
        ("Axial PNG fixture",            seed_axial_png),
    ]
    print("Seeding demo artifacts...", flush=True)
    for name, fn in seeds:
        try:
            path = fn()
            kb = path.stat().st_size // 1024 if path.is_file() else 0
            print(f"  OK   {name:35s} {path}  ({kb} KB)", flush=True)
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
            return 1
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
