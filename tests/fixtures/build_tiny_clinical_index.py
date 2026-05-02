"""Build a synthetic TF-IDF clinical-RAG index for tests.

Avoids needing real PDFs. Constructs the same payload schema the user's
rag.py produces so the loader can be tested independently of pypdf.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.clinical.types import ClinicalChunk


def build(path: Path) -> Path:
    """Save a tiny TF-IDF index at `path`."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

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
        "source_dir": str(path.parent),
        "chunk_words": 220,
        "overlap_words": 45,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path
