"""Smoke: ask the clinical corpus a question from the terminal.

Usage:
    python scripts/clinical_rag_smoke.py "egzersiz Alzheimer feedback"
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.rag.clinical.loader import load_index
from src.rag.clinical.retrieve import retrieve_clinical


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    payload = load_index(Path("data/external_rag/index/rag_index.pkl"))
    result = retrieve_clinical(payload, query, top_k=5, evidence_limit=5)
    print(result.summary_text or "(no matches)")


if __name__ == "__main__":
    main()
