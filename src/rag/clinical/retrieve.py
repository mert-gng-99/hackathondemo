"""TF-IDF retrieval over the clinical-paper corpus, with Turkish→English query expansion."""
from __future__ import annotations

import re
from textwrap import shorten
from typing import Any

import numpy as np

from src.core.logger import get_logger
from src.rag.clinical.types import ClinicalEvidence, ClinicalRetrievalResult

logger = get_logger(__name__)

# Mirrors the table in /Users/mertgungor/Downloads/rag/rag.py so the same
# Turkish keyword set produces the same expansion in both pipelines.
_QUERY_EXPANSIONS: dict[str, str] = {
    "alzheimer": "alzheimer dementia cognitive impairment mild cognitive impairment mci memory",
    "demans": "dementia alzheimer cognitive impairment memory cognition",
    "unutkanlik": "memory impairment cognitive decline dementia alzheimer",
    "parkinson": "parkinson disease movement disorder tremor motor symptoms non motor symptoms",
    "titreme": "tremor parkinson motor symptoms movement disorder",
    "egzersiz": "exercise physical activity training aerobic resistance cognition",
    "beslenme": "nutrition diet lifestyle metabolic risk factors",
    "risk": "risk factors lifestyle metabolic nutrition prevention",
    "tani": "diagnosis diagnostic criteria assessment screening",
    "tedavi": "treatment management therapy intervention",
}


def _expand_query(query: str) -> str:
    normalized = query.casefold()
    extras = [exp for key, exp in _QUERY_EXPANSIONS.items() if key in normalized]
    return f"{query} {' '.join(extras)}" if extras else query


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 6]


def _query_terms(expanded: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9]+", expanded.lower()) if len(t) >= 4}


def retrieve_clinical(
    payload: dict[str, Any],
    query: str,
    top_k: int = 5,
    evidence_limit: int = 5,
) -> ClinicalRetrievalResult:
    """Run TF-IDF search over the clinical corpus, return evidence + a feedback summary."""
    if not query.strip():
        return ClinicalRetrievalResult(query=query, evidence=[], summary_text="")

    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]
    chunks = payload["chunks"]

    expanded = _expand_query(query)
    qv = vectorizer.transform([expanded])
    scores = (matrix @ qv.T).toarray().ravel()
    if not np.any(scores):
        return ClinicalRetrievalResult(query=query, evidence=[], summary_text="")

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [(chunks[int(i)], float(scores[int(i)])) for i in top_indices if scores[int(i)] > 0]

    terms = _query_terms(expanded)
    candidates: list[tuple[float, str, Any, float]] = []
    for chunk, chunk_score in top_chunks:
        for sentence in _split_sentences(chunk.text):
            sent_terms = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
            overlap = len(terms & sent_terms)
            if overlap == 0:
                continue
            candidates.append((overlap + chunk_score, sentence, chunk, chunk_score))

    candidates.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    evidence: list[ClinicalEvidence] = []
    for _, sent, chunk, sc in candidates:
        fp = sent[:120].lower()
        if fp in seen:
            continue
        seen.add(fp)
        evidence.append(ClinicalEvidence(
            sentence=shorten(sent, width=420, placeholder="..."),
            source=chunk.source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            score=sc,
        ))
        if len(evidence) >= evidence_limit:
            break

    if not evidence:
        for chunk, sc in top_chunks[:evidence_limit]:
            evidence.append(ClinicalEvidence(
                sentence=shorten(chunk.text, width=420, placeholder="..."),
                source=chunk.source,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=sc,
            ))

    lines = ["Clinical RAG evidence (not a medical diagnosis):"]
    for ev in evidence:
        page = (
            f"p.{ev.page_start}" if ev.page_start == ev.page_end
            else f"pp.{ev.page_start}-{ev.page_end}"
        )
        lines.append(f"- {ev.sentence} [{ev.source}, {page} | score={ev.score:.3f}]")
    summary = "\n".join(lines)

    return ClinicalRetrievalResult(query=query, evidence=evidence, summary_text=summary)
