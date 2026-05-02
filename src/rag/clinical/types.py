"""Types shared across clinical-RAG modules."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ClinicalChunk:
    """Mirrors the Chunk dataclass produced by the user's rag.py builder."""
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    text: str


class ClinicalEvidence(BaseModel):
    sentence: str
    source: str
    page_start: int
    page_end: int
    score: float = Field(..., ge=0.0)


class ClinicalRetrievalResult(BaseModel):
    query: str
    evidence: list[ClinicalEvidence]
    summary_text: str = Field(..., description="Pre-formatted RAG feedback for the agent")
