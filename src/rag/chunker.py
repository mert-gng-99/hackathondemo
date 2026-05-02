"""Paragraph-aware recursive character splitter for RAG ingestion.

Public entry: `chunk_text(text, max_chars, overlap)`. Splits on the first
of [paragraph break, sentence end, newline, space] that fits inside the
window. Empty / whitespace-only inputs return [].
"""
from __future__ import annotations


_SEPARATORS: tuple[str, ...] = ("\n\n", ". ", "\n", " ")


def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """Split `text` into chunks of at most `max_chars`, with `overlap` carry-over."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # try to land on a clean boundary inside [start, end]
            for sep in _SEPARATORS:
                last = text.rfind(sep, start, end)
                if last > start:
                    end = last + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks
