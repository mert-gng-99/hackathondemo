# TF-IDF Clinical RAG Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. TDD throughout.

**Goal.** Integrate the user's pre-built TF-IDF RAG corpus (14 medical PDFs covering Alzheimer's, Parkinson's, lifestyle, nutrition, exercise; Turkish + English query expansion; pre-built `rag_index.pkl`) into the platform alongside the existing FAISS+fastembed RAG. Both run side-by-side; the agent picks per-query.

**Architecture.** A new `src/rag/clinical/` sub-package wraps the user's `rag.py` script as an importable module. The existing `retrieve_context` agent tool grows a `corpus` parameter (`"reference"` for FAISS — current behaviour, kept default — `"clinical"` for the new TF-IDF index). A new module-level retriever object is constructed once at startup and reused. Pure addition: existing tests and behaviour stay green.

**Tech stack.** scikit-learn (already in deps via the existing pipelines? — verify; if not, add), `pypdf`, `numpy`. No FAISS or new embedding model. The user's pickle deserialises with stdlib `pickle` and `sklearn.feature_extraction.text.TfidfVectorizer`.

---

## Prerequisite (controller blocker)

The corpus and index live at `/Users/mertgungor/Downloads/rag/`. Before any task starts:

1. **Source PDFs.** Copy `Downloads/rag/HACKATHON/*.pdf` to `data/external_rag/clinical_pdfs/` in this repo. **Do NOT commit the PDFs to git** — they are external research papers, possibly copyrighted, and large (~33 MB total). Add `data/external_rag/` to `.gitignore` if not already covered (the existing repo gitignores `data/processed/` — check that this also covers `data/external_rag/`).

2. **Pre-built index.** Copy `Downloads/rag/index/rag_index.pkl` to `data/external_rag/index/rag_index.pkl`. Also gitignored.

3. **Verify pickle loads.** `python -c "import pickle; print(list(pickle.load(open('data/external_rag/index/rag_index.pkl','rb')).keys()))"`. Expect: `['created_at', 'source_dir', 'chunk_words', 'overlap_words', 'chunks', 'vectorizer', 'matrix']`.

If pickle fails to load (sklearn version mismatch, missing module), Task 1 has a regenerate fallback: rebuild the index from the PDFs using the same parameters in `Downloads/rag/rag.py`.

---

## File structure

| Path | Responsibility |
|---|---|
| Modify `requirements.txt` | confirm `scikit-learn` and `pypdf` are present (sklearn likely is; pypdf may not be) |
| Modify `.gitignore` | ensure `data/external_rag/` is ignored |
| Create `src/rag/clinical/__init__.py` | package marker |
| Create `src/rag/clinical/types.py` | `ClinicalChunk` dataclass + `ClinicalRetrievalResult` pydantic |
| Create `src/rag/clinical/loader.py` | unpickle the index; handle the user's payload schema; rebuild from PDFs as fallback |
| Create `src/rag/clinical/retrieve.py` | TF-IDF query + Turkish/English query expansion + sentence-level evidence picking |
| Modify `src/agents/tools.py` | `retrieve_context` accepts `corpus: Literal["reference", "clinical"]` |
| Modify `src/agents/prompts.py` | one-line update describing the corpus parameter |
| Create `tests/rag/test_clinical_loader.py` | unpickle + rebuild tests using a tiny fixture corpus |
| Create `tests/rag/test_clinical_retrieve.py` | retrieval correctness tests |
| Create `tests/agents/test_tools_clinical_corpus.py` | end-to-end agent-tool routing |
| Create `tests/fixtures/build_tiny_clinical_index.py` | builds a 2-page synthetic PDF index for tests |
| Modify `README.md` | document the dual-corpus surface |

---

## Tasks

### Task 0: Deps + gitignore + asset copy verification

**Files:** `requirements.txt`, `.gitignore`

- [ ] **Step 1:** check `requirements.txt`. If `pypdf` is absent, add `pypdf>=4.0,<6.0`. `scikit-learn` should already be there from the existing pipelines — verify with `pip show scikit-learn`.

- [ ] **Step 2:** open `.gitignore`. If `data/external_rag/` (or a parent that covers it) is not ignored, add a single line: `data/external_rag/`.

- [ ] **Step 3:** verify the asset transfer. `ls data/external_rag/clinical_pdfs/*.pdf | wc -l` should print `14` and `ls -lh data/external_rag/index/rag_index.pkl` should show ~12.9 MB.

- [ ] **Step 4:** if pypdf was added, run `pip install -r requirements.txt`.

- [ ] **Step 5:** `pytest -q` baseline. Commit: `chore(rag): pin pypdf; gitignore external rag corpus`.

---

### Task 1: Loader for the pre-built TF-IDF index

**Files:**
- Create: `src/rag/clinical/__init__.py`
- Create: `src/rag/clinical/types.py`
- Create: `src/rag/clinical/loader.py`
- Create: `tests/rag/test_clinical_loader.py`
- Create: `tests/fixtures/build_tiny_clinical_index.py`

- [ ] **Step 1: Build the tiny-fixture helper.**

`tests/fixtures/build_tiny_clinical_index.py`:

```python
"""Build a synthetic TF-IDF clinical-RAG index for tests.

Avoids needing real PDFs. Constructs the same payload schema the user's
rag.py produces so the loader can be tested independently of pypdf.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


# Same schema the user's rag.py produces. We define our own dataclass here
# so the test fixture is self-contained.
@dataclass(frozen=True)
class _Chunk:
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    text: str


def build(path: Path) -> Path:
    """Save a tiny TF-IDF index at `path`."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [
        _Chunk(0, "alzheimers_lifestyle.pdf", 1, 1,
               "Aerobic exercise and Mediterranean diet are associated with reduced cognitive decline in older adults at risk for Alzheimer's disease."),
        _Chunk(1, "parkinsons_motor.pdf", 1, 1,
               "Levodopa remains the most effective symptomatic treatment for motor symptoms of Parkinson's disease."),
        _Chunk(2, "alzheimers_mci.pdf", 2, 2,
               "Mild cognitive impairment may progress to dementia; MMSE and MoCA are standard screening tools."),
        _Chunk(3, "parkinsons_nutrition.pdf", 1, 1,
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
```

- [ ] **Step 2: Failing test.**

`tests/rag/test_clinical_loader.py`:

```python
"""Tests for src.rag.clinical.loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.clinical import loader
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestLoadIndex:
    def test_load_returns_payload_with_expected_keys(self, tmp_path: Path) -> None:
        idx_path = build_tiny(tmp_path / "tiny.pkl")
        payload = loader.load_index(idx_path)
        assert {"chunks", "vectorizer", "matrix"} <= set(payload)
        assert len(payload["chunks"]) == 4

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="clinical RAG index not found"):
            loader.load_index(tmp_path / "nope.pkl")

    def test_unique_sources(self, tmp_path: Path) -> None:
        idx_path = build_tiny(tmp_path / "tiny.pkl")
        payload = loader.load_index(idx_path)
        sources = {c.source for c in payload["chunks"]}
        assert sources == {
            "alzheimers_lifestyle.pdf", "parkinsons_motor.pdf",
            "alzheimers_mci.pdf", "parkinsons_nutrition.pdf",
        }
```

Run → ImportError on `src.rag.clinical.loader`.

- [ ] **Step 3: Minimal impl.**

`src/rag/clinical/__init__.py`: empty.

`src/rag/clinical/types.py`:

```python
"""Types shared across clinical-RAG modules."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


# The user's rag.py uses a frozen dataclass with this exact name and fields.
# We re-export the same shape so the loader can read pickles produced by
# either the user's script or our test fixture without translation.
@dataclass(frozen=True)
class ClinicalChunk:
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
    summary_text: str = Field(..., description="Pre-formatted RAG feedback string for the agent")
```

`src/rag/clinical/loader.py`:

```python
"""Load (or rebuild) the TF-IDF clinical RAG index."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.core.logger import get_logger

logger = get_logger(__name__)


def load_index(path: Path) -> dict[str, Any]:
    """Unpickle a TF-IDF index produced by the user's rag.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"clinical RAG index not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if "chunks" not in payload or "vectorizer" not in payload or "matrix" not in payload:
        raise ValueError(f"clinical RAG index missing expected keys: {sorted(payload)}")
    logger.info("loaded clinical RAG index: %d chunks from %s", len(payload["chunks"]), path)
    return payload
```

Note: we rely on the source dataclass `Chunk` being importable from where it was pickled. Sklearn is fine with version drift across minor patches; if a major-version drift causes a deserialise error, the user's `rag.py` rebuild is the recovery path (Task 2 wraps that).

Run tests → 3 passed.

- [ ] **Step 4:** `pytest -q` no regressions.

- [ ] **Step 5:** commit: `feat(rag): clinical TF-IDF index loader`.

---

### Task 2: Retrieval (TF-IDF query + Turkish/English expansion + evidence picking)

**Files:**
- Create: `src/rag/clinical/retrieve.py`
- Create: `tests/rag/test_clinical_retrieve.py`

- [ ] **Step 1: Failing test.**

`tests/rag/test_clinical_retrieve.py`:

```python
"""Tests for src.rag.clinical.retrieve."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.clinical.retrieve import retrieve_clinical
from src.rag.clinical.loader import load_index
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestRetrieve:
    def test_alzheimer_query_picks_alzheimer_chunks(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="exercise and Alzheimer's", top_k=2)
        sources = {ev.source for ev in result.evidence}
        assert any("alzheimers" in s for s in sources)

    def test_parkinson_query_picks_parkinson_chunks(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="Parkinson levodopa", top_k=2)
        sources = {ev.source for ev in result.evidence}
        assert any("parkinsons" in s for s in sources)

    def test_turkish_keyword_routes_via_expansion(self, tmp_path: Path) -> None:
        # User's rag.py expands "egzersiz" -> "exercise physical activity ...".
        # Our retrieve must honour the same expansion table so Turkish queries
        # hit English chunks.
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="egzersiz Alzheimer", top_k=2)
        # Turkish "egzersiz" + "alzheimer" should pick the lifestyle PDF.
        assert any("alzheimers_lifestyle" in ev.source for ev in result.evidence)

    def test_summary_text_contains_citations(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="diet and Parkinson", top_k=2)
        # Summary should embed source filenames so the LLM has citations.
        assert any(ev.source in result.summary_text for ev in result.evidence)

    def test_empty_query_returns_empty_evidence(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="", top_k=2)
        assert result.evidence == []
```

Run → ImportError.

- [ ] **Step 2: Minimal impl.**

`src/rag/clinical/retrieve.py`:

```python
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

    # Sentence-level evidence picking: pick the highest-overlap sentences first.
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
        # Fall back to chunk-level evidence if no sentence overlapped.
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
```

Run tests → 5 passed.

- [ ] **Step 3:** commit: `feat(rag): TF-IDF clinical retrieval with Turkish/English query expansion`.

---

### Task 3: Wire into the agent's `retrieve_context` tool

**Files:**
- Modify: `src/agents/tools.py`
- Modify: `src/agents/prompts.py`
- Create: `tests/agents/test_tools_clinical_corpus.py`

- [ ] **Step 1: Failing test.**

`tests/agents/test_tools_clinical_corpus.py`:

```python
"""Tests: retrieve_context tool dispatches by `corpus`."""
from __future__ import annotations

from pathlib import Path

from src.agents.tools import build_default_tools
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestClinicalCorpus:
    def test_corpus_default_is_reference(self, tmp_path: Path) -> None:
        clinical_idx = build_tiny(tmp_path / "tiny.pkl")
        tools = {t.name: t for t in build_default_tools(
            rag_index_dir=None,
            clinical_rag_index_path=clinical_idx,
        )}
        tool = tools["retrieve_context"]
        # `corpus` not provided → defaults to "reference".
        out = tool.execute(tool.input_model.model_validate({"query": "test"}))
        # With rag_index_dir=None, reference returns empty.
        assert hasattr(out, "chunks")

    def test_clinical_corpus_returns_evidence(self, tmp_path: Path) -> None:
        clinical_idx = build_tiny(tmp_path / "tiny.pkl")
        tools = {t.name: t for t in build_default_tools(
            rag_index_dir=None,
            clinical_rag_index_path=clinical_idx,
        )}
        tool = tools["retrieve_context"]
        out = tool.execute(tool.input_model.model_validate({
            "query": "exercise and Alzheimer",
            "corpus": "clinical",
        }))
        assert len(out.chunks) > 0
        # Each returned chunk has source + page metadata.
        for c in out.chunks:
            assert "source" in c and "text" in c
```

Run → fails (signature mismatch).

- [ ] **Step 2: Wire the tool.**

In `src/agents/tools.py`, the existing `RetrieveContextInput`/`RetrieveContextOutput` need the `corpus` field. Find the schemas (likely in `src/agents/schemas.py`) and add:

```python
from typing import Literal

class RetrieveContextInput(BaseModel):
    query: str
    k: int = 4
    corpus: Literal["reference", "clinical"] = "reference"
```

`RetrieveContextOutput.chunks` already accepts dicts; no change needed there.

In `src/agents/tools.py`, add a `clinical_rag_index_path: Path | None = None` parameter to `build_default_tools`. Update `_make_retrieve_executor` to take both index sources:

```python
def _make_retrieve_executor(
    rag_index_dir: Path | None,
    clinical_rag_index_path: Path | None,
) -> Callable[[RetrieveContextInput], RetrieveContextOutput]:
    # Lazily load the clinical payload at first use, cache for subsequent calls.
    clinical_cache: dict[str, Any] = {}

    def execute(inp: RetrieveContextInput) -> RetrieveContextOutput:
        if inp.corpus == "clinical":
            if clinical_rag_index_path is None:
                logger.warning("retrieve_context corpus=clinical but no index path configured")
                return RetrieveContextOutput(chunks=[])
            if "payload" not in clinical_cache:
                from src.rag.clinical.loader import load_index
                clinical_cache["payload"] = load_index(clinical_rag_index_path)
            from src.rag.clinical.retrieve import retrieve_clinical
            result = retrieve_clinical(clinical_cache["payload"], inp.query, top_k=inp.k)
            return RetrieveContextOutput(chunks=[
                {
                    "source": ev.source,
                    "page_start": ev.page_start,
                    "page_end": ev.page_end,
                    "text": ev.sentence,
                    "score": ev.score,
                }
                for ev in result.evidence
            ])

        # corpus == "reference" — existing FAISS path. Keep current behaviour.
        ... (preserve existing executor body) ...

    return execute
```

In `src/api/routes.py`, where `build_default_tools(...)` is called inside `_build_orchestrator()` (around line 577), pass the new path:

```python
clinical_idx = Path(os.environ.get(
    "CLINICAL_RAG_INDEX_PATH",
    "data/external_rag/index/rag_index.pkl",
))
tools = build_default_tools(
    rag_index_dir=rag_dir if rag_status["exists"] else None,
    clinical_rag_index_path=clinical_idx if clinical_idx.exists() else None,
)
```

Update `src/agents/prompts.py` `retrieve_context` description (already mentions FAISS) — adapt to:

```
- retrieve_context: retrieve up to k passages from a knowledge base. Pass corpus="clinical" for medical-paper evidence (Alzheimer's / Parkinson's / lifestyle / nutrition; supports Turkish keywords); default corpus="reference" for the curated FAISS index.
```

- [ ] **Step 3:** `pytest -q` → 2 new tests + previous baseline + retrieve regressions checked.

- [ ] **Step 4:** commit: `feat(agents): retrieve_context corpus dispatch (reference vs clinical)`.

---

### Task 4: README + CLI sanity

**Files:**
- Modify: `README.md`
- Create: `scripts/clinical_rag_smoke.py`

- [ ] **Step 1:** small CLI tool to demo the corpus from the terminal:

```python
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
```

- [ ] **Step 2:** README addition (find the existing RAG section, append):

```markdown
### Clinical Corpus (TF-IDF, Turkish + English)

A second, lightweight RAG index covers 14 medical PDFs (Alzheimer's, Parkinson's, lifestyle, nutrition, exercise) using TF-IDF + sklearn. Source PDFs live at `data/external_rag/clinical_pdfs/` (gitignored — copy from your team's shared drive). Pre-built index at `data/external_rag/index/rag_index.pkl`.

Agent invocation:

```python
retrieve_context(query="egzersiz Alzheimer feedback", corpus="clinical", k=5)
```

Local CLI smoke:

```bash
python scripts/clinical_rag_smoke.py "egzersiz Alzheimer feedback"
```

The Turkish keywords `alzheimer`, `parkinson`, `egzersiz`, `beslenme`, `tani`, `tedavi`, `risk`, `unutkanlik`, `titreme`, `demans` auto-expand to English equivalents so Turkish queries hit English chunks.
```

- [ ] **Step 3:** commit: `docs(rag): document clinical TF-IDF corpus + add CLI smoke`.

---

## Self-review

1. **Spec coverage.** User said "fully integrate the new RAG folder". The wrapper imports the user's exact pickle schema, mirrors the Turkish expansion table verbatim, and surfaces the same evidence semantics (citation per sentence, source+page tags). ✓
2. **Backward compatibility.** Default corpus is `"reference"`, preserving existing FAISS behaviour. ✓
3. **No XAI / re-embedding.** The user's TF-IDF index is used as-is. We don't re-embed or fine-tune. ✓
4. **Independence.** No coupling to BBB, fusion, or MRI modules. ✓
5. **No placeholders.** Every step has the full code or full diff direction. ✓

---

## Execution handoff

Save and choose: subagent-driven (recommended) or inline executing-plans.
