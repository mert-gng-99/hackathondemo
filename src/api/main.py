"""NeuroBridge FastAPI entrypoint.

Exposes /health for liveness. Pipeline routes are mounted in Task 8.
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import (
    router as pipeline_router,
    predict_router,
    explain_router,
    experiments_router,
    agent_router,
)
from src.api.schemas import HealthResponse

app = FastAPI(
    title="NeuroBridge Enterprise",
    description="Three-modality clinical-ML pipeline surface (BBB / EEG / MRI).",
    version="0.4.0",
)

app.include_router(pipeline_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(experiments_router)
app.include_router(agent_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — used by docker-compose health checks and Streamlit."""
    return HealthResponse(status="ok", pipelines=["bbb", "eeg", "mri"])


@app.get("/diag/openrouter")
def diag_openrouter() -> dict:
    """One-shot OpenRouter reachability probe — diagnostic only.

    Reports whether the explainer can reach OpenRouter from this container.
    Returns key presence (length + first 12 chars only — never the full
    secret), kill-switch state, the first model in the chain, and the
    HTTP/error status of an 8-token probe call against that model. Used
    to diagnose why /explain/* falls back to template in production.
    """
    import os as _os
    from src.llm import explainer as _ex

    key = _os.environ.get("OPENROUTER_API_KEY") or ""
    chain = _ex._free_model_chain()
    first_model = chain[0] if chain else None

    out: dict = {
        "has_key": bool(key),
        "key_len": len(key),
        "key_prefix": key[:12] if key else None,
        "kill_switch_on": _os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1",
        "should_use_llm": _ex._should_use_llm(),
        "chain_len": len(chain),
        "first_model": first_model,
        "probe": None,
    }

    if not key or not first_model:
        out["probe"] = "skipped (no key or empty chain)"
        return out

    try:
        from openai import (
            OpenAI,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )
    except ImportError as e:
        out["probe"] = f"openai SDK not importable: {e}"
        return out

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=8.0,
        )
        c = client.chat.completions.create(
            model=first_model,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=8,
            temperature=0,
        )
        text = (c.choices[0].message.content or "").strip()
        out["probe"] = {"status": "OK", "preview": text[:60]}
    except RateLimitError:
        out["probe"] = {"status": "429", "note": "rate-limited"}
    except APIStatusError as e:
        out["probe"] = {"status": str(getattr(e, "status_code", "?")), "message": str(e)[:200]}
    except (APIConnectionError, APITimeoutError) as e:
        out["probe"] = {"status": "CONN", "exception": type(e).__name__}
    except Exception as e:
        out["probe"] = {"status": "ERR", "exception": type(e).__name__, "message": str(e)[:200]}

    return out


@app.get("/diag/agent")
def diag_agent() -> dict:
    """Reachability probe for the orchestrator agent surface.

    Reports key presence (length + 12-char prefix only — never the full
    secret), the configured agent model, knowledge-base index status,
    and the registered tool names.
    """
    import os as _os
    from pathlib import Path as _Path

    from src.agents.tools import build_default_tools

    key = _os.environ.get("OPENROUTER_API_KEY") or ""
    model = _os.environ.get("NEUROBRIDGE_AGENT_MODEL", "google/gemini-2.0-flash-exp:free")

    rag_dir = _Path("data/processed/faiss_index")
    rag_status: dict = {"index_dir": str(rag_dir), "exists": False, "chunk_count": 0}
    if (rag_dir / "index.bin").exists() and (rag_dir / "chunks.json").exists():
        rag_status["exists"] = True
        try:
            import json as _json
            rag_status["chunk_count"] = len(_json.loads((rag_dir / "chunks.json").read_text()))
        except Exception as e:
            rag_status["error"] = f"chunks.json unreadable: {e}"

    tools = build_default_tools(rag_index_dir=rag_dir if rag_status["exists"] else None)
    return {
        "has_key": bool(key),
        "key_len": len(key),
        "key_prefix": key[:12] if key else None,
        "agent_model": model,
        "rag": rag_status,
        "tool_names": [t.name for t in tools],
    }
