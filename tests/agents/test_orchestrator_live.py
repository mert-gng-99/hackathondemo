"""Live integration test — hits real OpenRouter, picks pipeline, retrieves chunks.

Skipped unless BOTH OPENROUTER_API_KEY is set AND the BBB model artifact
is built (the `run_bbb_pipeline` tool can't run without it). Marked `slow`
(network round-trips).

The dual gate matters because src/llm/explainer.py auto-loads .env at
import time; without the model-artifact gate, this test would attempt a
real OpenRouter call in CI/dev and then fail because the BBB tool can't
execute. In the deployed Docker image both conditions are satisfied
(secret + build-time training).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from openai import NotFoundError, OpenAI

from src.agents.orchestrator import Orchestrator
from src.agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from src.agents.tools import build_default_tools
from src.rag.ingest import ingest_directory


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"
_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"
_FALLBACK_MODEL = "anthropic/claude-haiku-4-5"
_BBB_MODEL_PATH = Path(
    os.environ.get("BBB_MODEL_PATH", "data/processed/bbb_model.joblib")
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
@pytest.mark.skipif(
    not _BBB_MODEL_PATH.exists(),
    reason=f"BBB model artifact missing at {_BBB_MODEL_PATH} — run python -m src.models.bbb_model",
)
class TestOrchestratorLive:
    @pytest.fixture(scope="class")
    def rag_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        d = tmp_path_factory.mktemp("rag_live")
        ingest_directory(_FIXTURE_KB, d)
        return d

    @pytest.fixture(scope="class")
    def client(self) -> OpenAI:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=30.0,
        )

    def test_smiles_input_picks_bbb_then_retrieves(self, client: OpenAI, rag_dir: Path) -> None:
        tools = build_default_tools(rag_index_dir=rag_dir)
        orch = Orchestrator(
            llm_client=client,
            tools=tools,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            model=os.environ.get("NEUROBRIDGE_AGENT_MODEL", _DEFAULT_MODEL),
            max_steps=5,
        )
        try:
            result = orch.run("CCO")
        except NotFoundError as e:
            # Free-tier model availability churns on OpenRouter — a 404 here
            # means the configured model isn't on this account, not that the
            # orchestrator code is broken. Verify with `GET /diag/agent` and
            # override via NEUROBRIDGE_AGENT_MODEL.
            pytest.skip(f"agent model unavailable on OpenRouter: {e}")
        # Soft assertions — model behavior varies but the workflow shape is fixed.
        assert result.finish_reason == "complete", f"got {result.finish_reason}, trace={result.trace}"
        tool_names = [t.name for t in result.trace]
        assert "run_bbb_pipeline" in tool_names, f"BBB pipeline not called; trace={tool_names}"
        assert "retrieve_context" in tool_names, f"RAG not called; trace={tool_names}"
        assert result.text, "empty final text"
