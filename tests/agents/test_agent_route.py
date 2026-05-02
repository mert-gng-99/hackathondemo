"""Tests for POST /agent/run — uses a stub orchestrator factory."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.agents.schemas import AgentResult, ToolTraceItem
from src.api.main import app


client = TestClient(app)


class _FakeOrchestrator:
    """Returns a canned AgentResult; ignores input."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, user_input: str, context: dict[str, Any] | None = None) -> AgentResult:
        return AgentResult(
            text=f"Synthesized answer for: {user_input}",
            trace=[
                ToolTraceItem(name="run_bbb_pipeline", args={"smiles": user_input},
                              result={"label": 1, "label_text": "permeable"}),
                ToolTraceItem(name="retrieve_context", args={"query": "BBB"},
                              result={"chunks": []}),
            ],
            model="stub-model",
            finish_reason="complete",
        )


class TestAgentRoute:
    def test_post_returns_synthesized_text_and_trace(self) -> None:
        with patch("src.api.routes._build_orchestrator", return_value=_FakeOrchestrator()):
            r = client.post("/agent/run", json={"user_input": "CCO"})
        assert r.status_code == 200
        body = r.json()
        assert "Synthesized answer for: CCO" in body["text"]
        assert len(body["trace"]) == 2
        assert body["trace"][0]["name"] == "run_bbb_pipeline"
        assert body["model"] == "stub-model"
        assert body["finish_reason"] == "complete"

    def test_empty_user_input_422(self) -> None:
        r = client.post("/agent/run", json={"user_input": ""})
        assert r.status_code == 422

    def test_missing_user_input_422(self) -> None:
        r = client.post("/agent/run", json={})
        assert r.status_code == 422
