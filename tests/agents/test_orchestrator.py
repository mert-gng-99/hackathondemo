"""Tests for src.agents.orchestrator — agent loop with stubbed LLM client.

We do NOT hit OpenRouter here. We construct a fake client that returns
scripted tool-call responses, then verify the orchestrator dispatches
tools and assembles the trace correctly.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.agents.orchestrator import Orchestrator
from src.agents.tools import Tool


# --- Helpers ----------------------------------------------------------------


def _fake_choice_with_tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> Any:
    msg = MagicMock()
    msg.content = None
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    tc.model_dump = MagicMock(return_value={"id": call_id, "type": "function",
                                            "function": {"name": name,
                                                         "arguments": json.dumps(args)}})
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_choice_with_text(text: str) -> Any:
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class _PingInput(BaseModel):
    msg: str


class _PingOutput(BaseModel):
    echo: str


def _make_ping_tool() -> Tool:
    return Tool(
        name="ping",
        description="Echo a string back.",
        input_model=_PingInput,
        output_model=_PingOutput,
        execute=lambda inp: _PingOutput(echo=f"pong:{inp.msg}"),
    )


class _BBBInput(BaseModel):
    smiles: str


class _BBBOutput(BaseModel):
    label_text: str
    confidence: float


class _RetrieveInput(BaseModel):
    query: str
    k: int = 4


class _RetrieveOutput(BaseModel):
    chunks: list[dict[str, Any]]


def _make_workflow_tools() -> list[Tool]:
    return [
        Tool(
            name="run_bbb_pipeline",
            description="Run BBB.",
            input_model=_BBBInput,
            output_model=_BBBOutput,
            execute=lambda inp: _BBBOutput(label_text="permeable", confidence=0.82),
        ),
        Tool(
            name="retrieve_context",
            description="Retrieve context.",
            input_model=_RetrieveInput,
            output_model=_RetrieveOutput,
            execute=lambda inp: _RetrieveOutput(
                chunks=[{"source": "lipinski.md", "text": "BBB context"}]
            ),
        ),
    ]


# --- Tests ------------------------------------------------------------------


class TestOrchestrator:
    def test_single_tool_then_text_response(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"msg": "hello"}),
            _fake_choice_with_text("All done."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test input")
        assert result.text == "All done."
        assert result.finish_reason == "complete"
        assert len(result.trace) == 1
        assert result.trace[0].name == "ping"
        assert result.trace[0].args == {"msg": "hello"}
        assert result.trace[0].result == {"echo": "pong:hello"}

    def test_unknown_tool_recorded_as_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("nonexistent_tool", {"x": 1}),
            _fake_choice_with_text("Done."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test")
        assert result.trace[0].error is not None
        assert "unknown tool" in result.trace[0].error
        assert result.text == "Done."

    def test_invalid_tool_args_recorded_as_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"wrong_field": "x"}),
            _fake_choice_with_text("Recovered."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test")
        assert result.trace[0].error is not None
        assert result.text == "Recovered."

    def test_max_steps_exhausted_returns_finish_reason(self) -> None:
        client = MagicMock()
        # Always return another tool call — never terminates with text
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"msg": f"{i}"}, call_id=f"c{i}")
            for i in range(10)
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=3,
        )
        result = orch.run("test")
        assert result.finish_reason == "max_steps"
        assert len(result.trace) == 3

    def test_first_response_is_text_no_tools(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_text("Direct answer."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
        )
        result = orch.run("trivial input")
        assert result.text == "Direct answer."
        assert result.trace == []

    def test_enforced_workflow_falls_back_when_model_skips_tool_calls(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_text("I will answer directly."),
            _fake_choice_with_text("Still no retrieval."),
            _fake_choice_with_text("Grounded final answer."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=_make_workflow_tools(),
            system_prompt="sys",
            model="stub-model",
            max_steps=5,
            enforce_workflow=True,
            workflow_pipeline_tools={"run_bbb_pipeline"},
            workflow_retrieval_tool="retrieve_context",
            workflow_router=lambda user_input, context: (
                "run_bbb_pipeline",
                {"smiles": user_input},
            ),
            workflow_query_builder=lambda user_input, pipeline_trace, context: (
                "BBB permeability of small lipophilic molecules"
            ),
        )
        result = orch.run("CCO")
        assert result.finish_reason == "complete"
        assert result.text == "Grounded final answer."
        assert [t.name for t in result.trace] == ["run_bbb_pipeline", "retrieve_context"]
        assert result.trace[0].result == {"label_text": "permeable", "confidence": 0.82}
        assert result.trace[1].args["query"] == "BBB permeability of small lipophilic molecules"

    def test_workflow_drops_out_of_stage_tool_call_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            # During the pipeline stage the model wrongly calls retrieve_context
            _fake_choice_with_tool_call("retrieve_context", {"query": "x", "k": 4}),
            # After the workflow guard runs the BBB pipeline, model produces text
            _fake_choice_with_text("Skipping retrieval."),
            # Then the guard runs retrieve_context, model finalizes
            _fake_choice_with_text("Final answer."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=_make_workflow_tools(),
            system_prompt="sys",
            model="stub-model",
            max_steps=5,
            enforce_workflow=True,
            workflow_pipeline_tools={"run_bbb_pipeline"},
            workflow_retrieval_tool="retrieve_context",
            workflow_router=lambda user_input, context: (
                "run_bbb_pipeline",
                {"smiles": user_input},
            ),
            workflow_query_builder=lambda user_input, pipeline_trace, context: "q",
        )
        from src.agents import orchestrator as orch_module
        caplog.handler.setLevel(logging.INFO)
        orch_module.logger.addHandler(caplog.handler)
        try:
            result = orch.run("CCO")
        finally:
            orch_module.logger.removeHandler(caplog.handler)
        assert result.finish_reason == "complete"
        assert any(
            "dropped out-of-stage tool call" in rec.message
            and "retrieve_context" in rec.message
            and "stage=pipeline" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
