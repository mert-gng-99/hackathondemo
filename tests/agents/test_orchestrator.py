"""Tests for src.agents.orchestrator — agent loop with stubbed LLM client.

We do NOT hit OpenRouter here. We construct a fake client that returns
scripted tool-call responses, then verify the orchestrator dispatches
tools and assembles the trace correctly.
"""
from __future__ import annotations

import json
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
