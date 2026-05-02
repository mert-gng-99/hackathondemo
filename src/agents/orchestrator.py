"""Orchestrator agent: function-calling loop over a list of Tools.

No agent framework — uses the openai SDK's chat-completions function-calling
interface directly. This is the same SDK already used by src/llm/explainer.py,
keeping the dependency surface minimal.

Public entry: `Orchestrator(llm_client, tools, system_prompt, model).run(user_input)`.
Returns an `AgentResult` with synthesized text + full tool-call trace.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.schemas import AgentResult, ToolTraceItem
from src.agents.tools import Tool
from src.core.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Single-agent function-calling loop. Stops on (a) text response, (b) max steps."""

    def __init__(
        self,
        llm_client: Any,
        tools: list[Tool],
        system_prompt: str,
        model: str,
        max_steps: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._client = llm_client
        self._tools_by_name = {t.name: t for t in tools}
        self._tool_schemas = [t.openai_schema() for t in tools]
        self._system_prompt = system_prompt
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature

    def run(self, user_input: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]
        trace: list[ToolTraceItem] = []

        for _step in range(self._max_steps):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._tool_schemas,
                tool_choice="auto",
                temperature=self._temperature,
            )
            msg = response.choices[0].message

            if not getattr(msg, "tool_calls", None):
                return AgentResult(
                    text=(msg.content or "").strip(),
                    trace=trace,
                    model=self._model,
                    finish_reason="complete",
                )

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                tool = self._tools_by_name.get(name)
                if tool is None:
                    err = f"unknown tool: {name}"
                    trace.append(ToolTraceItem(name=name, args={}, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": err}),
                    })
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = tool.invoke(args)
                    trace.append(ToolTraceItem(name=name, args=args, result=result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"result": result}, default=str),
                    })
                except Exception as e:
                    err = str(e)
                    trace.append(ToolTraceItem(name=name, args={}, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": err}),
                    })

        return AgentResult(
            text="Max steps reached without a final answer.",
            trace=trace,
            model=self._model,
            finish_reason="max_steps",
        )
