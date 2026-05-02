"""Orchestrator agent: function-calling loop over a list of Tools.

No agent framework — uses the openai SDK's chat-completions function-calling
interface directly. This is the same SDK already used by src/llm/explainer.py,
keeping the dependency surface minimal.

Public entry: `Orchestrator(llm_client, tools, system_prompt, model).run(user_input)`.
Returns an `AgentResult` with synthesized text + full tool-call trace.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from src.agents.schemas import AgentResult, ToolTraceItem
from src.agents.tools import Tool
from src.core.logger import get_logger

logger = get_logger(__name__)


WorkflowRouter = Callable[[str, dict[str, Any] | None], tuple[str, dict[str, Any]] | None]
WorkflowQueryBuilder = Callable[[str, ToolTraceItem, dict[str, Any] | None], str]


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
        enforce_workflow: bool = False,
        workflow_pipeline_tools: set[str] | None = None,
        workflow_retrieval_tool: str | None = None,
        workflow_router: WorkflowRouter | None = None,
        workflow_query_builder: WorkflowQueryBuilder | None = None,
    ) -> None:
        self._client = llm_client
        self._tools_by_name = {t.name: t for t in tools}
        self._tool_schemas = [t.openai_schema() for t in tools]
        self._tool_schemas_by_name = {
            t.name: t.openai_schema()
            for t in tools
        }
        self._system_prompt = system_prompt
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature
        self._enforce_workflow = enforce_workflow
        self._workflow_pipeline_tools = workflow_pipeline_tools or set()
        self._workflow_retrieval_tool = workflow_retrieval_tool
        self._workflow_router = workflow_router
        self._workflow_query_builder = workflow_query_builder

    def run(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]
        trace: list[ToolTraceItem] = []

        for _step in range(self._max_steps):
            stage = self._workflow_stage(trace)
            request_kwargs = self._completion_kwargs(messages, stage)
            response = self._client.chat.completions.create(**request_kwargs)
            msg = response.choices[0].message

            if not getattr(msg, "tool_calls", None):
                if self._enforce_workflow and stage == "pipeline":
                    if self._invoke_routed_pipeline(user_input, context, trace, messages):
                        continue
                    return AgentResult(
                        text=(
                            "Cannot identify modality. Provide a SMILES, .fif/.edf "
                            "path, or NIfTI directory."
                        ),
                        trace=trace,
                        model=self._model,
                        finish_reason="error",
                    )
                if self._enforce_workflow and stage == "retrieve":
                    if self._invoke_fallback_retrieval(user_input, context, trace, messages):
                        continue
                    return AgentResult(
                        text="Pipeline completed, but retrieval could not be executed.",
                        trace=trace,
                        model=self._model,
                        finish_reason="error",
                    )
                return AgentResult(
                    text=(msg.content or "").strip(),
                    trace=trace,
                    model=self._model,
                    finish_reason="complete",
                )

            selected_tool_calls = self._select_tool_calls(msg.tool_calls, stage)
            if self._enforce_workflow and not selected_tool_calls:
                if stage == "pipeline":
                    if self._invoke_routed_pipeline(user_input, context, trace, messages):
                        continue
                    return AgentResult(
                        text=(
                            "Cannot identify modality. Provide a SMILES, .fif/.edf "
                            "path, or NIfTI directory."
                        ),
                        trace=trace,
                        model=self._model,
                        finish_reason="error",
                    )
                if stage == "retrieve":
                    if self._invoke_fallback_retrieval(user_input, context, trace, messages):
                        continue
                    return AgentResult(
                        text="Pipeline completed, but retrieval could not be executed.",
                        trace=trace,
                        model=self._model,
                        finish_reason="error",
                    )

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in selected_tool_calls],
            })

            for tc in selected_tool_calls:
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

    def _completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        stage: str,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if not self._enforce_workflow:
            kwargs["tools"] = self._tool_schemas
            kwargs["tool_choice"] = "auto"
            return kwargs

        schemas = self._schemas_for_stage(stage)
        if schemas:
            kwargs["tools"] = schemas
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _schemas_for_stage(self, stage: str) -> list[dict[str, Any]]:
        if stage == "pipeline":
            return [
                self._tool_schemas_by_name[name]
                for name in sorted(self._workflow_pipeline_tools)
                if name in self._tool_schemas_by_name
            ]
        if stage == "retrieve" and self._workflow_retrieval_tool:
            schema = self._tool_schemas_by_name.get(self._workflow_retrieval_tool)
            return [schema] if schema else []
        return []

    def _workflow_stage(self, trace: list[ToolTraceItem]) -> str:
        if not self._enforce_workflow:
            return "open"
        has_pipeline = any(
            t.name in self._workflow_pipeline_tools and t.result is not None and t.error is None
            for t in trace
        )
        if not has_pipeline:
            return "pipeline"
        has_retrieval = any(
            t.name == self._workflow_retrieval_tool and t.result is not None and t.error is None
            for t in trace
        )
        return "final" if has_retrieval else "retrieve"

    def _select_tool_calls(self, tool_calls: list[Any], stage: str) -> list[Any]:
        if not self._enforce_workflow:
            return list(tool_calls)
        if stage == "pipeline":
            for tc in tool_calls:
                if tc.function.name in self._workflow_pipeline_tools:
                    return [tc]
            for tc in tool_calls:
                logger.info(
                    "dropped out-of-stage tool call: name=%s stage=%s",
                    tc.function.name,
                    stage,
                )
            return []
        if stage == "retrieve":
            for tc in tool_calls:
                if tc.function.name == self._workflow_retrieval_tool:
                    return [tc]
            for tc in tool_calls:
                logger.info(
                    "dropped out-of-stage tool call: name=%s stage=%s",
                    tc.function.name,
                    stage,
                )
            return []
        for tc in tool_calls:
            logger.info(
                "dropped out-of-stage tool call: name=%s stage=%s",
                tc.function.name,
                stage,
            )
        return []

    def _invoke_routed_pipeline(
        self,
        user_input: str,
        context: dict[str, Any] | None,
        trace: list[ToolTraceItem],
        messages: list[dict[str, Any]],
    ) -> bool:
        if self._workflow_router is None:
            return False
        routed = self._workflow_router(user_input, context)
        if routed is None:
            return False
        name, args = routed
        tool = self._tools_by_name.get(name)
        if tool is None:
            trace.append(ToolTraceItem(name=name, args=args, error=f"unknown tool: {name}"))
            return False
        try:
            result = tool.invoke(args)
            trace.append(ToolTraceItem(name=name, args=args, result=result))
            messages.append({
                "role": "user",
                "content": (
                    "Workflow guard executed the required pipeline tool. "
                    f"Tool: {name}. Result: {json.dumps(result, default=str)}. "
                    "Now call retrieve_context with a focused scientific query."
                ),
            })
            return True
        except Exception as e:
            trace.append(ToolTraceItem(name=name, args=args, error=str(e)))
            return False

    def _invoke_fallback_retrieval(
        self,
        user_input: str,
        context: dict[str, Any] | None,
        trace: list[ToolTraceItem],
        messages: list[dict[str, Any]],
    ) -> bool:
        if self._workflow_retrieval_tool is None or self._workflow_query_builder is None:
            return False
        pipeline_trace = next(
            (
                t for t in trace
                if t.name in self._workflow_pipeline_tools and t.result is not None and t.error is None
            ),
            None,
        )
        if pipeline_trace is None:
            return False
        tool = self._tools_by_name.get(self._workflow_retrieval_tool)
        if tool is None:
            return False
        query = self._workflow_query_builder(user_input, pipeline_trace, context)
        args = {"query": query, "k": 4}
        try:
            result = tool.invoke(args)
            trace.append(ToolTraceItem(
                name=self._workflow_retrieval_tool,
                args=args,
                result=result,
            ))
            messages.append({
                "role": "user",
                "content": (
                    "Workflow guard executed retrieve_context. "
                    f"Result: {json.dumps(result, default=str)}. "
                    "Now synthesize the final answer in the user's language."
                ),
            })
            return True
        except Exception as e:
            trace.append(ToolTraceItem(
                name=self._workflow_retrieval_tool,
                args=args,
                error=str(e),
            ))
            return False
