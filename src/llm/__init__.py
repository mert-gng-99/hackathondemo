"""LLM-backed natural-language explainers (Day 7).

`explain()` is the ONLY public entry point. It guarantees a non-empty
rationale every call: tries OpenRouter when available, falls back to a
deterministic template otherwise. The deterministic path is the source
of truth for tests; the LLM path is gated behind env config.
"""
from src.llm.explainer import ExplainPayload, ExplainResult, explain  # noqa: F401
