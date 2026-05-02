"""Probe OpenRouter for which free-tier model IDs are reachable today.

Reads OPENROUTER_API_KEY from .env (or process env). Issues a single
8-token chat completion against a candidate list and prints one line per
model: status (OK / HTTP-code / exception name) + a 30-char preview of
the response when OK.

Use:
    python scripts/diagnose_openrouter.py

Or to probe a custom list:
    python scripts/diagnose_openrouter.py google/gemma-2-9b-it:free meta-llama/llama-3.2-3b-instruct:free
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Manually parse .env without python-dotenv (some envs choke on its
# frame-introspection in heredocs / non-stack-rooted callers).
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for raw in _env_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("OPENROUTER_API_KEY not set (looked in env and .env)")

# Candidate list: well-known stable free-tier IDs as of 2026-Q2.
# Update by replacing this list — script is a probe, not a config source.
DEFAULT_CANDIDATES = [
    "google/gemma-2-9b-it:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

candidates = sys.argv[1:] or DEFAULT_CANDIDATES

from openai import (  # noqa: E402  (after env load)
    OpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError,
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=15.0,
)

for m in candidates:
    try:
        c = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=8,
            temperature=0,
        )
        text = (c.choices[0].message.content or "").strip()
        print(f"  OK     {m}  →  {text[:30]!r}")
    except APIStatusError as e:
        code = getattr(e, "status_code", "?")
        print(f"  {code:<5}  {m}")
    except RateLimitError:
        print(f"  429    {m}  (rate-limited)")
    except (APIConnectionError, APITimeoutError) as e:
        print(f"  CONN   {m}  ({type(e).__name__})")
    except Exception as e:
        print(f"  ERR    {m}  ({type(e).__name__}: {e})")
