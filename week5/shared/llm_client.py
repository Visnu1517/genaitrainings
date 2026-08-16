"""
llm_client.py
-------------
Anthropic client shared by all three tasks.

The class reference used the OpenAI SDK pointed at a local Copilot proxy; this
uses Claude directly, staying consistent with weeks 3 and 4.

Exposes one helper, `chat()`, for plain system+user calls that don't need tools.
Tool-calling lives in agent_loop.py.

The key is read from ANTHROPIC_API_KEY (a .env file is loaded automatically).
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

MODEL_NAME = os.getenv("MODEL_NAME", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

_client = None


def get_client():
    """Create the Anthropic client on first use, with a clear error if unset."""
    global _client
    if _client is not None:
        return _client

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'anthropic' package is missing. Run: pip install -r requirements.txt"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            "  Create a .env file containing:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  or set it as an environment variable."
        )

    _client = anthropic.Anthropic()
    return _client


def extract_text(response) -> str:
    """Join every text block of an Anthropic response into one string."""
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()


def chat(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    """A single system + user call with no tools. Returns the text reply."""
    response = get_client().messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens or MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return extract_text(response)
