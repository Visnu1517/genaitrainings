"""
llm_client.py
-------------
Wrapper around the Anthropic Messages API, now tool-aware.

Two methods:
    create()   - returns the full response object, so the caller can inspect
                 tool_use blocks and stop_reason. Used by the agentic loop.
    generate() - convenience wrapper returning just the text.

Anthropic's tool format differs from the OpenAI format used in the class
reference demo:

    tool spec      {"name", "description", "input_schema"}
                   vs OpenAI's {"type":"function","function":{...,"parameters"}}
    model requests content blocks with type == "tool_use" (stop_reason == "tool_use")
                   vs OpenAI's message.tool_calls
    arguments      already a dict in block.input
                   vs OpenAI's JSON string needing json.loads()
    sending result {"role":"user","content":[{"type":"tool_result","tool_use_id",...}]}
                   vs OpenAI's {"role":"tool","tool_call_id",...}
    system prompt  a separate `system=` parameter, not a message

The API key is read from ANTHROPIC_API_KEY. Never hard-code it.
"""

from __future__ import annotations

import os

# Load variables from a .env file in the project root, if present, so
# ANTHROPIC_API_KEY can be set there instead of exported manually every time.
# Safe to skip if python-dotenv isn't installed - the key can still be set
# as a normal environment variable.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LLMClient:
    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 1000):
        # Imported lazily so the package (and the offline tests, which inject a
        # fake client) works without the SDK installed.
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is missing. Run: pip install -r requirements.txt"
            ) from exc
        self._anthropic = anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  macOS/Linux : export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  Windows     : setx ANTHROPIC_API_KEY \"sk-ant-...\"  (then reopen the terminal)"
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    def create(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        """Return the raw response object (needed to read tool_use blocks)."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system": system if system else self._anthropic.NOT_GIVEN,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)

    # ------------------------------------------------------------------ #
    def generate(self, messages: list[dict], system: str | None = None) -> str:
        """Convenience: return only the text of a non-tool response."""
        return extract_text(self.create(messages, system=system))


def extract_text(response) -> str:
    """Join every text block in a response into one string."""
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
