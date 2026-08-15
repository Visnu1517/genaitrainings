"""
llm_client.py
-------------
Wrapper around the Anthropic Messages API.

Same idea as the class example, with one deliberate change: the system prompt is
passed via the `system=` parameter instead of being concatenated onto the front of
the user's message (`prompt + user_prompt`). Keeping instructions in `system` and
the conversation in `messages` is what makes chat history — and therefore session
memory — possible, and it's the pattern Anthropic documents.

The API key is read from the ANTHROPIC_API_KEY environment variable. Never
hard-code it; use a .env file and keep it out of git.
"""

from __future__ import annotations

import os


class LLMClient:
    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 1000):
        # Imported here rather than at module level so the rest of the package
        # (and the offline tests, which inject a fake LLM) works without it.
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

    def generate(self, messages: list[dict], system: str | None = None) -> str:
        """
        Send the conversation and return the model's plain-text reply.

        messages : [{"role": "user"|"assistant", "content": str}, ...]
                   Must start with a user turn and alternate roles.
        system   : the grounding + guardrails prompt.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system if system else self._anthropic.NOT_GIVEN,
            messages=messages,
        )
        return response.content[0].text
