"""
chatbot.py — the week-4 AT&T assistant, now MCP-backed.

Keeps the week-4 behaviour that matters:
  * guardrails — the system prompt returns "NO-OP" for dangerous or health
    requests, checked before any tool runs
  * session memory — plain text turns only

What changed: tools arrive from the MCP server instead of a local registry.

Session memory deliberately stores plain text, never tool_use/tool_result blocks.
Anthropic requires every tool_use to be answered by a matching tool_result in the
next turn, so if the memory cap trimmed between them the next call would 400.
Keeping the tool round-trip local to one get_answer() avoids that entirely.
"""

from __future__ import annotations

import os

from shared.llm import run_tool_loop

GUARDRAIL_TOKEN = "NO-OP"
GUARDRAIL_REPLY = "Sorry, I can't help with that."
EMPTY_REPLY = "Sorry, can't answer that yet."

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.md")


class SessionMemory:
    def __init__(self, max_messages: int = 20):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get(self) -> list[dict]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)


class ChatBot:
    def __init__(self, mcp_client, verbose: bool = False):
        self.mcp = mcp_client
        self.verbose = verbose
        self.session = SessionMemory()
        with open(_PROMPT_PATH, "r", encoding="utf-8") as handle:
            self.system_prompt = handle.read()

    async def get_answer(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return EMPTY_REPLY

        messages = self.session.get()
        messages.append({"role": "user", "content": query})

        answer = await run_tool_loop(
            self.mcp, self.system_prompt, messages, verbose=self.verbose
        )

        # Guardrail: the model returns the bare token for refused requests.
        if answer.strip() == GUARDRAIL_TOKEN:
            return GUARDRAIL_REPLY

        if not answer:
            answer = EMPTY_REPLY

        self.session.add("user", query)
        self.session.add("assistant", answer)
        return answer

    def reset(self) -> None:
        self.session.clear()

    def status(self) -> str:
        return (
            f"tools (via MCP)={len(self.mcp.tools)} | "
            f"session messages={len(self.session)}"
        )
