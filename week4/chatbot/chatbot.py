"""
chatbot.py
----------
The AT&T assistant. Week 3 gave it grounding, guardrails and memory; week 4 adds
tool calling, so it can look things up instead of only reciting documents.

Full flow for one get_answer(query):

    1. GUARDRAILS   - the system prompt makes the model reply with exactly "NO-OP"
                      for dangerous or health requests. Checked on the first
                      response, before any tool runs.
    2. RETRIEVAL    - relevant AT&T chunks (knowledge store) and user memories
                      (long-term store) are pulled and injected into the system
                      prompt.
    3. AGENTIC LOOP - the model may request tools. We validate the arguments,
                      execute, feed results back, and repeat until it stops asking
                      (capped by MAX_TOOL_ROUNDS).
    4. MEMORY       - the final answer is saved to session and long-term memory.

Two design decisions worth calling out:

* Session memory stores only plain user/assistant text turns, never the raw
  tool_use / tool_result blocks. Anthropic requires every tool_use block to be
  answered by a matching tool_result in the next turn, so keeping those blocks in
  trimmable history risks a 400 error the moment the cap chops between them. The
  tool round-trip stays local to a single get_answer() call.

* Long-term memory stores the final answer, not raw tool JSON, so retrieval isn't
  polluted with unreadable dicts.
"""

from __future__ import annotations

import json
import os

from .embeddings import Embedder
from .llm_client import LLMClient, extract_text
from .memory import LongTermMemory, SessionMemory
from .tool_executor import execute_tool
from .tool_registry import all_specs
from .vector_store import VectorStore

# Importing tools registers them via the @tool decorator. Keep this import even
# though the names aren't referenced directly here.
from . import tools as _tools  # noqa: F401

GUARDRAIL_TOKEN = "NO-OP"
GUARDRAIL_REPLY = "Sorry, I can't help with that."
EMPTY_REPLY = "Sorry, can't answer that yet."

KNOWLEDGE_K = 3
MEMORY_K = 3
MIN_SCORE = 0.15

# Safety valve: stop an agent that keeps calling tools forever.
MAX_TOOL_ROUNDS = 5


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChatBot:
    def __init__(
        self,
        prompt_path: str | None = None,
        docs_path: str | None = None,
        data_dir: str | None = None,
        llm=None,
        prefer_ollama: bool = True,
        verbose: bool = False,
    ):
        root = _project_root()
        prompt_path = prompt_path or os.path.join(root, "prompts", "system_prompt.md")
        docs_path = docs_path or os.path.join(root, "docs")
        data_dir = data_dir or os.path.join(root, "data")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

        self.verbose = verbose
        self.embedder = Embedder(prefer_ollama=prefer_ollama)

        self.knowledge = VectorStore(os.path.join(data_dir, "knowledge"), self.embedder)
        self.ingested = self.knowledge.ingest_folder(docs_path)

        self.long_term = LongTermMemory(
            VectorStore(os.path.join(data_dir, "user_memory"), self.embedder),
            top_k=MEMORY_K,
            min_score=MIN_SCORE,
        )
        self.session = SessionMemory()
        self.llm = llm if llm is not None else LLMClient()
        self.tools = all_specs()

        # Populated on each call so the CLI/tests can show what happened.
        self.last_tool_calls: list[dict] = []

    # ------------------------------------------------------------------ #
    def _build_system_prompt(self, query: str) -> str:
        parts = [self.base_prompt]

        hits = self.knowledge.search(query, k=KNOWLEDGE_K, min_score=MIN_SCORE)
        if hits:
            block = "\n\n".join(h["content"] for h in hits)
            parts.append(
                "\n\n## Grounding Context (retrieved)\n"
                "Use the information below for AT&T background facts.\n\n"
                f"{block}\n"
            )

        memories = self.long_term.recall(query)
        if memories:
            lines = "\n".join(f"- {m}" for m in memories)
            parts.append(
                "\n\n## Long-term memory (from earlier conversations with this user)\n"
                f"{lines}\n"
            )

        return "".join(parts)

    # ------------------------------------------------------------------ #
    def _run_tool_round(self, response, messages: list[dict]) -> None:
        """Execute every tool_use block in `response` and append the results."""
        tool_results = []

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            if self.verbose:
                print(f"\n  MODEL SELECTED TOOL: {block.name}")
                print(f"  MODEL GENERATED ARGUMENTS: {json.dumps(block.input)}")

            result = execute_tool(block.name, block.input, verbose=self.verbose)

            self.last_tool_calls.append(
                {
                    "tool": block.name,
                    "arguments": block.input,
                    "ok": result.ok,
                    "error": result.error,
                }
            )

            if self.verbose:
                print(f"  BACKEND TOOL RESULT: {result.to_json()}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.to_json(),
                    # Flagging errors helps the model notice it must correct itself.
                    "is_error": not result.ok,
                }
            )

        # All tool results for a turn go back in a single user message.
        messages.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------ #
    def get_answer(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return EMPTY_REPLY

        self.last_tool_calls = []
        system_prompt = self._build_system_prompt(query)

        messages = self.session.get_messages()
        messages.append({"role": "user", "content": query})

        answer = ""
        for round_num in range(MAX_TOOL_ROUNDS):
            response = self.llm.create(messages, system=system_prompt, tools=self.tools)
            text = extract_text(response)

            # Guardrail check happens before any tool executes.
            if text == GUARDRAIL_TOKEN:
                return GUARDRAIL_REPLY

            # Record the assistant turn (including tool_use blocks) so the next
            # API call has the context it needs.
            messages.append({"role": "assistant", "content": response.content})

            if getattr(response, "stop_reason", None) != "tool_use":
                answer = text
                break

            self._run_tool_round(response, messages)
        else:
            # Loop finished without the model settling on an answer.
            answer = (
                "I wasn't able to finish that request — it required too many lookups. "
                "Could you narrow it down?"
            )

        if not answer:
            answer = EMPTY_REPLY

        # Store plain text only, never tool blocks.
        self.session.add("user", query)
        self.session.add("assistant", answer)
        self.long_term.remember_exchange(query, answer)
        return answer

    # ------------------------------------------------------------------ #
    def reset_session(self) -> None:
        self.session.clear()

    def forget(self) -> None:
        self.long_term.clear()

    def status(self) -> str:
        return (
            f"embeddings={self.embedder.backend} | "
            f"tools={len(self.tools)} | "
            f"knowledge chunks={len(self.knowledge.chunks)} | "
            f"memories={len(self.long_term)} | "
            f"session messages={len(self.session)}"
        )
