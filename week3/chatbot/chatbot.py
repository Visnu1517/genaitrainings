"""
chatbot.py
----------
The AT&T assistant. Keeps the get_answer() interface from the week-2 FAQ bot, but
every answer now flows through four layers:

    1. GUARDRAILS  - the system prompt tells the model to reply with exactly
                     "NO-OP" for dangerous or health-related requests. We catch
                     that token and return a refusal. Blocked turns are never
                     written to memory.

    2. GROUNDING   - instead of pasting a fixed grounding.md into every prompt,
                     we retrieve only the AT&T chunks relevant to this question
                     from a FAISS knowledge store, and inject those. This is RAG,
                     and it scales past what fits in a prompt.

    3. MEMORY      - relevant long-term memories are retrieved and injected, and
                     the current conversation is sent as message history
                     (session memory).

    4. LLM         - Claude generates the grounded answer.

Flow:

    query
      -> retrieve knowledge chunks      (VectorStore: docs/)
      -> retrieve user memories         (VectorStore: user memory)
      -> system prompt = base + grounding + memory
      -> messages = session history + query
      -> LLM
      -> if "NO-OP": refuse, store nothing
         else: save to session memory + long-term memory
"""

from __future__ import annotations

import os

from .embeddings import Embedder
from .llm_client import LLMClient
from .memory import LongTermMemory, SessionMemory
from .vector_store import VectorStore

GUARDRAIL_TOKEN = "NO-OP"
GUARDRAIL_REPLY = "Sorry, I can't help with that."
EMPTY_REPLY = "Sorry, can't answer that yet."

# How many chunks to pull from each store.
KNOWLEDGE_K = 3
MEMORY_K = 3
# Cosine-similarity floor. Below this a "match" is usually noise, and injecting
# noise into the prompt is worse than injecting nothing.
MIN_SCORE = 0.15


def _project_root() -> str:
    """The week3/ folder — one level above this package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChatBot:
    def __init__(
        self,
        prompt_path: str | None = None,
        docs_path: str | None = None,
        data_dir: str | None = None,
        llm=None,
        prefer_ollama: bool = True,
    ):
        root = _project_root()
        prompt_path = prompt_path or os.path.join(root, "prompts", "system_prompt.md")
        docs_path = docs_path or os.path.join(root, "docs")
        data_dir = data_dir or os.path.join(root, "data")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

        # One embedder shared by both stores so their vectors are comparable.
        self.embedder = Embedder(prefer_ollama=prefer_ollama)

        # Store 1: curated AT&T knowledge (read-only, ingested from docs/).
        self.knowledge = VectorStore(os.path.join(data_dir, "knowledge"), self.embedder)
        self.ingested = self.knowledge.ingest_folder(docs_path)

        # Store 2: things learned about this user (written at runtime).
        self.long_term = LongTermMemory(
            VectorStore(os.path.join(data_dir, "user_memory"), self.embedder),
            top_k=MEMORY_K,
            min_score=MIN_SCORE,
        )

        self.session = SessionMemory()
        # llm can be injected for offline testing.
        self.llm = llm if llm is not None else LLMClient()

    # ------------------------------------------------------------------ #
    def _build_system_prompt(self, query: str) -> str:
        """Base prompt + retrieved AT&T grounding + retrieved user memory."""
        parts = [self.base_prompt]

        hits = self.knowledge.search(query, k=KNOWLEDGE_K, min_score=MIN_SCORE)
        if hits:
            block = "\n\n".join(h["content"] for h in hits)
            parts.append(
                "\n\n## Grounding Context (retrieved)\n"
                "Answer AT&T questions using only the information below.\n\n"
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
    def get_answer(self, query: str) -> str:
        """Answer one question, using and updating memory."""
        query = (query or "").strip()
        if not query:
            return EMPTY_REPLY

        system_prompt = self._build_system_prompt(query)

        messages = self.session.get_messages()
        messages.append({"role": "user", "content": query})

        answer = self.llm.generate(messages, system=system_prompt).strip()

        # Guardrail fired -> refuse, and deliberately store nothing.
        if answer == GUARDRAIL_TOKEN:
            return GUARDRAIL_REPLY

        self.session.add("user", query)
        self.session.add("assistant", answer)
        self.long_term.remember_exchange(query, answer)
        return answer

    # ------------------------------------------------------------------ #
    def reset_session(self) -> None:
        """Start a new conversation; long-term memory is untouched."""
        self.session.clear()

    def forget(self) -> None:
        """Erase everything learned about the user. Knowledge base is untouched."""
        self.long_term.clear()

    def status(self) -> str:
        return (
            f"embeddings={self.embedder.backend} | "
            f"knowledge chunks={len(self.knowledge.chunks)} | "
            f"memories={len(self.long_term)} | "
            f"session messages={len(self.session)}"
        )
