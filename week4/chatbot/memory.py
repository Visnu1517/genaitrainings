"""
memory.py
---------
The two memory types this assignment asks for.

SessionMemory  (short-term)
    The current conversation, held in RAM as a list of {role, content} messages.
    This is what gets sent to the model as chat history, so follow-up questions
    like "and how much does that cost?" resolve correctly. It disappears when the
    program exits.

LongTermMemory (persistent)
    Facts and past exchanges written to a FAISS vector store on disk. On each new
    question we retrieve only the most relevant memories rather than replaying the
    entire history — that's the RAG part, and it's what lets the bot remember you
    across restarts without blowing up the prompt.

Design note: knowledge (AT&T facts) and user memory live in SEPARATE stores.
They have different lifecycles — knowledge is curated and read-only, user memory
is written at runtime and should be wipeable — and keeping them apart stops a
user's stray sentence from polluting the grounding context.
"""

from __future__ import annotations

import re

# Phrases that signal the user is stating a durable fact about themselves.
# These get saved with higher priority than ordinary chit-chat.
_FACT_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am called\b",
    r"\bcall me\b",
    r"\bi have\b",
    r"\bi'm on\b",
    r"\bi am on\b",
    r"\bmy plan\b",
    r"\bmy account\b",
    r"\bmy number\b",
    r"\bi prefer\b",
    r"\bi live in\b",
    r"\bi use\b",
    r"\bremember that\b",
]


class SessionMemory:
    """Short-term memory: the running conversation."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        # Keep only the most recent turns so the prompt stays a sane size.
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """Persistent memory backed by a vector store."""

    def __init__(self, store, top_k: int = 3, min_score: float = 0.15):
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    # ------------------------------------------------------------------ #
    @staticmethod
    def looks_like_fact(text: str) -> bool:
        """Heuristic: is the user telling us something durable about themselves?"""
        lowered = text.lower()
        return any(re.search(p, lowered) for p in _FACT_PATTERNS)

    # ------------------------------------------------------------------ #
    def remember_exchange(self, query: str, answer: str) -> None:
        """
        Save one turn. Personal facts are stored separately and phrased plainly,
        because a short clean sentence retrieves far better than a whole exchange.
        """
        if self.looks_like_fact(query):
            self.store.add_text(f"User fact: {query.strip()}", doc="user_fact")

        self.store.add_text(
            f"Earlier the user asked: {query.strip()}\nThe assistant answered: {answer.strip()}",
            doc="exchange",
        )

    def remember_fact(self, fact: str) -> None:
        """Explicitly store a fact (used by the /remember command)."""
        self.store.add_text(f"User fact: {fact.strip()}", doc="user_fact")

    # ------------------------------------------------------------------ #
    def recall(self, query: str) -> list[str]:
        """Return the text of the most relevant stored memories for this query."""
        hits = self.store.search(query, k=self.top_k, min_score=self.min_score)
        return [h["content"] for h in hits]

    def clear(self) -> None:
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store.chunks)
