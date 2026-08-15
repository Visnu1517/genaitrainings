"""
test_chatbot.py
---------------
Offline tests. They use a FAKE LLM, so they run without an API key and without
Ollama — proving the memory, guardrail, and retrieval plumbing works.

Run:
    python test_chatbot.py
"""

import os
import shutil
import sys

from chatbot import ChatBot
from chatbot.embeddings import Embedder
from chatbot.memory import SessionMemory
from chatbot.vector_store import VectorStore, chunk_text

TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_test")


class FakeLLM:
    """Records what it was sent and replies from a simple script."""

    def __init__(self, script=None):
        self.script = script or {}
        self.last_system = None
        self.last_messages = None

    def generate(self, messages, system=None):
        self.last_system = system
        self.last_messages = messages
        text = messages[-1]["content"].lower()
        for key, reply in self.script.items():
            if key in text:
                return reply
        return "OK."


def fresh_bot(script=None):
    shutil.rmtree(TEST_DATA, ignore_errors=True)
    return ChatBot(llm=FakeLLM(script), data_dir=TEST_DATA, prefer_ollama=False)


def reopen_bot(script=None):
    """Same data dir = simulates restarting the program."""
    return ChatBot(llm=FakeLLM(script), data_dir=TEST_DATA, prefer_ollama=False)


# --------------------------------------------------------------------------- #
def test_chunking():
    text = " ".join(str(i) for i in range(1200))
    chunks = chunk_text(text, word_count=512, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 512 for c in chunks)
    # overlap means consecutive chunks share words
    assert set(chunks[0].split()) & set(chunks[1].split())
    print("PASS  chunking produces overlapping windows")


def test_vector_store_retrieves_correct_chunk():
    """The class template's bug returned chunk 0 every time. Ours must not."""
    shutil.rmtree(TEST_DATA, ignore_errors=True)
    store = VectorStore(os.path.join(TEST_DATA, "vs"), Embedder(prefer_ollama=False))
    store.add_text("Paris is the capital city of France", doc="a")
    store.add_text("AT&T Fiber provides high speed home internet", doc="b")
    store.add_text("Bananas are a yellow tropical fruit", doc="c")

    hits = store.search("tell me about fiber internet", k=1)
    assert hits, "no results returned"
    assert "Fiber" in hits[0]["content"], f"wrong chunk retrieved: {hits[0]['content']}"
    print("PASS  vector store returns the RELEVANT chunk (not just the first)")


def test_knowledge_ingested_and_grounding_injected():
    bot = fresh_bot()
    assert len(bot.knowledge.chunks) > 0, "docs/ was not ingested"
    bot.get_answer("what home internet does AT&T offer?")
    assert "Grounding Context (retrieved)" in bot.llm.last_system
    assert "Fiber" in bot.llm.last_system
    print("PASS  AT&T knowledge ingested and injected as grounding")


def test_session_memory_sends_history():
    bot = fresh_bot()
    bot.get_answer("what internet plans do you have?")
    bot.get_answer("is that available everywhere?")
    roles = [m["role"] for m in bot.llm.last_messages]
    assert roles == ["user", "assistant", "user"], roles
    assert bot.llm.last_messages[-1]["content"] == "is that available everywhere?"
    print("PASS  session memory sends conversation history to the model")


def test_session_memory_is_capped():
    sm = SessionMemory(max_messages=4)
    for i in range(10):
        sm.add("user", f"msg {i}")
    assert len(sm) == 4
    assert sm.get_messages()[-1]["content"] == "msg 9"
    print("PASS  session memory caps old turns")


def test_guardrail_blocks_and_does_not_store():
    bot = fresh_bot(script={"headache": "NO-OP"})
    answer = bot.get_answer("I have a headache, what medicine should I take?")
    assert answer == "Sorry, I can't help with that.", answer
    assert len(bot.session) == 0, "blocked turn leaked into session memory"
    assert len(bot.long_term) == 0, "blocked turn leaked into long-term memory"
    print("PASS  guardrail NO-OP refuses and stores nothing")


def test_long_term_memory_survives_restart():
    bot = fresh_bot()
    bot.get_answer("my name is Vishnu")
    assert len(bot.long_term) > 0

    bot2 = reopen_bot()          # restart
    bot2.get_answer("what is my name?")
    assert "Vishnu" in bot2.llm.last_system, "memory not recalled after restart"
    assert "Long-term memory" in bot2.llm.last_system
    print("PASS  long-term memory persists across restart and is recalled")


def test_fact_detection():
    bot = fresh_bot()
    bot.get_answer("my name is Vishnu")
    docs = [c["doc"] for c in bot.long_term.store.chunks]
    assert "user_fact" in docs, "personal fact was not stored separately"
    bot.get_answer("what is 2 plus 2?")
    print("PASS  personal facts stored separately from ordinary exchanges")


def test_forget_clears_memory_not_knowledge():
    bot = fresh_bot()
    bot.get_answer("my name is Vishnu")
    knowledge_before = len(bot.knowledge.chunks)
    bot.forget()
    assert len(bot.long_term) == 0
    assert len(bot.knowledge.chunks) == knowledge_before
    print("PASS  /forget clears user memory but keeps the knowledge base")


def test_empty_query():
    bot = fresh_bot()
    assert bot.get_answer("   ") == "Sorry, can't answer that yet."
    print("PASS  empty query handled")


def test_ingest_is_cached():
    bot = fresh_bot()
    first = len(bot.knowledge.chunks)
    bot2 = reopen_bot()
    assert len(bot2.knowledge.chunks) == first, "docs were re-ingested (cache broken)"
    assert bot2.ingested == 0
    print("PASS  knowledge ingestion is cached across runs")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(TEST_DATA, ignore_errors=True)
    print(f"\nAll {len(tests)} tests passed.")
