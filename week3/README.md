# Week 3 — AT&T Assistant with Grounding, Guardrails and Memory

Upgrades the week-2 FAQ chatbot into an AT&T assistant. Answers now come from a real
LLM (Claude Haiku 4.5) instead of a dictionary lookup, and the bot has **two kinds of
memory** plus **RAG-based grounding** and **guardrails**.

## What was added

| Layer | What it does | Where |
|---|---|---|
| **Guardrails** | Dangerous and health-related requests return `NO-OP`, which the app turns into a refusal | `prompts/system_prompt.md`, `chatbot/chatbot.py` |
| **Grounding (RAG)** | AT&T facts are retrieved from a FAISS index per question instead of pasting a fixed block into every prompt | `chatbot/vector_store.py`, `docs/` |
| **Session memory** | The running conversation, sent as message history so follow-ups resolve | `chatbot/memory.py` |
| **Long-term memory** | Facts and past exchanges saved to disk and retrieved by similarity — survives restarts | `chatbot/memory.py` |

## Architecture

```
user query
   │
   ├─► retrieve AT&T chunks      ── FAISS knowledge store  (docs/)
   ├─► retrieve user memories    ── FAISS user-memory store (runtime writes)
   │
   ├─► system prompt = base prompt + grounding + memory
   ├─► messages     = session history + this query
   │
   ├─► Claude Haiku 4.5
   │
   ├─► if reply == "NO-OP"  →  refuse, store NOTHING
   └─► else                 →  save to session + long-term memory
```

**Two separate vector stores, deliberately.** Knowledge is curated and read-only;
user memory is written at runtime and must be wipeable. Keeping them apart stops a
user's stray sentence from polluting the AT&T grounding context.

## Project layout

```
week3/
├── main.py                    # interactive chat
├── test_chatbot.py            # offline tests (no API key needed)
├── requirements.txt
├── .env.example
├── .gitignore
├── prompts/
│   └── system_prompt.md       # RTF role/tasks/format + guardrails
├── docs/                      # the AT&T knowledge base (RAG source)
│   ├── att_consumer.md
│   └── att_support.md
└── chatbot/
    ├── __init__.py
    ├── embeddings.py          # Ollama nomic-embed-text, with fallback
    ├── vector_store.py        # FAISS index, chunking, persistence
    ├── memory.py              # SessionMemory + LongTermMemory
    ├── llm_client.py          # Anthropic Messages API wrapper
    └── chatbot.py             # ChatBot.get_answer() — ties it together
```

## Setup

```bash
cd week3
pip install -r requirements.txt

# API key
export ANTHROPIC_API_KEY=sk-ant-...          # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."

# Embeddings (recommended — matches the class RAG template)
# install Ollama, then:
ollama pull nomic-embed-text
```

If Ollama isn't running, the app automatically falls back to a built-in hashing
embedder so it still works — it just matches on shared words rather than meaning.
`/status` shows which backend is active.

## Run

```bash
python main.py
```

Commands: `/status`, `/reset`, `/remember <fact>`, `/forget`, `/quit`

### Demonstrating each feature

**Session memory (short-term)**
```
You: what home internet options does AT&T offer?
You: is that available everywhere?        ← "that" resolves from history
```

**Long-term memory (persists across restarts)**
```
You: my name is Vishnu
You: /quit
$ python main.py
You: what is my name?                     ← still knows
```

**Guardrails**
```
You: I have a headache, what medicine should I take?   → refusal
```

**Grounding**
```
You: who is Narendra Modi?     → answers, then notes it's outside AT&T scope
You: what is AT&T's CEO's salary?  → says it doesn't have that information
```

## Tests

```bash
python test_chatbot.py
```

11 tests covering chunking, retrieval correctness, grounding injection, session
history, memory persistence across restarts, guardrail blocking, and cache behaviour.
They use a fake LLM, so **no API key or Ollama is required to run them**.

## Notes on the RAG implementation

Built from scratch following the ideas in the class template
([ChandrahaasJ/RAG_template](https://github.com/ChandrahaasJ/RAG_template)) — same
Ollama `nomic-embed-text` embeddings, same 512-word/50-overlap chunking, same FAISS
index and JSON caching — with three changes:

1. **Retrieval indexing fixed.** The template's `queryDB()` appends `lst[i]` using the
   loop counter rather than `lst[indices[i]]`, so it returns the first three chunks
   regardless of the query. This version indexes by the position FAISS actually returns.
2. **Incremental writes.** The template only ingests files from a folder at startup.
   Long-term memory needs to save a single fact mid-conversation, so `add_text()` exists
   alongside `ingest_folder()`.
3. **Cosine similarity** (`IndexFlatIP` on normalized vectors) instead of raw L2, which
   behaves better for text, plus a relevance floor so weak matches aren't injected into
   the prompt.
