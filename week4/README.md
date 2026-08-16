# Week 4 — Tool Calling with Pydantic Validation

Adds **agentic tool calling** to the week-3 AT&T assistant. The bot can now look
things up — coverage, plans, outages, balances, stores — instead of only reciting
documents. Every tool argument the model generates is validated with **Pydantic**
before any Python code runs.

Grounding, guardrails, session memory and long-term memory from week 3 all still work.

## The tool-calling loop

```
user prompt
   → model sees the tool specs
   → model selects a tool and generates arguments
   → PYDANTIC VALIDATES THE ARGUMENTS        ← the gate
   → Python function executes
   → result (or structured error) goes back to the model
   → repeat until the model stops asking, then final answer
```

The model never runs your code. It only decides *which* tool and *what arguments*.

## Tools

| Tool | Purpose | Validation highlights |
|---|---|---|
| `check_service_availability` | Fiber/internet/wireless at a ZIP | 5-digit ZIP pattern, service enum |
| `get_plan_details` | Plan pricing and features | plan enum, `num_lines` 1–10 |
| `check_outage_status` | Known outages in an area | 5-digit ZIP pattern |
| `lookup_account_balance` | Balance, due date, autopay | `ACC-######` pattern + normalisation |
| `find_nearest_store` | Retail stores near a ZIP | ZIP pattern, radius 1–50 miles |
| `search_web` | Mocked web search *(from the class demo)* | query length 3–200, blank rejected |
| `get_oncall_engineer` | Internal on-call lookup *(from the class demo)* | service enum |

The last two are carried over from the reference demo so the two styles sit side by side.

## What this does differently from the reference demo

**1. One source of truth for schemas.** The reference keeps Pydantic models in
`schemas.py` *and* hand-written JSON in `tool_specs.py` — the same parameters
described twice, free to drift apart. Here the `@tool` decorator derives the JSON
Schema from the Pydantic model via `model_json_schema()`. There is no second file.

```python
@tool("check_outage_status", "Check for network outages...", OutageStatusInput)
def check_outage_status(args: OutageStatusInput) -> dict:
    ...   # args is already validated
```

**2. Validation that actually rejects things.** The reference models declare bare
`str` fields, so nothing is ever refused. These carry patterns, enums, ranges and
custom validators — because an LLM will confidently invent `"account_id": "12345"`.

**3. Errors become data, not crashes.** The reference raises on an unknown tool
(`KeyError`), a wrong argument name (`TypeError`), or an invalid value
(`ValidationError`). Here all three are caught and returned to the model as a
`ToolResult` with `ok: false`, so it can recover:

```
You: what's the balance on account 12345?
  [validation failed] account_id: String should match pattern '^ACC-\d{6}$'
Bot: That account ID doesn't look right — it should be in the format ACC-100001.
```

**4. A real loop.** The reference executes one round of tools then makes a final
call. This loops until the model stops requesting tools, capped by
`MAX_TOOL_ROUNDS = 5` so a runaway agent can't hang.

## Anthropic vs OpenAI tool format

The reference uses the OpenAI SDK; this uses Anthropic, to stay consistent with
week 3. The differences are not cosmetic:

| | OpenAI (reference) | Anthropic (here) |
|---|---|---|
| Tool spec | `{"type":"function","function":{...,"parameters"}}` | `{"name","description","input_schema"}` |
| Model requests a tool | `message.tool_calls[]` | content blocks with `type == "tool_use"` |
| Detecting it | truthy `tool_calls` | `stop_reason == "tool_use"` |
| Arguments | JSON string → `json.loads()` | already a dict in `block.input` |
| Returning a result | `{"role":"tool","tool_call_id":...}` | `{"role":"user","content":[{"type":"tool_result","tool_use_id":...}]}` |
| System prompt | a message in the array | separate `system=` parameter |

## Project layout

```
week4/
├── main.py                    # interactive chat (--verbose for tool tracing)
├── test_chatbot.py            # 23 offline tests, no API key needed
├── requirements.txt
├── .env.example
├── .gitignore
├── prompts/
│   └── system_prompt.md       # RTF + guardrails + tool-use rules
├── docs/                      # RAG knowledge base
│   ├── att_consumer.md
│   └── att_support.md
└── chatbot/
    ├── __init__.py
    ├── schemas.py             # Pydantic input models + ToolResult envelope
    ├── mock_data.py           # stand-in AT&T data
    ├── tools.py               # the tool functions
    ├── tool_registry.py       # @tool decorator → Anthropic specs
    ├── tool_executor.py       # validation + error handling
    ├── llm_client.py          # Anthropic Messages API (tool-aware)
    ├── embeddings.py          # Ollama nomic-embed-text, with fallback
    ├── vector_store.py        # FAISS index
    ├── memory.py              # SessionMemory + LongTermMemory
    └── chatbot.py             # the agentic loop
```

## Setup

```bash
cd week4
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."
ollama pull nomic-embed-text             # optional; falls back automatically
```

## Run

```bash
python main.py --verbose
```

Commands: `/tools`, `/status`, `/verbose`, `/reset`, `/remember <fact>`, `/forget`, `/quit`

### Demo prompts

```
Tool calling      Is fiber available in 75201?
                  How much is Unlimited Extra for 3 lines?
                  What's the balance on account ACC-100001?

Validation        What's the balance on account 12345?     → model asks you to fix it
                  Is fiber available in ZIP 752?

Grounding         What is AT&T Fiber?                      → answered from docs, no tool

Guardrails        I have a headache, what should I take?   → refused

Memory            my name is Vishnu   → /quit → restart → what is my name?
```

## Tests

```bash
python test_chatbot.py
```

23 tests, no API key or Ollama required. Coverage: valid input and defaults,
malformed ZIP, bad account ID, out-of-range numbers, invalid enums, missing required
fields, hallucinated tool names, the executor never raising, correct tool maths and
filtering, auto-generated schemas, single/multi/parallel tool rounds, the
`tool_result` message shape, validation errors reaching the model as `is_error`, the
loop cap, guardrails blocking before tools run, and all week-3 memory behaviour.

## Design decisions

**Session memory stores plain text only, never `tool_use` / `tool_result` blocks.**
Anthropic requires every `tool_use` block to be answered by a matching `tool_result`
in the next turn. If the session cap trimmed history between the two, the next API
call would 400. Keeping the tool round-trip local to one `get_answer()` call avoids
that class of bug entirely.

**Guardrails are checked before any tool executes.** A health or dangerous request is
refused on the first response, so no lookup ever runs and nothing is written to memory.

**The grounding rule had to be relaxed.** Week 3's prompt said "answer only using the
Grounding Context." Left unchanged, the model would refuse to use its own tool
results. It now reads "the Grounding Context **or the results returned by your
tools**" — without that edit, tools silently do nothing.

**Long-term memory stores the final answer, not raw tool JSON**, so retrieval isn't
polluted with unreadable dicts.
