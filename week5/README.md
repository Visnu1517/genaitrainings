# Week 5 — Agentic Workflows: Planning and Reasoning

The **same** complex request answered three ways, to show that richer
orchestration produces better answers:

> Compare AT&T's mobile connection plans with its competitors and make a detailed
> report of how much market capitalization AT&T has compared to its competitors.
> Also give areas where AT&T can capitalize in order to capture more market.

Expected quality: **TASK-3 > TASK-2 > TASK-1**.

## The three architectures

### TASK-1 — one-shot
```
input → LLM calls tools → answer
```
One conversation, one pass at everything. No decomposition, so a request with
three deliverables gets whatever attention the model happens to spread across them.

### TASK-2 — hardcoded workflow
```
input → [ plans ] [ market cap ] [ growth ] → join
```
Better, because each branch focuses on one job. But the three branches were chosen
**at coding time** — that only works because we already knew the question. Ask
something else and the branches are simply wrong. Aggregation is also a plain
join, so nothing reconciles contradictions between sections.

### TASK-3 — planning and reasoning
```
input → PLAN → RESEARCH each question → REASON: enough?
                    ↑                        │ no → research the gaps
                    └────────────────────────┘
                                             │ yes
                                             ↓
                                        SYNTHESIZE
```
Nothing is hardcoded. The planner reads the request and decides the decomposition
at runtime, so the same script handles a question nobody anticipated.

## The loop-back

The class reference describes this behaviour in its PRD:

> reason whether this information is enough → if enough, aggregate → **otherwise
> research what is missing then aggregate**

…but its reasoner prompt says *"Do NOT request more research"*, and the code never
loops. An insufficient result just gets labelled insufficient.

**This version implements it properly.** The reasoner returns a structured verdict:

```json
{"sufficient": false, "gaps": ["What is AT&T's churn rate?"], "notes": "..."}
```

A `false` verdict genuinely sends those gaps back through the researcher, and the
new findings join the rest before synthesis. `MAX_REASONING_ROUNDS = 2` stops a
never-satisfied reasoner from spinning forever.

That loop is what makes this *reasoning* rather than just a longer pipeline.

## Tools

Both read a local markdown file, so nothing needs network or database access:

| Tool | Reads | Contains |
|---|---|---|
| `web_search` | `webresults.md` | Verizon + T-Mobile mock data |
| `query_db` | `query.md` | AT&T mock data |

All figures are fictional, created for this assignment.

## Layout

```
week5/
├── task1_llm_with_tools.py       # one-shot
├── task2_workflow.py             # hardcoded branches
├── task3_planning_reasoning.py   # plan → research → reason → synthesize
├── run_all.py                    # run all three, save to outputs/
├── test_workflow.py              # 19 offline tests, no API key needed
├── query.md, webresults.md       # mock tool data
├── requirements.txt, .env.example, .gitignore
├── Prompts/
│   ├── task1.md
│   ├── task2_compare_plans.md
│   ├── task2_market_cap.md
│   ├── task2_growth_areas.md
│   ├── task3_planner.md          # how to decompose
│   ├── task3_researcher.md       # how to research one question
│   ├── task3_reasoner.md         # how to judge sufficiency
│   └── task3_synthesizer.md      # how to write the final report
└── shared/
    ├── llm_client.py             # Anthropic client
    ├── tools.py                  # the two tools
    ├── tool_specs.py             # Anthropic tool schemas
    ├── tool_executor.py          # dispatch + error handling
    ├── prompt_loader.py          # loads Prompts/*.md
    ├── schemas.py                # Pydantic validation of plan + verdict
    └── agent_loop.py             # multi-round tool loop
```

Prompts live in markdown rather than Python strings on purpose: the planning and
reasoning behaviour lives almost entirely in the prompts, so they should be
editable without touching code.

## Setup

```bash
cd week5
pip install -r requirements.txt
cp .env.example .env        # then paste your real key into .env
```

## Run

```bash
python task1_llm_with_tools.py
python task2_workflow.py
python task3_planning_reasoning.py

python run_all.py                    # all three, saved to outputs/
python run_all.py --prompt "Which carrier has the widest 5G coverage and why?"
```

Running a **custom prompt** is the most revealing test: TASK-2 will still produce
a plans/market-cap/growth report regardless of what you asked, because its
branches are hardcoded. TASK-3 adapts.

## Tests

```bash
python test_workflow.py
```

19 tests, no API key required. Coverage includes tool file reading, unknown-tool
and bad-argument handling, planner output parsed through markdown fences and
surrounding prose, the question cap, multi-round tool looping and its cap, TASK-2's
three fixed branches, TASK-3's happy path, **the loop-back actually feeding gap
research into synthesis**, the reasoning round cap, and graceful recovery when the
planner or reasoner returns unparseable output.

## Differences from the class reference

| | Reference | This version |
|---|---|---|
| LLM | OpenAI SDK + local proxy | Anthropic (consistent with weeks 3–4) |
| TASK-3 loop-back | described in the PRD, never coded | implemented, with a round cap |
| Plan parsing | manual fence-stripping, newline fallback, no cap | Pydantic model, 2–6 questions enforced |
| Reasoner output | free prose | structured JSON verdict |
| Synthesis | reasoner also writes the answer | separate synthesizer prompt |
| Tool loop | single round | multi-round, capped |
| Tool errors | `KeyError` / `TypeError` crash | returned as data |
