# Week 6 — Tools Behind an MCP Server

All tools from weeks 4 and 5 now live behind a **Model Context Protocol** server.
The agentic workflow and the chatbot reach them through an **MCP client** instead
of importing Python functions.

## Before and after

```
WEEK 5                              WEEK 6
app ──import──> tools.py            app ──MCP client──┐
                                                      │  (stdio, JSON-RPC)
                                    MCP server <──────┘
                                       └── 9 tools
```

Week 5 did `from shared.tools import web_search` — a hard import. Now the app asks
the server *"what tools do you have?"* at runtime. Add a tool to the server and
the workflow picks it up on the next run with **no change to the app code**.

## The 9 tools

| From | Tool | Purpose |
|---|---|---|
| Week 4 | `check_service_availability` | Fiber/internet/wireless at a ZIP |
| Week 4 | `get_plan_details` | Plan pricing and features |
| Week 4 | `check_outage_status` | Known outages in an area |
| Week 4 | `lookup_account_balance` | Balance, due date, autopay |
| Week 4 | `find_nearest_store` | Retail stores near a ZIP |
| Week 4 | `search_web` | General mocked search |
| Week 4 | `get_oncall_engineer` | Internal on-call lookup |
| Week 5 | `web_search` | Competitor carrier research |
| Week 5 | `query_db` | AT&T corporate database |

## Layout

```
week6/
├── mcp_server/
│   ├── server.py      # THE SERVER — all 9 tools behind FastMCP
│   └── mock_data.py   # AT&T mock data (from week 4)
├── mcp_client/
│   └── client.py      # connects, discovers, converts MCP -> Anthropic format
├── workflow/          # week-5 plan -> research -> reason -> synthesize
│   ├── workflow.py
│   └── prompts/
├── chatbot/           # week-4 assistant (guardrails + session memory)
│   ├── chatbot.py
│   └── system_prompt.md
├── shared/llm.py      # Anthropic client + MCP-backed agent loop
├── data/              # query.md, webresults.md (week-5 markdown data)
├── main.py            # entry point — chat and workflow share ONE connection
├── test_mcp.py        # 20 offline tests
└── claude_desktop_config.example.json
```

## Three things worth knowing

**1. The SDK version matters.** `pip install mcp` now installs **v2.0.0**, where
`FastMCP` was renamed to `MCPServer` and `mcp.server.fastmcp` no longer exists.
`requirements.txt` pins `mcp>=1.28,<2` on purpose so `FastMCP` stays available and
matches course material. Installing unpinned gives `ModuleNotFoundError`.

**2. Launch the server with `sys.executable`, not `"python"`.** `"python"` resolves
to whatever is first on PATH — usually the *system* Python, which has no `mcp`
inside a virtual environment. The server then dies on import and the client
reports a very unhelpful `McpError: Connection closed`.

**3. Declare constraints on the parameters, not inside the function.** This is
what keeps week 4's validation alive through MCP:

| Style | Schema the model sees |
|---|---|
| Validate inside the function body | `{"type": "string"}` — constraints lost |
| Pydantic model as the parameter | full, but nested under `$ref`/`$defs` |
| **`Field(...)` on each parameter** | **full, flat, readable** |

So `find_nearest_store` reaches the model as:

```json
{"zip_code":     {"type": "string",  "pattern": "^\\d{5}$"},
 "radius_miles": {"type": "integer", "minimum": 1, "maximum": 50}}
```

FastMCP then rejects bad arguments **before the function body runs**, and the
error comes back as `isError: true` — which the client turns into `is_error` on
the tool result so the model can correct itself.

## Setup

```bash
cd week6
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # paste your real ANTHROPIC_API_KEY into it
```

## Run

```bash
python main.py --list-tools         # connect, print the catalogue, exit (no API key needed)
python main.py                      # interactive chat (week-4 assistant)
python main.py --verbose            # show every MCP tool call
python main.py --workflow           # week-5 research workflow, tools over MCP
python main.py --workflow --prompt "Which carrier has the widest 5G coverage?"
```

You never start the server yourself — the client launches it as a subprocess and
shuts it down on exit.

In-chat commands: `/tools`, `/status`, `/workflow <question>`, `/reset`, `/quit`

## Tests

```bash
python test_mcp.py
```

20 tests, **no API key required**. They start a real MCP server subprocess (so
discovery, transport and validation are genuinely exercised) with a fake LLM.
Coverage: all 9 tools discovered, Anthropic spec conversion, patterns/ranges/enums
surviving the round trip, correct tool data, server-side validation returned as
data, unknown tools, disconnected client, multi-round tool looping and its cap,
`is_error` propagation, chatbot guardrails and plain-text session memory, workflow
research and loop-back, and chatbot + workflow sharing one connection.

## Bonus: use these tools in Claude Desktop

Because this is a standard MCP server, Claude Desktop can use it too — that is the
real payoff of the protocol. See `claude_desktop_config.example.json`, replace the
two paths with your own, and restart Claude Desktop.

## Known rough edge

Week 4 has `search_web` and week 5 has `web_search` — near-identical names on the
same server, which invites the model to pick the wrong one. They are kept as-is
for fidelity to the earlier assignments, and their descriptions are written to
disambiguate (`search_web` = general mocked search; `web_search` = competitor
carriers). Renaming one would be the cleaner fix in a real system: **tool name
collisions are a genuine integration problem once tools from different sources
share a server.**
