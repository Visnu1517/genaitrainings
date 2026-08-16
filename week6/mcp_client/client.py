"""
client.py — the MCP client.

Connects to the MCP server over stdio, discovers whatever tools it offers, and
translates them into the format the Anthropic API expects.

    MCP tool                          Anthropic tool
    ---------------------------       -----------------------------
    tool.name                    ->   "name"
    tool.description             ->   "description"
    tool.inputSchema             ->   "input_schema"

Nothing in this file knows what the tools actually *are*. That is the whole point
of MCP: add a tool to the server and the workflow picks it up on the next run
with no code change here.

IMPORTANT: the server subprocess is launched with `sys.executable`, not "python".
Using "python" resolves to whatever is first on PATH — usually the SYSTEM Python,
which does not have `mcp` installed inside a virtual environment. The server then
dies on import and the client reports the very unhelpful "Connection closed".
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The SDK emits a harmless pydantic-settings warning about an unresolved forward
# reference on its own internal model, and the server logs every request at INFO.
# Neither is actionable here and both drown out the actual output.
warnings.filterwarnings("ignore", message=".*incomplete definition.*")
logging.getLogger("mcp").setLevel(logging.WARNING)


class MCPClient:
    """A connection to the MCP server, plus helpers for using its tools."""

    def __init__(self, verbose: bool = False):
        self.session: ClientSession | None = None
        self.tools: list = []
        self.verbose = verbose
        self._stack: AsyncExitStack | None = None

    # ------------------------------------------------------------------ #
    async def connect(self) -> "MCPClient":
        """Start the server subprocess, handshake, and discover its tools."""
        self._stack = AsyncExitStack()

        params = StdioServerParameters(
            command=sys.executable,               # not "python" — see module docstring
            args=["-m", "mcp_server.server"],
            cwd=_PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": _PROJECT_ROOT, "MCP_LOG_LEVEL": "WARNING"},
        )

        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        self.tools = (await self.session.list_tools()).tools

        if self.verbose:
            print(f"[mcp] connected — {len(self.tools)} tools discovered:")
            for tool in self.tools:
                print(f"        - {tool.name}")

        return self

    async def close(self) -> None:
        """Shut the server subprocess down."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.session = None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *_exc):
        await self.close()

    # ------------------------------------------------------------------ #
    def anthropic_specs(self) -> list[dict]:
        """Convert the discovered MCP tools into Anthropic tool specs."""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in self.tools
        ]

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    # ------------------------------------------------------------------ #
    async def call(self, name: str, arguments: dict) -> dict:
        """
        Call a tool on the server.

        Returns {"ok": bool, "content": str}. Errors are returned as data rather
        than raised, so a bad argument becomes something the model can read and
        correct — the same principle as week 4's tool executor.
        """
        if self.session is None:
            return {"ok": False, "content": "MCP client is not connected."}

        if self.verbose:
            print(f"      [tool] {name}({json.dumps(arguments)})")

        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 — unknown tool, transport failure, etc.
            return {"ok": False, "content": f"Tool call failed: {exc}"}

        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )

        # FastMCP sets isError when the tool raised — including Pydantic
        # validation failures, which happen before the function body runs.
        ok = not getattr(result, "isError", False)

        if self.verbose and not ok:
            print(f"      [tool error] {text[:160]}")

        return {"ok": ok, "content": text}
