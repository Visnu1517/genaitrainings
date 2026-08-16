"""
main.py — entry point. One MCP connection shared by both consumers.

    python main.py                      interactive chat (week-4 assistant)
    python main.py --workflow           run the week-5 research workflow
    python main.py --workflow --prompt "..."
    python main.py --list-tools         connect, print the tool catalogue, exit
    python main.py --verbose            show every MCP tool call

Both the chatbot and the workflow use the SAME MCPClient, so the server
subprocess starts once rather than twice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from chatbot.chatbot import ChatBot
from mcp_client.client import MCPClient
from workflow.workflow import run_workflow

DEFAULT_WORKFLOW_PROMPT = (
    "Compare AT&T's mobile connection plans with its competitor's and make a "
    "detailed report of how much of market capitalization AT&T has compared to "
    "its competitors. Also give areas where AT&T can capitalize in order to "
    "capture more market."
)

BANNER = """AT&T Assistant (week 6: tools served over MCP)
Commands: /tools  /status  /workflow <question>  /reset  /quit
"""


def print_tools(client: MCPClient) -> None:
    for spec in client.anthropic_specs():
        print(f"\n  {spec['name']}")
        desc = " ".join(spec["description"].split())
        print(f"    {desc[:150]}")
        props = spec["input_schema"].get("properties", {})
        required = set(spec["input_schema"].get("required", []))
        for field, meta in props.items():
            mark = "required" if field in required else "optional"
            limits = {
                k: v for k, v in meta.items()
                if k in ("pattern", "enum", "minimum", "maximum", "minLength", "maxLength")
            }
            extra = f"  {json.dumps(limits)}" if limits else ""
            print(f"      - {field} ({meta.get('type', '?')}, {mark}){extra}")
    print()


async def interactive(client: MCPClient, verbose: bool) -> None:
    bot = ChatBot(client, verbose=verbose)
    print(BANNER)
    print(bot.status(), "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            continue
        if query == "/quit":
            return
        if query == "/tools":
            print_tools(client)
            continue
        if query == "/status":
            print(bot.status(), "\n")
            continue
        if query == "/reset":
            bot.reset()
            print("(session cleared)\n")
            continue
        if query.startswith("/workflow "):
            report = await run_workflow(client, query[len("/workflow "):], verbose=True)
            print(f"\n{report}\n")
            continue

        try:
            print(f"Bot: {await bot.get_answer(query)}\n")
        except Exception as exc:  # noqa: BLE001 — keep the chat alive
            print(f"[error] {exc}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Week 6 — MCP-backed assistant.")
    parser.add_argument("--workflow", action="store_true", help="run the research workflow")
    parser.add_argument("--prompt", type=str, help="prompt for the workflow")
    parser.add_argument("--list-tools", action="store_true", help="print tools and exit")
    parser.add_argument("--verbose", action="store_true", help="show MCP tool calls")
    args = parser.parse_args()

    async with MCPClient(verbose=args.verbose) as client:
        if args.list_tools:
            print(f"\n{len(client.tools)} tools discovered over MCP:")
            print_tools(client)
            return

        if args.workflow:
            report = await run_workflow(
                client, args.prompt or DEFAULT_WORKFLOW_PROMPT, verbose=True
            )
            print("\n" + "=" * 78)
            print("FINAL REPORT")
            print("=" * 78)
            print(report)
            return

        await interactive(client, args.verbose)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
