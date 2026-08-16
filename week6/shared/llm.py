"""
llm.py — Anthropic client and the MCP-backed tool loop.

The agent loop is the bridge between the two protocols. Claude asks for a tool in
Anthropic's format; the MCP client executes it over MCP; the result goes back to
Claude as a tool_result block. Neither side knows about the other.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_NAME = os.getenv("MODEL_NAME", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
MAX_TOOL_ROUNDS = 4

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing 'anthropic'. Run: pip install -r requirements.txt") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            "  Create a .env file containing:  ANTHROPIC_API_KEY=sk-ant-..."
        )
    _client = anthropic.Anthropic()
    return _client


def extract_text(response) -> str:
    return "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()


def chat(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    """A plain system + user call with no tools."""
    response = get_client().messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens or MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return extract_text(response)


async def run_tool_loop(
    mcp_client,
    system_prompt: str,
    messages: list[dict],
    verbose: bool = False,
) -> str:
    """
    Let Claude call MCP tools until it produces a final answer.

    `messages` is mutated in place with the assistant and tool_result turns, so
    callers can inspect the exchange afterwards.
    """
    llm = get_client()
    tools = mcp_client.anthropic_specs()

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if getattr(response, "stop_reason", None) != "tool_use":
            return extract_text(response)

        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            outcome = await mcp_client.call(block.name, block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": outcome["content"],
                    "is_error": not outcome["ok"],
                }
            )
        messages.append({"role": "user", "content": results})

    return "Stopped: the tool loop hit its round limit before producing an answer."
