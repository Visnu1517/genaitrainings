"""
agent_loop.py
-------------
A reusable tool-calling loop: let the model call tools until it has what it
needs, then return its final text answer. Used by TASK-1 and by each researcher
in TASK-3.

The class reference runs exactly ONE round of tools and then forces a final
answer, so a question needing a second lookup gets truncated. This loops until
the model stops asking, capped by MAX_TOOL_ROUNDS.
"""

from __future__ import annotations

import json

from .llm_client import MAX_TOKENS, MODEL_NAME, extract_text, get_client
from .tool_executor import execute_tool_call
from .tool_specs import TOOLS

MAX_TOOL_ROUNDS = 4


def run_tool_loop(system_prompt: str, user_prompt: str, verbose: bool = True) -> str:
    """Run a conversation where the model may call tools, and return its answer."""
    client = get_client()
    messages = [{"role": "user", "content": user_prompt}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Record the assistant turn (may contain tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        if getattr(response, "stop_reason", None) != "tool_use":
            return extract_text(response)

        # Execute every tool the model asked for this round.
        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            if verbose:
                print(f"      [tool] {block.name}({json.dumps(block.input)})")

            result = execute_tool_call(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": "error" in result,
                }
            )

        # All results for a turn go back in one user message.
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: the tool loop hit its round limit before producing an answer."
