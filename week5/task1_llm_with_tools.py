"""
TASK-1: Just an LLM client with tools. No workflow, no planning.

    receive input -> LLM calls tools -> LLM answers from the results

This is the baseline. Everything happens in a single conversation: one model,
one system prompt, one shot at the whole request. There is no decomposition, so
a request containing three separate deliverables gets whatever attention the
model happens to distribute across them.

Run:
    python task1_llm_with_tools.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.agent_loop import run_tool_loop
from shared.prompt_loader import load_prompt

USER_PROMPT = (
    "Compare AT&T's mobile connection plans with its competitor's and make a "
    "detailed report of how much of market capitalization AT&T has compared to "
    "its competitors. Also give areas where AT&T can capitalize in order to "
    "capture more market."
)


def run(user_prompt: str = USER_PROMPT, verbose: bool = True) -> str:
    if verbose:
        print("\n" + "=" * 78)
        print("TASK-1: ONE-SHOT LLM WITH TOOLS")
        print("=" * 78)
        print(f"\nUSER PROMPT:\n{user_prompt}\n")
        print("--- ANSWERING (single pass) ---")

    answer = run_tool_loop(load_prompt("task1.md"), user_prompt, verbose=verbose)

    if verbose:
        print("\n" + "=" * 78)
        print("FINAL ANSWER")
        print("=" * 78)
        print(answer)

    return answer


if __name__ == "__main__":
    run()
