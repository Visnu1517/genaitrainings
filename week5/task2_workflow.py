"""
TASK-2: A hardcoded workflow.

    receive input -> split into 3 FIXED branches -> run each -> join the results

The three branches are decided at coding time, not at runtime:
    1. compare mobile connection plans
    2. report market capitalization
    3. identify growth areas

That is the whole point of this task, and also its limitation. We can hardcode
this split only because we already know what the question is going to be. Give
this script a different question and the branches are simply wrong — it would
still dutifully produce a plans comparison and a market cap report.

Each branch gets the same tool data and a narrow system prompt telling it to
cover its section and nothing else. The sections are then joined under headings.
Like the class reference, aggregation here is a plain join with no extra model
call, which means nothing reconciles contradictions between branches. TASK-3
fixes both problems.

Run:
    python task2_workflow.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.llm_client import chat
from shared.prompt_loader import load_prompt
from shared.tools import query_db, web_search

USER_PROMPT = (
    "Compare AT&T's mobile connection plans with its competitor's and make a "
    "detailed report of how much of market capitalization AT&T has compared to "
    "its competitors. Also give areas where AT&T can capitalize in order to "
    "capture more market."
)

# The pre-determined decomposition. Hardcoded on purpose.
BRANCHES = [
    ("Mobile Connection Plans", "task2_compare_plans.md"),
    ("Market Capitalization", "task2_market_cap.md"),
    ("Growth Opportunities", "task2_growth_areas.md"),
]


def _gather_tool_context() -> str:
    """Run both tools once up front; every branch shares the same data."""
    att = query_db("AT&T market cap, subscribers and mobile plans")
    competitors = web_search("Verizon and T-Mobile market cap and mobile plans")
    return (
        "AT&T data (query_db):\n"
        f"{json.dumps(att)}\n\n"
        "Competitor data (web_search):\n"
        f"{json.dumps(competitors)}"
    )


def _run_branch(title: str, prompt_file: str, tool_context: str, verbose: bool) -> str:
    if verbose:
        print(f"\n--- BRANCH: {title} ---")

    section = chat(
        load_prompt(prompt_file),
        f"User request:\n{USER_PROMPT}\n\nTool data available to you:\n{tool_context}",
    )

    if verbose:
        print(section)

    return section


def run(user_prompt: str = USER_PROMPT, verbose: bool = True) -> str:
    if verbose:
        print("\n" + "=" * 78)
        print("TASK-2: HARDCODED 3-BRANCH WORKFLOW")
        print("=" * 78)
        print(f"\nUSER PROMPT:\n{user_prompt}\n")
        print("Branches (fixed at coding time):")
        for i, (title, _) in enumerate(BRANCHES, 1):
            print(f"  {i}. {title}")

    tool_context = _gather_tool_context()

    sections = [
        f"## {title}\n\n{_run_branch(title, prompt_file, tool_context, verbose)}"
        for title, prompt_file in BRANCHES
    ]

    # Aggregation is a plain join — no synthesis pass.
    report = "\n\n".join(sections)

    if verbose:
        print("\n" + "=" * 78)
        print("FINAL REPORT (branches joined)")
        print("=" * 78)
        print(report)

    return report


if __name__ == "__main__":
    run()
