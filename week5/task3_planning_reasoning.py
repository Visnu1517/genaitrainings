"""
TASK-3: Planning and reasoning. Paths decided at runtime.

    receive input
      -> PLAN      : decompose the request into research questions
      -> RESEARCH  : answer each question independently, using tools
      -> REASON    : are the findings enough?
                       yes -> synthesize
                       no  -> research the gaps, then reason again
      -> SYNTHESIZE: write the final report

The difference from TASK-2 is that nothing is hardcoded. The planner decides the
decomposition after reading the request, so this same script works for a question
nobody anticipated.

The loop-back is the part that makes this "reasoning" rather than a longer
pipeline. The class reference describes this behaviour in its PRD but its
reasoner prompt says "Do NOT request more research", and the code never loops —
so an insufficient result just gets labelled insufficient. Here the reasoner
returns a structured verdict, and a false verdict genuinely sends work back to
the researcher.

MAX_REASONING_ROUNDS caps it, so a reasoner that is never satisfied cannot spin
forever.

Run:
    python task3_planning_reasoning.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import ValidationError

from shared.agent_loop import run_tool_loop
from shared.llm_client import chat
from shared.prompt_loader import load_prompt
from shared.schemas import ResearchPlan, SufficiencyCheck, parse_plan, parse_sufficiency

USER_PROMPT = (
    "Compare AT&T's mobile connection plans with its competitor's and make a "
    "detailed report of how much of market capitalization AT&T has compared to "
    "its competitors. Also give areas where AT&T can capitalize in order to "
    "capture more market."
)

# How many times the reasoner may send work back for more research.
MAX_REASONING_ROUNDS = 2


# --------------------------------------------------------------------------- #
# Step 1: PLAN
# --------------------------------------------------------------------------- #
def plan(user_prompt: str) -> ResearchPlan:
    """Decompose the request into research questions, validated by Pydantic."""
    raw = chat(load_prompt("task3_planner.md"), user_prompt)
    try:
        return parse_plan(raw)
    except (ValidationError, ValueError, Exception) as exc:  # noqa: BLE001
        # If the planner returns something unusable, fall back to researching the
        # request as a single question rather than crashing the workflow.
        print(f"  [planner output could not be parsed: {exc}]")
        print("  [falling back to a single research question]")
        return ResearchPlan(questions=[user_prompt])


# --------------------------------------------------------------------------- #
# Step 2: RESEARCH
# --------------------------------------------------------------------------- #
def research(question: str, verbose: bool = True) -> str:
    """Answer one research question using the tools."""
    return run_tool_loop(load_prompt("task3_researcher.md"), question, verbose=verbose)


# --------------------------------------------------------------------------- #
# Step 3: REASON
# --------------------------------------------------------------------------- #
def _format_findings(findings: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"Q{i}: {q}\nFinding: {a}" for i, (q, a) in enumerate(findings, 1)
    )


def reason(user_prompt: str, findings: list[tuple[str, str]]) -> SufficiencyCheck:
    """Judge whether the findings cover the request; name the gaps if not."""
    raw = chat(
        load_prompt("task3_reasoner.md"),
        f"Original user request:\n{user_prompt}\n\n"
        f"Research questions and findings:\n{_format_findings(findings)}",
    )
    try:
        return parse_sufficiency(raw)
    except Exception as exc:  # noqa: BLE001
        # An unreadable verdict should not stall the workflow; treat it as
        # sufficient and proceed to synthesis with what we have.
        print(f"  [reasoner output could not be parsed: {exc} - proceeding]")
        return SufficiencyCheck(sufficient=True, gaps=[], notes="verdict unparseable")


# --------------------------------------------------------------------------- #
# Step 4: SYNTHESIZE
# --------------------------------------------------------------------------- #
def synthesize(user_prompt: str, findings: list[tuple[str, str]]) -> str:
    """Write the final report from everything gathered."""
    return chat(
        load_prompt("task3_synthesizer.md"),
        f"Original user request:\n{user_prompt}\n\n"
        f"All research findings:\n{_format_findings(findings)}",
        max_tokens=4000,
    )


# --------------------------------------------------------------------------- #
def run(user_prompt: str = USER_PROMPT, verbose: bool = True) -> str:
    if verbose:
        print("\n" + "=" * 78)
        print("TASK-3: PLANNING AND REASONING")
        print("=" * 78)
        print(f"\nUSER PROMPT:\n{user_prompt}")

    # ---- PLAN ---------------------------------------------------------- #
    if verbose:
        print("\n--- STEP 1: PLAN ---")
    research_plan = plan(user_prompt)
    if verbose:
        for i, q in enumerate(research_plan.questions, 1):
            print(f"  {i}. {q}")

    # ---- RESEARCH ------------------------------------------------------ #
    if verbose:
        print("\n--- STEP 2: RESEARCH ---")
    findings: list[tuple[str, str]] = []
    for i, question in enumerate(research_plan.questions, 1):
        if verbose:
            print(f"\n  Q{i}: {question}")
        answer = research(question, verbose=verbose)
        findings.append((question, answer))
        if verbose:
            print(f"  -> {answer}")

    # ---- REASON (with loop-back) --------------------------------------- #
    for round_num in range(1, MAX_REASONING_ROUNDS + 1):
        if verbose:
            print(f"\n--- STEP 3: REASON (round {round_num}) ---")

        verdict = reason(user_prompt, findings)

        if verbose:
            print(f"  Verdict: {'SUFFICIENT' if verdict.sufficient else 'INSUFFICIENT'}")
            if verdict.notes:
                print(f"  Notes: {verdict.notes}")

        if verdict.sufficient or not verdict.gaps:
            break

        if verbose:
            print(f"  Gaps found ({len(verdict.gaps)}) - researching them:")

        # This is the loop-back: gaps go back through the researcher.
        for gap in verdict.gaps:
            if verbose:
                print(f"\n  GAP: {gap}")
            answer = research(gap, verbose=verbose)
            findings.append((gap, answer))
            if verbose:
                print(f"  -> {answer}")
    else:
        if verbose:
            print("  [reasoning round limit reached - synthesizing with what we have]")

    # ---- SYNTHESIZE ---------------------------------------------------- #
    if verbose:
        print("\n--- STEP 4: SYNTHESIZE ---")
    report = synthesize(user_prompt, findings)

    if verbose:
        print("\n" + "=" * 78)
        print(f"FINAL REPORT  (built from {len(findings)} findings)")
        print("=" * 78)
        print(report)

    return report


if __name__ == "__main__":
    run()
