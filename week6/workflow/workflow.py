"""
workflow.py — the week-5 agentic workflow, now MCP-backed.

    PLAN -> RESEARCH each question -> REASON: enough?
                  ^                      | no -> research the gaps
                  |______________________|
                                         | yes -> SYNTHESIZE

The only thing that changed from week 5 is where tools come from. Week 5 did
`from shared.tools import web_search` — a hard import. Here the researcher is
handed an MCP client and uses whatever tools the server advertises.
"""

from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field, field_validator

from shared.llm import chat, run_tool_loop

_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

MAX_QUESTIONS = 6
MAX_REASONING_ROUNDS = 2


def load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS, name), "r", encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------- #
# Validated structured output (carried over from week 5)
# --------------------------------------------------------------------------- #
class ResearchPlan(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=MAX_QUESTIONS)

    @field_validator("questions")
    @classmethod
    def clean(cls, values):
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            raise ValueError("plan contains no usable questions")
        return cleaned


class SufficiencyCheck(BaseModel):
    sufficient: bool
    gaps: list[str] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    notes: str = ""


def extract_json(raw: str) -> str:
    """Pull JSON out of a reply that may be fenced or padded with prose."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.lstrip().lower().startswith("json"):
                block = block.lstrip()[4:]
            text = block.strip()
    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append((start, text[start: end + 1]))
    if candidates:
        text = min(candidates, key=lambda c: c[0])[1]
    return text.strip()


def parse_plan(raw: str) -> ResearchPlan:
    data = json.loads(extract_json(raw))
    if isinstance(data, list):
        data = {"questions": data}
    return ResearchPlan(**data)


def parse_sufficiency(raw: str) -> SufficiencyCheck:
    return SufficiencyCheck(**json.loads(extract_json(raw)))


def _format(findings: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"Q{i}: {q}\nFinding: {a}" for i, (q, a) in enumerate(findings, 1))


# --------------------------------------------------------------------------- #
async def run_workflow(mcp_client, user_prompt: str, verbose: bool = True) -> str:
    """Plan, research over MCP, reason with loop-back, then synthesize."""

    # ---- PLAN ---------------------------------------------------------- #
    if verbose:
        print("\n--- STEP 1: PLAN ---")
    try:
        plan = parse_plan(chat(load_prompt("planner.md"), user_prompt))
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [plan unparseable: {exc} — researching the request as one question]")
        plan = ResearchPlan(questions=[user_prompt])

    if verbose:
        for i, q in enumerate(plan.questions, 1):
            print(f"  {i}. {q}")

    # ---- RESEARCH (over MCP) ------------------------------------------- #
    if verbose:
        print("\n--- STEP 2: RESEARCH (tools via MCP) ---")

    async def research(question: str) -> str:
        return await run_tool_loop(
            mcp_client,
            load_prompt("researcher.md"),
            [{"role": "user", "content": question}],
            verbose=verbose,
        )

    findings: list[tuple[str, str]] = []
    for i, question in enumerate(plan.questions, 1):
        if verbose:
            print(f"\n  Q{i}: {question}")
        answer = await research(question)
        findings.append((question, answer))
        if verbose:
            print(f"  -> {answer[:400]}")

    # ---- REASON, with loop-back ---------------------------------------- #
    for round_num in range(1, MAX_REASONING_ROUNDS + 1):
        if verbose:
            print(f"\n--- STEP 3: REASON (round {round_num}) ---")
        try:
            verdict = parse_sufficiency(
                chat(
                    load_prompt("reasoner.md"),
                    f"Original user request:\n{user_prompt}\n\n"
                    f"Research questions and findings:\n{_format(findings)}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [verdict unparseable: {exc} — proceeding]")
            break

        if verbose:
            print(f"  Verdict: {'SUFFICIENT' if verdict.sufficient else 'INSUFFICIENT'}")
            if verdict.notes:
                print(f"  Notes: {verdict.notes}")

        if verdict.sufficient or not verdict.gaps:
            break

        for gap in verdict.gaps:
            if verbose:
                print(f"\n  GAP: {gap}")
            answer = await research(gap)
            findings.append((gap, answer))
            if verbose:
                print(f"  -> {answer[:400]}")

    # ---- SYNTHESIZE ---------------------------------------------------- #
    if verbose:
        print("\n--- STEP 4: SYNTHESIZE ---")
    return chat(
        load_prompt("synthesizer.md"),
        f"Original user request:\n{user_prompt}\n\nAll research findings:\n{_format(findings)}",
        max_tokens=4000,
    )
