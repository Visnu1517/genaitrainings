"""
schemas.py
----------
Validated parsing of the LLM's structured output for TASK-3.

The planner and the reasoner are both asked to reply in JSON. An LLM will
occasionally wrap that JSON in markdown fences, add a sentence before it, or
return the wrong shape entirely — so the raw text is extracted, then parsed
through a Pydantic model that enforces the structure.

The class reference parses the planner output by hand (stripping fences, hunting
for brackets, falling back to splitting on newlines) with no schema and no cap on
how many questions come back. This is the same idea done with validation.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator

MAX_QUESTIONS = 6


class ResearchPlan(BaseModel):
    """The planner's decomposition of the user's request."""

    questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTIONS,
        description="Focused, independently answerable research questions.",
    )

    @field_validator("questions")
    @classmethod
    def clean(cls, values: list[str]) -> list[str]:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            raise ValueError("plan contains no usable questions")
        return cleaned


class SufficiencyCheck(BaseModel):
    """The reasoner's verdict on whether the findings answer the request."""

    sufficient: bool = Field(..., description="True if the findings fully cover the request.")
    gaps: list[str] = Field(
        default_factory=list,
        max_length=MAX_QUESTIONS,
        description="Follow-up questions needed to close the gaps. Empty when sufficient.",
    )
    notes: str = Field(default="", description="Short explanation of the verdict.")

    @field_validator("gaps")
    @classmethod
    def clean_gaps(cls, values: list[str]) -> list[str]:
        return [str(v).strip() for v in values if str(v).strip()]


def extract_json(raw: str) -> str:
    """
    Pull the JSON out of an LLM reply that may be wrapped in markdown fences or
    padded with prose.
    """
    text = raw.strip()

    if "```" in text:
        # Take the content of the first fenced block.
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.lstrip().lower().startswith("json"):
                block = block.lstrip()[4:]
            text = block.strip()

    # Narrow to the outermost JSON object or array.
    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append((start, text[start: end + 1]))
    if candidates:
        # Prefer whichever structure appears first in the text.
        text = min(candidates, key=lambda c: c[0])[1]

    return text.strip()


def parse_plan(raw: str) -> ResearchPlan:
    """Parse the planner's reply into a validated ResearchPlan."""
    text = extract_json(raw)
    data = json.loads(text)

    # Accept either a bare array or {"questions": [...]}.
    if isinstance(data, list):
        data = {"questions": data}

    return ResearchPlan(**data)


def parse_sufficiency(raw: str) -> SufficiencyCheck:
    """Parse the reasoner's reply into a validated SufficiencyCheck."""
    return SufficiencyCheck(**json.loads(extract_json(raw)))
