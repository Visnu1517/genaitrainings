"""
prompt_loader.py
----------------
Loads prompt text from the Prompts/ directory.

Keeping prompts in markdown files rather than Python strings is deliberate: the
planning and reasoning behaviour this assignment is about lives almost entirely
in the prompts, so they should be editable without touching code.
"""

from __future__ import annotations

import os

_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Prompts"
)


def load_prompt(name: str) -> str:
    """Load a prompt file by name, e.g. load_prompt("task3_planner.md")."""
    path = os.path.join(_PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
