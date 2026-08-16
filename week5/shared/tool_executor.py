"""
tool_executor.py
----------------
Dispatch a tool call by name to its Python function.

Unlike the class reference — which does `TOOL_REGISTRY[tool_name]` and would
raise KeyError on a hallucinated tool name, or TypeError on wrong arguments —
this returns the error as data so the model can read it and correct itself.
"""

from __future__ import annotations

from .tools import query_db, web_search

TOOL_REGISTRY = {
    "web_search": web_search,
    "query_db": query_db,
}


def execute_tool_call(tool_name: str, tool_args: dict) -> dict:
    """Run a tool. Always returns a dict; never raises."""
    tool_function = TOOL_REGISTRY.get(tool_name)

    if tool_function is None:
        return {
            "error": f"Unknown tool '{tool_name}'.",
            "available_tools": list(TOOL_REGISTRY),
        }

    try:
        return tool_function(**(tool_args or {}))
    except TypeError as exc:
        return {"error": f"Invalid arguments for '{tool_name}': {exc}"}
    except Exception as exc:  # noqa: BLE001 — report, don't crash the workflow
        return {"error": f"Tool '{tool_name}' failed: {exc}"}
