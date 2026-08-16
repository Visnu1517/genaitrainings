"""
tool_executor.py
----------------
Runs a tool the model asked for, and — importantly — never lets a bad request
crash the program.

Three failure modes are handled explicitly, because all three genuinely happen
when an LLM is generating the arguments:

  1. Unknown tool      - the model hallucinated a tool name that does not exist.
  2. Invalid arguments - the model sent a malformed ZIP, a bad account ID, an
                         out-of-range number, or omitted a required field.
                         Pydantic raises ValidationError; we translate it into a
                         readable message listing the offending fields.
  3. Execution error   - the function itself blew up.

In every case the failure is returned as a ToolResult with ok=False and fed back
to the model as the tool result. That is the difference between an agent that
recovers ("that account ID isn't valid — could you check it?") and one that dies
with a traceback. The reference demo does none of this: an invalid argument raises
straight out of `tool_function(**tool_args)`.
"""

from __future__ import annotations

from pydantic import ValidationError

from .schemas import ToolResult
from .tool_registry import get_tool, tool_names


def _format_validation_error(exc: ValidationError) -> tuple[str, list[dict]]:
    """Turn a Pydantic ValidationError into a short message plus per-field detail."""
    details = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ())) or "(input)"
        details.append(
            {
                "field": field,
                "problem": err.get("msg", "invalid value"),
                "received": repr(err.get("input")),
            }
        )
    summary = "; ".join(f"{d['field']}: {d['problem']}" for d in details)
    return f"Invalid arguments - {summary}", details


def execute_tool(name: str, raw_args: dict, verbose: bool = False) -> ToolResult:
    """
    Validate `raw_args` against the tool's Pydantic model, then run it.
    Always returns a ToolResult; never raises.
    """
    tool_obj = get_tool(name)

    # 1. Unknown tool
    if tool_obj is None:
        return ToolResult.failure(
            tool=name,
            error=f"Unknown tool '{name}'. Available tools: {', '.join(tool_names())}",
        )

    # 2. Argument validation — this is where Pydantic earns its keep
    try:
        args = tool_obj.input_model(**(raw_args or {}))
    except ValidationError as exc:
        message, details = _format_validation_error(exc)
        if verbose:
            print(f"  [validation failed] {message}")
        return ToolResult.failure(tool=name, error=message, details=details)
    except TypeError as exc:
        # e.g. the model sent a completely wrong argument shape
        return ToolResult.failure(tool=name, error=f"Invalid arguments - {exc}")

    # 3. Execution
    try:
        data = tool_obj.func(args)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; report, don't crash
        if verbose:
            print(f"  [execution failed] {exc}")
        return ToolResult.failure(tool=name, error=f"Tool execution failed - {exc}")

    if verbose:
        print(f"  [validated args] {args.model_dump()}")

    return ToolResult.success(tool=name, data=data)
