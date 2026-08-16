"""
tool_registry.py
----------------
Turns plain Python functions into tools Claude can call.

A function only becomes a "tool" once the model is told its name, what it does,
and what arguments it takes. The reference demo supplied that description by
hand-writing JSON in a separate file, so the Pydantic models and the JSON could
drift apart. Here the decorator takes the Pydantic model and derives the JSON
Schema from it with `model_json_schema()`, so there is exactly one source of truth.

Usage:

    @tool("check_outage_status", "Check for network outages.", OutageStatusInput)
    def check_outage_status(args: OutageStatusInput) -> dict:
        ...

The decorated function receives an already-validated model instance, so the body
never has to defend against bad input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable

    def anthropic_spec(self) -> dict:
        """
        The shape Claude expects:
            {"name": ..., "description": ..., "input_schema": {JSON Schema}}

        Note this differs from the OpenAI format used in the class reference,
        which nests everything under {"type": "function", "function": {...}}
        and calls the schema "parameters" instead of "input_schema".
        """
        schema = self.input_model.model_json_schema()
        # "title" keys are Pydantic bookkeeping and add noise to the model's context.
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


# name -> Tool
REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, input_model: type[BaseModel]):
    """Decorator that registers a function as a callable tool."""

    def decorator(func: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"Tool '{name}' is already registered")
        REGISTRY[name] = Tool(
            name=name,
            description=description,
            input_model=input_model,
            func=func,
        )
        return func

    return decorator


def get_tool(name: str) -> Tool | None:
    return REGISTRY.get(name)


def all_specs() -> list[dict]:
    """Every registered tool, in the format sent to the Anthropic API."""
    return [t.anthropic_spec() for t in REGISTRY.values()]


def tool_names() -> list[str]:
    return list(REGISTRY.keys())
