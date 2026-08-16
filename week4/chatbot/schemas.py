"""
schemas.py
----------
Pydantic models describing every tool's INPUT.

These do double duty, which is the central design idea of this week's work:

  1. VALIDATION  - arguments the model generates are parsed through these models
                   before the Python function runs. Bad values are rejected with
                   a clear, structured error instead of crashing or silently
                   producing nonsense.

  2. SCHEMA      - `model_json_schema()` generates the JSON Schema we hand to
                   Claude. The reference demo wrote that JSON by hand in a second
                   file, which means two things to keep in sync; here the Pydantic
                   model is the single source of truth.

Note the constraints are real, not decorative. A ZIP code must be five digits, an
account ID must match ACC-######, a plan type must be one of a fixed set. That is
what makes validation meaningful: an LLM can and will invent plausible-looking
arguments, and these models are the gate that stops them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Shared field types
# --------------------------------------------------------------------------- #
ZipCode = Field(
    ...,
    description="5-digit US ZIP code, for example 75201.",
    pattern=r"^\d{5}$",
    min_length=5,
    max_length=5,
)


# --------------------------------------------------------------------------- #
# AT&T tools
# --------------------------------------------------------------------------- #
class ServiceAvailabilityInput(BaseModel):
    """Input for checking which AT&T services are offered at a location."""

    zip_code: str = ZipCode
    service_type: Literal["fiber", "internet", "wireless", "all"] = Field(
        default="all",
        description="Which service to check. Use 'all' if the user did not specify.",
    )


class PlanDetailsInput(BaseModel):
    """Input for looking up a specific AT&T plan."""

    plan_type: Literal[
        "unlimited_starter",
        "unlimited_extra",
        "unlimited_premium",
        "prepaid",
        "fiber_300",
        "fiber_1gig",
    ] = Field(..., description="The exact plan identifier to look up.")
    num_lines: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of wireless lines, between 1 and 10. Ignored for fiber plans.",
    )


class OutageStatusInput(BaseModel):
    """Input for checking network outages in an area."""

    zip_code: str = ZipCode


class AccountBalanceInput(BaseModel):
    """Input for looking up an account balance."""

    account_id: str = Field(
        ...,
        description="Account ID in the format ACC-###### (ACC- followed by exactly 6 digits).",
        pattern=r"^ACC-\d{6}$",
    )

    @field_validator("account_id", mode="before")
    @classmethod
    def uppercase_prefix(cls, v):
        # mode="before" matters: this must run BEFORE the pattern constraint is
        # checked, otherwise "acc-100001" is rejected before we can normalise it.
        return v.upper() if isinstance(v, str) else v


class StoreLocatorInput(BaseModel):
    """Input for finding AT&T retail stores near a ZIP code."""

    zip_code: str = ZipCode
    radius_miles: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Search radius in miles, between 1 and 50.",
    )


# --------------------------------------------------------------------------- #
# Tools carried over from the class reference demo, for comparison
# --------------------------------------------------------------------------- #
class WebSearchInput(BaseModel):
    """Input for the mocked web search tool."""

    query: str = Field(
        ...,
        description="Search query.",
        min_length=3,
        max_length=200,
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query cannot be blank")
        return cleaned


class OnCallInput(BaseModel):
    """Input for looking up the on-call engineer for an internal service."""

    service_name: Literal["payment", "billing", "checkout"] = Field(
        ...,
        description="Name of the internal service.",
    )


# --------------------------------------------------------------------------- #
# Output envelope
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """
    A consistent shape for every tool result, success or failure.

    Giving errors the same structure as successes matters in an agentic loop: the
    model receives the failure as data it can read and react to (retry with a
    corrected argument, or ask the user for the missing detail) instead of the
    program raising and dying.
    """

    ok: bool
    tool: str
    data: dict | list | None = None
    error: str | None = None
    details: list[dict] | None = None

    @classmethod
    def success(cls, tool: str, data) -> "ToolResult":
        return cls(ok=True, tool=tool, data=data)

    @classmethod
    def failure(cls, tool: str, error: str, details: list[dict] | None = None) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, details=details)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)
