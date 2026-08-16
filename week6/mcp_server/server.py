"""
server.py — the MCP server.

Every tool built in weeks 4 and 5 now lives behind the Model Context Protocol
instead of being imported directly by the app:

    Week 4 (AT&T assistant)   check_service_availability, get_plan_details,
                              check_outage_status, lookup_account_balance,
                              find_nearest_store, search_web, get_oncall_engineer
    Week 5 (market research)  web_search, query_db

WHY THIS MATTERS
Previously the workflow did `from tools import get_plan_details` — a hard Python
import. Now it asks a server "what tools do you have?" over a protocol. The tools
can be swapped, extended, or reused by any other MCP client (Claude Desktop,
Cursor) with no change to the app.

HOW ARGUMENTS ARE DECLARED
Each parameter uses `Field(...)` with its constraints. This matters: if you
validate *inside* the function body instead, FastMCP only sees `str` and the
model receives a schema with no pattern, no min/max and no enum — losing all the
validation work from week 4. Declaring constraints on the parameters keeps the
full schema visible to the model AND makes FastMCP reject bad input before the
function body ever runs.

Run standalone (it will wait on stdin — that is correct, it speaks stdio):
    python -m mcp_server.server
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .mock_data import (
    ACCOUNTS,
    ONCALL_ENGINEERS,
    OUTAGES,
    PLANS,
    SERVICE_AVAILABILITY,
    STORES,
    WEB_RESULTS,
)

warnings.filterwarnings("ignore", message=".*incomplete definition.*")
logging.getLogger("mcp").setLevel(logging.WARNING)

mcp = FastMCP("att-tools")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# --------------------------------------------------------------------------- #
# Week 4 tools — the AT&T customer assistant
# --------------------------------------------------------------------------- #
@mcp.tool()
def check_service_availability(
    zip_code: str = Field(..., pattern=r"^\d{5}$", description="5-digit US ZIP code, e.g. 75201"),
    service_type: Literal["fiber", "internet", "wireless", "all"] = Field(
        "all", description="Which service to check. Use 'all' if unspecified."
    ),
) -> dict:
    """Check which AT&T services (fiber, internet, wireless) are available at a
    specific 5-digit US ZIP code. Use when a user asks whether a service is
    offered at their address or area."""
    area = SERVICE_AVAILABILITY.get(zip_code)
    if area is None:
        return {"zip_code": zip_code, "found": False,
                "message": "No coverage data on file for this ZIP code."}

    if service_type == "all":
        available = {k: v for k, v in area.items() if k != "city"}
    else:
        available = {service_type: area.get(service_type, False)}

    return {"zip_code": zip_code, "city": area["city"], "found": True, "available": available}


@mcp.tool()
def get_plan_details(
    plan_type: Literal[
        "unlimited_starter", "unlimited_extra", "unlimited_premium",
        "prepaid", "fiber_300", "fiber_1gig",
    ] = Field(..., description="The exact plan identifier to look up."),
    num_lines: int = Field(1, ge=1, le=10, description="Number of wireless lines (1-10)."),
) -> dict:
    """Get pricing and features for a specific AT&T plan. Wireless plans are
    priced per line, so pass num_lines when the user mentions multiple lines."""
    plan = PLANS[plan_type]
    result = {
        "plan_id": plan_type, "name": plan["name"], "category": plan["category"],
        "data": plan["data"], "features": plan["features"],
        "price_per_line": plan["price_per_line"],
    }
    if plan["category"] == "wireless":
        result["num_lines"] = num_lines
        result["monthly_total"] = round(plan["price_per_line"] * num_lines, 2)
        result["hotspot_gb_per_line"] = plan["hotspot_gb"]
    else:
        result["monthly_total"] = plan["price_per_line"]
        result["note"] = "Internet plans are billed at a flat monthly rate."
    return result


@mcp.tool()
def check_outage_status(
    zip_code: str = Field(..., pattern=r"^\d{5}$", description="5-digit US ZIP code."),
) -> dict:
    """Check whether there is a known AT&T network or internet outage affecting a
    5-digit US ZIP code, including the estimated restore time."""
    outage = OUTAGES.get(zip_code)
    if outage is None:
        return {"zip_code": zip_code, "affected": False,
                "summary": "No known outages reported in this area."}
    return {"zip_code": zip_code, **outage}


@mcp.tool()
def lookup_account_balance(
    account_id: str = Field(
        ..., pattern=r"^ACC-\d{6}$",
        description="Account ID in the format ACC-###### (ACC- followed by exactly 6 digits).",
    ),
) -> dict:
    """Look up the current balance, due date and autopay status for an AT&T
    account. If the user has not provided an account ID, ask for it rather than
    guessing."""
    account = ACCOUNTS.get(account_id.upper())
    if account is None:
        return {"account_id": account_id, "found": False,
                "message": "No account found with that ID."}
    return {"found": True, **account}


@mcp.tool()
def find_nearest_store(
    zip_code: str = Field(..., pattern=r"^\d{5}$", description="5-digit US ZIP code."),
    radius_miles: int = Field(10, ge=1, le=50, description="Search radius in miles (1-50)."),
) -> dict:
    """Find AT&T retail stores near a 5-digit US ZIP code, optionally within a
    given radius in miles."""
    nearby = STORES.get(zip_code, [])
    within = [s for s in nearby if s["distance_miles"] <= radius_miles]
    return {"zip_code": zip_code, "radius_miles": radius_miles,
            "count": len(within), "stores": within}


@mcp.tool()
def search_web(
    query: str = Field(..., min_length=3, max_length=200, description="Search query."),
) -> list[dict]:
    """General mocked web search for miscellaneous questions that are NOT about
    competitor carriers. For Verizon or T-Mobile data use web_search instead."""
    return WEB_RESULTS.get(
        query.lower(),
        [{"title": "No mocked result found",
          "summary": "This is a mocked web search result for demo purposes."}],
    )


@mcp.tool()
def get_oncall_engineer(
    service_name: Literal["payment", "billing", "checkout"] = Field(
        ..., description="Name of the internal service."
    ),
) -> dict:
    """Get the current on-call engineer for an internal service. This is an
    internal engineering tool, not customer-facing."""
    return {"service_name": service_name,
            "oncall_engineer": ONCALL_ENGINEERS.get(service_name, "unknown")}


# --------------------------------------------------------------------------- #
# Week 5 tools — market research
# --------------------------------------------------------------------------- #
def _read(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return {"content": handle.read()}
    except FileNotFoundError:
        return {"error": "Data file not found", "path": path}


@mcp.tool()
def web_search(
    query: str = Field(..., min_length=3, description="What to look up about competitors."),
) -> dict:
    """Research COMPETITOR mobile carriers — Verizon and T-Mobile — including
    their market capitalization, subscriber numbers, ARPU, 5G coverage and
    mobile connection plans. Use this for any competitor comparison."""
    result = _read("webresults.md")
    result["query"] = query
    return result


@mcp.tool()
def query_db(
    query: str = Field(..., min_length=3, description="What to look up about AT&T."),
) -> dict:
    """Query the internal AT&T company database for corporate-level information:
    market capitalization, total subscribers, ARPU, fiber footprint, business
    segment revenue and the full plan lineup. Use this for AT&T company analysis."""
    result = _read("query.md")
    result["query"] = query
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
