"""
tools.py
--------
The tools themselves: ordinary Python functions.

Each one is registered with @tool(name, description, InputModel). By the time the
body runs, `args` is a validated Pydantic instance — the executor has already
rejected anything malformed — so these functions contain business logic only, with
no defensive argument checking.

Five AT&T tools, plus two carried over from the class reference demo so the two
styles can be compared side by side.

The `description` text matters more than it looks: it is the only thing the model
reads when deciding whether a tool is relevant. Vague descriptions produce wrong
tool selection, which is the most common failure in agentic systems.
"""

from __future__ import annotations

from .mock_data import (
    ACCOUNTS,
    ONCALL_ENGINEERS,
    OUTAGES,
    PLANS,
    SERVICE_AVAILABILITY,
    STORES,
    WEB_RESULTS,
)
from .schemas import (
    AccountBalanceInput,
    OnCallInput,
    OutageStatusInput,
    PlanDetailsInput,
    ServiceAvailabilityInput,
    StoreLocatorInput,
    WebSearchInput,
)
from .tool_registry import tool


# --------------------------------------------------------------------------- #
# AT&T tools
# --------------------------------------------------------------------------- #
@tool(
    "check_service_availability",
    "Check which AT&T services (fiber, internet, wireless) are available at a "
    "specific 5-digit US ZIP code. Use this whenever a user asks whether a service "
    "is offered at their address or area.",
    ServiceAvailabilityInput,
)
def check_service_availability(args: ServiceAvailabilityInput) -> dict:
    area = SERVICE_AVAILABILITY.get(args.zip_code)
    if area is None:
        return {
            "zip_code": args.zip_code,
            "found": False,
            "message": "No coverage data on file for this ZIP code.",
        }

    if args.service_type == "all":
        available = {k: v for k, v in area.items() if k != "city"}
    else:
        available = {args.service_type: area.get(args.service_type, False)}

    return {
        "zip_code": args.zip_code,
        "city": area["city"],
        "found": True,
        "available": available,
    }


@tool(
    "get_plan_details",
    "Get pricing and features for a specific AT&T plan. Wireless plans are priced "
    "per line, so pass num_lines when the user mentions multiple lines.",
    PlanDetailsInput,
)
def get_plan_details(args: PlanDetailsInput) -> dict:
    plan = PLANS[args.plan_type]  # Literal in the schema guarantees this key exists

    result = {
        "plan_id": args.plan_type,
        "name": plan["name"],
        "category": plan["category"],
        "data": plan["data"],
        "features": plan["features"],
        "price_per_line": plan["price_per_line"],
    }

    if plan["category"] == "wireless":
        result["num_lines"] = args.num_lines
        result["monthly_total"] = round(plan["price_per_line"] * args.num_lines, 2)
        result["hotspot_gb_per_line"] = plan["hotspot_gb"]
    else:
        result["monthly_total"] = plan["price_per_line"]
        result["note"] = "Internet plans are billed at a flat monthly rate."

    return result


@tool(
    "check_outage_status",
    "Check whether there is a known AT&T network or internet outage affecting a "
    "5-digit US ZIP code, including the estimated restore time.",
    OutageStatusInput,
)
def check_outage_status(args: OutageStatusInput) -> dict:
    outage = OUTAGES.get(args.zip_code)
    if outage is None:
        return {
            "zip_code": args.zip_code,
            "affected": False,
            "summary": "No known outages reported in this area.",
        }
    return {"zip_code": args.zip_code, **outage}


@tool(
    "lookup_account_balance",
    "Look up the current balance, due date and autopay status for an AT&T account. "
    "Requires an account ID in the format ACC-###### . If the user has not provided "
    "one, ask for it rather than guessing.",
    AccountBalanceInput,
)
def lookup_account_balance(args: AccountBalanceInput) -> dict:
    account = ACCOUNTS.get(args.account_id)
    if account is None:
        return {
            "account_id": args.account_id,
            "found": False,
            "message": "No account found with that ID.",
        }
    return {"found": True, **account}


@tool(
    "find_nearest_store",
    "Find AT&T retail stores near a 5-digit US ZIP code, optionally within a given "
    "radius in miles.",
    StoreLocatorInput,
)
def find_nearest_store(args: StoreLocatorInput) -> dict:
    nearby = STORES.get(args.zip_code, [])
    within = [s for s in nearby if s["distance_miles"] <= args.radius_miles]
    return {
        "zip_code": args.zip_code,
        "radius_miles": args.radius_miles,
        "count": len(within),
        "stores": within,
    }


# --------------------------------------------------------------------------- #
# Tools carried over from the class reference demo, for comparison
# --------------------------------------------------------------------------- #
@tool(
    "search_web",
    "Search the web for general or latest information that is not specific to AT&T "
    "accounts or coverage. Results are mocked for this assignment.",
    WebSearchInput,
)
def search_web(args: WebSearchInput) -> list[dict]:
    return WEB_RESULTS.get(
        args.query.lower(),
        [
            {
                "title": "No mocked result found",
                "summary": "This is a mocked web search result for demo purposes.",
            }
        ],
    )


@tool(
    "get_oncall_engineer",
    "Get the current on-call engineer for an internal service (payment, billing or "
    "checkout). This is an internal engineering tool, not a customer-facing one.",
    OnCallInput,
)
def get_oncall_engineer(args: OnCallInput) -> dict:
    return {
        "service_name": args.service_name,
        "oncall_engineer": ONCALL_ENGINEERS.get(args.service_name, "unknown"),
    }
