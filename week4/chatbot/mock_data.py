"""
mock_data.py
------------
Stand-in data for the tools. In a real assistant these lookups would hit AT&T
systems; mocking them keeps the assignment self-contained and deterministic, which
is exactly what the class reference demo does.

The important part of the exercise is the tool-calling loop and validation, not
where the bytes come from.
"""

# --------------------------------------------------------------------------- #
# AT&T mock data
# --------------------------------------------------------------------------- #

# ZIP -> which services are available there
SERVICE_AVAILABILITY = {
    "75201": {"fiber": True, "internet": True, "wireless": True, "city": "Dallas, TX"},
    "30301": {"fiber": True, "internet": True, "wireless": True, "city": "Atlanta, GA"},
    "10001": {"fiber": False, "internet": True, "wireless": True, "city": "New York, NY"},
    "90001": {"fiber": False, "internet": True, "wireless": True, "city": "Los Angeles, CA"},
    "60601": {"fiber": True, "internet": True, "wireless": True, "city": "Chicago, IL"},
}

# plan id -> details. Wireless plans price per line; fiber plans are flat.
PLANS = {
    "unlimited_starter": {
        "name": "Unlimited Starter",
        "category": "wireless",
        "price_per_line": 65.00,
        "data": "Unlimited talk, text and data",
        "hotspot_gb": 0,
        "features": ["5G access on compatible devices", "Standard definition streaming"],
    },
    "unlimited_extra": {
        "name": "Unlimited Extra",
        "category": "wireless",
        "price_per_line": 75.00,
        "data": "Unlimited talk, text and data",
        "hotspot_gb": 30,
        "features": ["5G access", "30GB hotspot per line", "50GB premium data"],
    },
    "unlimited_premium": {
        "name": "Unlimited Premium",
        "category": "wireless",
        "price_per_line": 85.00,
        "data": "Unlimited talk, text and data",
        "hotspot_gb": 60,
        "features": ["5G access", "60GB hotspot per line", "High definition streaming"],
    },
    "prepaid": {
        "name": "Prepaid Unlimited",
        "category": "wireless",
        "price_per_line": 50.00,
        "data": "Unlimited talk, text and data",
        "hotspot_gb": 10,
        "features": ["No annual contract", "No credit check"],
    },
    "fiber_300": {
        "name": "AT&T Fiber 300",
        "category": "internet",
        "price_per_line": 55.00,
        "data": "300 Mbps symmetrical",
        "hotspot_gb": 0,
        "features": ["Unlimited data", "Equipment included"],
    },
    "fiber_1gig": {
        "name": "AT&T Fiber 1 GIG",
        "category": "internet",
        "price_per_line": 80.00,
        "data": "1000 Mbps symmetrical",
        "hotspot_gb": 0,
        "features": ["Unlimited data", "Equipment included", "Suited to gaming and streaming"],
    },
}

# ZIP -> current outage information
OUTAGES = {
    "75201": {
        "affected": True,
        "service": "internet",
        "started": "2026-08-16T09:15:00Z",
        "estimated_restore": "2026-08-16T16:00:00Z",
        "summary": "Fiber maintenance affecting some customers in downtown Dallas.",
    },
    "10001": {
        "affected": True,
        "service": "wireless",
        "started": "2026-08-16T11:00:00Z",
        "estimated_restore": "unknown",
        "summary": "Elevated wireless error rates under investigation.",
    },
}

# account id -> billing snapshot
ACCOUNTS = {
    "ACC-100001": {
        "account_id": "ACC-100001",
        "balance_due": 128.45,
        "currency": "USD",
        "due_date": "2026-09-01",
        "autopay_enabled": False,
        "plan": "unlimited_extra",
        "lines": 3,
    },
    "ACC-100002": {
        "account_id": "ACC-100002",
        "balance_due": 0.00,
        "currency": "USD",
        "due_date": "2026-09-05",
        "autopay_enabled": True,
        "plan": "fiber_1gig",
        "lines": 1,
    },
}

# ZIP -> nearby retail stores
STORES = {
    "75201": [
        {"name": "AT&T Store - Main Street", "address": "1512 Main St, Dallas, TX", "distance_miles": 1.2},
        {"name": "AT&T Store - Uptown", "address": "2401 McKinney Ave, Dallas, TX", "distance_miles": 3.8},
        {"name": "AT&T Store - Deep Ellum", "address": "2800 Elm St, Dallas, TX", "distance_miles": 12.4},
    ],
    "30301": [
        {"name": "AT&T Store - Peachtree", "address": "1100 Peachtree St NE, Atlanta, GA", "distance_miles": 2.1},
        {"name": "AT&T Store - Midtown", "address": "800 W Peachtree St, Atlanta, GA", "distance_miles": 4.6},
    ],
}


# --------------------------------------------------------------------------- #
# Data carried over from the class reference demo (for comparison)
# --------------------------------------------------------------------------- #
ONCALL_ENGINEERS = {
    "payment": "payment-oncall@company.com",
    "billing": "billing-oncall@company.com",
    "checkout": "checkout-oncall@company.com",
}

WEB_RESULTS = {
    "what is tool calling": [
        {
            "title": "Tool calling overview",
            "summary": "Tool calling allows an LLM to request external functions or APIs "
                       "instead of only generating text.",
        }
    ],
    "agentic ai tools": [
        {
            "title": "Agentic AI and tools",
            "summary": "Agentic systems use tools to access data, perform actions, observe "
                       "results, and continue the task.",
        }
    ],
}
