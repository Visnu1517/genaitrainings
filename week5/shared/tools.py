"""
tools.py
--------
The two tools, matching the class reference demo.

Both read a local markdown file and return its content, so the whole assignment
runs with no network or database:

    web_search -> webresults.md   (Verizon + T-Mobile mock data)
    query_db   -> query.md        (AT&T mock data)

The `query` argument does not filter anything — the full file comes back either
way. That is intentional in the reference: the point is to exercise the
tool-calling flow, not to build a search engine.
"""

from __future__ import annotations

import os

# week5/ — one level above this shared/ package, where the .md files live.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEB_RESULTS_FILE = os.path.join(_BASE_DIR, "webresults.md")
QUERY_DB_FILE = os.path.join(_BASE_DIR, "query.md")


def _read_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return {"content": handle.read()}
    except FileNotFoundError:
        return {"error": "Data file not found", "path": path}


def web_search(query: str) -> dict:
    """Return competitor carrier information (Verizon, T-Mobile)."""
    result = _read_file(WEB_RESULTS_FILE)
    result["query"] = query
    return result


def query_db(query: str) -> dict:
    """Return internal AT&T company information."""
    result = _read_file(QUERY_DB_FILE)
    result["query"] = query
    return result
