"""
tool_specs.py
-------------
What the model sees: each tool's name, description, and argument schema.

Anthropic's format differs from the OpenAI format used in the class reference:
    Anthropic : {"name", "description", "input_schema"}
    OpenAI    : {"type": "function", "function": {..., "parameters"}}
"""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for information about competitor mobile carriers such as "
            "Verizon and T-Mobile, including their market capitalization, subscriber "
            "numbers, and mobile connection plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search the web for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_db",
        "description": (
            "Query the internal company database for AT&T information, including its "
            "market capitalization, subscriber numbers, fiber footprint, and mobile "
            "connection plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up in the AT&T database.",
                }
            },
            "required": ["query"],
        },
    },
]
