"""
test_mcp.py — offline tests.

These start a REAL MCP server subprocess (so the protocol, discovery and
validation are genuinely exercised) but use a FAKE Anthropic client, so no API
key is needed and no credits are spent.

Run:
    python test_mcp.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared.llm as llm
from chatbot.chatbot import ChatBot, SessionMemory
from mcp_client.client import MCPClient
from workflow.workflow import extract_json, parse_plan, parse_sufficiency, run_workflow


# --------------------------------------------------------------------------- #
# Fake Anthropic client
# --------------------------------------------------------------------------- #
def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, payload):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


def response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.owner.calls.append(snapshot)
        return self.owner.responses.pop(0) if self.owner.responses else response(
            [text_block("(default)")]
        )


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.messages = FakeMessages(self)


def install(responses):
    fake = FakeLLM(responses)
    llm._client = fake
    return fake


def restore():
    llm._client = None


# --------------------------------------------------------------------------- #
# MCP server + client
# --------------------------------------------------------------------------- #
async def t_discovery():
    async with MCPClient() as client:
        names = client.tool_names()
        assert len(names) == 9, f"expected 9 tools, got {len(names)}: {names}"
        for expected in (
            "check_service_availability", "get_plan_details", "check_outage_status",
            "lookup_account_balance", "find_nearest_store", "search_web",
            "get_oncall_engineer", "web_search", "query_db",
        ):
            assert expected in names, f"missing tool {expected}"
    print("PASS  all 9 tools from weeks 4 and 5 discovered over MCP")


async def t_specs_are_anthropic_shaped():
    async with MCPClient() as client:
        for spec in client.anthropic_specs():
            assert set(spec) == {"name", "description", "input_schema"}, spec
            assert spec["input_schema"]["type"] == "object"
            assert spec["description"], f"{spec['name']} has no description"
    print("PASS  MCP tools converted to Anthropic spec format")


async def t_constraints_survive_the_protocol():
    """Week 4's Pydantic constraints must still reach the model through MCP."""
    async with MCPClient() as client:
        specs = {s["name"]: s["input_schema"]["properties"] for s in client.anthropic_specs()}

        assert specs["find_nearest_store"]["zip_code"]["pattern"] == r"^\d{5}$"
        assert specs["find_nearest_store"]["radius_miles"]["minimum"] == 1
        assert specs["find_nearest_store"]["radius_miles"]["maximum"] == 50
        assert specs["lookup_account_balance"]["account_id"]["pattern"] == r"^ACC-\d{6}$"
        assert "enum" in specs["get_plan_details"]["plan_type"]
        assert "enum" in specs["get_oncall_engineer"]["service_name"]
    print("PASS  patterns, ranges and enums survive the MCP round trip")


async def t_tool_calls_work():
    async with MCPClient() as client:
        r = await client.call("get_plan_details", {"plan_type": "unlimited_extra", "num_lines": 3})
        assert r["ok"] and "225" in r["content"], r["content"]

        r = await client.call("check_service_availability", {"zip_code": "75201"})
        assert r["ok"] and "Dallas" in r["content"]

        r = await client.call("web_search", {"query": "competitors"})
        assert r["ok"] and "Verizon" in r["content"] and "T-Mobile" in r["content"]

        r = await client.call("query_db", {"query": "market cap"})
        assert r["ok"] and "148.0 billion" in r["content"]
    print("PASS  tools return correct data through MCP")


async def t_validation_errors_come_back_as_data():
    async with MCPClient() as client:
        for name, args in [
            ("find_nearest_store", {"zip_code": "75201", "radius_miles": 999}),
            ("check_service_availability", {"zip_code": "752"}),
            ("lookup_account_balance", {"account_id": "12345"}),
            ("get_plan_details", {"plan_type": "made_up_plan"}),
            ("check_outage_status", {}),
        ]:
            r = await client.call(name, args)
            assert r["ok"] is False, f"{name} should have failed"
            assert r["content"], "an error message should be returned"
    print("PASS  invalid arguments rejected server-side, returned as data not crashes")


async def t_unknown_tool():
    async with MCPClient() as client:
        r = await client.call("teleport_user", {"x": 1})
        assert r["ok"] is False and "teleport_user" in r["content"]
    print("PASS  unknown tool handled without crashing")


async def t_disconnected_client():
    client = MCPClient()
    r = await client.call("get_plan_details", {"plan_type": "prepaid"})
    assert r["ok"] is False and "not connected" in r["content"]
    print("PASS  calling a disconnected client fails gracefully")


# --------------------------------------------------------------------------- #
# Agent loop over MCP
# --------------------------------------------------------------------------- #
async def t_agent_loop_multi_round():
    async with MCPClient() as client:
        fake = install([
            response([tool_block("t1", "query_db", {"query": "att"})], "tool_use"),
            response([tool_block("t2", "web_search", {"query": "rivals"})], "tool_use"),
            response([text_block("Final answer.")]),
        ])
        messages = [{"role": "user", "content": "compare them"}]
        answer = await llm.run_tool_loop(client, "sys", messages)
        assert answer == "Final answer."
        assert len(fake.calls) == 3

        tool_msg = fake.calls[1]["messages"][-1]
        assert tool_msg["role"] == "user"
        block = tool_msg["content"][0]
        assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
        assert "AT&T" in block["content"]
        restore()
    print("PASS  agent loop runs multiple MCP tool rounds, tool_result shape correct")


async def t_agent_loop_flags_errors():
    async with MCPClient() as client:
        fake = install([
            response([tool_block("t1", "lookup_account_balance", {"account_id": "bad"})], "tool_use"),
            response([text_block("That account ID looks wrong.")]),
        ])
        messages = [{"role": "user", "content": "balance for bad"}]
        await llm.run_tool_loop(client, "sys", messages)
        block = fake.calls[1]["messages"][-1]["content"][0]
        assert block["is_error"] is True, "validation failure should be flagged to the model"
        restore()
    print("PASS  MCP validation failure reaches the model as is_error")


async def t_agent_loop_round_cap():
    async with MCPClient() as client:
        install([
            response([tool_block(f"t{i}", "query_db", {"query": "x"})], "tool_use")
            for i in range(10)
        ])
        answer = await llm.run_tool_loop(client, "sys", [{"role": "user", "content": "q"}])
        assert "round limit" in answer
        restore()
    print("PASS  agent loop cap prevents an endless loop")


# --------------------------------------------------------------------------- #
# Chatbot (week 4 behaviour, MCP tools)
# --------------------------------------------------------------------------- #
async def t_chatbot_uses_mcp_tools():
    async with MCPClient() as client:
        install([
            response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})], "tool_use"),
            response([text_block("There is a fiber maintenance outage in Dallas.")]),
        ])
        bot = ChatBot(client)
        answer = await bot.get_answer("any outage in 75201?")
        assert "outage" in answer.lower()
        assert len(bot.session) == 2
        restore()
    print("PASS  chatbot answers using tools served over MCP")


async def t_chatbot_guardrail():
    async with MCPClient() as client:
        install([response([text_block("NO-OP")])])
        bot = ChatBot(client)
        answer = await bot.get_answer("I have a headache, what medicine should I take?")
        assert answer == "Sorry, I can't help with that."
        assert len(bot.session) == 0, "a refused request must not be stored"
        restore()
    print("PASS  guardrail still blocks and stores nothing")


async def t_chatbot_session_is_plain_text():
    async with MCPClient() as client:
        install([
            response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})], "tool_use"),
            response([text_block("No outage.")]),
        ])
        bot = ChatBot(client)
        await bot.get_answer("outage in 75201?")
        for msg in bot.session.get():
            assert isinstance(msg["content"], str), "tool blocks leaked into session memory"
        restore()
    print("PASS  session memory stores plain text only")


def t_session_cap():
    memory = SessionMemory(max_messages=4)
    for i in range(10):
        memory.add("user", f"m{i}")
    assert len(memory) == 4
    print("PASS  session memory cap enforced")


async def t_chatbot_empty_query():
    async with MCPClient() as client:
        install([])
        bot = ChatBot(client)
        assert await bot.get_answer("   ") == "Sorry, can't answer that yet."
        restore()
    print("PASS  empty query handled")


# --------------------------------------------------------------------------- #
# Workflow (week 5 behaviour, MCP tools)
# --------------------------------------------------------------------------- #
def t_parsers():
    assert parse_plan('```json\n["Q1?", "Q2?"]\n```').questions == ["Q1?", "Q2?"]
    assert parse_plan('{"questions": ["A?"]}').questions == ["A?"]
    assert extract_json('prose {"a": 1} tail') == '{"a": 1}'
    v = parse_sufficiency('{"sufficient": false, "gaps": ["g"], "notes": "n"}')
    assert v.sufficient is False and v.gaps == ["g"]
    try:
        parse_plan('["a","b","c","d","e","f","g","h"]')
        raise AssertionError("question cap not enforced")
    except AssertionError:
        raise
    except Exception:
        pass
    print("PASS  planner and verdict parsing (fences, prose, cap)")


async def t_workflow_over_mcp():
    async with MCPClient() as client:
        install([
            response([text_block('{"questions": ["What are the market caps?"]}')]),
            response([tool_block("t1", "query_db", {"query": "market cap"})], "tool_use"),
            response([text_block("AT&T is $148B.")]),
            response([text_block('{"sufficient": true, "gaps": [], "notes": "ok"}')]),
            response([text_block("FINAL REPORT")]),
        ])
        report = await run_workflow(client, "compare market caps", verbose=False)
        assert report == "FINAL REPORT"
        restore()
    print("PASS  workflow researches over MCP and synthesizes")


async def t_workflow_loopback():
    """The reasoner's gaps must go back through MCP research and reach synthesis."""
    async with MCPClient() as client:
        fake = install([
            response([text_block('{"questions": ["Q1?"]}')]),
            response([text_block("Finding 1")]),
            response([text_block('{"sufficient": false, "gaps": ["What is the churn rate?"], '
                                 '"notes": "missing churn"}')]),
            response([text_block("Churn is 0.85%")]),
            response([text_block('{"sufficient": true, "gaps": [], "notes": "ok"}')]),
            response([text_block("REPORT WITH GAP FILLED")]),
        ])
        report = await run_workflow(client, "analyse AT&T", verbose=False)
        assert report == "REPORT WITH GAP FILLED"
        synth_input = fake.calls[-1]["messages"][0]["content"]
        assert "Finding 1" in synth_input
        assert "Churn is 0.85%" in synth_input, "gap research never reached synthesis"
        restore()
    print("PASS  workflow loop-back researches gaps and feeds them to synthesis")


async def t_workflow_bad_plan_fallback():
    async with MCPClient() as client:
        install([
            response([text_block("I cannot produce JSON.")]),
            response([text_block("Finding")]),
            response([text_block('{"sufficient": true, "gaps": []}')]),
            response([text_block("REPORT")]),
        ])
        assert await run_workflow(client, "q", verbose=False) == "REPORT"
        restore()
    print("PASS  workflow falls back gracefully on an unparseable plan")


async def t_shared_connection():
    """Chatbot and workflow must share one client, not spawn two servers."""
    async with MCPClient() as client:
        bot = ChatBot(client)
        assert bot.mcp is client
        install([
            response([text_block('{"questions": ["Q?"]}')]),
            response([text_block("F")]),
            response([text_block('{"sufficient": true, "gaps": []}')]),
            response([text_block("R")]),
        ])
        await run_workflow(client, "q", verbose=False)
        assert len(client.tools) == 9, "connection should still be live afterwards"
        restore()
    print("PASS  chatbot and workflow share a single MCP connection")


# --------------------------------------------------------------------------- #
async def main() -> None:
    sync_tests = [t_session_cap, t_parsers]
    async_tests = [
        t_discovery, t_specs_are_anthropic_shaped, t_constraints_survive_the_protocol,
        t_tool_calls_work, t_validation_errors_come_back_as_data, t_unknown_tool,
        t_disconnected_client, t_agent_loop_multi_round, t_agent_loop_flags_errors,
        t_agent_loop_round_cap, t_chatbot_uses_mcp_tools, t_chatbot_guardrail,
        t_chatbot_session_is_plain_text, t_chatbot_empty_query, t_workflow_over_mcp,
        t_workflow_loopback, t_workflow_bad_plan_fallback, t_shared_connection,
    ]

    for test in sync_tests:
        test()
    for test in async_tests:
        await test()

    restore()
    print(f"\nAll {len(sync_tests) + len(async_tests)} tests passed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
