"""
test_chatbot.py
---------------
Offline tests. A fake LLM stands in for Claude, so these run with no API key and
no Ollama, while still exercising the real validation and tool-execution paths.

Run:
    python test_chatbot.py
"""

import json
import os
import shutil
import sys
from types import SimpleNamespace

from chatbot import ChatBot, all_specs, execute_tool, tool_names
from chatbot.embeddings import Embedder
from chatbot.memory import SessionMemory
from chatbot.schemas import AccountBalanceInput, ServiceAvailabilityInput
from chatbot.vector_store import VectorStore, chunk_text

TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_test")


# --------------------------------------------------------------------------- #
# Fakes that mimic the shape of an Anthropic response
# --------------------------------------------------------------------------- #
def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_id, name, payload):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=payload)


def response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


class FakeLLM:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, responses=None):
        self.responses = list(responses or [response([text_block("OK.")])])
        self.calls = []
        self.last_system = None
        self.last_messages = None
        self.last_tools = None

    def create(self, messages, system=None, tools=None):
        self.last_system = system
        self.last_messages = messages
        self.last_tools = tools
        self.calls.append({"messages": list(messages), "system": system})
        return self.responses.pop(0) if self.responses else response([text_block("Done.")])


def fresh_bot(responses=None):
    shutil.rmtree(TEST_DATA, ignore_errors=True)
    return ChatBot(llm=FakeLLM(responses), data_dir=TEST_DATA, prefer_ollama=False)


# --------------------------------------------------------------------------- #
# Pydantic validation
# --------------------------------------------------------------------------- #
def test_valid_input_accepted():
    args = ServiceAvailabilityInput(zip_code="75201")
    assert args.zip_code == "75201"
    assert args.service_type == "all"          # default applied
    print("PASS  valid input accepted, defaults applied")


def test_invalid_zip_rejected():
    result = execute_tool("check_service_availability", {"zip_code": "752"})
    assert result.ok is False
    assert "zip_code" in result.error
    assert result.details[0]["field"] == "zip_code"
    print("PASS  malformed ZIP rejected by Pydantic with field detail")


def test_account_id_pattern_and_normalisation():
    bad = execute_tool("lookup_account_balance", {"account_id": "12345"})
    assert bad.ok is False, "invalid account id should fail"

    # lowercase is accepted then normalised by the field_validator
    good = AccountBalanceInput(account_id="acc-100001")
    assert good.account_id == "ACC-100001"
    print("PASS  account ID pattern enforced and normalised")


def test_out_of_range_rejected():
    result = execute_tool("find_nearest_store", {"zip_code": "75201", "radius_miles": 999})
    assert result.ok is False
    assert "radius_miles" in result.error
    print("PASS  out-of-range number rejected")


def test_bad_enum_rejected():
    result = execute_tool("get_plan_details", {"plan_type": "made_up_plan"})
    assert result.ok is False
    print("PASS  invalid enum value rejected")


def test_missing_required_field():
    result = execute_tool("check_outage_status", {})
    assert result.ok is False
    assert "zip_code" in result.error
    print("PASS  missing required field rejected")


def test_unknown_tool_handled():
    result = execute_tool("teleport_user", {"x": 1})
    assert result.ok is False
    assert "Unknown tool" in result.error
    print("PASS  hallucinated tool name handled without crashing")


def test_executor_never_raises():
    """Every bad-input shape must return a ToolResult, not raise."""
    for name, args in [
        ("check_service_availability", {"zip_code": None}),
        ("get_plan_details", {"plan_type": 123}),
        ("search_web", {"query": "a"}),           # below min_length
        ("get_oncall_engineer", {"service_name": "nope"}),
        ("find_nearest_store", {}),
    ]:
        result = execute_tool(name, args)
        assert result.ok is False, f"{name} should have failed"
    print("PASS  executor returns errors instead of raising")


# --------------------------------------------------------------------------- #
# Tool behaviour
# --------------------------------------------------------------------------- #
def test_tools_return_expected_data():
    r = execute_tool("check_service_availability", {"zip_code": "75201", "service_type": "fiber"})
    assert r.ok and r.data["available"]["fiber"] is True

    r = execute_tool("get_plan_details", {"plan_type": "unlimited_extra", "num_lines": 3})
    assert r.data["monthly_total"] == 225.0, r.data

    r = execute_tool("check_outage_status", {"zip_code": "99999"})
    assert r.data["affected"] is False

    r = execute_tool("find_nearest_store", {"zip_code": "75201", "radius_miles": 5})
    assert r.data["count"] == 2, r.data          # the 12.4-mile store is excluded
    print("PASS  tools return correct data (incl. per-line maths and radius filter)")


def test_schemas_generated_from_pydantic():
    specs = all_specs()
    assert len(specs) == len(tool_names())
    for spec in specs:
        assert set(spec) == {"name", "description", "input_schema"}, spec
        assert spec["input_schema"]["type"] == "object"
        assert "title" not in spec["input_schema"]
    plan = next(s for s in specs if s["name"] == "get_plan_details")
    assert "enum" in plan["input_schema"]["properties"]["plan_type"]
    print("PASS  Anthropic specs auto-generated from Pydantic models")


# --------------------------------------------------------------------------- #
# Agentic loop
# --------------------------------------------------------------------------- #
def test_single_tool_round():
    bot = fresh_bot([
        response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})],
                 stop_reason="tool_use"),
        response([text_block("There is a fiber maintenance outage in Dallas.")]),
    ])
    answer = bot.get_answer("any outage in 75201?")
    assert "outage" in answer.lower()
    assert bot.last_tool_calls[0]["tool"] == "check_outage_status"
    assert bot.last_tool_calls[0]["ok"] is True
    print("PASS  single tool round executes and produces a final answer")


def test_tool_result_message_shape():
    bot = fresh_bot([
        response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})],
                 stop_reason="tool_use"),
        response([text_block("Done.")]),
    ])
    bot.get_answer("outage in 75201?")
    sent = bot.llm.calls[1]["messages"]
    tool_msg = sent[-1]
    assert tool_msg["role"] == "user"
    block = tool_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert json.loads(block["content"])["ok"] is True
    print("PASS  tool_result block matches the Anthropic format")


def test_multi_round_loop():
    bot = fresh_bot([
        response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})],
                 stop_reason="tool_use"),
        response([tool_block("t2", "find_nearest_store", {"zip_code": "75201"})],
                 stop_reason="tool_use"),
        response([text_block("Outage plus 3 stores nearby.")]),
    ])
    answer = bot.get_answer("outage and stores in 75201?")
    assert len(bot.last_tool_calls) == 2
    assert "stores" in answer
    print("PASS  loop continues across multiple tool rounds")


def test_parallel_tool_calls_in_one_round():
    bot = fresh_bot([
        response(
            [
                tool_block("t1", "check_outage_status", {"zip_code": "75201"}),
                tool_block("t2", "check_service_availability", {"zip_code": "75201"}),
            ],
            stop_reason="tool_use",
        ),
        response([text_block("Both checked.")]),
    ])
    bot.get_answer("outage and availability in 75201?")
    assert len(bot.last_tool_calls) == 2
    results = bot.llm.calls[1]["messages"][-1]["content"]
    assert len(results) == 2, "both results must go back in one user message"
    print("PASS  parallel tool calls handled in a single round")


def test_validation_error_reaches_model_as_is_error():
    bot = fresh_bot([
        response([tool_block("t1", "lookup_account_balance", {"account_id": "oops"})],
                 stop_reason="tool_use"),
        response([text_block("That account ID doesn't look right - could you check it?")]),
    ])
    answer = bot.get_answer("what is my balance for account oops")
    block = bot.llm.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert json.loads(block["content"])["ok"] is False
    assert bot.last_tool_calls[0]["ok"] is False
    assert "check it" in answer
    print("PASS  validation failure is fed back as an error the model can recover from")


def test_loop_cap():
    """A model that never stops calling tools must not hang."""
    endless = [
        response([tool_block(f"t{i}", "check_outage_status", {"zip_code": "75201"})],
                 stop_reason="tool_use")
        for i in range(20)
    ]
    bot = fresh_bot(endless)
    answer = bot.get_answer("loop forever")
    assert "narrow it down" in answer
    assert len(bot.last_tool_calls) == 5, len(bot.last_tool_calls)
    print("PASS  MAX_TOOL_ROUNDS caps a runaway loop")


def test_tools_are_passed_to_the_model():
    bot = fresh_bot()
    bot.get_answer("hello")
    assert bot.llm.last_tools and len(bot.llm.last_tools) == len(tool_names())
    print("PASS  tool specs are sent on every request")


# --------------------------------------------------------------------------- #
# Week-3 behaviour must still hold
# --------------------------------------------------------------------------- #
def test_guardrail_blocks_before_tools_run():
    bot = fresh_bot([response([text_block("NO-OP")])])
    answer = bot.get_answer("I have a headache, what medicine should I take?")
    assert answer == "Sorry, I can't help with that."
    assert bot.last_tool_calls == [], "no tool should run on a blocked request"
    assert len(bot.session) == 0
    assert len(bot.long_term) == 0
    print("PASS  guardrail blocks before any tool executes and stores nothing")


def test_session_history_excludes_tool_blocks():
    """Stored history must be plain text, or Anthropic 400s on the next call."""
    bot = fresh_bot([
        response([tool_block("t1", "check_outage_status", {"zip_code": "75201"})],
                 stop_reason="tool_use"),
        response([text_block("No outage.")]),
    ])
    bot.get_answer("outage in 75201?")
    for msg in bot.session.get_messages():
        assert isinstance(msg["content"], str), "tool blocks leaked into session memory"
    print("PASS  session memory stores plain text only")


def test_grounding_and_memory_still_injected():
    bot = fresh_bot([response([text_block("AT&T offers fiber.")])])
    bot.get_answer("what home internet does AT&T offer?")
    assert "Grounding Context (retrieved)" in bot.llm.last_system

    bot2 = ChatBot(llm=FakeLLM([response([text_block("Hi Vishnu.")])]),
                   data_dir=TEST_DATA, prefer_ollama=False)
    bot2.get_answer("my name is Vishnu")
    bot3 = ChatBot(llm=FakeLLM([response([text_block("Vishnu.")])]),
                   data_dir=TEST_DATA, prefer_ollama=False)
    bot3.get_answer("what is my name?")
    assert "Vishnu" in bot3.llm.last_system
    print("PASS  RAG grounding and long-term memory still work")


def test_session_memory_capped():
    sm = SessionMemory(max_messages=4)
    for i in range(10):
        sm.add("user", f"msg {i}")
    assert len(sm) == 4
    print("PASS  session memory cap still enforced")


def test_chunking_and_retrieval():
    text = " ".join(str(i) for i in range(1200))
    assert len(chunk_text(text)) >= 2

    shutil.rmtree(TEST_DATA, ignore_errors=True)
    store = VectorStore(os.path.join(TEST_DATA, "vs"), Embedder(prefer_ollama=False))
    store.add_text("Paris is the capital of France")
    store.add_text("AT&T Fiber provides high speed home internet")
    hits = store.search("fiber internet", k=1)
    assert "Fiber" in hits[0]["content"]
    print("PASS  chunking and vector retrieval still correct")


def test_empty_query():
    bot = fresh_bot()
    assert bot.get_answer("  ") == "Sorry, can't answer that yet."
    print("PASS  empty query handled")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(TEST_DATA, ignore_errors=True)
    print(f"\nAll {len(tests)} tests passed.")
