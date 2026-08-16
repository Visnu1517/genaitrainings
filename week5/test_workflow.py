"""
test_workflow.py
----------------
Offline tests. A fake Anthropic client stands in for Claude, so these run with
no API key and no credits spent, while still exercising the real planning,
research, reasoning and loop-back logic.

Run:
    python test_workflow.py
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared.agent_loop as agent_loop
import shared.llm_client as llm_client
import task1_llm_with_tools as task1
import task2_workflow as task2
import task3_planning_reasoning as task3
from shared.schemas import extract_json, parse_plan, parse_sufficiency
from shared.tool_executor import execute_tool_call
from shared.tools import query_db, web_search


# --------------------------------------------------------------------------- #
# Fakes shaped like Anthropic responses
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
        # Record a snapshot: the caller keeps appending to the same messages
        # list, so storing it by reference would make every recorded call show
        # the final state instead of what was actually sent at the time.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.owner.calls.append(snapshot)
        if self.owner.responses:
            return self.owner.responses.pop(0)
        return response([text_block("(default reply)")])


class FakeClient:
    """Stands in for anthropic.Anthropic()."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.messages = FakeMessages(self)


def install(responses):
    """Point both modules at a fake client and return it."""
    fake = FakeClient(responses)
    llm_client._client = fake
    agent_loop.get_client = lambda: fake
    llm_client.get_client = lambda: fake
    task3.chat = llm_client.chat
    return fake


def restore():
    llm_client._client = None


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_tools_read_their_files():
    att = query_db("market cap")
    comp = web_search("competitors")
    assert "AT&T" in att["content"]
    assert "148.0 billion" in att["content"]
    assert "Verizon" in comp["content"] and "T-Mobile" in comp["content"]
    assert att["query"] == "market cap"
    print("PASS  tools return their markdown files")


def test_unknown_tool_returns_error():
    result = execute_tool_call("nonexistent_tool", {"query": "x"})
    assert "error" in result and "Unknown tool" in result["error"]
    print("PASS  unknown tool returns an error instead of raising")


def test_bad_arguments_return_error():
    result = execute_tool_call("query_db", {"wrong_arg": "x"})
    assert "error" in result
    print("PASS  bad tool arguments return an error instead of raising")


# --------------------------------------------------------------------------- #
# Plan / verdict parsing
# --------------------------------------------------------------------------- #
def test_parse_plan_handles_messy_output():
    cases = [
        '["Q1?", "Q2?"]',
        '```json\n["Q1?", "Q2?"]\n```',
        'Here is the plan:\n```json\n{"questions": ["Q1?", "Q2?"]}\n```\nDone.',
        '{"questions": ["Q1?", "Q2?"]}',
    ]
    for raw in cases:
        assert parse_plan(raw).questions == ["Q1?", "Q2?"], raw
    print("PASS  planner output parsed through fences, prose and both shapes")


def test_plan_question_cap_enforced():
    try:
        parse_plan('["a","b","c","d","e","f","g","h"]')
    except Exception:
        print("PASS  too many planned questions rejected")
        return
    raise AssertionError("question cap was not enforced")


def test_parse_sufficiency():
    v = parse_sufficiency('{"sufficient": false, "gaps": ["need churn"], "notes": "n"}')
    assert v.sufficient is False and v.gaps == ["need churn"]
    v2 = parse_sufficiency('```json\n{"sufficient": true, "gaps": []}\n```')
    assert v2.sufficient is True
    print("PASS  reasoner verdict parsed")


def test_extract_json_picks_first_structure():
    assert extract_json('prose {"a": 1} trailing') == '{"a": 1}'
    print("PASS  JSON extracted from surrounding prose")


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
def test_tool_loop_multi_round():
    """The reference stops after one round; ours must continue."""
    fake = install([
        response([tool_block("t1", "query_db", {"query": "att"})], "tool_use"),
        response([tool_block("t2", "web_search", {"query": "rivals"})], "tool_use"),
        response([text_block("Final answer using both.")]),
    ])
    answer = agent_loop.run_tool_loop("sys", "question", verbose=False)
    assert answer == "Final answer using both."
    assert len(fake.calls) == 3
    restore()
    print("PASS  tool loop runs multiple rounds")


def test_tool_loop_round_cap():
    endless = [
        response([tool_block(f"t{i}", "query_db", {"query": "x"})], "tool_use")
        for i in range(10)
    ]
    install(endless)
    answer = agent_loop.run_tool_loop("sys", "q", verbose=False)
    assert "round limit" in answer
    restore()
    print("PASS  tool loop cap prevents an endless loop")


def test_tool_result_block_shape():
    fake = install([
        response([tool_block("t1", "query_db", {"query": "att"})], "tool_use"),
        response([text_block("done")]),
    ])
    agent_loop.run_tool_loop("sys", "q", verbose=False)
    sent = fake.calls[1]["messages"][-1]
    assert sent["role"] == "user"
    block = sent["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
    assert "AT&T" in json.loads(block["content"])["content"]
    restore()
    print("PASS  tool_result block matches the Anthropic format")


# --------------------------------------------------------------------------- #
# TASK-1 and TASK-2
# --------------------------------------------------------------------------- #
def test_task1_single_pass():
    install([
        response([tool_block("t1", "query_db", {"query": "att"})], "tool_use"),
        response([text_block("One-shot answer.")]),
    ])
    assert task1.run(verbose=False) == "One-shot answer."
    restore()
    print("PASS  TASK-1 runs one-shot with tools")


def test_task2_runs_three_fixed_branches():
    fake = install([
        response([text_block("Plans section.")]),
        response([text_block("Market cap section.")]),
        response([text_block("Growth section.")]),
    ])
    report = task2.run(verbose=False)
    assert len(fake.calls) == 3, "expected exactly one call per branch"
    for heading in ("Mobile Connection Plans", "Market Capitalization", "Growth Opportunities"):
        assert f"## {heading}" in report
    assert "Plans section." in report and "Growth section." in report
    restore()
    print("PASS  TASK-2 runs 3 fixed branches and joins them under headings")


def test_task2_branches_share_tool_data():
    fake = install([response([text_block(f"S{i}")]) for i in range(3)])
    task2.run(verbose=False)
    for call in fake.calls:
        content = call["messages"][0]["content"]
        assert "query_db" in content and "web_search" in content
    restore()
    print("PASS  TASK-2 gives every branch both tools' data")


# --------------------------------------------------------------------------- #
# TASK-3
# --------------------------------------------------------------------------- #
def test_task3_happy_path_no_loopback():
    install([
        # plan
        response([text_block('{"questions": ["Q1?", "Q2?"]}')]),
        # research x2 (no tools used, straight answers)
        response([text_block("Finding 1")]),
        response([text_block("Finding 2")]),
        # reason -> sufficient
        response([text_block('{"sufficient": true, "gaps": [], "notes": "all covered"}')]),
        # synthesize
        response([text_block("FINAL REPORT")]),
    ])
    assert task3.run(verbose=False) == "FINAL REPORT"
    restore()
    print("PASS  TASK-3 plan -> research -> reason(sufficient) -> synthesize")


def test_task3_loopback_researches_gaps():
    """The behaviour the class reference documents but never implements."""
    fake = install([
        response([text_block('{"questions": ["Q1?"]}')]),          # plan
        response([text_block("Finding 1")]),                        # research Q1
        response([text_block('{"sufficient": false, "gaps": ["What is the churn rate?"], '
                             '"notes": "missing churn"}')]),        # reason -> INSUFFICIENT
        response([text_block("Churn is 0.85%")]),                   # research the gap
        response([text_block('{"sufficient": true, "gaps": [], "notes": "ok"}')]),  # reason again
        response([text_block("FINAL REPORT WITH GAP FILLED")]),     # synthesize
    ])
    report = task3.run(verbose=False)
    assert report == "FINAL REPORT WITH GAP FILLED"

    # The synthesizer must have received BOTH the original finding and the gap research.
    synth_input = fake.calls[-1]["messages"][0]["content"]
    assert "Finding 1" in synth_input
    assert "Churn is 0.85%" in synth_input, "gap research never reached the synthesizer"
    assert "What is the churn rate?" in synth_input
    restore()
    print("PASS  TASK-3 loop-back researches gaps and feeds them to synthesis")


def test_task3_reasoning_round_cap():
    """A reasoner that is never satisfied must not loop forever."""
    never_happy = [response([text_block('{"questions": ["Q1?"]}')]),
                   response([text_block("Finding 1")])]
    for _ in range(10):
        never_happy.append(
            response([text_block('{"sufficient": false, "gaps": ["more?"], "notes": "no"}')])
        )
        never_happy.append(response([text_block("gap finding")]))
    never_happy.append(response([text_block("REPORT ANYWAY")]))

    install(never_happy)
    report = task3.run(verbose=False)
    assert isinstance(report, str) and report
    restore()
    print("PASS  TASK-3 reasoning rounds are capped")


def test_task3_survives_unparseable_plan():
    install([
        response([text_block("I cannot produce JSON, sorry.")]),   # bad plan
        response([text_block("Finding for whole request")]),        # fallback research
        response([text_block('{"sufficient": true, "gaps": []}')]),
        response([text_block("REPORT")]),
    ])
    assert task3.run(verbose=False) == "REPORT"
    restore()
    print("PASS  TASK-3 falls back gracefully when the plan is unparseable")


def test_task3_survives_unparseable_verdict():
    install([
        response([text_block('{"questions": ["Q1?"]}')]),
        response([text_block("Finding 1")]),
        response([text_block("not json at all")]),   # bad verdict
        response([text_block("REPORT")]),
    ])
    assert task3.run(verbose=False) == "REPORT"
    restore()
    print("PASS  TASK-3 proceeds when the verdict is unparseable")


def test_task3_uses_all_four_prompts():
    from shared.prompt_loader import load_prompt

    for name in ("task3_planner.md", "task3_researcher.md",
                 "task3_reasoner.md", "task3_synthesizer.md"):
        assert len(load_prompt(name)) > 100, name
    print("PASS  all four TASK-3 prompt files load")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
    finally:
        restore()
    print(f"\nAll {len(tests)} tests passed.")
