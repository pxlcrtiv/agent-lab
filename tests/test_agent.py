"""Tests for the agent loop: tool calling, memory seeding, turn budget."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentlab.agent import Agent
from agentlab.builtin_tools import default_tool_registry
from agentlab.llm import MockBackend, tool_call_message
from agentlab.memory import Memory
from agentlab.tools import ToolRegistry, tool


def _make_agent(tmp: Path, max_turns: int = 6) -> Agent:
    mem = Memory(tmp / "memory.jsonl")
    reg = default_tool_registry(mem, tmp / "todo.json")
    return Agent(MockBackend(), reg, memory=mem, max_turns=max_turns)


@tool
def echo(value: str) -> str:
    """Echo a value.

    Args:
        value: Text to return unchanged.
    """
    return value


class AgentLoopTests(unittest.TestCase):
    def test_tool_call_then_answer(self):
        with tempfile.TemporaryDirectory() as d:
            agent = _make_agent(Path(d))
            result = agent.run("What time is it in London?")
            self.assertTrue(result.answer)
            self.assertGreaterEqual(result.tool_calls, 1)
            kinds = [s.kind for s in result.steps]
            self.assertIn("tool_call", kinds)
            self.assertIn("tool_result", kinds)
            self.assertEqual(result.turns, 2)  # tool round + answer round
            self.assertEqual(result.evidence.task, "What time is it in London?")
            self.assertEqual(result.evidence.answer, result.answer)
            self.assertEqual(result.evidence.stop_reason, "final")
            self.assertEqual(len(result.evidence.exchanges), result.tool_calls)

    def test_calculator_flow(self):
        with tempfile.TemporaryDirectory() as d:
            agent = _make_agent(Path(d))
            result = agent.run("Calculate (2 + 3) * 4 / 2 ** 2")
            self.assertIn("tool_call", [s.kind for s in result.steps])
            self.assertIn("5.0", result.answer)  # mock quotes the tool result

    def test_memory_write_and_recall(self):
        with tempfile.TemporaryDirectory() as d:
            agent = _make_agent(Path(d))
            agent.run("Remember that my focus area is AI agent tooling")
            mem = agent.memory
            assert mem is not None
            hits = mem.recall("agent tooling")
            self.assertTrue(hits, "expected a memory hit")
            self.assertIn("AI agent tooling", hits[0]["text"])
            # sessions are recorded too
            self.assertGreaterEqual(len(mem.all()), 2)

    def test_turn_budget_guard(self):
        class DumbBackend(MockBackend):
            def chat(self, messages, tools=None):
                # always request clock: the loop guard should stop the agent
                return tool_call_message("c1", "clock", {})

        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "m.jsonl")
            reg = default_tool_registry(mem, Path(d) / "t.json")
            agent = Agent(DumbBackend(), reg, memory=mem, max_turns=4)
            result = agent.run("loop me")
            self.assertIn("loop", result.answer.lower())
            # identical-call guard fires before the full budget burns down
            self.assertLess(result.turns, 4)
            self.assertEqual(result.evidence.stop_reason, "repeated_call")
            self.assertEqual(len(result.evidence.exchanges), 2)
            self.assertEqual(result.evidence.exchanges[0].arguments, {})

    def test_no_tools_needed(self):
        with tempfile.TemporaryDirectory() as d:
            agent = _make_agent(Path(d))
            result = agent.run("Hello")
            self.assertEqual(result.tool_calls, 0)
            self.assertTrue(result.answer)
            self.assertEqual(result.evidence.stop_reason, "final")
            self.assertEqual(result.evidence.exchanges, ())


class AgentEvidenceTests(unittest.TestCase):
    def test_long_tool_result_is_complete_in_evidence_but_bounded_in_steps(self):
        value = "x" * 500 + "evidence-tail"

        class LongResultBackend(MockBackend):
            def chat(self, messages, tools=None):
                if messages[-1].get("role") == "tool":
                    return {"role": "assistant", "content": "That result came back clean."}
                return tool_call_message("long_1", "echo", {"value": value})

        result = Agent(LongResultBackend(), ToolRegistry(echo)).run("echo a long value")
        exchange = result.evidence.exchanges[0]
        self.assertEqual(exchange.arguments, {"value": value})
        self.assertEqual(exchange.outcome["result"], value)
        self.assertIn("evidence-tail", result.evidence.render_tool_transcript())
        preview = next(s.detail for s in result.steps if s.kind == "tool_result")
        self.assertEqual(len(preview), 300)
        self.assertNotIn("evidence-tail", preview)

    def test_failed_tool_outcome_is_preserved(self):
        class MissingToolBackend(MockBackend):
            def chat(self, messages, tools=None):
                if messages[-1].get("role") == "tool":
                    return {"role": "assistant", "content": "The tool failed."}
                return tool_call_message("missing_1", "missing", {})

        result = Agent(MissingToolBackend(), ToolRegistry()).run("use a missing tool")
        outcome = result.evidence.exchanges[0].outcome
        self.assertFalse(outcome["ok"])
        self.assertIn("unknown tool", outcome["error"])
        self.assertIn("unknown tool", result.evidence.render_tool_transcript())

    def test_multiple_calls_keep_order_and_ids(self):
        class MultiCallBackend(MockBackend):
            def chat(self, messages, tools=None):
                if messages[-1].get("role") == "tool":
                    return {"role": "assistant", "content": "That result came back clean."}
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "first", "type": "function", "function": {"name": "clock", "arguments": "{}"}},
                        {"id": "second", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1 + 1"}'}},
                    ],
                }

        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "memory.jsonl")
            registry = default_tool_registry(mem, Path(d) / "todo.json")
            result = Agent(MultiCallBackend(), registry).run("use two tools")
        self.assertEqual([e.call_id for e in result.evidence.exchanges], ["first", "second"])
        self.assertEqual([e.name for e in result.evidence.exchanges], ["clock", "calculator"])
        self.assertTrue(all(e.outcome["ok"] for e in result.evidence.exchanges))

    def test_turn_budget_is_a_structured_stop_reason(self):
        class ChangingCallBackend(MockBackend):
            def __init__(self):
                self.count = 0

            def chat(self, messages, tools=None):
                self.count += 1
                return tool_call_message("call_%d" % self.count, "echo", {"value": str(self.count)})

        result = Agent(ChangingCallBackend(), ToolRegistry(echo), max_turns=2).run("keep calling")
        self.assertEqual(result.evidence.stop_reason, "turn_budget")
        self.assertTrue(result.evidence.stopped)
        self.assertEqual(len(result.evidence.exchanges), 2)
        self.assertEqual([e.arguments for e in result.evidence.exchanges], [{"value": "1"}, {"value": "2"}])


if __name__ == "__main__":
    unittest.main()