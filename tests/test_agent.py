"""Tests for the agent loop: tool calling, memory seeding, turn budget."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentlab.agent import Agent
from agentlab.builtin_tools import default_tool_registry
from agentlab.llm import MockBackend, tool_call_message
from agentlab.memory import Memory


def _make_agent(tmp: Path, max_turns: int = 6) -> Agent:
    mem = Memory(tmp / "memory.jsonl")
    reg = default_tool_registry(mem, tmp / "todo.json")
    return Agent(MockBackend(), reg, memory=mem, max_turns=max_turns)


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

    def test_no_tools_needed(self):
        with tempfile.TemporaryDirectory() as d:
            agent = _make_agent(Path(d))
            result = agent.run("Hello")
            self.assertEqual(result.tool_calls, 0)
            self.assertTrue(result.answer)


if __name__ == "__main__":
    unittest.main()