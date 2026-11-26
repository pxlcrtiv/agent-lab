"""Tests for the canonical record consumed by the critic."""

import unittest

from agentlab.run_evidence import RunEvidence, ToolExchange


class RunEvidenceTests(unittest.TestCase):
    def test_no_tools_renders_explicit_marker(self):
        evidence = RunEvidence(task="task", answer="answer")
        self.assertEqual(evidence.render_tool_transcript(), "(no tools were called)")

    def test_renders_complete_arguments_and_outcomes(self):
        value = "x" * 500 + "complete-tail"
        evidence = RunEvidence(
            task="task",
            answer="answer",
            exchanges=(
                ToolExchange(
                    call_id="call_1",
                    turn=1,
                    name="long_tool",
                    arguments={"value": value},
                    outcome={"ok": True, "tool": "long_tool", "result": value},
                ),
            ),
        )
        transcript = evidence.render_tool_transcript()
        self.assertIn(value, transcript)
        self.assertIn("complete-tail", transcript)
        self.assertIn('"ok": true', transcript)

    def test_preserves_failed_outcomes(self):
        evidence = RunEvidence(
            task="task",
            answer="answer",
            exchanges=(
                ToolExchange(
                    call_id="call_1",
                    turn=1,
                    name="missing",
                    arguments={},
                    outcome={"ok": False, "tool": "missing", "error": "unknown tool"},
                ),
            ),
        )
        transcript = evidence.render_tool_transcript()
        self.assertIn('"ok": false', transcript)
        self.assertIn("unknown tool", transcript)

    def test_preserves_exchange_order_and_call_ids(self):
        exchanges = (
            ToolExchange("first", 1, "clock", {}, {"ok": True, "result": "12:00"}),
            ToolExchange("second", 1, "calculator", {"expression": "1 + 1"}, {"ok": True, "result": 2}),
        )
        evidence = RunEvidence(task="task", answer="answer", exchanges=exchanges)
        self.assertEqual([e.call_id for e in evidence.exchanges], ["first", "second"])
        transcript = evidence.render_tool_transcript()
        self.assertLess(transcript.index("clock"), transcript.index("calculator"))

    def test_stop_reason_owns_early_termination(self):
        for reason, stopped in (("final", False), ("repeated_call", True), ("turn_budget", True)):
            with self.subTest(reason=reason):
                evidence = RunEvidence(task="task", answer="answer", stop_reason=reason)
                self.assertEqual(evidence.stopped, stopped)


if __name__ == "__main__":
    unittest.main()
