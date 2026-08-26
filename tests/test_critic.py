"""Unit tests for the critic (self-evaluation) module -- all offline."""

import unittest

from agentlab.critic import CRITIC_SYSTEM, Critic, MockCritic, REVISE_SYSTEM
from agentlab.llm import ChatBackend


class ScriptedBackend(ChatBackend):
    """Deterministic fake LLM: returns queued contents per system prompt."""

    def __init__(self, evaluator_replies, reviser_replies=None):
        self.evaluator_replies = list(evaluator_replies)
        self.reviser_replies = list(reviser_replies or [])
        self.calls = []
        self.users = []

    name = "scripted"

    def chat(self, messages, tools=None):
        system = messages[0]["content"]
        self.calls.append(system)
        self.users.append(messages[1]["content"] if len(messages) > 1 else "")
        if system == CRITIC_SYSTEM:
            return {"role": "assistant", "content": self.evaluator_replies.pop(0)}
        if system == REVISE_SYSTEM:
            return {"role": "assistant", "content": self.reviser_replies.pop(0)}
        return {"role": "assistant", "content": "?"}


def pass_json(scores=None, overall=None, verdict="pass", feedback="All good."):
    scores = scores or {"grounded": 5, "complete": 5, "concise": 5, "safe": 5}
    overall = overall or min(scores.values())
    return '{"scores": %s, "overall": %d, "verdict": "%s", "feedback": "%s"}' % (
        scores.__repr__().replace("'", '"'),
        overall,
        verdict,
        feedback,
    )


class MockCriticTests(unittest.TestCase):
    def test_passes_a_grounded_answer_without_revision(self):
        c = MockCritic()
        critique = c.run("What time is it in London?", "That result came back clean. 12:00 UTC.")
        self.assertTrue(critique.passed)
        self.assertGreaterEqual(critique.overall, 4)
        self.assertEqual(critique.revisions, [])

    def test_revises_a_loop_guard_answer_then_passes(self):
        c = MockCritic()
        critique = c.run(
            "What time is it in London?",
            "Stopped: repeated identical tool call (clock) -- possible loop.",
        )
        self.assertTrue(critique.passed)
        self.assertEqual(len(critique.revisions), 1)
        self.assertIn("revised", critique.answer)

    def test_max_revisions_zero_locks_verdict(self):
        c = MockCritic(max_revisions=0)
        critique = c.run("task", "Stopped: repeated identical tool call (clock) -- possible loop.")
        self.assertFalse(critique.passed)
        self.assertEqual(len(critique.revisions), 1)

    def test_long_answer_flagged_not_concise_and_revised(self):
        c = MockCritic()
        blob = "That result came back clean. " + "filler words " * 80  # > 400 chars
        critique = c.run("task", blob)
        self.assertEqual(len(critique.revisions), 1)  # flagged, then revised
        self.assertTrue(critique.passed)  # after one revision it passes
        self.assertEqual(critique.scores["concise"], 5)  # final round is concise
        self.assertLess(len(critique.answer), 400)  # and it was actually trimmed


class CriticJsonTests(unittest.TestCase):
    def test_parse_fenced_json(self):
        raw = "Here you go:\n```json\n%s\n```" % pass_json()
        parsed = Critic._parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["verdict"], "pass")
        self.assertEqual(parsed["overall"], 5)

    def test_parse_with_preamble_and_trailing_text(self):
        raw = "Evaluation: " + pass_json() + " hope that helps!"
        parsed = Critic._parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["verdict"], "pass")

    def test_parse_nested_braces(self):
        parsed = Critic._parse_json('{"scores": {"a": {"x": 1}}, "verdict": "revise"}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["verdict"], "revise")

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(Critic._parse_json("no json here at all"))
        self.assertIsNone(Critic._parse_json('{"broken": '))


class CriticLoopTests(unittest.TestCase):
    def test_pass_on_first_round_no_revision_call(self):
        backend = ScriptedBackend([pass_json()])
        c = Critic(backend)
        critique = c.run("task", "answer")
        self.assertTrue(critique.passed)
        self.assertEqual(critique.revisions, [])
        self.assertEqual(len([x for x in backend.calls if "evaluator" in x]), 1)
        self.assertEqual(len([x for x in backend.calls if "editor" in x]), 0)

    def test_revise_then_pass(self):
        backend = ScriptedBackend(
            [
                pass_json(verdict="revise", scores={"grounded": 2, "complete": 2, "concise": 5, "safe": 5}, overall=2, feedback="Ground it."),
                pass_json(),
            ],
            reviser_replies=["A grounded revised answer."],
        )
        c = Critic(backend)
        critique = c.run("task", "old answer")
        self.assertTrue(critique.passed)
        self.assertEqual(len(critique.revisions), 1)
        self.assertEqual(critique.answer, "A grounded revised answer.")
        self.assertEqual(critique.revisions[0]["feedback"], "Ground it.")

    def test_override_verdict_requires_min_scores(self):
        # Model says pass but a score is below min -> still revise; capped at 0 revisions.
        backend = ScriptedBackend(
            [pass_json(verdict="pass", scores={"grounded": 2, "complete": 2, "concise": 5, "safe": 5}, overall=2)]
        )
        c = Critic(backend, min_pass=4, max_revisions=0)
        critique = c.run("task", "answer")
        self.assertFalse(critique.passed)
        self.assertEqual(len(critique.revisions), 1)

    def test_malformed_evaluator_output_is_revise(self):
        backend = ScriptedBackend(["I think it's fine.", pass_json()], reviser_replies=["better answer"])
        c = Critic(backend)
        critique = c.run("task", "answer")
        self.assertEqual(len(critique.revisions), 1)
        self.assertIn("could not parse", critique.revisions[0]["feedback"])

    def test_revision_cap(self):
        backend = ScriptedBackend(
            [
                pass_json(verdict="revise", scores={"grounded": 2, "complete": 2, "concise": 5, "safe": 5}, overall=2),
                pass_json(verdict="revise", scores={"grounded": 2, "complete": 2, "concise": 5, "safe": 5}, overall=2),
            ],
            reviser_replies=["r1", "r2"],
        )
        c = Critic(backend, max_revisions=1)
        critique = c.run("task", "answer")
        self.assertEqual(len(critique.revisions), 2)  # initial + 1 revise round, then cap
        self.assertFalse(critique.passed)
        self.assertEqual(critique.answer, "r1")

    def test_system_prompts_are_used(self):
        backend = ScriptedBackend([pass_json()], reviser_replies=["x"])
        c = Critic(backend)
        c.run("task", "answer")
        evaluator_call = next(x for x in backend.calls if "evaluator" in x)
        self.assertEqual(evaluator_call, CRITIC_SYSTEM)
        self.assertIn("evaluator", CRITIC_SYSTEM)
        self.assertIn("editor", REVISE_SYSTEM)

    def test_transcript_is_passed_to_the_editor(self):
        from agentlab.agent import Step

        backend = ScriptedBackend(
            [
                pass_json(verdict="revise", scores={"grounded": 2, "complete": 2, "concise": 5, "safe": 5}, overall=2),
                pass_json(),
            ],
            reviser_replies=["improved and grounded"],
        )
        c = Critic(backend)
        steps = [
            Step(1, "tool_call", "clock", '{"timezone": "Europe/London"}'),
            Step(1, "tool_result", "clock", "03:29:12 BST (UTC+0100)"),
        ]
        c.run("task", "answer", steps=steps)
        editor_idx = next(i for i, s in enumerate(backend.calls) if s == REVISE_SYSTEM)
        editor_user = backend.users[editor_idx]
        self.assertIn("TOOL TRANSCRIPT", editor_user)
        self.assertIn("03:29:12 BST (UTC+0100)", editor_user)
        self.assertIn("TOOL TRANSCRIPT", backend.users[0])  # evaluator sees it too

    def test_transcript_rendering_no_tools(self):
        self.assertEqual(Critic._render_transcript([]), "(no tools were called)")


if __name__ == "__main__":
    unittest.main()