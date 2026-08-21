"""Tests for the workflow engine: retries, guards, compensation, persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentlab.workflow import Workflow


class WorkflowTests(unittest.TestCase):
    def test_happy_path(self):
        wf = Workflow("demo")
        wf.add_step("a", lambda s: {"x": s.get("x", 0) + 1})
        wf.add_step("b", lambda s: {"y": s["x"] * 2})
        result = wf.run({"x": 10}, persist_path=None)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.state["x"], 11)
        self.assertEqual(result.state["y"], 22)
        self.assertEqual([s.name for s in result.steps], ["a", "b"])
        self.assertTrue(all(s.status == "ok" for s in result.steps))

    def test_retry_then_success(self):
        calls = {"n": 0}

        def flaky(state):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("boom")
            return {"ok": True}

        wf = Workflow("flaky")
        wf.add_step("work", flaky, retries=2, backoff_s=0.01)
        result = wf.run()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.steps[0].attempts, 2)

    def test_fail_stops_pipeline(self):
        def boom(state):
            raise RuntimeError("kaput")

        wf = Workflow("stop")
        wf.add_step("boom", boom, retries=0)
        wf.add_step("never", lambda s: {"ran": True}, retries=0)
        result = wf.run()
        self.assertEqual(result.status, "failed")
        self.assertEqual([s.name for s in result.steps], ["boom"])  # pipeline stopped
        self.assertIn("kaput", result.steps[0].error or "")

    def test_skip_on_error(self):
        def flaky(state):
            raise RuntimeError("nope")

        wf = Workflow("skip")
        wf.add_step("flaky", flaky, retries=0, on_error="skip")
        wf.add_step("after", lambda s: {"done": True})
        result = wf.run()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.steps[0].status, "skipped")
        self.assertTrue(result.state["done"])

    def test_compensation_rolls_back(self):
        rolled_back = []

        def side_effect(state):
            raise RuntimeError("tag failed after creating resources")

        def compensate(state):
            rolled_back.append(state.get("version"))

        wf = Workflow("saga")
        wf.add_step("create", side_effect, retries=0, compensate=compensate)
        result = wf.run({"version": "1.2.0"})
        self.assertEqual(result.status, "partially-failed")
        self.assertEqual(result.steps[0].status, "compensated")
        self.assertEqual(rolled_back, ["1.2.0"])

    def test_guard_skips_step(self):
        wf = Workflow("guarded")
        wf.add_step("only_when_visa", lambda s: {"v": True}, if_=lambda s: s.get("visa"))
        result = wf.run({"visa": False})
        self.assertEqual(result.steps[0].status, "skipped")
        self.assertNotIn("v", result.state)
        result2 = wf.run({"visa": True})
        self.assertEqual(result2.steps[0].status, "ok")

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            wf = Workflow("persist")
            wf.add_step("set", lambda s: {"value": "stored"})
            wf.run(persist_path=str(path))
            import json

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["value"], "stored")
            self.assertIn("_finished_at", saved)


if __name__ == "__main__":
    unittest.main()