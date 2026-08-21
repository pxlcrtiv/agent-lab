"""Tests for persistent memory: append, recall ranking, TTL pruning."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agentlab.memory import Memory, MemoryError


class MemoryTests(unittest.TestCase):
    def test_remember_and_recall(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "mem.jsonl")
            mem.remember("project agent-lab uses zero runtime dependencies")
            mem.remember("favorite color is green")
            hits = mem.recall("agent-lab dependencies")
            self.assertTrue(hits)
            self.assertIn("zero runtime dependencies", hits[0]["text"])

    def test_recall_ranks_best_match_first(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "mem.jsonl")
            mem.remember("postgres is a relational database")
            mem.remember("redis is an in-memory cache")
            mem.remember("redis and postgres are both databases")
            hits = mem.recall("redis cache")
            self.assertEqual(hits[0]["text"], "redis is an in-memory cache")

    def test_no_match_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "mem.jsonl")
            mem.remember("bananas are yellow")
            self.assertEqual(mem.recall("quantum chromodynamics"), [])

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mem.jsonl"
            Memory(path).remember("persisted fact survives restarts", kind="fact")
            mem2 = Memory(path)
            self.assertEqual(len(mem2.all()), 1)
            self.assertEqual(mem2.all()[0]["kind"], "fact")

    def test_ttl_pruning(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "mem.jsonl")
            mem.remember("ephemeral note", ttl=1)
            mem.remember("durable note")
            time.sleep(1.1)
            texts = [e["text"] for e in mem.recall("note", min_score=0.0, k=10)]
            self.assertIn("durable note", texts)
            self.assertNotIn("ephemeral note", texts)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Memory(Path(d) / "mem.jsonl")
            mem.remember("x")
            self.assertEqual(mem.clear(), 1)
            self.assertEqual(mem.all(), [])
            self.assertFalse(Path(d, "mem.jsonl").exists())

    def test_corrupt_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mem.jsonl"
            path.write_text("{not json\n", encoding="utf-8")
            with self.assertRaises(MemoryError):
                Memory(path)


if __name__ == "__main__":
    unittest.main()