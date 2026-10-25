"""Tests for the built-in tools: each tool owns the store it was built with."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentlab.builtin_tools import build_tools, default_tool_registry
from agentlab.memory import Memory

BUILTIN_TOOL_NAMES = [
    "calculator",
    "clock",
    "fetch",
    "notes_add",
    "notes_search",
    "todo_add",
    "todo_list",
    "todo_remove",
]


class BuiltinToolTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.memory = Memory(self.tmp / "memory.jsonl")
        self.registry = default_tool_registry(self.memory, self.tmp / "todo.json")

    def tearDown(self):
        self._tmp.cleanup()


class ToolSetTests(BuiltinToolTestCase):
    def test_registry_exposes_every_builtin_tool(self):
        self.assertEqual(self.registry.names(), BUILTIN_TOOL_NAMES)

    def test_build_tools_returns_the_same_set(self):
        names = sorted(t.name for t in build_tools(self.memory, self.tmp / "todo.json"))
        self.assertEqual(names, BUILTIN_TOOL_NAMES)

    def test_schemas_keep_their_docstring_descriptions(self):
        schemas = {s["name"]: s for s in self.registry.schemas()}
        self.assertIn("Write a fact or note", schemas["notes_add"]["description"])
        self.assertIn("Search notes and facts", schemas["notes_search"]["description"])
        self.assertIn("Add a task to the todo list", schemas["todo_add"]["description"])

    def test_notes_add_argument_is_required(self):
        schemas = {s["name"]: s for s in self.registry.schemas()}
        self.assertIn("value", schemas["notes_add"]["parameters"]["required"])
        self.assertNotIn("k", schemas["notes_search"]["parameters"]["required"])


class NotesToolTests(BuiltinToolTestCase):
    def test_notes_add_writes_to_its_own_memory(self):
        out = self.registry.call("notes_add", {"value": "agent tooling is my focus"})
        self.assertTrue(out["ok"])
        self.assertIn("saved note #1", out["result"])
        entries = self.memory.all(kind="note")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "agent tooling is my focus")

    def test_notes_search_finds_the_note(self):
        self.registry.call("notes_add", {"value": "agent tooling is my focus"})
        out = self.registry.call("notes_search", {"query": "agent tooling"})
        self.assertTrue(out["ok"])
        self.assertIn("agent tooling is my focus", out["result"])

    def test_notes_search_without_matches(self):
        out = self.registry.call("notes_search", {"query": "nothing here"})
        self.assertTrue(out["ok"])
        self.assertIn("no memory matches", out["result"])


class TodoToolTests(BuiltinToolTestCase):
    def test_add_list_remove(self):
        added = self.registry.call("todo_add", {"item": "ship agent-lab demo"})
        self.assertTrue(added["ok"])
        self.assertIn("ship agent-lab demo", added["result"])

        listed = self.registry.call("todo_list", {})
        self.assertTrue(listed["ok"])
        self.assertIn("1. ship agent-lab demo", listed["result"])

        removed = self.registry.call("todo_remove", {"index": 1})
        self.assertTrue(removed["ok"])
        self.assertIn("0 item(s) left", removed["result"])
        self.assertIn("todo is empty", self.registry.call("todo_list", {})["result"])

    def test_remove_out_of_range_is_an_error_not_a_crash(self):
        out = self.registry.call("todo_remove", {"index": 3})
        self.assertFalse(out["ok"])
        self.assertIn("out of range", out["error"])

    def test_todo_file_is_written_where_it_was_built(self):
        self.registry.call("todo_add", {"item": "written to disk"})
        self.assertTrue((self.tmp / "todo.json").exists())


class ToolIndependenceTests(BuiltinToolTestCase):
    """Two tool sets built in one process must not share any state."""

    def test_two_tool_sets_do_not_share_memory(self):
        other_memory = Memory(self.tmp / "other-memory.jsonl")
        other = default_tool_registry(other_memory, self.tmp / "other-todo.json")

        self.registry.call("notes_add", {"value": "belongs to the first set"})

        self.assertEqual(len(self.memory.all(kind="note")), 1)
        self.assertEqual(other_memory.all(kind="note"), [])
        self.assertIn("no memory matches", other.call("notes_search", {"query": "belongs"})["result"])

    def test_two_tool_sets_do_not_share_the_todo_file(self):
        other = default_tool_registry(self.memory, self.tmp / "other-todo.json")

        self.registry.call("todo_add", {"item": "first set only"})

        self.assertIn("first set only", self.registry.call("todo_list", {})["result"])
        self.assertIn("todo is empty", other.call("todo_list", {})["result"])

    def test_rebuilding_a_tool_set_does_not_rebind_the_first(self):
        other = default_tool_registry(self.memory, self.tmp / "other-todo.json")
        other.call("notes_add", {"value": "written through the second set"})

        self.registry.call("notes_add", {"value": "written through the first set"})

        texts = [e["text"] for e in self.memory.all(kind="note")]
        self.assertIn("written through the first set", texts)
        self.assertIn("written through the second set", texts)


class StatelessToolTests(BuiltinToolTestCase):
    def test_clock_returns_a_timestamp(self):
        out = self.registry.call("clock", {})
        self.assertTrue(out["ok"])
        self.assertRegex(out["result"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_fetch_rejects_non_http_urls(self):
        out = self.registry.call("fetch", {"url": "file:///etc/passwd"})
        self.assertFalse(out["ok"])
        self.assertIn("only http(s) URLs allowed", out["error"])

    def test_calculator_is_never_eval(self):
        out = self.registry.call("calculator", {"expression": "__import__('os').system('true')"})
        self.assertFalse(out["ok"])
        self.assertIn("unsupported syntax", out["error"])


if __name__ == "__main__":
    unittest.main()
