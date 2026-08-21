"""Tests for the tool registry: schema derivation and safe invocation.

Run with:
    python3 -m unittest discover -s tests -v
or:
    python3 -m pytest tests -q
"""

from __future__ import annotations

import unittest

from agentlab.tools import ToolError, ToolRegistry, tool


@tool
def greet(name: str, excited: bool = False, times: int = 1) -> str:
    """Greet someone.

    Args:
        name: who to greet.
        excited: add enthusiasm.
        times: how many times.
    """
    msg = ("Hello, %s!" % name).upper() if excited else "Hello, %s!" % name
    return " ".join([msg] * times)


@tool
def add_items(items: list) -> int:
    """Add numbers.

    Args:
        items: numbers to add.
    """
    if not all(isinstance(i, (int, float)) for i in items):
        raise ValueError("all items must be numbers")
    return sum(items)


class ToolSchemaTests(unittest.TestCase):
    def test_schema_shape(self):
        schema = greet.schema()
        self.assertEqual(schema["name"], "greet")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertIn("name", schema["parameters"]["required"])
        self.assertNotIn("excited", schema["parameters"]["required"])  # has default
        props = schema["parameters"]["properties"]
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["excited"]["type"], "boolean")
        self.assertEqual(props["times"]["type"], "integer")
        self.assertIn("who to greet", props["name"]["description"])

    def test_list_param_schema(self):
        schema = add_items.schema()
        params = schema["parameters"]["properties"]["items"]
        self.assertEqual(params["type"], "array")
        self.assertEqual(params["items"]["type"], "string")  # bare `list` -> string items

    def test_docstring_only_description(self):
        self.assertIn("Greet someone", greet.description)


class ToolCallTests(unittest.TestCase):
    def test_ok(self):
        reg = ToolRegistry(greet)
        out = reg.call("greet", {"name": "pxlcrtiv", "excited": True})
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"], "HELLO, PXLCRTIV!")

    def test_coercion(self):
        reg = ToolRegistry(greet)
        out = reg.call("greet", {"name": "a", "times": "3"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"], "Hello, a! Hello, a! Hello, a!")

    def test_missing_required_returns_error(self):
        reg = ToolRegistry(greet)
        out = reg.call("greet", {})
        self.assertFalse(out["ok"])
        self.assertIn("name", out["error"])

    def test_unknown_tool(self):
        reg = ToolRegistry()
        out = reg.call("nope", {})
        self.assertFalse(out["ok"])
        self.assertIn("unknown tool", out["error"])

    def test_bad_type(self):
        reg = ToolRegistry(greet)
        out = reg.call("greet", {"name": 12})
        self.assertFalse(out["ok"])
        self.assertIn("invalid arguments", out["error"])

    def test_calculator_is_safe(self):
        from agentlab.builtin_tools import calculator

        reg = ToolRegistry(calculator)
        out = reg.call("calculator", {"expression": "__import__('os').system('true')"})
        self.assertFalse(out["ok"])
        self.assertIn("unsupported syntax", out["error"])

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ToolError):
            ToolRegistry(greet, greet)


if __name__ == "__main__":
    unittest.main()