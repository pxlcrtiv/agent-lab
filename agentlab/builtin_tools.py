"""Built-in tools bundled with agent-lab.

Every tool is declared with ``@tool``; the JSON schema the model sees is
derived from the annotated signature and the ``Args:`` section of the
docstring -- one definition, zero drift.

Security note: the ``calculator`` tool uses an AST whitelist, never
``eval()``. The ``fetch`` tool is a read-only GET with a hard size cap.
"""

from __future__ import annotations

import ast
import operator
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .memory import Memory
from .tools import tool

__all__ = ["default_tool_registry"]


# -- calculator: safe arithmetic via AST whitelist ---------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name)


def _eval_node(node, names: Dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers allowed")
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError("unknown variable: %s" % node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, names))
    raise ValueError("expression contains unsupported syntax")


@tool
def calculator(expression: str, variables: Optional[Dict[str, float]] = None) -> float:
    """Evaluate an arithmetic expression safely (no eval/exec).

    Args:
        expression: math expression, e.g. "(2 + 3) * 4 / 2 ** 2".
        variables: optional named constants, e.g. {"pi": 3.14159}.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        if not isinstance(tree.body, _ALLOWED):
            raise ValueError("expression contains unsupported syntax")
        return _eval_node(tree.body, variables or {})
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("cannot evaluate %r: %s" % (expression, exc)) from exc


# -- clock --------------------------------------------------------------------------

@tool
def clock(timezone: Optional[str] = None) -> str:
    """Get the current date and time.

    Args:
        timezone: IANA zone name (e.g. "Europe/London"). Defaults to local time.
    """
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown timezone: %s" % timezone) from exc
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC offset %z)")


# -- fetch: read-only HTTP with size cap ---------------------------------------------

@tool
def fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return its text (read-only, capped).

    Args:
        url: http(s) URL to fetch.
        max_chars: maximum characters to return.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "agent-lab/0.1 (demo)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(max_chars + 500)
    except urllib.error.URLError as exc:
        raise ValueError("fetch failed: %s" % exc.reason) from exc
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = data.decode("latin-1", errors="replace")
    return text[:max_chars]


# -- notes: memory-backed facts ---------------------------------------------------------

@tool
def notes_add(value: str) -> str:
    """Write a fact or note into agent memory.

    Args:
        value: the fact or note to remember.
    """
    entry = _memory().remember(value, kind="note")
    return "saved note #%d: %s" % (len(_memory().all(kind="note")), value)


@tool
def notes_search(query: str, k: int = 3) -> str:
    """Search notes and facts stored in agent memory.

    Args:
        query: keywords to match.
        k: how many matches to return.
    """
    hits = _memory().recall(query, k=k, kinds=("note", "fact"))
    return "\n".join("- [%s] %s" % (e["kind"], e["text"]) for e in hits) or "no memory matches"


# -- todo: a small per-session task list (persisted to the memory dir) -------------------

_TODO_PATH: Optional[Path] = None


def _todo_path() -> Path:
    if _TODO_PATH is None:
        raise RuntimeError("todo backend not wired (call set_todo_path)")
    return _TODO_PATH


def _load_todo() -> List[str]:
    p = _todo_path()
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save_todo(items: List[str]) -> None:
    _todo_path().write_text(json.dumps(items, indent=2), encoding="utf-8")


@tool
def todo_add(item: str) -> str:
    """Add a task to the todo list.

    Args:
        item: task description.
    """
    items = _load_todo()
    items.append(item)
    _save_todo(items)
    return "todo now has %d item(s): %s" % (len(items), "; ".join(items))


@tool
def todo_list() -> str:
    """List all todo items."""
    items = _load_todo()
    return "todo (%d):\n%s" % (len(items), "\n".join("  %d. %s" % (i + 1, x) for i, x in enumerate(items))) if items else "todo is empty"


@tool
def todo_remove(index: int) -> str:
    """Remove a todo item by its 1-based index.

    Args:
        index: 1-based position in the list.
    """
    items = _load_todo()
    if not 1 <= index <= len(items):
        raise ValueError("index %d out of range (have %d items)" % (index, len(items)))
    removed = items.pop(index - 1)
    _save_todo(items)
    return "removed %r; %d item(s) left" % (removed, len(items))


# -- wiring ---------------------------------------------------------------------------------

_memory_ref: Dict[str, Memory] = {}


def _memory() -> Memory:
    if "_mem" not in _memory_ref:
        raise RuntimeError("memory not wired (call bind_memory)")
    return _memory_ref["_mem"]


def set_todo_path(path: Path) -> None:
    global _TODO_PATH
    _TODO_PATH = path


def bind_memory(mem: Memory) -> None:
    _memory_ref["_mem"] = mem


def default_tool_registry(memory: Memory, todo_path: Path) -> "ToolRegistry":
    """Registry with all built-in tools wired to a shared memory store."""
    from .tools import ToolRegistry

    bind_memory(memory)
    set_todo_path(todo_path)
    return ToolRegistry(clock, calculator, fetch, notes_add, notes_search, todo_add, todo_list, todo_remove)