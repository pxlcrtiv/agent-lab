"""Tool registry: declare tools, expose OpenAI-style JSON schemas, call safely.

A tool is a plain function with type-annotated parameters and a docstring
that documents them::

    @tool
    def send(title: str, urgent: bool = False) -> str:
        \"\"\"Send a notification.

        Args:
            title: Message title.
            urgent: Whether to mark it high priority.
        \"\"\"
        ...

The JSON schema for the LLM is derived from the signature and docstring, so
tool definitions never drift from their implementation. ``ToolRegistry.call``
coerces arguments to the declared types and always returns a dict, never
raises -- the agent feeds that dict straight back to the model.

Annotations are resolved through ``inspect.get_annotations(..., eval_str=True)``
so ``from __future__ import annotations`` (PEP 563) does not break schema
derivation.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

__all__ = ["Tool", "ToolRegistry", "tool", "ToolError"]

_TYPE_MAP = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}

_UNION_NULL = type(None)


def _json_type(annotation: Any, has_default: bool) -> Dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [a for a in args if a is not _UNION_NULL]
        if non_none and len(non_none) == 1:
            schema = _json_type(non_none[0], True)
            schema["nullable"] = True
            return schema
        # multi-type union: fall back to the first known member
        members = [_json_type(a, True) for a in (non_none or [str])]
        return members[0] if members else {"type": "string"}

    if origin in (list, List) or annotation is list:
        item = _json_type(args[0], True) if args and args[0] is not _UNION_NULL else {"type": "string"}
        return {"type": "array", "items": item}

    if annotation is dict or origin in (dict, Dict):
        return {"type": "object"}

    base = _TYPE_MAP.get(annotation if origin is None else origin)
    if base is None:
        base = {"type": "string"}  # be permissive with exotic hints
    return dict(base)


def _docstring_args(doc: str) -> Dict[str, str]:
    """Parse the ``Args:`` section of a docstring -> {name: description}."""
    out: Dict[str, str] = {}
    if not doc:
        return out
    in_args = False
    for raw in doc.splitlines():
        line = raw.strip()
        if line.lower().startswith(("args:", "arguments:")):
            in_args = True
            continue
        if in_args:
            m = re.match(r"^([\w]+)\s*[:(]\s*(.*)$", line)
            if m and line and not line.startswith(("Returns", "Raises", "Example")):
                out[m.group(1)] = m.group(2).strip()
            elif line == "":
                continue
            elif not m:
                in_args = False
    return out


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def schema(self) -> Dict[str, Any]:
        """OpenAI function schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ToolError(Exception):
    pass


def _resolved_annotations(fn: Callable) -> Dict[str, Any]:
    """Type hints with PEP 563 string annotations evaluated."""
    return inspect.get_annotations(fn, eval_str=True)


def tool(fn: Callable) -> Tool:
    """Decorator: turn a typed function into a Tool with an auto-derived schema."""
    sig = inspect.signature(fn)
    annotations = _resolved_annotations(fn)
    doc = inspect.getdoc(fn) or fn.__name__
    arg_docs = _docstring_args(doc)
    desc_lines = [ln for ln in doc.splitlines() if not ln.strip().lower().startswith(("args:", "arguments:"))]
    description = " ".join(ln.strip() for ln in desc_lines if ln.strip()).strip() or fn.__name__

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        has_default = param.default is not inspect.Parameter.empty
        ann = annotations.get(pname, str)
        schema = _json_type(ann, has_default)
        if has_default and param.default is not None:
            schema["default"] = param.default
        if pname in arg_docs:
            schema["description"] = arg_docs[pname]
        properties[pname] = schema
        if not has_default:
            required.append(pname)

    return Tool(name=fn.__name__, description=description, fn=fn, parameters={"properties": properties, "required": required})


class ToolRegistry:
    """Holds tools, exposes their schemas, and executes calls safely."""

    def __init__(self, *tools: Tool):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ToolError("duplicate tool name: %s" % t.name)
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError("unknown tool: %s (have: %s)" % (name, ", ".join(sorted(self._tools)))) from None

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call. Never raises: returns {"ok": bool, ...}."""
        try:
            t = self.get(name)
            coerced = self._coerce(t, arguments)
            result = t(**coerced)
            return {"ok": True, "tool": name, "result": result}
        except ToolError as exc:  # unknown tool / bad args
            return {"ok": False, "tool": name, "error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "tool": name, "error": "invalid arguments: %s" % exc}
        except Exception as exc:  # tool implementation failure
            return {"ok": False, "tool": name, "error": "%s: %s" % (type(exc).__name__, exc)}

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _coerce(t: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(t.fn)
        annotations = _resolved_annotations(t.fn)
        coerced: Dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname not in arguments:
                if param.default is not inspect.Parameter.empty:
                    coerced[pname] = param.default
                continue
            raw = arguments[pname]
            ann = annotations.get(pname)
            if ann is None:
                coerced[pname] = raw
                continue
            coerced[pname] = _coerce_one(ann, raw, pname)
        return coerced


def _coerce_one(ann: Any, raw: Any, name: str) -> Any:
    origin = get_origin(ann)
    args = get_args(ann)

    # Optional[X] / X | None
    if origin is Union:
        non_none = [a for a in args if a is not _UNION_NULL]
        if raw is None:
            return None
        if non_none:
            return _coerce_one(non_none[0], raw, name)

    if origin in (list, List) or ann is list:
        if not isinstance(raw, list):
            raise TypeError("%s: expected a list, got %r" % (name, raw))
        item_t = args[0] if args and args[0] is not _UNION_NULL else str
        return [_coerce_one(item_t, item, "%s[]" % name) for item in raw]

    if ann is bool:
        if isinstance(raw, bool):
            return raw
        if raw in ("true", "True", "1"):
            return True
        if raw in ("false", "False", "0"):
            return False
        raise TypeError("%s: expected a boolean, got %r" % (name, raw))

    if ann is str:
        if isinstance(raw, str):
            return raw
        raise TypeError("%s: expected a string, got %r" % (name, raw))

    if ann is int:
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise TypeError("%s: expected an integer, got %r" % (name, raw))
        try:
            return int(raw)
        except ValueError:
            raise TypeError("%s: expected an integer, got %r" % (name, raw)) from None

    if ann is float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise TypeError("%s: expected a number, got %r" % (name, raw)) from None

    if isinstance(raw, (dict, list)):
        return json.dumps(raw)
    return raw