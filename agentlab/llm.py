"""LLM backends for the agent.

Two backends implement the same ``chat()`` contract:

``MockBackend``
    Deterministic and fully offline. It inspects the conversation and emits
    realistic tool calls / final answers from a small rule set, so tests, CI
    and the recorded demo are hermetic. Swap in a real model for production.

``OpenAICompatBackend``
    Talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint using
    only the standard library: OpenAI, OpenRouter (free tiers work), Ollama,
    vLLM, LM Studio... Configure with env vars:

    * ``OPENAI_API_KEY``   (required)
    * ``OPENAI_BASE_URL``  (default ``https://api.openai.com/v1``)
    * ``OPENAI_MODEL``     (default ``gpt-4o-mini``)

Messages use the OpenAI wire format (subset):

    {"role": "system"|"user"|"assistant"|"tool",
     "content": str | None,
     "tool_calls": [{"id", "function": {"name", "arguments"}}],   # assistant
     "tool_call_id": str}                                          # tool
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional

__all__ = ["ChatBackend", "MockBackend", "OpenAICompatBackend", "Message"]


# A message is an OpenAI-style dict (see module docstring).
Message = dict


class ChatBackend:
    """Minimum contract every backend must satisfy."""

    name: str = "base"

    def chat(self, messages: List[Message], tools: Optional[List[dict]] = None) -> Message:
        raise NotImplementedError


def tool_call_message(call_id: str, name: str, arguments: dict) -> Message:
    """Build an assistant message that requests one tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def tool_result_message(call_id: str, result: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": result}


class MockBackend(ChatBackend):
    """Deterministic offline backend.

    The default responder is a small rule engine over the last user message:

    * a query mentioning time/date            -> ``clock``
    * a query mentioning todo items           -> ``todo.add`` / ``todo.list``
    * a query asking to remember a fact       -> ``notes.add`` (memory)
    * a query asking to find/search something -> ``notes.search`` (memory)
    * a query mentioning math/calc            -> ``calculator``
    * anything else                           -> a plain final answer

    After a tool result arrives it produces a final answer that quotes the
    result, so the full loop (call -> result -> answer) is visible.
    """

    name = "mock"

    _call_seq = [0]

    def __init__(self, responder: Optional[Callable[[List[Message], List[dict]], Message]] = None):
        self.responder = responder or self._default_responder

    def chat(self, messages: List[Message], tools: Optional[List[dict]] = None) -> Message:
        return self.responder(messages, tools or [])

    # -- default rule engine -------------------------------------------------

    def _next_id(self) -> str:
        MockBackend._call_seq[0] += 1
        return "call_mock_%04d" % MockBackend._call_seq[0]

    def _last_user(self, messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    def _default_responder(self, messages: List[Message], tools: List[dict]) -> Message:
        names = [t["name"] for t in tools]
        last = messages[-1] if messages else {}
        raw_text = self._last_user(messages)
        text = raw_text.lower()

        # A tool result just came back -> summarize it and finish the turn.
        if last.get("role") == "tool" and last.get("content"):
            content = str(last["content"])
            try:
                parsed = json.loads(content)
                summary = parsed.get("result", content)
            except (ValueError, AttributeError):
                summary = content
            reply = "Done. The tool returned: %s" % (summary,)
            preview = str(summary)[:160]
            return {"role": "assistant", "content": "That result came back clean. %s" % preview}

        want = {
            "clock": any(k in text for k in ("time", "date", "when is it")),
            "calculator": any(k in text for k in ("calc", "calculate", "sum", "math", "compute", "plus", "times")),
            "todo_add": "todo" in text and any(k in text for k in ("add", "create", "new", "remind")),
            "todo_list": "todo" in text and any(k in text for k in ("list", "what", "show", "items", "on my")),
            "notes_add": any(k in text for k in ("remember", "note that", "write down", "save this")),
            "notes_search": any(k in text for k in ("find", "search", "recall", "what did i note", "look up")),
            "fetch": any(k in text for k in ("fetch", "http", "url", "download", "scrape")),
        }
        for name, chosen in want.items():
            if chosen and name in names:
                args = self._args_for(name, raw_text)
                if args is not None:
                    return tool_call_message(self._next_id(), name, args)

        reply = (
            "I processed that against my tools and memory. "
            "Try asking me to use a tool (time, todo, notes, calculator, fetch) "
            "or run a workflow (see `python -m agentlab workflow --help`)."
        )
        return {"role": "assistant", "content": reply}

    @staticmethod
    def _args_for(name: str, text: str) -> Optional[dict]:
        if name == "clock":
            return {}  # 'timezone' is optional in the schema; leave default
        if name == "calculator":
            m = re.search(r"(?:calculate|calc|compute|what is|sum of)[^\w()]*?(.{2,80})", text, re.IGNORECASE)
            return {"expression": m.group(1).strip().rstrip("?")} if m else {"expression": "1 + 1"}
        if name == "todo_add":
            m = re.search(r"(?:add|create|new|remind)[^\w]*['\"]?([\w .!-]{3,60})", text, re.IGNORECASE)
            return {"item": m.group(1).strip().strip("'\"").strip()} if m else {"item": "untitled task"}
        if name == "notes_add":
            m = re.search(r"(?:remember|note that|write down|save this)[^\w]*['\"]?(.{3,140})", text, re.IGNORECASE)
            return {"value": m.group(1).strip().strip("'\"").strip()} if m else {"value": "untitled note"}
        if name == "notes_search":
            m = re.search(r"(?:find|search|recall)[^\w]*['\"]?([\w .-]{1,60})", text, re.IGNORECASE)
            return {"query": m.group(1).strip().strip("'\"")} if m else {"query": text.strip()[:40]}
        return {}


class OpenAICompatBackend(ChatBackend):
    """OpenAI-compatible chat completions via stdlib urllib only."""

    name = "openai-compat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def chat(self, messages: List[Message], tools: Optional[List[dict]] = None) -> Message:
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatBackend needs OPENAI_API_KEY "
                "(or set OPENAI_BASE_URL/OPENAI_MODEL for OpenRouter, Ollama, ...)"
            )
        body = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError("LLM endpoint returned HTTP %s: %s" % (exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("LLM endpoint unreachable: %s" % exc.reason) from exc

        message = payload["choices"][0]["message"]
        # Some providers return 'arguments' as a dict already or no tool_calls key.
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"])
        return message