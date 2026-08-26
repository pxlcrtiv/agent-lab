"""The agent loop: tools + memory + a model backend under one API.

The loop is deliberately simple and inspectable -- every step is recorded
so you can see exactly what happened, which is the whole point of a demo:

    user task
      -> backends see [system + memory context + history]
      -> model may ask for tool calls  (function calling)
      -> tools run safely via ToolRegistry.call()
      -> results appended as 'tool' messages
      -> repeat until a final answer, or the turn budget runs out

Context handling: a hard char budget (``context_chars``) protects the model
from unbounded history. When the transcript exceeds the budget the oldest
tool exchanges are pruned first (they are the most redundant), keeping the
system prompt, the current task and the latest exchange intact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import ChatBackend, tool_result_message
from .memory import Memory
from .tools import ToolError, ToolRegistry

__all__ = ["Agent", "AgentConfig", "AgentResult", "Step"]

DEFAULT_SYSTEM = (
    "You are agent-lab, a tool-using AI assistant. You can call tools to get "
    "facts and take actions. Call at most one tool per turn, wait for its "
    "result, then answer the user's original question using that result. "
    "Be concise and concrete."
)


@dataclass
class Step:
    """One recorded event in an agent run (for logs / the demo)."""

    turn: int
    kind: str  # "tool_call" | "tool_result" | "message" | "trim"
    name: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    turns: int
    tool_calls: int = 0
    context_chars: int = 0
    steps: List[Step] = field(default_factory=list)


class Agent:
    """A minimal tool-using agent. Thread-safe enough for the demo (no shared mutable state)."""

    def __init__(
        self,
        backend: ChatBackend,
        tools: ToolRegistry,
        memory: Optional[Memory] = None,
        system_prompt: str = DEFAULT_SYSTEM,
        max_turns: int = 8,
        context_chars: int = 12_000,
    ):
        self.backend = backend
        self.tools = tools
        self.memory = memory
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.context_chars = context_chars

    # -- public --------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        steps: List[Step] = []
        tool_calls = 0
        messages = self._seed_context(task)
        answer = ""
        answer_turn = 0

        for turn in range(1, self.max_turns + 1):
            if not any(m.get("role") == "user" and m.get("content") == task for m in messages):
                messages.append({"role": "user", "content": task})
            self._trim(messages, steps, turn)

            reply = self.backend.chat(messages, tools=self.tools.schemas())
            calls = reply.get("tool_calls")

            if not calls:
                answer = str(reply.get("content") or "(no content)")
                # Some providers (Qwen3-style templates) append a stray
                # <tool_call> continuation tag after the final answer. Drop it.
                if "<tool_call" in answer:
                    answer = re.sub(r"\s*</?tool_call>.*$", "", answer, flags=re.S).strip()
                answer_turn = turn
                break

            # Function calling round: the model asked for tools.
            messages.append(reply)
            for tc in calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                if not tc.get("id"):
                    tc["id"] = "call_%d_%s" % (turn, name)
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                except ValueError:
                    arguments = {}
                call_id = tc["id"]
                steps.append(Step(turn, "tool_call", name, json.dumps(arguments)))
                result = self.tools.call(name, arguments)
                tool_calls += 1
                payload = result.get("result") if result.get("ok") else result.get("error")
                steps.append(Step(turn, "tool_result", name, str(payload)[:300]))
                messages.append(tool_result_message(call_id, json.dumps(result)))
            steps.append(Step(turn, "message", "assistant", "requested %d tool call(s)" % len(calls)))

            # Loop guard: same tool+args twice back-to-back means the model is stuck.
            tool_calls_log = [s for s in steps if s.kind == "tool_call"]
            if (len(tool_calls_log) >= 2
                    and tool_calls_log[-1].name == tool_calls_log[-2].name
                    and tool_calls_log[-1].detail == tool_calls_log[-2].detail):
                answer = "Stopped: repeated identical tool call (%s) -- possible loop." % tool_calls_log[-1].name
                answer_turn = turn
                break

        if not answer:
            answer = "Stopped: turn budget (%d) exceeded without a final answer." % self.max_turns
            answer_turn = self.max_turns

        if self.memory is not None:
            try:
                self.memory.remember("%s\n-> %s" % (task, answer), kind="session")
            except Exception:
                pass  # memory failures must never break the agent

        return AgentResult(
            answer=answer,
            turns=answer_turn,
            tool_calls=tool_calls,
            context_chars=sum(len(json.dumps(m)) for m in messages),
            steps=steps,
        )

    # -- context handling ------------------------------------------------------

    def _seed_context(self, task: str) -> List[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.memory is not None:
            hits = self.memory.recall(task, k=3, min_score=0.1, kinds=("note", "fact"))
            if hits:
                digest = "\n".join("- %s" % h["text"] for h in hits)
                messages.append(
                    {"role": "system", "content": "Relevant facts from agent memory:\n" + digest}
                )
        return messages

    def _trim(self, messages: List[dict], steps: List[Step], turn: int) -> None:
        """Drop oldest tool exchanges while over budget; never the system/user anchors."""
        while len(messages) > 4 and self._size(messages) > self.context_chars:
            # find the oldest removable index: skip system msgs and the first user task
            for i in range(2, len(messages) - 2):
                if messages[i].get("role") in ("assistant", "tool"):
                    removed = messages.pop(i)
                    steps.append(Step(turn, "trim", "context", "%d chars dropped" % len(json.dumps(removed))))
                    break
            else:
                break

    @staticmethod
    def _size(messages: List[dict]) -> int:
        return sum(len(json.dumps(m)) for m in messages)