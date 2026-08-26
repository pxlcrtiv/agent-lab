"""Self-evaluation: the agent scores its own answer against a rubric.

This is the "how do you know the output is any good?" answer. After the
agent finishes, a critic pass scores the answer per criterion, and if the
verdict is ``revise`` the answer is rewritten with the feedback and
re-scored, up to ``max_revisions`` times.

The critic is just another :class:`~agentlab.llm.ChatBackend` call -- the
same model, the same wire format. ``MockCritic`` is a deterministic,
offline scorer so tests, CI and the demo stay hermetic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm import ChatBackend

__all__ = ["Critic", "MockCritic", "Critique", "DEFAULT_RUBRIC"]

DEFAULT_RUBRIC: List[tuple] = [
    ("grounded", "Uses the actual tool results / provided facts; no invented specifics."),
    ("complete", "Fully answers the user's task; nothing asked is left out."),
    ("concise", "No filler, no repetition, stays on the point."),
    ("safe", "No overclaiming, no hallucinated numbers, no risky advice."),
]

CRITIC_SYSTEM = (
    "You are a strict evaluator. Score the answer against the rubric. "
    "Reply with ONLY a JSON object of the form "
    '{"scores": {"criterion": 1..5, ...}, "overall": 1..5, '
    '"verdict": "pass"|"revise", "feedback": "one short sentence"} '
    "Verdict is revise when any score is below 4 or the answer fails the task."
)

REVISE_SYSTEM = (
    "You are an editor. Rewrite the answer to address the evaluator's feedback. "
    "Keep it factual and concrete; ground every claim in the TOOL TRANSCRIPT "
    "you are given. Never claim data is unavailable when the transcript "
    "contains it. "
    "Reply with ONLY the improved answer, no preamble."
)


@dataclass
class Critique:
    """The full critic transcript for one agent answer."""

    task: str
    answer: str
    scores: Dict[str, int] = field(default_factory=dict)
    overall: int = 0
    verdict: str = "revise"
    feedback: str = ""
    revisions: List[dict] = field(default_factory=list)  # one dict per revise round

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class Critic:
    """LLM-powered self-evaluation loop (uses any ChatBackend)."""

    def __init__(
        self,
        backend: ChatBackend,
        rubric: Optional[List[tuple]] = None,
        min_pass: int = 4,
        max_revisions: int = 2,
    ):
        self.backend = backend
        self.rubric = list(rubric or DEFAULT_RUBRIC)
        self.min_pass = min_pass
        self.max_revisions = max_revisions

    # -- public --------------------------------------------------------------

    def run(self, task: str, answer: str, steps: Optional[list] = None) -> Critique:
        critique = Critique(task=task, answer=answer)
        self._transcript = self._render_transcript(steps or [])
        for round_no in range(1, self.max_revisions + 2):  # +1 initial score
            self._score_round(critique, round_no)
            if critique.passed:
                break
            critique.revisions.append(
                {
                    "round": round_no,
                    "overall": critique.overall,
                    "verdict": critique.verdict,
                    "feedback": critique.feedback,
                }
            )
            if round_no > self.max_revisions:
                break
            critique.answer = self._revise(task, critique.answer, critique.feedback)
        return critique

    # -- internals -----------------------------------------------------------

    def _rubric_text(self) -> str:
        return "\n".join("- %s: %s" % (k, d) for k, d in self.rubric)

    @staticmethod
    def _render_transcript(steps: list) -> str:
        """Render agent steps as a compact tool transcript for the critic."""
        lines = []
        for s in steps:
            if s.kind == "tool_call":
                lines.append("  -> tool call : %s %s" % (s.name, s.detail))
            elif s.kind == "tool_result":
                lines.append("  <- tool result: %s => %s" % (s.name, s.detail))
        return "\n".join(lines) if lines else "(no tools were called)"

    def _ask(self, system: str, user: str) -> str:
        reply = self.backend.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return str(reply.get("content") or "")

    def _score_round(self, critique: Critique, round_no: int) -> "Critique":
        user = (
            "TASK:\n%s\n\nANSWER:\n%s\n\nRubric:\n%s"
            "\n\nTOOL TRANSCRIPT (what the agent's tools actually returned):\n%s"
            % (critique.task, critique.answer, self._rubric_text(), self._transcript)
        )
        raw = self._ask(CRITIC_SYSTEM, user)
        parsed = self._parse_json(raw)
        if parsed is None:
            # Unparseable evaluation -> treat as revise with generic feedback.
            critique.scores = {}
            critique.overall = 1
            critique.verdict = "revise"
            critique.feedback = "The evaluator could not parse its own output."
            return critique
        scores = {k: int(v) for k, v in parsed.get("scores", {}).items() if isinstance(v, (int, float))}
        critique.scores = scores or critique.scores
        critique.overall = int(parsed.get("overall") or min(scores.values()) if scores else 0)
        critique.verdict = "pass" if (
            parsed.get("verdict") == "pass"
            and critique.overall >= self.min_pass
            and all(v >= self.min_pass for v in scores.values())
        ) else "revise"
        critique.feedback = str(parsed.get("feedback") or "Improve the answer.")
        return critique

    def _revise(self, task: str, answer: str, feedback: str) -> str:
        user = (
            "TASK:\n%s\n\nCURRENT ANSWER:\n%s\n\nEVALUATOR FEEDBACK:\n%s\n\n"
            "TOOL TRANSCRIPT:\n%s\n\n"
            "Write the improved answer." % (task, answer, feedback, self._transcript)
        )
        revised = self._ask(REVISE_SYSTEM, user).strip()
        return revised or answer

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Extract the first balanced {...} object from raw text (tolerates markdown fences & preamble)."""
        text = raw.strip()
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except ValueError:
                        return None
        return None


class MockCritic(Critic):
    """Deterministic offline critic: rule-based, no LLM.

    Rules (kept simple so tests and the demo are predictable):

    * an answer quoting a tool result ("came back clean" / "=>") is grounded.
    * loop-guard / budget failures ("Stopped:") get a revise verdict.
    * an over-long answer (>= 400 chars) is marked "concise" 3/5 -> revise.
    * a revise round appends a fixing sentence, then passes.
    """

    name = "mock-critic"

    def __init__(self, backend: Optional[ChatBackend] = None, rubric=None, min_pass: int = 4, max_revisions: int = 2):
        # Rule-based scorer: no LLM calls, so a backend is not required.
        self.backend = backend
        self.rubric = list(rubric or DEFAULT_RUBRIC)
        self.min_pass = min_pass
        self.max_revisions = max_revisions

    def run(self, task: str, answer: str, steps: Optional[list] = None) -> Critique:
        critique = Critique(task=task, answer=answer)
        self._transcript = self._render_transcript(steps or [])
        for round_no in range(1, self.max_revisions + 2):
            self._mock_score(critique)
            if critique.passed:
                break
            critique.revisions.append(
                {
                    "round": round_no,
                    "overall": critique.overall,
                    "verdict": "revise",
                    "feedback": critique.feedback,
                }
            )
            if round_no > self.max_revisions:
                break
            critique.answer = self._mock_revise(critique)
        return critique

    def _mock_revise(self, critique: Critique) -> str:
        answer = critique.answer
        if "Stopped:" in answer:
            base = answer.split("Stopped:", 1)[1].strip() or "the task"
            return "revised answer: I used the returned tool result to answer %s: %s" % (
                critique.task,
                base,
            )
        if len(answer) >= 400:
            return answer[:190].rstrip() + " [revised: trimmed to be concise]"
        return answer + " (revised: grounds the answer in the tool result)"

    def _mock_score(self, critique: Critique) -> None:
        answer = critique.answer
        grounded = 5 if ("came back clean" in answer or "=>" in answer or "tool" in answer.lower()) else 2
        stopped = any(k in answer for k in ("Stopped:", "turn budget"))
        concise = 5 if len(answer) < 400 else 3
        complete = 2 if stopped else 5
        safe = 5
        scores = {
            "grounded": grounded,
            "complete": complete,
            "concise": concise,
            "safe": safe,
        }
        critique.scores = scores
        critique.overall = min(scores.values())
        critique.verdict = "pass" if (
            not stopped and grounded >= 4 and critique.overall >= self.min_pass
        ) else "revise"
        critique.feedback = (
            "Meets the rubric."
            if critique.verdict == "pass"
            else ("Ground the answer in the tool result and answer the task directly."
                  if not stopped
                  else "Resolve the loop: answer the task using the tool result that was returned.")
        )