"""Authoritative evidence from an agent run.

``Step`` records are bounded previews for people reading a trace.  The critic
needs the complete tool exchanges instead: full arguments, full structured
outcomes, and why the run stopped.  This module owns that representation and
its rendering so display policy cannot silently become evaluation policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Tuple

__all__ = ["RunEvidence", "StopReason", "ToolExchange"]

StopReason = Literal["final", "repeated_call", "turn_budget"]


@dataclass(frozen=True)
class ToolExchange:
    """One complete request/outcome pair from a tool-using turn."""

    call_id: str
    turn: int
    name: str
    arguments: dict
    outcome: dict


@dataclass(frozen=True)
class RunEvidence:
    """The canonical record of what the agent did and why it finished."""

    task: str
    answer: str
    exchanges: Tuple[ToolExchange, ...] = field(default_factory=tuple)
    stop_reason: StopReason = "final"

    @property
    def stopped(self) -> bool:
        """Whether the run ended without a normal final answer."""
        return self.stop_reason != "final"

    def render_tool_transcript(self) -> str:
        """Render every complete exchange for evaluation, without previews."""
        if not self.exchanges:
            return "(no tools were called)"
        lines = []
        for exchange in self.exchanges:
            arguments = json.dumps(exchange.arguments, ensure_ascii=False, sort_keys=True)
            outcome = json.dumps(exchange.outcome, ensure_ascii=False, sort_keys=True)
            lines.append("  -> tool call : %s %s" % (exchange.name, arguments))
            lines.append("  <- tool result: %s => %s" % (exchange.name, outcome))
        return "\n".join(lines)
