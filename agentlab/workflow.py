"""Workflow engine: small, explicit state machines with retries & rollback.

A workflow is an ordered list of steps. Each step receives the shared state
dict and may return a dict that gets merged in. Steps can be guarded with
``if_=``, retried with backoff, and -- on hard failure -- rolled back via a
``compensate`` function (the saga pattern), so multi-step automation stays
safe when one stage breaks.

State is serialized to JSON (optionally persisted), which makes runs
inspectable and resumable -- the same ideas behind production orchestrators.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Workflow", "WorkflowResult", "StepResult"]


def _noop(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return None


@dataclass
class StepSpec:
    name: str
    fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    retries: int = 2
    backoff_s: float = 0.1
    on_error: str = "fail"  # "fail" | "skip" | "continue"
    if_: Optional[Callable[[Dict[str, Any]], bool]] = None
    compensate: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skipped" | "failed" | "compensated"
    attempts: int
    duration_s: float
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    name: str
    status: str  # "ok" | "failed" | "partially-failed"
    steps: List[StepResult] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_markdown(self) -> str:
        lines = ["# workflow: %s  (status: %s)" % (self.name, self.status)]
        for s in self.steps:
            mark = {"ok": "ok", "skipped": "skip", "failed": "FAIL", "compensated": "comp"}[s.status]
            extra = "  (%d attempt%s, %.2fs)" % (s.attempts, "s" if s.attempts != 1 else "", s.duration_s)
            lines.append("  [%s] %s%s" % (mark, s.name, extra))
            if s.error:
                lines.append("        error: %s" % s.error)
        lines.append("  total: %.2fs" % self.duration_s)
        return "\n".join(lines)


class Workflow:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._steps: List[StepSpec] = []

    # -- registration ----------------------------------------------------------

    def step(self, name: Optional[str] = None, retries: int = 2, backoff_s: float = 0.1,
             on_error: str = "fail", if_: Optional[Callable[[Dict[str, Any]], bool]] = None,
             compensate: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Decorator: register a function as a workflow step."""

        def wrap(fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Callable:
            step_name = name or fn.__name__
            self._steps.append(StepSpec(step_name, fn, retries, backoff_s, on_error, if_, compensate))
            return fn

        return wrap

    def add_step(self, step_name: str, fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
                 **kwargs: Any) -> None:
        self._steps.append(StepSpec(step_name, fn, **kwargs))

    # -- execution ---------------------------------------------------------------

    def run(self, initial_state: Optional[Dict[str, Any]] = None,
            persist_path: Optional[str] = None) -> WorkflowResult:
        state: Dict[str, Any] = dict(initial_state or {})
        state.setdefault("_started_at", time.time())
        step_results: List[StepResult] = []
        status = "ok"
        t0 = time.time()

        for spec in self._steps:
            if spec.if_ is not None and not spec.if_(state):
                step_results.append(StepResult(spec.name, "skipped", 0, 0.0))
                continue

            attempts = 0
            error: Optional[str] = None
            step_t0 = time.time()
            while attempts <= spec.retries:
                attempts += 1
                try:
                    outcome = spec.fn(state)
                    if isinstance(outcome, dict):
                        state.update(outcome)
                    step_results.append(StepResult(spec.name, "ok", attempts, time.time() - step_t0))
                    break
                except Exception as exc:  # noqa: BLE001 - workflow steps fail loudly by design
                    error = "%s: %s" % (type(exc).__name__, exc)
                    if attempts <= spec.retries:
                        time.sleep(spec.backoff_s * attempts)
            else:
                # all retries exhausted
                if spec.compensate is not None:
                    try:
                        spec.compensate(state)
                        step_results.append(StepResult(spec.name, "compensated", attempts, time.time() - step_t0, error))
                        status = "partially-failed"
                        continue
                    except Exception as comp_exc:  # compensation itself failed
                        error = "%s (compensation failed: %s)" % (error, comp_exc)
                if spec.on_error == "skip":
                    step_results.append(StepResult(spec.name, "skipped", attempts, time.time() - step_t0, error))
                    continue
                if spec.on_error == "continue":
                    step_results.append(StepResult(spec.name, "failed", attempts, time.time() - step_t0, error))
                    status = "partially-failed"
                    continue
                step_results.append(StepResult(spec.name, "failed", attempts, time.time() - step_t0, error))
                status = "failed"
                break

        state["_finished_at"] = time.time()
        state["_status"] = status
        if persist_path:
            Path(persist_path).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

        return WorkflowResult(self.name, status, step_results, state, time.time() - t0)

    def describe(self) -> str:
        lines = ["%s: %s" % (self.name, self.description or "(no description)")]
        for s in self._steps:
            lines.append("  %-28s retries=%d on_error=%s%s" % (
                s.name, s.retries, s.on_error, "  (guarded)" if s.if_ else ""))
        return "\n".join(lines)