"""Job-lead triage workflow.

Scores a lead against a role profile and decides: apply / watch / pass.
Guarded steps keep the pipeline honest when inputs are missing, and the
decision is logged for later review. Deterministic -- safe to run offline.

Run with:

    python -m agentlab workflow examples/lead_triage.py \
        --state role_type=ai-agent-engineer remote=true visa=true
"""

from __future__ import annotations

import re
from typing import Dict

from agentlab.workflow import Workflow

LEAD_KEYWORDS = {
    "visa": ["visa", "sponsorship", "relocation", "work permit", "authorization"],
    "remote": ["remote", "distributed", "anywhere", "hybrid"],
    "ai": ["ai", "agent", "llm", "ml", "model", "function calling", "tool use", "pipeline"],
}


def score_lead(state: Dict) -> Dict:
    """Keyword-score a lead description against the role profile."""
    text = " ".join(str(state.get(k, "")) for k in ("title", "description", "company")).lower()
    hits = {key: sum(1 for kw in kws if kw in text) for key, kws in LEAD_KEYWORDS.items()}
    score = 0
    score += 25 if state.get("visa") else (8 if hits["visa"] else 0)
    score += 15 if state.get("remote") else (6 if hits["remote"] else 0)
    score += min(30, 10 * hits["ai"])
    role = str(state.get("role_type", ""))
    score += 20 if role and re.search(r"agent|ai engineer|ml|llm", role) else 0
    state["score"] = score
    state["hits"] = hits
    return {"score": score, "hits": hits}


def decide(state: Dict) -> Dict:
    score = int(state.get("score", 0))
    if score >= 60:
        decision = "apply"
    elif score >= 40:
        decision = "watch"
    else:
        decision = "pass"
    state["decision"] = decision
    return {"decision": decision}


def log_decision(state: Dict) -> Dict:
    line = "score=%d decision=%s [role=%s remote=%s visa=%s]" % (
        int(state.get("score", 0)),
        state.get("decision"),
        state.get("role_type", "?"),
        state.get("remote", "?"),
        state.get("visa", "?"),
    )
    state["log"] = line
    return {"log": line}


def notify(state: Dict) -> Dict:
    """Print the decision; in a real deployment this posts to Slack/email."""
    print("  [notify] sending decision to pipeline logger: %s" % state.get("log"))
    return {}


workflow = Workflow("lead_triage", "Score a job lead and route it: apply / watch / pass.")
workflow.add_step("score_lead", score_lead)
workflow.add_step("decide", decide, if_=lambda s: s.get("score") is not None)
workflow.add_step("log_decision", log_decision, if_=lambda s: s.get("decision") is not None)
workflow.add_step("notify", notify, retries=1, on_error="skip")