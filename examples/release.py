"""Release pipeline workflow.

Demonstrates the robustness features of the workflow engine: retries with
backoff, a guarded step, and saga-style compensation when a step fails after
side effects. Everything is a dry-run -- safe to run anywhere.

Run with:

    python -m agentlab workflow examples/release.py --state version=1.2.0
"""

from __future__ import annotations

import re
from typing import Dict

from agentlab.workflow import Workflow


def validate_version(state: Dict) -> Dict:
    version = str(state.get("version", ""))
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("version %r is not semver (need X.Y.Z)" % version)
    state["version"] = version
    return {"version": version}


def run_tests(state: Dict) -> Dict:
    """Simulated test gate -- consider it flaky for the retry demo: fails once."""
    attempts = int(state.get("_test_attempts", 0)) + 1
    state["_test_attempts"] = attempts
    if attempts < 2:
        raise RuntimeError("flaky test runner: connection reset (attempt %d)" % attempts)
    state["tests"] = "ok (%d attempts, backoff applied)" % attempts
    return {"tests": state["tests"]}


def create_tag(state: Dict) -> Dict:
    # Side effect: in reality this would push `git tag v<version>`.
    state["tag"] = "v%s" % state.get("version")
    return {"tag": state["tag"]}
    # Uncomment to demo compensation:
    # raise RuntimeError("tag server 500")


def revert_tag(state: Dict) -> None:
    print("  [compensate] rolling back tag v%s" % state.get("version"))


def notify(state: Dict) -> Dict:
    print("  [notify] release v%s shipped (tests: %s)" % (state.get("version"), state.get("tests")))
    return {}


workflow = Workflow("release", "Dry-run release pipeline with retry + compensation.")
workflow.add_step("validate_version", validate_version, retries=0)
workflow.add_step("run_tests", run_tests, retries=2, backoff_s=0.05)
workflow.add_step("create_tag", create_tag, retries=0, compensate=revert_tag)
workflow.add_step(
    "notify", notify, if_=lambda s: s.get("tag") is not None, retries=1, on_error="skip"
)