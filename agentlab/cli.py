"""Command-line interface for agent-lab.

Usage:
    python -m agentlab run "your task"          one-shot agent run
    python -m agentlab chat                     interactive REPL
    python -m agentlab workflow examples/lead_triage.py [--state k=v ...]
    python -m agentlab demo                     scripted offline demo (used by demo/make_demo.sh)

Backend selection: --backend mock|openai, or env AGENTLAB_BACKEND (default mock).
The mock backend is deterministic and offline; the openai backend talks to any
OpenAI-compatible endpoint (see agentlab/llm.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .builtin_tools import default_tool_registry
from .critic import Critic, MockCritic
from .llm import MockBackend, OpenAICompatBackend
from .memory import Memory

DEMO_MEM_PATH = ".agentlab-demo/memory.jsonl"
DEMO_TODO_PATH = ".agentlab-demo/todo.json"


def _agent_dir() -> Path:
    base = Path(os.environ.get("AGENTLAB_DIR", Path.home() / ".agentlab"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_agent(backend_name: str):
    backend_name = backend_name or os.environ.get("AGENTLAB_BACKEND", "mock")
    if backend_name == "mock":
        backend: object = MockBackend()
    elif backend_name in ("openai", "openai-compat"):
        backend = OpenAICompatBackend()
    else:
        sys.exit("unknown backend %r (use mock or openai)" % backend_name)

    mem_path = _agent_dir() / "memory.jsonl"
    todo_path = _agent_dir() / "todo.json"
    if backend_name == "mock":
        mem_path = Path(DEMO_MEM_PATH)  # keep the demo hermetic: memory lives in cwd
        todo_path = Path(DEMO_TODO_PATH)
        mem_path.parent.mkdir(parents=True, exist_ok=True)

    mem = Memory(mem_path)
    registry = default_tool_registry(mem, todo_path)
    return Agent(backend, registry, memory=mem), backend  # type: ignore[arg-type]


def _print_run(result) -> None:
    for s in result.steps:
        if s.kind == "tool_call":
            print("  -> tool call : %s %s" % (s.name, s.detail))
        elif s.kind == "tool_result":
            print("  <- tool res  : %s => %s" % (s.name, s.detail[:120]))
        elif s.kind == "message":
            print("  ... assistant requests tooling" if s.detail else "")
        elif s.kind == "trim":
            print("  !! context trim: %s" % s.detail)
    print("turns=%d tool_calls=%d context_chars=%d" % (result.turns, result.tool_calls, result.context_chars))
    print("final: %s" % result.answer)


def _print_critic(critique) -> None:
    scores = " ".join("%s=%d/5" % (k, v) for k, v in sorted(critique.scores.items()))
    print("critic: %s overall=%d/5 verdict=%s" % (scores, critique.overall, critique.verdict))
    for i, rev in enumerate(critique.revisions, 1):
        print("  critic round %d: overall=%d/5 verdict=revise" % (i, rev["overall"]))
    print("  critic feedback: %s" % critique.feedback)
    if critique.revisions:
        print("final (revised): %s" % critique.answer)


def _cmd_run(args: argparse.Namespace) -> int:
    agent, backend = _build_agent(args.backend)
    result = agent.run(args.task)
    _print_run(result)
    if args.critic:
        critic = MockCritic(backend) if backend.name == "mock" else Critic(backend, max_revisions=args.max_revisions)
        critique = critic.run(args.task, result.answer, steps=result.steps)
        _print_critic(critique)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    agent, _ = _build_agent(args.backend)
    print("agent-lab %s -- tools: %s" % (__version__, ", ".join(agent.tools.names())))
    print("commands: /tools /memory /quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/tools":
            for t in agent.tools.names():
                print("  -", t)
            continue
        if line == "/memory":
            mem = agent.memory
            if mem is None:
                print("  (no memory)")
                continue
            for e in mem.all():
                print("  [%s] %s" % (e["kind"], e["text"][:100]))
            continue
        result = agent.run(line)
        _print_run(result)
    return 0


def _cmd_workflow(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        sys.exit("workflow file not found: %s" % path)
    spec = importlib.util.spec_from_file_location("workflow_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wf = getattr(mod, "workflow", None)
    if wf is None:
        sys.exit("%s must define a module-level `workflow` object" % path)

    if args.describe:
        print(wf.describe())
        return 0

    state: dict = {}
    for kv in args.state or []:
        if "=" not in kv:
            sys.exit("--state entries must be key=value, got %r" % kv)
        key, _, raw = kv.partition("=")
        try:
            state[key] = json.loads(raw)
        except ValueError:
            state[key] = raw
    persist = None
    if args.persist:
        persist = str(path.parent / (path.stem + ".state.json"))
    result = wf.run(initial_state=state, persist_path=persist)
    print(result.to_markdown())
    print("\nfinal state:")
    for k, v in sorted(result.state.items()):
        print("  %s = %s" % (k, json.dumps(v, default=str)[:160]))
    return 0 if result.status in ("ok", "partially-failed") else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    """The recorded demo: deterministic offline session shown in README / demo.gif."""
    mem = Memory("demo/session-memory.jsonl")
    mem.clear()
    todo = Path("demo/session-todo.json")
    if todo.exists():
        todo.unlink()
    registry = default_tool_registry(mem, todo)
    agent = Agent(MockBackend(), registry, memory=mem, max_turns=6)

    print("agent-lab %s - deterministic offline demo (backend: mock)" % __version__)
    print("=" * 72)
    queries = [
        "What time is it in London?",
        "Calculate (2 + 3) * 4 / 2 ** 2",
        "Add 'ship agent-lab demo' to my todos",
        "Remember that AI agent tooling is my focus area",
        "Find 'agent tooling' in my notes",
    ]
    for q in queries:
        print("\n$ python -m agentlab run %r" % q)
        result = agent.run(q)
        for s in result.steps:
            if s.kind == "tool_call":
                print("  -> tool call : %s %s" % (s.name, s.detail))
            elif s.kind == "tool_result":
                print("  <- tool res  : %s => %s" % (s.name, s.detail[:110]))
        print("final: %s" % result.answer)
        print("turns=%d tool_calls=%d" % (result.turns, result.tool_calls))

    print("\n$ python -m agentlab workflow examples/lead_triage.py \\")
    print("      --state role_type=ai-agent-engineer remote=true visa=true")
    spec = importlib.util.spec_from_file_location("lead_triage", "examples/lead_triage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wf = mod.workflow
    wf_result = wf.run(initial_state={"role_type": "ai-agent-engineer", "remote": True, "visa": True})
    print(wf_result.to_markdown())
    print("\ndemo complete - all output generated locally, no network, no API keys.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab", description="Lean AI agent framework demo")
    parser.add_argument("--version", action="version", version="agent-lab %s" % __version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="one-shot agent run")
    p_run.add_argument("task", help="the task to run")
    p_run.add_argument("--backend", choices=["mock", "openai"], default=None, help="LLM backend (default: env AGENTLAB_BACKEND or mock)")
    p_run.add_argument("--critic", action="store_true", help="self-evaluate the answer against a rubric (scores + optional revision)")
    p_run.add_argument("--max-revisions", type=int, default=2, help="max revise rounds for --critic (default: 2)")
    p_run.set_defaults(fn=_cmd_run)

    p_chat = sub.add_parser("chat", help="interactive REPL")
    p_chat.add_argument("--backend", choices=["mock", "openai"], default=None)
    p_chat.set_defaults(fn=_cmd_chat)

    p_wf = sub.add_parser("workflow", help="run one of the example workflows")
    p_wf.add_argument("path", help="path to a workflow module (must expose `workflow`)")
    p_wf.add_argument("--state", nargs="+", default=[], metavar="k=v", help="initial state entries, e.g. --state key1=val1 key2=val2")
    p_wf.add_argument("--describe", action="store_true", help="print the step plan and exit")
    p_wf.add_argument("--persist", action="store_true", help="write final state JSON next to the module")
    p_wf.set_defaults(fn=_cmd_workflow)

    p_demo = sub.add_parser("demo", help="scripted deterministic demo (feeds demo/make_demo.sh)")
    p_demo.set_defaults(fn=_cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())