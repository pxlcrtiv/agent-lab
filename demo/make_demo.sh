#!/usr/bin/env bash
# Regenerates demo/agent-lab-demo.gif from a fresh, deterministic CLI run.
#
# Pipeline: `python -m agentlab demo` -> demo/session.txt
#           -> demo/render_gif.py (Pillow, Monaco, green-on-dark) -> GIF
#
# Requires:
#   - PY  : python3 (3.10+) for the demo itself      (default: python3)
#   - PY_RENDER: a python3 WITH Pillow installed     (default: python3)
# The library itself stays dependency-free; Pillow is only needed to render.
#
# Usage: bash demo/make_demo.sh (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
PY_RENDER="${PY_RENDER:-python3}"

# 1. Capture a fresh deterministic session (mock backend, no network).
rm -f demo/session.txt demo/session-memory.jsonl demo/session-todo.json
"$PY" -m agentlab demo > demo/session.txt
echo "session captured: $(wc -l < demo/session.txt | tr -d ' ') lines"

# 2. Render the animated GIF.
"$PY_RENDER" demo/render_gif.py --hold 6 --fps 15 --width 720

# 3. Tidy the scratch state the demo writes.
rm -f demo/session-memory.jsonl demo/session-todo.json
echo "done: demo/agent-lab-demo.gif ($(du -h demo/agent-lab-demo.gif | cut -f1))"