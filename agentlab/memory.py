"""Persistent memory: a JSONL store of facts, notes and session records.

Entries are plain JSON lines with a timestamp, a kind, free text and
optional metadata. ``recall`` scores entries against a query using
token overlap plus a recency bonus -- no external dependencies, fully
deterministic, and good enough to demonstrate the memory/context pattern:

    >>> mem.remember("project agent-lab uses zero runtime dependencies", kind="fact")
    >>> mem.recall("agent-lab dependencies", k=3)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Memory", "MemoryError"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


class MemoryError(Exception):
    pass


class Memory:
    """Append-only JSONL memory store with keyword+recency recall."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # -- write path -----------------------------------------------------------

    def remember(self, text: str, kind: str = "fact", meta: Optional[Dict[str, Any]] = None,
                 ttl: Optional[int] = None) -> Dict[str, Any]:
        """Store an entry. ``ttl`` seconds from now marks it for pruning."""
        entry = {
            "ts": time.time(),
            "kind": kind,
            "text": text,
            "meta": meta or {},
            "ttl": (time.time() + ttl) if ttl else None,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._entries.append(entry)
        return entry

    def clear(self) -> int:
        """Delete all entries. Returns how many were removed."""
        n = len(self._entries)
        self._entries = []
        self.path.unlink(missing_ok=True)
        return n

    # -- read path ------------------------------------------------------------

    def recall(self, query: str, k: int = 5, min_score: float = 0.15,
               kinds: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Top-k entries matching ``query`` by token overlap + recency.

        ``kinds`` optionally whitelists entry kinds (e.g. ("note", "fact"))
        so session *logs* never masquerade as *facts* in LLM context.
        """
        q = _tokens(query)
        if not q:
            return []
        now = time.time()
        scored = []
        for e in self._entries:
            if kinds is not None and e.get("kind") not in kinds:
                continue
            if e.get("ttl") and e["ttl"] < now:
                continue
            et = _tokens(str(e.get("text", ""))) | _tokens(str(e.get("meta", {})))
            if not et:
                continue
            overlap = len(q & et)
            if overlap == 0:
                continue
            base = overlap / max(len(q), len(et))
            age = now - float(e.get("ts", now))
            recency = max(0.0, 0.10 - age / 86400.0 * 0.02)  # up to +0.10, decays over ~5 days
            scored.append((base + recency, e))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for _, e in scored[:k] if _ >= min_score]

    def all(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if kind is None:
            return list(self._entries)
        return [e for e in self._entries if e.get("kind") == kind]

    def stats(self) -> Dict[str, int]:
        from collections import Counter

        kinds = Counter(e.get("kind", "?") for e in self._entries)
        return {"entries": len(self._entries), "kinds": dict(kinds)}

    # -- internals ------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise MemoryError("corrupt memory file %s: %s" % (self.path, exc)) from exc