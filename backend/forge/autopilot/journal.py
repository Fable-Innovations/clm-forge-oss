"""The autopilot journal: every need, remedy, round, deferral and decision, with its reason and counts.

Entries live in the ``autopilot_journal`` table. For project targets each entry is also emitted into the
project's event stream as an ``autopilot`` event (``action, reason, counts, head, round``).

The journal holds counts, ids and reasons only, never real rows or message text: callers pass ids and numbers, an
adapter failure reaches it as the adapter's one-line ``ERROR`` statement (the full stderr goes to a local log file,
``forge.autopilot.adapter``), and ``bounded`` caps every string so an unexpected long text is cut, not stored.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from forge.store import Store

REASON_MAX = 1000
VALUE_MAX = 200


def bounded(v: Any, limit: int = VALUE_MAX) -> Any:
    """``v`` with every string cut to ``limit`` characters (nested lists and dicts too)."""
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + f"… ({len(v)} chars)"
    if isinstance(v, dict):
        return {str(k)[:limit]: bounded(x, limit) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [bounded(x, limit) for x in v]
    return v


class Journal:
    def __init__(self, store: Store, emit: Callable[[str, dict], None] | None = None):
        self.store = store
        self.emit = emit          # (project id, event) for project targets
        self.project_of: Callable[[str], str | None] = lambda _tid: None

    def add(self, target_id: str, action: str, reason: str, *, run_id: str | None = None,
            head: str | None = None, round: int | None = None, counts: dict | None = None) -> dict:
        reason, counts = bounded(reason, REASON_MAX), bounded(counts or {})
        e = self.store.add_journal({"target_id": target_id, "run_id": run_id, "head": head, "round": round,
                                    "action": action, "reason": reason, "counts": counts})
        pid = self.project_of(target_id)
        if pid and self.emit:
            self.emit(pid, {"type": "autopilot", "action": action, "reason": reason, "counts": counts or {},
                            "head": head, "round": round, "target": target_id, "run": run_id})
        return e

    def query(self, **kw) -> list[dict]:
        return self.store.journal(**kw)
