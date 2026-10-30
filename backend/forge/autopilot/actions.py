"""One-time autopilot actions the user approved.

**Actions** (``autopilot_actions`` table): something a target must do once, because the user decided so, not because
a detector found a need. Each has the target, the action, who approved it and when (``approved``, in words), why,
and a state (``queued`` until a run completes it, then ``done`` with the run and its result; a failed attempt keeps
it ``queued`` with ``last_error``, so the next qualifying run tries again). Kinds (``KINDS``):

* ``resplit_val``: redraw every head's validation split from the same generation procedure as its frozen test
  split (the adapter's ``resplit-val`` verb), archive the old val with hashes, journal per head; training and
  selection use the new val from that run on. It needs generator / teacher CLI calls, so it runs in a **scheduled**
  or **manual** run (``WHEN["resplit_val"]``); a data-triggered run journals that it waits. Queued through
  ``Actions.queue`` by whoever approved it.

The runner (``forge.autopilot.runner``) asks ``pending(target)`` at the start of a head-set run, journals
``action_waiting`` / ``action_started`` / ``action_done`` / ``action_failed``, and marks the action through
``complete`` / ``failed``.
"""
from __future__ import annotations

from forge.autopilot.journal import Journal
from forge.store import Store, now

KINDS = ("resplit_val",)
#: the run triggers an action may run in (``resplit_val`` needs generator / teacher calls: a scheduled run or one the
#: user started, never a data-triggered one)
WHEN: dict[str, tuple[str, ...]] = {"resplit_val": ("schedule", "manual")}


class Actions:
    def __init__(self, store: Store, journal: Journal):
        self.store, self.journal = store, journal

    def queue(self, target_id: str, action: str, approved: str, reason: str) -> dict:
        if action not in KINDS:
            raise ValueError(f"unknown action {action!r} (known: {', '.join(KINDS)})")
        a = self.store.add_action(target_id, action, approved, reason)
        self.journal.add(target_id, "action_queued", f"{action} queued ({approved}): {reason}",
                         counts={"action": action, "action_id": a["id"], "approved": approved,
                                 "runs_in": list(WHEN[action])})
        return a

    def pending(self, target_id: str) -> list[dict]:
        return self.store.actions(target_id, states=("queued",))

    def all(self, target_id: str | None = None) -> list[dict]:
        return self.store.actions(target_id)

    def complete(self, a: dict, run_id: str, result: dict) -> None:
        self.store.save_action({**a, "state": "done", "run_id": run_id, "done": now(), "last_error": None,
                                "result": result})

    def failed(self, a: dict, run_id: str, error: str) -> None:
        self.store.save_action({**a, "state": "queued", "run_id": run_id, "last_error": error[:2000]})


__all__ = ["Actions", "KINDS", "WHEN"]
