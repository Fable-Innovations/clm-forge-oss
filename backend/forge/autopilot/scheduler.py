"""Cadence loop: enqueue autopilot runs when their schedule is due.

A schedule is ``daily`` or ``weekly`` (``weekday`` 0 = Monday) at ``time_local`` (HH:MM, this machine's local
time). ``tick`` runs every 60 s (started from the API lifespan): a schedule whose ``next_run`` has passed
enqueues ONE run (trigger ``schedule``) and moves ``next_run`` to the next future slot. Runs missed while the
server was down therefore enqueue once on the first tick, never one per missed slot.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import datetime, timedelta

from forge.store import Store

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def local_now() -> datetime:
    return datetime.now().astimezone()


def validate(cadence: str, weekday: int | None, time_local: str) -> None:
    if cadence not in ("daily", "weekly"):
        raise ValueError("cadence must be daily or weekly")
    if not _HHMM.match(time_local or ""):
        raise ValueError("time_local must be HH:MM (24 h)")
    if cadence == "weekly" and (weekday is None or not 0 <= int(weekday) <= 6):
        raise ValueError("a weekly schedule needs weekday 0 (Monday) .. 6 (Sunday)")


def next_occurrence(cadence: str, weekday: int | None, time_local: str, after: datetime) -> datetime:
    """The first slot strictly after ``after`` (an aware local datetime)."""
    validate(cadence, weekday, time_local)
    hh, mm = map(int, time_local.split(":"))
    cand = after.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cadence == "daily":
        return cand if cand > after else cand + timedelta(days=1)
    cand += timedelta(days=(int(weekday) - cand.weekday()) % 7)   # type: ignore[arg-type]
    return cand if cand > after else cand + timedelta(days=7)


class Scheduler:
    def __init__(self, store: Store, enqueue: Callable[[str, str], dict | None],
                 target_enabled: Callable[[str], bool], tick_s: float = 60.0):
        self.store, self.enqueue, self.target_enabled, self.tick_s = store, enqueue, target_enabled, tick_s

    def put(self, target_id: str, cadence: str, weekday: int | None, time_local: str, enabled: bool = True,
            now: datetime | None = None) -> dict:
        validate(cadence, weekday, time_local)
        now = now or local_now()
        old = next((s for s in self.store.schedules() if s["target_id"] == target_id), {})
        sched = {"target_id": target_id, "cadence": cadence, "weekday": weekday if cadence == "weekly" else None,
                 "time_local": time_local, "enabled": enabled, "last_run": old.get("last_run"),
                 "next_run": next_occurrence(cadence, weekday, time_local, now).isoformat()}
        self.store.save_schedule(sched)
        return sched

    def get(self, target_id: str) -> dict | None:
        return next((s for s in self.store.schedules() if s["target_id"] == target_id), None)

    def tick(self, now: datetime | None = None) -> list[str]:
        """Enqueue every due schedule once; -> the target ids enqueued."""
        now = now or local_now()
        fired = []
        for s in self.store.schedules():
            if not s["enabled"] or not self.target_enabled(s["target_id"]):
                continue
            due = s.get("next_run")
            if due and datetime.fromisoformat(due) <= now:
                self.enqueue(s["target_id"], "schedule")
                s["last_run"] = now.isoformat()
                fired.append(s["target_id"])
            if not due or datetime.fromisoformat(due) <= now:
                s["next_run"] = next_occurrence(s["cadence"], s["weekday"], s["time_local"], now).isoformat()
                self.store.save_schedule(s)
        return fired

    async def loop(self) -> None:
        while True:
            self.tick()
            await asyncio.sleep(self.tick_s)
