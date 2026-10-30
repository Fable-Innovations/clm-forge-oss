"""In-process pub/sub -> SSE.

`bus.emit(pid, {"type": ..., ...})` stamps ts + seq, persists (except ephemeral high-rate types),
and fans out to every subscriber. Safe to call from worker threads: delivery to each subscriber's
asyncio queue goes through that subscriber's loop with `call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from forge.store import Store, now

# High-rate / transient events: streamed live but not replayed on reconnect.
EPHEMERAL = frozenset({"train_step", "chat_delta", "embed"})


@dataclass(eq=False)
class Subscription:
    pid: str
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop


class EventBus:
    def __init__(self, store: Store):
        self.store = store
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._subs: dict[str, set[Subscription]] = {}

    def _next_seq(self, pid: str) -> int:
        if pid not in self._seq:
            self._seq[pid] = self.store.max_seq(pid)
        self._seq[pid] += 1
        return self._seq[pid]

    def emit(self, pid: str, ev: dict[str, Any]) -> dict:
        ev = dict(ev)
        with self._lock:
            ev["seq"] = self._next_seq(pid)
            ev.setdefault("ts", now())
            if ev.get("type") not in EPHEMERAL:
                self.store.add_event(pid, ev)
            subs = list(self._subs.get(pid, ()))
        for s in subs:
            try:
                s.loop.call_soon_threadsafe(s.queue.put_nowait, ev)
            except RuntimeError:  # loop closed
                self.unsubscribe(s)
        return ev

    def subscribe(self, pid: str) -> Subscription:
        s = Subscription(pid, asyncio.Queue(maxsize=10000), asyncio.get_running_loop())
        with self._lock:
            self._subs.setdefault(pid, set()).add(s)
        return s

    def unsubscribe(self, s: Subscription) -> None:
        with self._lock:
            self._subs.get(s.pid, set()).discard(s)

    def forget(self, pid: str) -> None:
        with self._lock:
            self._seq.pop(pid, None)
