"""The one lock around a labelling queue's ``real_rows.jsonl`` (SPEC "Labelling real rows" > "Locking").

Two kinds of writer edit the same file: Forge's ``LabelQueue`` (the UI, the curator, a project's labelling stage,
threads of the API process) and the head-set adapter's processes (its ingest, labelling agent and
teacher). Every read-modify-write of ``<dir>/real_rows.jsonl`` holds ``real_rows_lock(dir)``: an
``fcntl.flock`` on ``<dir>/.real_rows.lock`` (excludes other processes) plus a process-local re-entrant lock
(excludes other threads; ``flock`` locks belong to the open file, so two threads of one process would not
exclude each other through it). The writer re-reads the file inside the lock.

Re-entrant within a thread: a nested ``real_rows_lock`` on the same directory takes neither lock again.
"""
from __future__ import annotations

import contextlib
import fcntl
import threading
from collections.abc import Iterator
from pathlib import Path

LOCK_FILE = ".real_rows.lock"

_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_depth = threading.local()


def lock_path(d: Path | str) -> Path:
    return Path(d) / LOCK_FILE


def _thread_lock(key: str) -> threading.RLock:
    with _guard:
        return _locks.setdefault(key, threading.RLock())


@contextlib.contextmanager
def real_rows_lock(d: Path | str) -> Iterator[None]:
    """Hold the cross-process and cross-thread lock of ``<d>/real_rows.jsonl``."""
    path = lock_path(d)
    key = str(path.parent.resolve())
    tl = _thread_lock(key)
    with tl:
        held: dict[str, int] = getattr(_depth, "held", None) or {}
        _depth.held = held
        if held.get(key):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            held[key] = 1
            try:
                yield
            finally:
                held[key] = 0
                fcntl.flock(f, fcntl.LOCK_UN)


__all__ = ["LOCK_FILE", "lock_path", "real_rows_lock"]
