"""Machine-wide "one loaded model at a time" lock, shared by every process that imports forge.

The lock is an ``fcntl.flock`` on ``~/.cache/clm-forge/model-job.lock`` (``FORGE_MODEL_LOCK_DIR``
overrides the directory, for tests). It is held by the embedding worker (``forge.embed_worker``)
from before the model loads until the worker exits, so two Forge servers, a test run and a server,
or two projects can never hold two base models in RAM at once. The kernel drops a flock when its
holder dies, so a crashed or killed worker never leaves the lock stuck.

Waiters announce themselves with a file ``waiters/<pid>``. A holder that is *idle* (loaded but not
embedding) checks ``others_waiting()`` and exits to let them in; a busy holder finishes its request
first. The holder writes ``{"pid", "owner", "since"}`` into the lock file so a waiter can say who
it is waiting for.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import psutil

Notify = Callable[[str, str], None]


class ModelLockTimeout(RuntimeError):
    """The machine-wide model lock stayed taken for longer than the configured wait."""


def lock_dir() -> Path:
    return Path(os.environ.get("FORGE_MODEL_LOCK_DIR", Path.home() / ".cache" / "clm-forge")).expanduser()


class ModelJobLock:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory else lock_dir()
        self.path = self.dir / "model-job.lock"
        self.waiters = self.dir / "waiters"
        self._fd: int | None = None

    # -- holder --------------------------------------------------------------------------------
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> dict:
        try:
            return json.loads(self.path.read_text() or "{}")
        except (OSError, ValueError):
            return {}

    def acquire(self, owner: str, timeout: float, notify: Notify, *, poll_s: float = 0.25,
                report_every_s: float = 15.0) -> None:
        """Block until the lock is ours (reporting who holds it), or raise ``ModelLockTimeout``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.waiters.mkdir(exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        me = self.waiters / str(os.getpid())
        t0, last, waiting = time.monotonic(), -1e9, False
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
                if not waiting:
                    me.write_text(owner)
                    waiting = True
                now = time.monotonic()
                if now - t0 >= timeout:
                    h = self.holder()
                    raise ModelLockTimeout(f"model lock still held by pid {h.get('pid', '?')} ({h.get('owner', '?')}) "
                                           f"after {timeout:.0f}s; another model job is running on this machine")
                if now - last >= report_every_s:
                    h = self.holder()
                    notify("warn", f"waiting for model lock: pid {h.get('pid', '?')} is running "
                                   f"{h.get('owner', 'a model job')}")
                    last = now
                time.sleep(poll_s)
        except BaseException:
            os.close(fd)
            raise
        finally:
            if waiting:
                me.unlink(missing_ok=True)
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps({"pid": os.getpid(), "owner": owner, "since": time.time()}).encode(), 0)
        self._fd = fd
        if waiting:
            notify("info", f"model lock acquired after {time.monotonic() - t0:.0f}s")

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.ftruncate(self._fd, 0)
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    # -- waiters --------------------------------------------------------------------------------
    def others_waiting(self) -> list[int]:
        """Live pids (other than ours) queued for the lock; stale waiter files are removed."""
        out = []
        for f in self.waiters.glob("*") if self.waiters.exists() else []:
            try:
                pid = int(f.name)
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            if psutil.pid_exists(pid):
                out.append(pid)
            else:
                f.unlink(missing_ok=True)
        return out
