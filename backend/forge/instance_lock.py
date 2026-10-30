"""One Forge server per data directory.

A server owns its data directory's database: on start it marks left-over model jobs interrupted and re-queues
"running" project stages and autopilot runs (``Orchestrator.restore_all``, ``AutopilotService.restore``). A second
server on the same directory would take the first server's live work for crashed work and run it a second time,
in parallel (two runners, two job queues, two model jobs at once). So ``forge-api`` holds an exclusive
``fcntl.flock`` on ``<data>/forge-server.lock`` for its whole life and refuses to start when another server
holds it, naming that server's pid and port. The kernel drops the lock when its holder exits or is killed.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import IO

LOCK_NAME = "forge-server.lock"


class InstanceLockError(RuntimeError):
    pass


def acquire(data_dir: Path, port: int) -> IO[str]:
    """Hold ``<data_dir>/forge-server.lock``; the returned file must stay open for the server's life."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / LOCK_NAME
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "{}"
        f.close()
        try:
            info = json.loads(holder)
        except ValueError:
            info = {}
        raise InstanceLockError(f"another Forge server (pid {info.get('pid', '?')}, port {info.get('port', '?')}) "
                                f"is already serving {data_dir}. Stop that server first (or `forge service "
                                "uninstall` if it is the launchd agent), or start this one with FORGE_DATA set to "
                                "another folder") from None
    f.seek(0)
    f.truncate()
    f.write(json.dumps({"pid": os.getpid(), "port": port}))
    f.flush()
    return f


def holder(data_dir: Path) -> dict | None:
    """``{pid, port}`` of the server holding ``<data_dir>/forge-server.lock``, or None when no server holds it
    (``forge service status``). Probes without keeping the lock."""
    path = Path(data_dir) / LOCK_NAME
    if not path.is_file():
        return None
    with open(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            try:
                return json.loads(f.read() or "{}")
            except ValueError:
                return {}
        fcntl.flock(f, fcntl.LOCK_UN)
    return None


__all__ = ["InstanceLockError", "LOCK_NAME", "acquire", "holder"]
