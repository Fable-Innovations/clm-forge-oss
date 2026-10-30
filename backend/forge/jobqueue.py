"""The global model-job queue: one model or teacher job at a time, first come first served.

Every job that loads a model or drives the teacher LLM — project stages (curating, labelling, selecting,
training, evaluating, the label-review teacher job) and autopilot steps (label, generate, train, gate) —
runs inside ``JobQueue.slot(kind, owner)``. The queue:

- is FIFO with concurrency 1 (``asyncio.Lock`` wakes its waiters in order);
- persists every job in the ``model_jobs`` table (queued -> deferred? -> running -> done | failed |
  cancelled); a server restart marks left-over rows ``interrupted`` (``Store.interrupt_model_jobs``) and
  their owners (``Orchestrator.restore_all``, ``AutopilotService.restore``) queue them again;
- runs a preflight before each job: memory pressure must be normal and no foreign model or training
  process may be running (processes matching ``Settings.preflight_ignore`` are ignored).
  Otherwise the job is ``deferred`` with the reason (``on_defer`` journals it) and re-checked every
  ``defer_s`` seconds. Nothing is ever killed.

The queue never holds the machine-wide flock (``forge.model_lock.ModelJobLock``): the child process that
loads a model still takes it itself. The queue orders this server's jobs; the flock guards the machine.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import psutil

from forge import config
from forge.store import Store, now

# Processes that hold a model or train one. Matched against the full command line.
MODEL_PROCESS = re.compile(r"(?i)\b(mlx_lm|mlx[-_.]|llama-server|llama_cpp|llama\.cpp|forge\.embed_worker|"
                           r"embed_worker|train_heads|train_v4|ollama runner|ollama_llama_server|torchrun)")


@dataclass
class Preflight:
    ok: bool
    reason: str = ""


PreflightFn = Callable[[], Preflight]
DeferFn = Callable[[str], None]


def memory_pressure_level() -> int:
    """macOS ``kern.memorystatus_vm_pressure_level``: 1 normal, 2 warn, 4 critical."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True,
                         timeout=10)
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        raise RuntimeError(f"cannot read memory pressure: {out.stderr.strip() or out.stdout.strip() or out.returncode}")
    return int(out.stdout.strip())


def foreign_model_processes(ignore: list[str]) -> list[str]:
    """Model/training processes that are neither ours (this server and its children) nor ignored."""
    me = psutil.Process(os.getpid())
    ours = {me.pid} | {p.pid for p in me.children(recursive=True)}
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if not cmd or p.info["pid"] in ours or not MODEL_PROCESS.search(cmd):
            continue
        if any(marker in cmd for marker in ignore):
            continue
        found.append(f"pid {p.info['pid']}: {cmd[:160]}")
    return found


def system_preflight() -> Preflight:
    s = config.load_settings()
    try:
        level = memory_pressure_level()
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        return Preflight(False, str(e))
    if level > s.preflight_max_pressure_level:
        name = {2: "warn", 4: "critical"}.get(level, str(level))
        return Preflight(False, f"memory pressure is {name} (level {level}); waiting for normal")
    foreign = foreign_model_processes(list(s.preflight_ignore))
    if foreign:
        return Preflight(False, f"another model process is running ({len(foreign)}): {foreign[0]}")
    return Preflight(True)


class JobQueue:
    def __init__(self, store: Store, preflight: PreflightFn | None = None, defer_s: float | None = None):
        self.store = store
        self.preflight = preflight or system_preflight
        self.defer_s = defer_s
        self._locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if loop not in self._locks:
            self._locks = {lp: lk for lp, lk in self._locks.items() if not lp.is_closed()}
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    def _defer_seconds(self) -> float:
        return self.defer_s if self.defer_s is not None else config.load_settings().job_defer_s

    @asynccontextmanager
    async def slot(self, kind: str, owner: str, on_defer: DeferFn | None = None) -> AsyncIterator[int]:
        """Wait for this job's turn (and a passing preflight), run the body, record the outcome."""
        jid = self.store.add_model_job(kind, owner)
        lock = self._lock()
        try:
            await lock.acquire()
        except asyncio.CancelledError:
            self.store.update_model_job(jid, state="cancelled", ended=now())
            raise
        try:
            last = None
            while True:
                pf = await asyncio.to_thread(self.preflight)
                if pf.ok:
                    break
                self.store.update_model_job(jid, state="deferred", error=pf.reason)
                if on_defer and pf.reason != last:
                    on_defer(pf.reason)
                last = pf.reason
                await asyncio.sleep(self._defer_seconds())
            self.store.update_model_job(jid, state="running", started=now(), error=None)
            try:
                yield jid
            except asyncio.CancelledError:
                self.store.update_model_job(jid, state="cancelled", ended=now())
                raise
            except BaseException as e:
                self.store.update_model_job(jid, state="failed", ended=now(), error=f"{type(e).__name__}: {e}"[:2000])
                raise
            self.store.update_model_job(jid, state="done", ended=now())
        except asyncio.CancelledError:
            self.store.update_model_job(jid, state="cancelled", ended=now())
            raise
        finally:
            lock.release()

    def snapshot(self, limit: int = 100) -> dict:
        """Active jobs in queue order plus the most recent finished ones (GET /api/autopilot/queue)."""
        active = sorted(self.store.model_jobs(("queued", "deferred", "running")), key=lambda j: j["id"])
        recent = self.store.model_jobs(("done", "failed", "cancelled", "interrupted"), limit=limit)
        return {"active": active, "recent": recent}
