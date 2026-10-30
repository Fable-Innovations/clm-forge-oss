"""The Forge autopilot: detects what each target needs, acts on it, and journals why (SPEC "Autopilot").

Public API: ``AutopilotService``. Owner modules:

- ``targets``   target registry and policy (project | headset);
- ``detect``    pure detectors over a head's status;
- ``balance``   balance target maths;
- ``diagnose``  gate-failure diagnosis from val evidence;
- ``promotion`` the promotion rule;
- ``runner``    the bounded, resumable run loop (adapter or orchestrator);
- ``adapter``   subprocess client for the head-set adapter contract;
- ``scheduler`` daily / weekly cadence;
- ``journal``   append / query journal entries;
- ``groups``    call-group readiness;
- ``presets``   named head sets (adapter, roots, sources) the target form starts from;
- ``validation`` field-level errors for the target form;
- ``describe``  a run in words: step labels, progress, timeline, counts;
- ``actions``   one-time actions the user approved (``resplit_val``).

Target edits go through ``create`` / ``replace`` / ``set_enabled``: a changed encoder is journalled
(``encoder_changed``, with names) and re-derives a derived output root; any other change is journalled
``target_edited`` with the fields that changed.

The service runs one autopilot run at a time (``_work``); inside a run every model or teacher step also
waits for the global ``forge.jobqueue.JobQueue``, shared with project stages.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from forge import encoders
from forge.autopilot import describe
from forge.autopilot.actions import Actions
from forge.autopilot.adapter import Adapter, AdapterError
from forge.autopilot.journal import Journal
from forge.autopilot.runner import AdapterFactory, Runner, default_adapter
from forge.autopilot.scheduler import Scheduler
from forge.autopilot.targets import Policy, Target, Targets
from forge.jobqueue import JobQueue
from forge.store import Store, now

__all__ = ["AutopilotService", "Adapter", "AdapterError", "Policy", "Target"]

ACTIVE_RUNS = ("queued", "running")


class AutopilotService:
    def __init__(self, store: Store, jobs: JobQueue, *, emit=None, orchestrator=None,
                 adapter_factory: AdapterFactory = default_adapter, tick_s: float = 60.0):
        self.store, self.jobs, self.orch = store, jobs, orchestrator
        self.targets = Targets(store)
        self.journal = Journal(store, emit)
        self.journal.project_of = self._project_of
        self.actions = Actions(store, self.journal)
        self.runner = Runner(store, jobs, self.journal, orchestrator, adapter_factory, actions=self.actions)
        self.scheduler = Scheduler(store, self.enqueue, self._enabled, tick_s)
        self._wake: asyncio.Event | None = None
        self._tasks: list[asyncio.Task] = []
        self.current: str | None = None

    def _project_of(self, tid: str) -> str | None:
        t = self.targets.get(tid)
        return t.project.project_id if t and t.project else None

    def _enabled(self, tid: str) -> bool:
        t = self.targets.get(tid)
        return bool(t and t.enabled)

    # ---- lifecycle ------------------------------------------------------------------------------
    def restore(self) -> None:
        """Runs a restart interrupted are queued again; they resume at their step cursor."""
        for run in self.store.runs(states=("running", "interrupted")):
            run["state"] = "queued"
            self.store.save_run(run)
            self.journal.add(run["target_id"], "interrupted", f"server restarted during the run; re-queued at "
                             f"step {run.get('step_cursor') or '(start)'}", run_id=run["id"])

    def start(self) -> None:
        self.restore()
        self._wake = asyncio.Event()
        self._tasks = [asyncio.get_running_loop().create_task(self._work()),
                       asyncio.get_running_loop().create_task(self.scheduler.loop())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutting down
                pass
        self._tasks = []

    async def _work(self) -> None:
        while True:
            queued = sorted(self.store.runs(states=("queued",), limit=1000), key=lambda r: (r["created"], r["id"]))
            if not queued:
                assert self._wake is not None
                self._wake.clear()
                await self._wake.wait()
                continue
            await self.execute(queued[0]["id"])

    async def execute(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        t = self.targets.get(run["target_id"])
        if t is None:
            run.update({"state": "failed", "ended": now(), "result": {**run["result"], "error": "target removed"}})
            self.store.save_run(run)
            return run
        self.current = run_id
        try:
            return await self.runner.execute(run, t)
        finally:
            self.current = None

    # ---- runs -------------------------------------------------------------------------------------
    def enqueue(self, target_id: str, trigger: str) -> dict:
        """Queue a run (schedule | manual | data). A target with a run already queued or running gets that run."""
        if trigger not in ("schedule", "manual", "data"):
            raise ValueError("trigger must be schedule, manual or data")
        t = self.targets.get(target_id)
        if t is None:
            raise KeyError(target_id)
        active = self.store.runs(target_id, states=ACTIVE_RUNS, limit=1)
        if active:
            self.journal.add(target_id, "coalesced", f"{trigger} trigger joined the {active[0]['state']} run",
                             run_id=active[0]["id"])
            return active[0]
        run = {"id": uuid.uuid4().hex[:12], "target_id": target_id, "trigger": trigger, "state": "queued",
               "step_cursor": None, "round": 0, "result": {}, "started": None, "ended": None, "created": now()}
        self.store.save_run(run)
        self.journal.add(target_id, "queued", f"{trigger} run queued", run_id=run["id"])
        if self._wake is not None:
            self._wake.set()
        return run

    def run_view(self, run: dict) -> dict:
        """A run with its derived views (``describe``): progress, timeline, counts, the encoder's name and the
        target's name (``target_removed`` once the target is deleted; its runs and journal stay)."""
        live = self.runner.live.get(run["id"])
        detail = (live.progress or {}).get("message") if live else None
        key = (run.get("result") or {}).get("encoder")
        t = self.targets.get(run["target_id"])
        return {**run, "progress": describe.progress(run, detail), "timeline": describe.timeline(run),
                "counts": describe.counts(run), "results": describe.results(run), "encoder": _encoder_name(key), "target_name": t.name if t else None,
                "target_removed": t is None}

    # ---- target edits -------------------------------------------------------------------------------
    def create(self, t: Target) -> Target:
        if self.targets.get(t.id) is not None:
            raise FileExistsError(f"a target with id {t.id} already exists")
        self.targets.put(t)
        self.journal.add(t.id, "target_created", f"target {t.name} created ({t.kind}"
                         + (f", {_encoder_name(t.headset.encoder)}" if t.headset else "") + ")")
        return t

    def replace(self, old: Target, new: Target) -> Target:
        """Full replace of ``old`` by ``new`` (same id). An encoder change re-derives the output root when it was
        the derived default for the old encoder, and is journalled ``encoder_changed``; the other changed fields
        are journalled ``target_edited``."""
        switched = bool(old.headset and new.headset and old.headset.encoder != new.headset.encoder)
        rederived = False
        if switched and new.headset.output_root == old.headset.output_root:
            follow = output_root_for(old, new.headset.encoder)
            if follow != old.headset.output_root:
                new = Target.model_validate({**new.model_dump(), "headset": {
                    **new.headset.model_dump(), "output_root": follow}})
                rederived = True
        self.targets.put(new)
        changed = _changed_fields(old.model_dump(), new.model_dump())
        if switched:
            a, b = old.headset.encoder, new.headset.encoder
            self.journal.add(new.id, "encoder_changed", f"encoder changed from {_encoder_name(a)} to "
                             f"{_encoder_name(b)}; the next run trains fresh controls"
                             + (f" and writes to {new.headset.output_root}" if rederived else ""),
                             counts={"from": a, "to": b, "from_name": _encoder_name(a), "to_name": _encoder_name(b)})
            drop = {"headset.encoder"} | ({"headset.output_root"} if rederived else set())
            changed = [f for f in changed if f not in drop]
        if changed:
            self.journal.add(new.id, "target_edited", "edited " + ", ".join(changed), counts={"fields": changed})
        return new

    def set_enabled(self, t: Target, enabled: bool) -> Target:
        if t.enabled == enabled:
            return t
        new = t.model_copy(update={"enabled": enabled})
        self.targets.put(new)
        self.journal.add(t.id, "target_enabled" if enabled else "target_disabled",
                         f"target {'enabled' if enabled else 'disabled'}")
        return new

    def project_encoder_changed(self, pid: str, old_spec: str | None, new_spec: str) -> None:
        """A project's encoder changed (``PUT /api/projects/{id}/encoder``): journal it on its targets."""
        for t in self.targets.list():
            if t.project and t.project.project_id == pid:
                a, b = encoders.name_of_spec(old_spec) or "none", encoders.name_of_spec(new_spec)
                ka, kb = encoders.by_spec(old_spec), encoders.by_spec(new_spec)
                self.journal.add(t.id, "encoder_changed", f"project encoder changed from {a} to {b}",
                                 counts={"from": ka.key if ka else old_spec, "to": kb.key if kb else new_spec,
                                         "from_name": a, "to_name": b})

    # ---- reports ----------------------------------------------------------------------------------
    def target_view(self, t: Target) -> dict:
        runs = self.store.runs(t.id, limit=1)
        return {**t.model_dump(), "encoder_name": _encoder_name(t.headset.encoder) if t.headset else None,
                "output_roots": ({e.key: output_root_for(t, e.key) for e in encoders.all()} if t.headset else None),
                "schedule": self.scheduler.get(t.id), "last_run": self.run_view(runs[0]) if runs else None,
                "pending_actions": [{k: a[k] for k in ("id", "action", "approved", "reason", "queued", "last_error")}
                                    for a in self.actions.pending(t.id)]}

    def project_name(self, pid: str) -> str | None:
        rec = self.store.get_project(pid)
        return rec.get("name") if rec else None

    def groups(self) -> list[dict]:
        out = []
        for t in self.targets.list():
            if t.kind != "headset":
                continue
            run = next((r for r in self.store.runs(t.id, limit=50) if r["result"].get("groups")), None)
            out.append({"target": t.id, "run": run["id"] if run else None, "computed_at": run["ended"] if run else None,
                        "report": run["result"]["groups"] if run else None})
        return out

    def provenance(self, head: str, target_id: str | None = None) -> dict:
        from forge.labelling import _HEAD_ID
        if not _HEAD_ID.match(head or ""):
            raise ValueError(f"bad head id {head!r}")
        cands = [t for t in self.targets.list() if t.kind == "headset" and (target_id is None or t.id == target_id)]
        for t in cands:
            assert t.headset is not None
            p = Path(t.headset.output_root).expanduser() / head / "provenance.json"
            if p.is_file():
                return {"target": t.id, "path": str(p), "provenance": json.loads(p.read_text())}
        raise FileNotFoundError(f"no provenance.json for head {head!r}"
                                + (f" in target {target_id}" if target_id else ""))


def _encoder_name(key: str | None) -> str | None:
    if not key:
        return None
    if encoders.known(key):
        return encoders.get(key).name
    return encoders.name_of_spec(key) if ":" in key else key


def output_root_for(t: Target, key: str) -> str:
    """Where ``t`` would write on encoder ``key``: a derived output root (``encoders.output_root``) follows the
    encoder, a root set by hand stays. The target form shows this, ``AutopilotService.replace`` applies it."""
    assert t.headset is not None
    base = _derived_base(t.headset.output_root, t.headset.encoder)
    return encoders.output_root(base, key) if base is not None else t.headset.output_root


def _derived_base(output_root: str, key: str) -> str | None:
    """The base an output root was derived from for ``key`` (``encoders.output_root``), or None when the root was
    set by hand."""
    suffix = f"-{key}"
    return output_root[: -len(suffix)] if output_root.endswith(suffix) else None


def _changed_fields(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb, name = a.get(k), b.get(k), f"{prefix}{k}"
        if isinstance(va, dict) and isinstance(vb, dict) and k != "settings":
            out += _changed_fields(va, vb, name + ".")
        elif va != vb:
            out.append(name)
    return out
