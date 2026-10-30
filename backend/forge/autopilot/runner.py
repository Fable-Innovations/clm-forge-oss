"""The bounded autopilot run loop (SPEC "Autopilot" > "Run loop").

A run is a sequence of steps, each keyed (``pull``, ``label:<head>``, ``gen:<head>:<round>:<i>``,
``train:<head>:<round>``, ``gate:<head>``, ...). A finished step's output is stored in the run's
``result.steps`` and ``step_cursor`` names the last one, so every step is idempotent: a resumed run (after a
restart) replays the stored outputs and continues at the first step that has none. Every decision is a pure
function of stored outputs, so a replay takes the same branch. (The adapter is idempotent too: a staged
candidate's gate, a control run and a generation batch are stored on its side and returned again.)

Head-set run (verbs and shapes: SPEC "Autopilot adapter contract"), per head:

1. ``pull-sources`` and ``ingest`` (once per run), ``inspect`` -> ``detect`` -> journal the needs; a head with
   no need, or a retired head, is skipped.
2. ``label`` (the labelling agent, with adjudication) when rows are undisposed or labels are noisy.
3. Rounds (at most ``Policy.max_rounds``): inspect; generate (balance deficits, shape gaps steered to the
   uncovered templates / channels, and the previous round's remedies; at most ``gen_cap`` rows, the rest
   journalled as carry; each ask goes out under the generate label ``inspect`` reports for its balance group,
   and an ask for a group no label fills, or for a head ``generate`` cannot serve, is journalled ``skip``); a control run per control mix (the adapter reuses one whose fingerprint holds);
   ``train --mix all`` (attempts chosen on val by the adapter); the val gate: val macro-F1 >= the head's
   ``gate_threshold``. A failed round is diagnosed from VAL evidence only (``diagnose``).
4. The best val-passing round is the candidate. ``gate`` reads TEST exactly once per head per run (a gate
   call interrupted mid-way is never repeated in that run); ``promotion.decide`` promotes it (``promote``) or
   leaves it in staging.
5. ``groups`` builds the call-group readiness report.

One-time actions the user approved (``forge.autopilot.actions``) run after the first ``inspect``, before any head:
``resplit_val`` (scheduled runs only; a manual or data run journals ``action_waiting``) calls ``resplit-val`` for
every head that is not retired (a model slot: it drives the generator and the teacher), journals ``resplit_val``
per head, and the heads are inspected again so this run trains and selects on the new val. The step may create
only the new val files (``RESPLIT_FILES``, under the writer guard's protected globs from then on); anything else it
changed fails the run, as for every step. A failed ``resplit-val`` fails the run (training on the val the user
decided to replace would be the masking path) and leaves the action queued for the next scheduled run.

Heads are isolated: a verb that exits non-zero (``AdapterError``) fails that head (``outcome=failed``, its
error, a ``head_failed`` journal entry with the adapter's stderr tail) and the run goes on with the next head;
``groups`` is still built, and the run ends ``done_with_failures``. A broken contract (``AdapterContractError``,
``ContractError``) or a writer-guard breach fails the whole run.

Every model or teacher step (label, generate, train, gate) runs inside a ``JobQueue`` slot. After every
adapter call the writer guard checks that no protected file (``rows.jsonl``, frozen val/test, files named in
``protected_files``) changed; a change fails the run.

Project targets are inspected and journalled (needs per the same detectors); the project's own pipeline
acts on them (its labelling stage is automatic).

Progress record (read by ``forge.autopilot.describe``): ``result.plan`` (written once: the heads the first
``inspect`` reported and ``PHASES``; a project run: its head and the orchestrator's stages), ``result.timings``
(per step key ``{s, e}``; a step with ``s`` and no ``e`` is the one running), per head ``started`` / ``ended`` in
``result.heads``, ``result.encoder`` (the registry key the run trains on) and ``run.first_started`` (set once; a
resumed run keeps it, while ``started`` is when this execution began).
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from forge.autopilot.adapter import Adapter, AdapterContractError, AdapterError
from forge.autopilot.detect import HeadStatus, Need, SplitMetrics, balance_plan, detect
from forge.autopilot.diagnose import Remedy, diagnose
from forge.autopilot.groups import readiness
from forge.autopilot.journal import Journal
from forge.autopilot.promotion import decide
from forge.autopilot.targets import HeadsetTarget, Target
from forge.jobqueue import JobQueue
from forge.store import Store, now

AdapterFactory = Callable[[Target], Adapter]
#: per head, the files the adapter's ``resplit-val`` creates (plus the archived old val,
#: ``_val_archive/val-<run>.jsonl``); the preset's protected globs freeze them after
RESPLIT_FILES = ("val_redrawn.jsonl", "val_split.json")
#: the phases of one head in a head-set run, in order (``describe.phase_of`` maps a step key to one)
PHASES = ["needs", "label", "generate", "controls", "train", "gate", "decision", "promote"]


class WriterGuardError(RuntimeError):
    """A step changed a file the autopilot must never write (rows.jsonl, a frozen val/test split)."""


class ContractError(RuntimeError):
    """The adapter's output lacks what the contract promises (a head's status, a gate threshold, val scores).
    Like ``AdapterContractError`` and ``WriterGuardError`` it fails the whole run: the adapter is wrong, not one
    head's data."""


class WriterGuard:
    def __init__(self, cfg: HeadsetTarget):
        self.root = Path(cfg.dataset_root).expanduser()
        self.globs = list(cfg.protected_globs)
        self.extra = [Path(f).expanduser() for f in cfg.protected_files]

    def files(self) -> list[Path]:
        out = {p for g in self.globs for p in self.root.glob(g) if p.is_file()} if self.root.is_dir() else set()
        return sorted(out | set(self.extra))

    def snapshot(self) -> dict[str, str]:
        snap = {}
        for p in self.files():
            if not p.exists():
                snap[str(p)] = "missing"
                continue
            h = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            snap[str(p)] = h.hexdigest()
        return snap

    def verify(self, before: dict[str, str], step: str, may_create: set[str] = frozenset()) -> dict[str, str]:
        """Refuse any change to a protected file since ``before``, except creating one named in ``may_create`` (a
        file that did not exist). -> the snapshot now."""
        after = self.snapshot()
        changed = sorted(p for p in set(before) | set(after) if before.get(p, "missing") != after.get(p, "missing")
                         and not (p in may_create and before.get(p, "missing") == "missing"))
        if changed:
            raise WriterGuardError(f"step {step} changed protected files: {changed[:5]}"
                                   + (f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""))
        return after


def _check_counts(check: dict | None) -> dict | None:
    """The real check record as journal counts: per stage n / candidate / incumbent macro-F1, and the verdict."""
    if not check:
        return None
    stages = {s: {"n": v.get("n", 0), "candidate": (v.get("candidate") or {}).get("macro_f1"),
                  "incumbent": (v.get("incumbent") or {}).get("macro_f1")}
              for s, v in (check.get("stages") or {}).items()}
    return {"deciding": check.get("deciding"), "passed": check.get("passed"), "stages": stages}


class NoPresetError(RuntimeError):
    """A head set without a preset: Forge does not run an adapter command that came from anywhere else."""


def default_adapter(t: Target) -> Adapter:
    """The head set's adapter, its command, working folder and environment taken from its preset (server-side),
    never from the stored target."""
    from forge.autopilot import presets
    cfg = t.headset
    assert cfg is not None
    if cfg.preset is None:
        raise NoPresetError(f"target {t.id} names no head-set preset; Forge runs only preset adapters (recreate it "
                            "from a preset)")
    p = presets.get(cfg.preset)
    if why := p.output_root_refusal(Path(cfg.output_root).expanduser().resolve()):
        raise NoPresetError(f"target {t.id}: {why}; edit its output folder")
    return Adapter(list(p.adapter), p.cwd, dict(p.env), cfg.target_config())


class _Run:
    """One run's mutable record plus the step memo."""

    def __init__(self, store: Store, run: dict):
        self.store, self.run = store, run
        res = run.setdefault("result", {})
        self.steps: dict[str, Any] = res.setdefault("steps", {})
        self.res = res
        self.progress: dict | None = None

    def save(self) -> None:
        self.store.save_run(self.run)

    async def step(self, key: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        if key in self.steps:
            return self.steps[key]
        timings = self.res.setdefault("timings", {})
        timings[key] = {"s": now(), "e": None}
        self.save()
        out = await fn()
        self.steps[key] = out
        timings[key]["e"] = now()
        self.run["step_cursor"] = key
        self.save()
        return out


class Runner:
    def __init__(self, store: Store, jobs: JobQueue, journal: Journal, orchestrator=None,
                 adapter_factory: AdapterFactory = default_adapter, actions=None):
        self.store, self.jobs, self.journal = store, jobs, journal
        self.actions = actions
        self.orch = orchestrator
        self.adapter_factory = adapter_factory
        self.live: dict[str, _Run] = {}

    # ---- entry ------------------------------------------------------------------------------------
    async def execute(self, run: dict, target: Target) -> dict:
        r = _Run(self.store, run)
        self.live[run["id"]] = r
        resumed = bool(r.steps)
        started = now()
        run.update({"state": "running", "started": started, "ended": None,
                    "first_started": run.get("first_started") or run.get("started") or started})
        r.res.setdefault("encoder", self._encoder_key(target))
        r.save()
        self._j(r, "run_started" if not resumed else "run_resumed",
                f"{run['trigger']} run" + (f", resuming after {run.get('step_cursor')}" if resumed else ""))
        try:
            if target.kind == "headset":
                await self._headset(r, target)
            else:
                await self._project(r, target)
            failed = sorted(h for h, v in (r.res.get("heads") or {}).items() if v.get("outcome") == "failed")
            run["state"] = "done_with_failures" if failed else "done"
            self._j(r, "run_done_with_failures" if failed else "run_done", self._summary(r),
                    counts={"failed_heads": failed} if failed else None)
        except asyncio.CancelledError:
            run["state"] = "interrupted"
            raise
        except Exception as e:  # noqa: BLE001 - every failure is recorded and journalled, never swallowed
            run["state"] = "failed"
            r.res["error"] = f"{type(e).__name__}: {e}"
            self._j(r, "run_failed", r.res["error"][:2000])
        finally:
            run["ended"] = now()
            r.save()
            self.live.pop(run["id"], None)
        return run

    def _encoder_key(self, t: Target) -> str | None:
        """The registry key the run trains on (a project's ``base_model`` spec when it has no registry entry)."""
        if t.headset is not None:
            return t.headset.encoder
        from forge import encoders
        rec = self.store.get_project(t.project.project_id) if t.project else None
        spec = (rec or {}).get("base_model")
        e = encoders.by_spec(spec)
        return e.key if e else spec

    def _j(self, r: _Run, action: str, reason: str, *, head: str | None = None, round: int | None = None,
           counts: dict | None = None) -> None:
        self.journal.add(r.run["target_id"], action, reason, run_id=r.run["id"], head=head, round=round,
                         counts=counts)

    @staticmethod
    def _summary(r: _Run) -> str:
        heads = r.res.get("heads") or {}
        if not heads:
            return "done"
        parts = [f"{h}: {v.get('outcome')}" for h, v in sorted(heads.items())]
        return "; ".join(parts)

    # ---- head sets ----------------------------------------------------------------------------------
    async def _headset(self, r: _Run, t: Target) -> None:
        cfg = t.headset
        assert cfg is not None
        pol = cfg.policy
        ad = self.adapter_factory(t)
        guard = WriterGuard(cfg)
        if "protected" not in r.res:
            r.res["protected"] = await asyncio.to_thread(guard.snapshot)
            r.save()
        protected = r.res["protected"]
        owner = f"autopilot:{t.id}:{r.run['id']}"

        def on_progress(ev: dict) -> None:
            r.progress = ev

        async def call(verb: str, key: str, model: bool, config_overrides: dict | None = None,
                       may_create: set[str] = frozenset(), **kw) -> dict:
            async def go() -> dict:
                if model:
                    async with self.jobs.slot(verb, owner, on_defer=lambda why: self._j(
                            r, "defer", f"{verb} deferred: {why}", head=kw.get("head"))):
                        out = await ad.call(verb, run_id=r.run["id"], on_progress=on_progress,
                                            config_overrides=config_overrides, **kw)
                else:
                    out = await ad.call(verb, run_id=r.run["id"], on_progress=on_progress,
                                        config_overrides=config_overrides, **kw)
                now_ = await asyncio.to_thread(guard.verify, protected, key, may_create)
                if may_create and now_ != protected:     # created files are protected from the next step on
                    protected.clear()
                    protected.update(now_)
                    r.res["protected"] = protected
                    r.save()
                return out
            return await r.step(key, go)

        if "pull" not in r.steps:
            out = await call("pull-sources", "pull", False, out=str(Path(cfg.dataset_root) / "_raw" / "real"))
            new = {src: v.get("new_rows", 0) for src, v in out.items() if isinstance(v, dict)}
            self._j(r, "pull", f"exported real sources (read-only snapshots): new rows {new}", counts=new)
            for src, v in sorted(out.items()):
                if isinstance(v, dict) and v.get("snapshot_restamped"):
                    rs = v["snapshot_restamped"]
                    self._j(r, "snapshot_restamped", f"{src}: the install's store is stamped schema {rs.get('stamped')} "
                            f"but holds schema {rs.get('holds')} columns (a build skew in the app that wrote it); the snapshot copy was "
                            "re-stamped before export, the live store was not touched", counts=rs)
        if "ingest" not in r.steps:
            out = await call("ingest", "ingest", False)
            if out.get("disabled_exports_removed"):
                self._j(r, "source_disabled_removed", "removed the pulled exports of decision sources disabled on "
                        f"this target: {', '.join(out['disabled_exports_removed'])}",
                        counts={"files": out["disabled_exports_removed"]})
            for h, v in sorted((out.get("heads") or {}).items()):
                if v.get("source_disabled_removed"):
                    self._j(r, "source_disabled_removed", f"removed {v['source_disabled_removed']} stored real rows "
                            "whose decision source is disabled on this target (with "
                            f"{v.get('source_disabled_shaped', 0)} rows generated from them and "
                            f"{v.get('source_disabled_artifacts', 0)} unpromoted heads trained on them); a promoted "
                            "head that learned from them is retrained", head=h,
                            counts={"removed": v["source_disabled_removed"],
                                    "shaped_removed": v.get("source_disabled_shaped", 0),
                                    "artifacts_removed": v.get("source_disabled_artifacts", 0)})
                fmt = v.get("format_excluded") or {}
                self._j(r, "ingest", f"{v.get('added', 0)} new real rows; dropped {v.get('dropped') or {}}; "
                        f"deleted at the source {v.get('pruned_deleted', 0)} (with {v.get('shaped_removed', 0)} rows "
                        f"generated from them, {v.get('stage_folders_removed', 0)} generation stage folders and "
                        f"{v.get('artifacts_removed', 0)} unpromoted heads trained on them); "
                        "same content as a kept row "
                        f"{v.get('same_content', 0)}; excluded by state format {sum(fmt.values())} {fmt or ''}"
                        f"; check sets {(v.get('check') or {}).get('stages') or {}}",
                        head=h, counts={"added": v.get("added", 0), "dropped": v.get("dropped") or {},
                                        "pruned_deleted": v.get("pruned_deleted", 0),
                                        "shaped_removed": v.get("shaped_removed", 0),
                                        "stage_folders_removed": v.get("stage_folders_removed", 0),
                                        "artifacts_removed": v.get("artifacts_removed", 0),
                                        "same_content": v.get("same_content", 0), "format_excluded": fmt,
                                        "check": v.get("check") or {}})
        status0 = await call("inspect", "inspect", False, heads=cfg.heads or None)
        statuses = [HeadStatus.model_validate(s) for s in status0.get("heads") or []]
        if cfg.heads:
            missing = sorted(set(cfg.heads) - {s.head for s in statuses})
            if missing:
                raise ContractError(f"inspect did not report heads {missing}")
        if "plan" not in r.res:
            r.res["plan"] = {"heads": [s.head for s in statuses], "phases": list(PHASES), "started_at": now()}
            r.save()
        if await self._run_actions(r, t, statuses, call):
            status0 = await call("inspect", "inspect:after_actions", False, heads=cfg.heads or None)
            statuses = [HeadStatus.model_validate(s) for s in status0.get("heads") or []]
        heads = r.res.setdefault("heads", {})
        for st in statuses:
            rec = heads.setdefault(st.head, {"rounds": []})
            if not rec.get("outcome") and not rec.get("started"):
                rec["started"] = now()
                r.save()
            try:
                await self._head(r, t, st, call, heads)
            except AdapterContractError:
                raise
            except AdapterError as e:          # one head's verb failed: record it, go on with the other heads
                self._head_failed(r, st.head, heads, e)
            if rec.get("outcome") and not rec.get("ended"):
                rec["ended"] = now()
                r.save()
        out = await call("groups", "groups", False)
        r.res["groups"] = readiness(out, pol, cfg.encoder)
        r.save()
        ready = [g["name"] for g in r.res["groups"]["groups"] if g["ready"]]
        self._j(r, "groups", f"{len(ready)}/{len(r.res['groups']['groups'])} call groups ready on {cfg.encoder}",
                counts={"ready": ready})

    async def _run_actions(self, r: _Run, t: Target, statuses: list[HeadStatus], call) -> bool:
        """The target's queued one-time actions (``forge.autopilot.actions``; module docstring). -> whether one ran
        in this run (the heads are inspected again)."""
        if self.actions is None:
            return False
        from forge.autopilot.actions import WHEN
        ran = False
        for a in self.actions.pending(t.id):
            key = f"action:{a['id']}"
            if r.run["trigger"] not in WHEN[a["action"]]:
                if f"{key}:waiting" not in r.steps:
                    when = " or ".join("scheduled" if w == "schedule" else w for w in WHEN[a["action"]])
                    self._j(r, "action_waiting", f"{a['action']} ({a['approved']}) waits for the next {when} run; "
                            f"this is a {r.run['trigger']} run",
                            counts={"action": a["action"], "action_id": a["id"]})
                    r.steps[f"{key}:waiting"] = True
                    r.save()
                continue
            if a["action"] == "resplit_val":
                await self._resplit_val(r, t, a, statuses, call, key)
                ran = True
        return ran

    async def _resplit_val(self, r: _Run, t: Target, a: dict, statuses: list[HeadStatus], call, key: str) -> None:
        cfg = t.headset
        assert cfg is not None and self.actions is not None
        heads = [s.head for s in statuses if not s.retired]
        for s in statuses:
            if s.retired and f"{key}:retired:{s.head}" not in r.steps:
                self._j(r, "skip", f"resplit_val: retired ({s.retired}), its val is not redrawn", head=s.head)
                r.steps[f"{key}:retired:{s.head}"] = True
        if f"{key}:started" not in r.steps:
            self._j(r, "need", f"resplit_val ({a['approved']}): {a['reason']}",
                    counts={"kind": "resplit_val", "remedy": "resplit_val", "heads": heads, "action_id": a["id"]})
            r.steps[f"{key}:started"] = True
            r.save()
        root = Path(cfg.dataset_root).expanduser()
        creates = {str(root / h / f) for h in heads
                   for f in (*RESPLIT_FILES, f"_val_archive/val-{r.run['id']}.jsonl")}
        fresh = key not in r.steps
        try:
            out = await call("resplit-val", key, True, may_create=creates, heads=heads)
        except AdapterError as e:
            self.actions.failed(a, r.run["id"], str(e))
            self._j(r, "action_failed", f"resplit_val failed; it stays queued for the next scheduled run: "
                    f"{(e.stderr or str(e)).strip()[-1500:]}", counts={"action": "resplit_val", "action_id": a["id"]})
            raise
        if fresh:
            for h, v in sorted((out.get("heads") or {}).items()):
                if not v.get("applicable", True):
                    self._j(r, "skip", f"resplit_val: {v.get('reason')}", head=h)
                    continue
                old, new = v.get("archived") or {}, v.get("val") or {}
                self._j(r, "resplit_val", f"val redrawn the way the frozen test was built ({v.get('procedure')}): "
                        f"{new.get('rows')} rows {new.get('labels')}, replacing {old.get('rows')} archived rows "
                        f"({old.get('file')}); leak check dropped {v.get('leak_check') or {}}; test untouched "
                        f"(sha256 {str(v.get('test_sha256'))[:12]})", head=h,
                        counts={"val": new, "archived": old, "leak_check": v.get("leak_check") or {},
                                "test_sha256": v.get("test_sha256"), "action_id": a["id"]})
        self.actions.complete(a, r.run["id"], {"heads": sorted((out.get("heads") or {}))})
        if fresh:
            self._j(r, "action_done", f"resplit_val done for {len(out.get('heads') or {})} heads; training and "
                    "selection use the new val from this run on", counts={"action": "resplit_val",
                                                                           "action_id": a["id"]})

    def _head_failed(self, r: _Run, h: str, heads: dict, e: AdapterError) -> None:
        """A head whose adapter call exited non-zero: ``outcome=failed`` with the error, journalled
        ``head_failed`` (the verb and the adapter's stderr tail). Not retried in this run (a resumed run keeps
        the outcome); the next run starts it afresh."""
        rec = heads.setdefault(h, {"rounds": []})
        tail = (e.stderr or str(e)).strip()[-1500:]
        rec["outcome"], rec["error"] = "failed", str(e)[:2000]
        self._j(r, "head_failed", f"{e.verb or 'adapter'} failed: {tail}", head=h,
                counts={"verb": e.verb, "round": r.run.get("round")})
        r.save()

    async def _head(self, r: _Run, t: Target, st0: HeadStatus, call, heads: dict) -> None:
        cfg = t.headset
        assert cfg is not None
        pol, h, run_id = cfg.policy, st0.head, r.run["id"]
        rec = heads.setdefault(h, {"rounds": []})
        if rec.get("outcome"):
            return
        if st0.retired:
            rec["outcome"] = "retired"
            self._j(r, "skip", f"retired ({st0.retired}): not trained", head=h)
            r.save()
            return
        trigger_due = r.run["trigger"] == "schedule"
        needs = detect(st0, pol, schedule_due=trigger_due)
        if f"needs:{h}" not in r.steps:
            for n in needs:
                self._j(r, "need", n.reason, head=h, counts={"kind": n.kind, "remedy": n.remedy, **n.counts})
            r.steps[f"needs:{h}"] = [n.model_dump() for n in needs]
            r.save()
        if not needs:
            rec["outcome"] = "skipped"
            self._j(r, "skip", "no need detected", head=h)
            r.save()
            return
        kinds = {n.kind for n in needs}
        if kinds & {"smoke_rows", "reshape"}:
            key = f"clean:{h}"
            fresh = key not in r.steps
            out = await call("clean", key, False, head=h)
            if fresh:
                smoke, bad = out.get("smoke") or [], out.get("reshape") or {}
                stages = out.get("smoke_stages") or []
                self._j(r, "clean", f"removed {len(smoke)} smoke rows, {len(stages)} smoke stage folders and "
                        f"{sum(map(len, bad.values()))} rows the real-source guard v{out.get('guard_version')} "
                        f"rejects ({out.get('checked', 0)} rechecked); a head trained on them is retrained", head=h,
                        counts={"smoke": len(smoke), "smoke_stages": len(stages),
                                "reshape": {k: len(v) for k, v in bad.items()},
                                "checked": out.get("checked", 0), "kept": out.get("kept", 0)})
        if kinds & {"unlabelled", "label_noise"}:
            key = f"label:{h}"
            fresh = key not in r.steps
            out = await call("label", key, True, head=h, **({"adjudicate": True} if pol.adjudicate
                                                            else {"no_adjudicate": True}))
            got = (out.get("heads") or {}).get(h) or {}
            if fresh:
                self._j(r, "label", f"agent labelling: {got.get('agent_labelled', 0)} labelled, "
                        f"{got.get('adjudicated', 0)} adjudicated, {got.get('excluded_by_agent', 0)} excluded, "
                        f"{got.get('undisposed', 0)} still undisposed", head=h,
                        counts={k: v for k, v in got.items() if isinstance(v, (int, float))})
                contra = got.get("export_contradictions") or []
                if contra:
                    self._j(r, "export_contradicted", f"{len(contra)} export labels contradicted by the teacher "
                            "(export kept)", head=h, counts={"n": len(contra), "first": contra[:3]})
            # the adapter fails a label step that leaves rows undisposed; an output that still reports some is
            # the same failure (never trained on half-labelled data, never skipped silently)
            left = int(got.get("undisposed") or 0) - (0 if pol.adjudicate else int(got.get("still_disputed") or 0))
            if left > 0:
                raise AdapterError(f"label left {left} real rows of {h} without a final disposition", verb="label",
                                   stderr=f"label left {left} rows undisposed: teacher_failed "
                                          f"{got.get('teacher_failed', 0)}, still_disputed {got.get('still_disputed', 0)}"
                                          f", export_check_failed {got.get('export_check_failed', 0)}")
        remedies: list[Remedy] = []
        real_cap: int | None = None
        candidates = []
        for rnd in range(1, pol.max_rounds + 1):
            r.run["round"] = rnd
            st = _one(await call("inspect", f"inspect:{h}:{rnd}", False, heads=[h]), h)
            if st.gate_threshold is None:
                raise ContractError(f"adapter inspect reported no gate_threshold for {h} (contract: HeadStatus)")
            rneeds = detect(st, pol, schedule_due=trigger_due)
            gens = self._generation_plan(st, rneeds, remedies, pol, r, h, rnd)
            if gens and pol.generate_kinds and st.model_extra and st.model_extra.get("kind") not in pol.generate_kinds:
                if f"nogen:{h}" not in r.steps:
                    self._j(r, "skip", f"no generator for {st.model_extra.get('kind')!r} heads: {len(gens)} generation "
                            "asks not made", head=h, round=rnd)
                    r.steps[f"nogen:{h}"] = True
                gens = []
            for i, g in enumerate(gens):
                key = f"gen:{h}:{rnd}:{i}"
                fresh = key not in r.steps
                out = await call("generate", key, True, head=h, label=g["label"], n=g["n"], exemplars=pol.exemplars,
                                 batch=f"{run_id}-{rnd}-{i}", focus_template=tuple(g.get("templates") or ()),
                                 focus_channel=tuple(g.get("channels") or ()))
                if fresh:
                    self._j(r, "generate", f"{g['why']}: {out.get('accepted', 0)}/{g['n']} {g['label']} rows accepted"
                            + (f" (confused with {g['focus']})" if g.get("focus") else ""),
                            head=h, round=rnd, counts={"accepted": out.get("accepted", 0), "asked": g["n"],
                                                       "drops": out.get("drops") or {}})
            if gens:
                st = _one(await call("inspect", f"inspect:{h}:{rnd}:post", False, heads=[h]), h)
            controls: dict[str, float | None] = {}
            control_fps: dict[str, str | None] = {}
            for mix in st.required_controls():
                key = f"control:{h}:{rnd}:{mix}"
                fresh = key not in r.steps
                # like for like: the candidate's seed set (``attempts``) and its val-based choice among them
                out = await call("train", key, True, head=h, encoder=cfg.encoder, mix=mix, attempts=pol.attempts)
                controls[mix] = (out.get("val") or {}).get("macro_f1")
                control_fps[mix] = out.get("fingerprint")
                if fresh:
                    self._j(r, "control", f"control ({mix}){' reused' if out.get('reused') else ''}: val macro-F1 "
                            f"{controls[mix]}, best of {pol.attempts} seeds on val", head=h, round=rnd,
                            counts={"mix": mix, "val_macro_f1": controls[mix], "fingerprint": out.get("fingerprint"),
                                    "attempts": pol.attempts})
            key = f"train:{h}:{rnd}"
            fresh = key not in r.steps
            tr = await call("train", key, True, config_overrides={"real_cap": real_cap} if real_cap else None,
                            head=h, encoder=cfg.encoder, mix="all", attempts=pol.attempts)
            vm = SplitMetrics.model_validate(tr.get("val") or _fail(f"train for {h} returned no val scores"))
            val_ok = vm.macro_f1 >= st.gate_threshold
            why = [f"val macro-F1 {vm.macro_f1:.3f} {'>=' if val_ok else '<'} gate threshold {st.gate_threshold:.3f}"]
            # a control trained on the candidate's very training set is the candidate: delta 0, not seed noise
            identical = sorted(m for m, f in control_fps.items() if f and f == tr.get("fingerprint"))
            deltas = {m: 0.0 if m in identical else round(vm.macro_f1 - c, 4)
                      for m, c in controls.items() if c is not None}
            cand = {"round": rnd, "staging": tr.get("path"), "val_macro_f1": vm.macro_f1, "val_gate": val_ok,
                    "val_reasons": why, "control_val_deltas": deltas, "attempts": tr.get("attempts"),
                    "chosen_attempt": tr.get("chosen"), "fingerprint": tr.get("fingerprint"), "real_cap": real_cap,
                    "same_as_controls": identical}
            candidates.append(cand)
            if fresh:
                rec["rounds"].append(cand)
                same = (f" (same training set as the {', '.join(identical)} control: reused, delta 0)"
                        if identical else "")
                self._j(r, "round", f"round {rnd}: {why[0]}; vs controls {deltas or 'n/a'}{same}", head=h, round=rnd,
                        counts={"val_macro_f1": vm.macro_f1, "val_gate": val_ok, "control_deltas": deltas,
                                "same_as_controls": identical})
                r.save()
            if val_ok:
                break
            if rnd < pol.max_rounds:
                plan = balance_plan(st, pol)
                remedies = diagnose(h, vm, threshold=st.gate_threshold, policy=pol,
                                    control_val_macro_f1=controls.get("synthetic"),
                                    real_counts=st.per("real_labelled"), target_shares=plan.shares,
                                    train_total=sum(st.have().values()))
                cap = next((x for x in remedies if x.kind == "cap_real" and x.n), None)
                if cap:
                    real_cap = cap.n
                if fresh:
                    for x in remedies:
                        self._j(r, "remedy", x.reason, head=h, round=rnd,
                                counts={"kind": x.kind, "label": x.label, "focus": x.focus, "n": x.n})
                    if not remedies:
                        self._j(r, "remedy", "no val-evidenced remedy; the next round retrains with the data as it is",
                                head=h, round=rnd)
        passing = [c for c in candidates if c["val_gate"]]
        if not passing:
            rec["outcome"] = "rejected"
            rec["why"] = f"no round passed the val gate in {len(candidates)} rounds; test not read"
            self._j(r, "reject", rec["why"], head=h)
            r.save()
            return
        best = max(passing, key=lambda c: (c["val_macro_f1"], -c["round"]))
        rec["chosen"] = best
        gkey = f"gate:{h}"
        started = r.res.setdefault("gate_started", [])
        if gkey not in r.steps and h in started:
            rec["outcome"] = "rejected"
            rec["why"] = "the test read was interrupted earlier in this run; test is read once per run, not again"
            self._j(r, "reject", rec["why"], head=h)
            r.save()
            return
        if gkey not in r.steps:
            started.append(h)
            r.save()
        gate = await call("gate", gkey, True, head=h, staging=best["staging"])
        if "passed" not in gate or gate.get("test_macro_f1") is None:
            raise ContractError(f"gate for {h} returned no passed / test_macro_f1")
        test_f1 = float(gate["test_macro_f1"])
        reasons = list((gate.get("gate_check") or {}).get("reasons") or [])
        if gate.get("extra_test"):
            ra = gate["extra_test"]
            reasons.append(f"extra test {'passed' if ra.get('passed') else 'regressed'} (macro-F1 {ra.get('macro_f1')})")
        inc = (gate.get("incumbent") or {}).get("macro_f1")
        check = gate.get("check")
        d = decide(bool(gate["passed"]), reasons, test_f1, inc, pol.tol_for(h), check,
                   incumbent_gate_passed=(gate.get("incumbent") or {}).get("gate_passed"),
                   over_failing_incumbent=pol.promote_over_failing_incumbent)
        rec["gate"] = {"test_macro_f1": test_f1, "reference": gate.get("reference"), "check": check,
                       "decision": d.model_dump()}
        if f"decision:{h}" not in r.steps:
            self._j(r, "gate", f"test macro-F1 {test_f1:.3f} (read once); " + "; ".join(d.reasons), head=h,
                    counts={"test_macro_f1": test_f1, "incumbent": inc, "delta": d.delta,
                            "reference": (gate.get("reference") or {}).get("macro_f1"), "cached": gate.get("cached"),
                            "check": _check_counts(check)})
            if d.policy:
                self._j(r, "gate_policy", f"target policy {d.policy} applied: {d.reasons[-1]}", head=h,
                        counts={"policy": d.policy, "test_macro_f1": test_f1, "incumbent": inc})
            r.steps[f"decision:{h}"] = d.model_dump()
            r.save()
        if d.promote:
            fresh = f"promote:{h}" not in r.steps
            out = await call("promote", f"promote:{h}", False, head=h, staging=best["staging"],
                             **({"over_failing_incumbent": True} if d.policy == "promote_over_failing_incumbent"
                                else {}))
            rec["outcome"], rec["path"] = "promoted", out.get("path")
            if fresh:
                self._j(r, "promote", f"promoted to {out.get('path')}"
                        + (f" (incumbent moved to {out['replaced']})" if out.get("replaced") else ""), head=h,
                        counts={"provenance_class": (out.get("provenance") or {}).get("class")})
        else:
            rec["outcome"] = "rejected"
            rec["why"] = "; ".join(d.reasons)
            if f"rejected:{h}" not in r.steps:
                self._j(r, "reject", f"candidate stays in staging ({best['staging']}): {rec['why']}", head=h)
                r.steps[f"rejected:{h}"] = True
        r.save()

    def _generation_plan(self, st: HeadStatus, needs: list[Need], remedies: list[Remedy], pol, r: _Run, h: str,
                         rnd: int) -> list[dict]:
        """This round's generation asks, capped at ``gen_cap`` rows; the rest is journalled as carry. Shape gaps
        steer every ask toward the uncovered key templates and channels."""
        plan = balance_plan(st, pol)
        gap = next((n for n in needs if n.kind == "shape_gap"), None)
        templates = list(st.shape.top_uncovered)[:5] if gap and st.shape else []
        channels = list(st.shape.channel_gaps) if gap and st.shape else []
        asks = [{"label": l, "n": n, "why": "balance"} for l, n in sorted(plan.allocation.items()) if n > 0]
        if not asks and gap:
            per = max(1, min(50, pol.gen_cap // max(1, len(st.labels))))
            asks = [{"label": l, "n": per, "why": "shape_gap"} for l in st.labels]
        for x in remedies:
            if x.kind == "hard_negatives":
                asks.append({"label": x.label, "n": x.n, "focus": x.focus, "why": "hard_negatives"})
            elif x.kind == "shaped_generation":
                asks.append({"label": x.label, "n": x.n, "why": "weak_label"})
        asks = self._generatable(st, asks, r, h, rnd)
        # only what a later round could generate is carried (a group no label fills was journalled as skipped)
        carried = {g: n for g, n in plan.carry.items() if st.generate is not None and st.generate.label_for(g)[0]}
        out, left = [], pol.gen_cap
        for a in asks:
            n = min(a["n"], left)
            if n < a["n"]:
                carried[a["label"]] = carried.get(a["label"], 0) + a["n"] - n
            if n > 0:
                out.append({**a, "n": n, "templates": templates, "channels": channels})
                left -= n
        carried = {k: v for k, v in carried.items() if v > 0}
        key = f"carry:{h}:{rnd}"
        if carried and key not in r.steps:
            self._j(r, "carry", f"{sum(carried.values())} rows over the {pol.gen_cap}-row round cap carried to the "
                    "next round or run", head=h, round=rnd, counts=carried)
            r.steps[key] = carried
            r.res.setdefault("heads", {}).setdefault(h, {})["carry"] = carried
        return out

    def _generatable(self, st: HeadStatus, asks: list[dict], r: _Run, h: str, rnd: int) -> list[dict]:
        """Map each ask's balance group to the generate label ``inspect`` reported for it; asks for a group no
        label fills, or for a head ``generate`` cannot serve, are dropped and journalled once per round
        (``skip``), never sent to the adapter to be refused."""
        if not asks:
            return []
        if st.generate is None:
            raise ContractError(f"adapter inspect reported no generate block for {h} (contract: HeadStatus)")
        out: list[dict] = []
        skipped: dict[str, dict] = {}
        for a in asks:
            label, why = st.generate.label_for(a["label"])
            if label is None:
                s = skipped.setdefault(a["label"], {"n": 0, "why": why})
                s["n"] += a["n"]
            else:
                out.append({**a, "label": label, "group": a["label"]})
        key = f"nogen:{h}:{rnd}"
        if skipped and key not in r.steps:
            if not st.generate.available:
                reason = f"no generation for this head: {st.generate.reason}"
            else:
                reason = "no generate label for " + "; ".join(f"{g} ({v['why']})" for g, v in sorted(skipped.items()))
            self._j(r, "skip", reason + f" ({sum(v['n'] for v in skipped.values())} rows not asked)", head=h,
                    round=rnd, counts={g: v["n"] for g, v in sorted(skipped.items())})
            r.steps[key] = {g: v["n"] for g, v in skipped.items()}
            r.save()
        return out

    # ---- Forge projects ------------------------------------------------------------------------------
    async def _project(self, r: _Run, t: Target) -> None:
        cfg = t.project
        assert cfg is not None
        if self.orch is None or cfg.project_id not in self.orch.rt:
            raise RuntimeError(f"no Forge project {cfg.project_id!r}")
        st = await asyncio.to_thread(project_status, self.orch, cfg.project_id)
        state = self.orch.rt[cfg.project_id].machine.state
        if "plan" not in r.res:
            from forge.orchestrator import PIPELINE_STAGES
            r.res["plan"] = {"heads": [st.head], "phases": list(PIPELINE_STAGES), "started_at": now()}
        needs = detect(st, cfg.policy, schedule_due=r.run["trigger"] == "schedule", fair_baselines=False)
        if "needs" not in r.steps:
            for n in needs:
                self._j(r, "need", n.reason, head=st.head, counts={"kind": n.kind, "remedy": n.remedy, **n.counts})
            r.steps["needs"] = [n.model_dump() for n in needs]
        r.res["heads"] = {st.head: {"outcome": f"inspected ({state})", "needs": [n.kind for n in needs],
                                    "started": r.res["plan"]["started_at"], "ended": now()}}
        self._j(r, "project", f"project {cfg.project_id} is {state}; its pipeline acts on these needs "
                "(labelling is automatic)", head=st.head, counts={"state": state, "needs": len(needs)})
        r.save()


def project_status(orch, pid: str) -> HeadStatus:
    """A Forge project's dataset and labelling queue in the detectors' shape."""
    from forge.models import REAL_SHAPED, REAL_SOURCES
    from forge.store import read_dataset
    ts = orch.task_schema(pid)
    if ts is None:
        raise RuntimeError(f"project {pid} has no accepted task yet")
    counts: dict[str, dict[str, int]] = {l: {"synthetic": 0, "shaped": 0, "real_labelled": 0} for l in ts.labels()}
    for e in read_dataset(pid):
        if e.split != "train" or e.label not in counts:
            continue
        src = ("real_labelled" if e.source in REAL_SOURCES else "shaped" if e.provenance == REAL_SHAPED
               else "synthetic")
        counts[e.label][src] += 1
    return HeadStatus(head=ts.name, counts=counts, undisposed=orch.label_queue(pid).counts()["needs_label"])


def _one(out: dict, head: str) -> HeadStatus:
    got = [s for s in out.get("heads") or [] if s.get("head") == head]
    if len(got) != 1:
        raise ContractError(f"inspect --heads {head} returned {len(got)} statuses for it")
    return HeadStatus.model_validate(got[0])


def _fail(msg: str):
    raise ContractError(msg)


__all__ = ["PHASES", "ContractError", "Runner", "WriterGuard", "WriterGuardError", "default_adapter", "project_status"]
