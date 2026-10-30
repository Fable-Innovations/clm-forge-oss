"""Drives projects through the rules layer.

- Every transition goes through `Orchestrator.fire`, which refreshes the guard facts, asks the
  machine (`rules.Project.fire`), persists, and emits a `state` (or `rejected`) event.
- Entering a state launches its job: deterministic stage work (curation, training, evaluation,
  packaging — CPU/GPU heavy work runs in worker threads) or an agent decision point.
- Agent decision points (`analyzing`, `reviewing`, and `proposed` after a chat revision to no_fit)
  are a LangGraph `decide -> act -> route` loop: the LLM proposes one of the machine's currently
  allowed *agent* events with a required reason; the machine decides; rejections are logged as
  `rejected` events, never coerced. Bounded by "state changed / no agent moves" and recursion_limit.
- Human events (accept, revise, retry, cancel, review_done, label_override) come from the API.
- `labelling` is a system stage between curation and base-model selection: its job applies recorded
  `label_override`s, then the labelling agent (`forge.labelling.agent_label`, the default `agent` policy)
  gives every real row a final disposition and fires `labels_settled`. Under an opt-in human policy
  (`user` | `teacher` | `mixed`) rows that remain undisposed send the project to the `label_review` gate
  (`needs_review`); the human labels them there (or asks the teacher, `label_with_teacher`) and fires
  `review_done`, which re-enters `labelling` to settle.
- Every stage that loads a model or drives the teacher waits for its turn in the global model-job queue
  (`forge.jobqueue.JobQueue`, concurrency 1, preflight before each job). A server restart re-queues an
  interrupted stage instead of failing it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from forge import analyst, config
from forge.events import EventBus
from forge.jobqueue import JobQueue
from forge.labelling import DEFAULT_POLICY, POLICIES, LabelError, LabelQueue, agent_label
from forge.llm import get_llm
from forge.llm.base import LLM, short
from forge.models import EvalReport, Opportunity, Proposal, TaskSchema
from forge.rules import FINAL_STATES, STAY_EVENTS, Facts, Project, Rejected, gate_check
from forge.store import Store, dataset_counts, now, read_dataset, write_dataset

# Stages with a running job: a restart re-enters them (the job runs again from its start). `proposed` and
# `label_review` are human gates; a restart leaves them exactly as they were.
ACTIVE_STAGES = ("analyzing", "accepted", "curating", "labelling", "selecting", "training", "evaluating",
                 "reviewing", "packaging")
# The pipeline's working stages in order (an autopilot project run's plan, forge.autopilot.runner).
PIPELINE_STAGES = ("curating", "labelling", "selecting", "training", "evaluating", "reviewing", "packaging")
# Dataset and labelling-queue edits are refused while a job reads or writes the dataset (record a
# `label_override` instead: it is applied by the next labelling stage).
DATASET_LOCKED = ("curating", "labelling", "selecting", "training", "evaluating")
# Stages whose whole job loads a model or drives the teacher: they run inside a model-job queue slot.
# (`labelling` takes a slot itself, only when the teacher has rows to answer.)
QUEUED_STAGES = ("curating", "selecting", "training", "evaluating")


class Decision(BaseModel):
    """The only thing the model may say at a decision point."""

    event: str = Field(description="exactly one of the allowed events")
    reason: str = Field(description="why, citing the evidence; this is the audit trail")
    guidance: str | None = Field(default=None, description="for recurate only: what the new data should fix")


class ChatReply(BaseModel):
    reply: str = Field(description="your answer to the user, markdown allowed")
    revised_proposal: Proposal | None = Field(
        default=None, description="only when the user asked for a change to the proposal; else null")


class S(TypedDict, total=False):
    """LangGraph state for one decision point."""

    decision: Decision | None
    rejections: list[str]


class ProjectBusy(RuntimeError):
    """A change that must not land under a running stage job (e.g. the project's encoder)."""


class NoEncoder(RuntimeError):
    """An experiment with no encoder (created before per-experiment encoders, and no migration source had one)."""


class LabelJobRunning(RuntimeError):
    """A teacher labelling job conflicts with the request (already running, or outside the review gate)."""


@dataclass
class Runtime:
    rec: dict
    machine: Project
    job: asyncio.Task | None = None
    gen: int = 0
    stop: threading.Event = field(default_factory=threading.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    chat_task: asyncio.Task | None = None
    label_job: asyncio.Task | None = None     # human-triggered teacher labelling at label_review (no state change)


class Orchestrator:
    def __init__(self, store: Store | None = None, bus: EventBus | None = None, jobs: JobQueue | None = None):
        config.ensure_dirs()
        self.store = store or Store()
        self.bus = bus or EventBus(self.store)
        self.jobs = jobs or JobQueue(self.store)
        self.rt: dict[str, Runtime] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self._llm_override: LLM | None = None  # tests

    # ---- plumbing --------------------------------------------------------------------------------
    def emit(self, pid: str, ev: dict) -> dict:
        return self.bus.emit(pid, ev)

    def log(self, pid: str, stage: str, message: str, level: str = "info") -> None:
        self.emit(pid, {"type": "log", "stage": stage, "level": level, "message": message})

    def settings(self) -> config.Settings:
        return config.load_settings()

    def llm_for(self, pid: str) -> LLM:
        if self._llm_override is not None:
            return self._llm_override
        rec = self.rt[pid].rec
        return get_llm(self.settings(), provider=rec.get("provider"), model=rec.get("model"))

    def pdir(self, pid: str) -> Path:
        p = config.project_dir(pid)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, pid: str) -> None:
        rt = self.rt[pid]
        rt.rec["machine"] = rt.machine.snapshot()
        saved = self.store.save_project(rt.rec)
        rt.rec["created_at"], rt.rec["updated_at"] = saved["created_at"], saved["updated_at"]

    def proposal(self, pid: str) -> Proposal | None:
        p = self.rt[pid].rec.get("proposal")
        return Proposal.model_validate(p) if p else None

    def task_schema(self, pid: str) -> TaskSchema | None:
        t = self.rt[pid].rec.get("task_schema")
        return TaskSchema.model_validate(t) if t else None

    def accepted_opportunity(self, pid: str) -> Opportunity | None:
        p, oid = self.proposal(pid), self.rt[pid].rec.get("accepted_opportunity_id")
        return analyst.opportunity(p, oid) if p and oid else None

    def report(self, pid: str) -> EvalReport | None:
        f = self.pdir(pid) / "eval_report.json"
        return EvalReport.model_validate_json(f.read_text()) if f.exists() else None

    def manifest(self, pid: str) -> dict | None:
        f = self.pdir(pid) / "manifest.json"
        return json.loads(f.read_text()) if f.exists() else None

    def label_policy(self, pid: str) -> str:
        """The project's label policy; a project that never chose one is run by the labelling agent."""
        return self.rt[pid].rec.get("label_policy") or DEFAULT_POLICY

    def label_queue(self, pid: str) -> LabelQueue:
        """The project's labelling queue; labelled rows are upserted into its dataset."""
        return LabelQueue(self.pdir(pid), self.task_schema(pid), dataset_sync=True)

    def facts(self, pid: str) -> Facts:
        rec, s = self.rt[pid].rec, self.settings()
        p = self.proposal(pid)
        ts = self.task_schema(pid)
        rows = read_dataset(pid)
        rep = self.report(pid)
        return Facts(
            proposal_verdict=p.verdict if p else None,
            opportunity_ids=[o.id for o in p.opportunities] if p else [],
            accepted_opportunity_id=rec.get("accepted_opportunity_id"),
            dataset_size=len(rows),
            labels=ts.labels() if ts else [],
            train_labels=sorted({r.label for r in rows if r.split == "train"}),
            min_examples=s.min_examples,
            undisposed_rows=(lc := self.label_queue(pid).counts())["needs_label"],
            excluded_by_agent=lc["excluded_by_agent"],
            label_policy=self.label_policy(pid),
            base_model=rec.get("base_model"),
            base_model_candidates=self._candidates(pid),
            checkpoint_exists=(self.pdir(pid) / "best_head.pt").exists(),
            report=rep.model_dump() if rep else None,
            manifest_exists=self.manifest(pid) is not None,
            attempts=int(rec.get("attempts", 0)),
            max_attempts=s.max_attempts,
        )

    def _encoder_view(self, pid: str) -> dict | None:
        """The experiment's encoder for the UI: registry key, name and spec (None only for an experiment the
        per-experiment encoder migration could not give one: it must be picked before training)."""
        from forge import encoders
        spec = self.rt[pid].rec.get("base_model")
        if not spec:
            return None
        e = encoders.by_spec(spec)
        return {"key": e.key if e else None, "name": encoders.name_of_spec(spec), "spec": spec,
                "source": self.rt[pid].rec.get("encoder_source")}

    def encoder_recommendations(self, pid: str) -> dict[str, dict]:
        """Forge's encoder recommendation for each opportunity of the proposal (``forge.encoder_choice``: the task,
        the encoder sizes and this machine's model budget): ``{opportunity id: {key, name, reason, profile}}``."""
        from forge.encoder_choice import recommend
        p = self.proposal(pid)
        if p is None:
            return {}
        out = {}
        for o in p.opportunities:
            r = recommend(o.task)
            out[o.id] = {"key": r.key, "name": r.name, "reason": r.reason, "profile": r.profile}
        return out

    def _candidates(self, pid: str, strict: bool = False) -> list[str]:
        """Encoder specs base-model selection compares: the experiment's own encoder. ``strict`` (the selecting
        job) refuses an experiment without one (``NoEncoder``); the rules' facts just get no candidate."""
        spec = self.rt[pid].rec.get("base_model")
        if spec:
            return [spec]
        if strict:
            raise NoEncoder(f"experiment {pid} has no encoder; pick one in its header")
        return []

    def encoder_spec(self, pid: str) -> str:
        """The spec this experiment embeds with: its own ``base_model`` (``NoEncoder`` without one)."""
        spec = self.rt[pid].rec.get("base_model")
        if not spec:
            raise NoEncoder(f"experiment {pid} has no encoder; pick one in its header")
        return spec

    # ---- the single door to the rules layer ------------------------------------------------------
    def fire(self, pid: str, event: str, actor: str, reason: str = "", **kw) -> tuple[str, str]:
        rt = self.rt[pid]
        rt.machine.facts = self.facts(pid)
        try:
            src, dst = rt.machine.fire(event, actor, reason, **kw)
        except Rejected as e:
            self.emit(pid, {"type": "rejected", "event": event, "actor": actor, "reason": reason, "error": str(e)})
            raise
        self.save(pid)
        self.emit(pid, {"type": "state", "from": src, "event": event, "to": dst, "actor": actor, "reason": reason})
        if event not in STAY_EVENTS:   # a recorded decision (label_override) starts no job
            self._enter(pid, dst)
        return src, dst

    def _enter(self, pid: str, state: str) -> None:
        rt = self.rt[pid]
        rt.gen += 1
        if state in ("cancelled", "errored") or state in FINAL_STATES:
            rt.stop.set()
            if rt.job and not rt.job.done() and rt.job is not asyncio.current_task():
                rt.job.cancel()
            self._cancel_label_job(rt)
            return
        if state != "label_review":
            # the teacher job writes the queue and the dataset; it belongs to the review gate only
            self._cancel_label_job(rt)
        rt.stop = threading.Event()
        handler = {
            "analyzing": self._job_analyze, "accepted": self._job_accepted, "curating": self._job_curate,
            "labelling": self._job_label, "selecting": self._job_select, "training": self._job_train, "evaluating": self._job_evaluate, "reviewing": self._job_review,
            "packaging": self._job_package,
        }.get(state)
        if handler is None:
            return
        gen = rt.gen
        rt.job = asyncio.get_event_loop().create_task(self._guard_job(pid, state, gen, handler))

    async def _guard_job(self, pid: str, state: str, gen: int, handler) -> None:
        """Run a stage job (inside a model-job queue slot for QUEUED_STAGES); any exception fails the project
        (unless the job is stale)."""
        try:
            if state in QUEUED_STAGES:
                async with self.jobs.slot(state, f"project:{pid}", on_defer=self._on_defer(pid, state)):
                    if not self._current(pid, gen):
                        return
                    await handler(pid, gen)
            else:
                await handler(pid, gen)
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001
            if not self._current(pid, gen):
                return
            self.emit(pid, {"type": "error", "stage": state, "message": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-6000:]})
            self.rt[pid].rec["error"] = f"{state}: {type(e).__name__}: {e}"
            try:
                self.fire(pid, "fail", "system", f"{state} failed: {short(str(e), 300)}")
            except Rejected:
                pass

    def _on_defer(self, pid: str, stage: str):
        def on_defer(reason: str) -> None:
            self.log(pid, stage, f"model job deferred: {reason}", "warn")
        return on_defer

    @staticmethod
    def _cancel_label_job(rt: Runtime) -> None:
        job = rt.label_job
        if job and not job.done() and job is not asyncio.current_task():
            job.cancel()

    def _current(self, pid: str, gen: int) -> bool:
        rt = self.rt.get(pid)
        return rt is not None and rt.gen == gen

    def _complete(self, pid: str, gen: int, event: str, reason: str) -> None:
        """Fire a system completion event unless the job went stale (cancelled / superseded)."""
        if not self._current(pid, gen):
            return
        try:
            self.fire(pid, event, "system", reason)
        except Rejected as e:
            # A completion that fails its guard is a stage failure, made visible.
            self.rt[pid].rec["error"] = str(e)
            self.emit(pid, {"type": "error", "stage": self.rt[pid].machine.state, "message": str(e), "traceback": ""})
            self.fire(pid, "fail", "system", str(e))

    def _thread_emit(self, pid: str):
        def emit(ev: dict) -> None:
            self.bus.emit(pid, ev)
        return emit

    # ---- projects --------------------------------------------------------------------------------
    async def create(self, *, name: str | None, repo: str | None, task: str | None,
                     provider: str | None, model: str | None, encoder: str) -> dict:
        """A new experiment on registry encoder ``encoder`` (required: there is no app-wide default; the create
        dialog preselects ``forge.encoder_choice.recommend``, the proposal recommends one per opportunity)."""
        from forge import encoders
        if not (repo or task):
            raise ValueError("give a repo (https git URL or absolute path) or a task description")
        if not encoders.known(encoder):
            raise ValueError(f"encoder: {encoder or 'no encoder'} is not a registered encoder; pick one")
        s = self.settings()
        provider = provider or s.provider
        model = model or (s.model if provider == s.provider else None)
        if not model:
            from forge.llm import catalog
            model = await catalog.default_model(provider, s)
        if not model and not provider.endswith("-subscription"):
            raise ValueError(f"no {provider} models available; check Settings")
        model = model or "default"
        pid = uuid.uuid4().hex[:10]
        default_name = (repo.rstrip("/").split("/")[-1].removesuffix(".git") if repo else (task or "")[:40]) or pid
        rec = {"id": pid, "name": name or default_name, "repo": repo, "repo_path": None, "task": task,
               "provider": provider, "model": model, "attempts": 0, "feedback": [], "guidance": None,
               "error": None, "base_model": encoders.get(encoder).spec,
               "encoder_source": {"from": "create", "at": now()}}
        m = Project(name=rec["name"])
        self.rt[pid] = Runtime(rec=rec, machine=m)
        self.save(pid)
        self.emit(pid, {"type": "log", "stage": "created", "level": "info",
                        "message": f"project created ({provider} / {model}; encoder "
                                   f"{encoders.get(encoder).name})"})
        self.fire(pid, "start", "system", "project created; starting analysis")
        return self.summary(pid)

    def delete(self, pid: str) -> None:
        rt = self.rt.pop(pid, None)
        if rt:
            rt.stop.set()
            rt.gen += 1
            if rt.job and not rt.job.done():
                rt.job.cancel()
        self.store.delete_project(pid)
        self.bus.forget(pid)
        import shutil
        shutil.rmtree(config.project_dir(pid), ignore_errors=True)

    def restore_all(self) -> None:
        """Load every project; a stage interrupted by the restart is queued again from its start (a project
        still in `created` starts its analysis). Must run on the server's event loop."""
        for rec in self.store.list_projects():
            pid = rec["id"]
            if pid in self.rt:
                continue
            m = Project.restore(rec.get("machine", {}), name=rec.get("name", ""))
            self.rt[pid] = Runtime(rec=rec, machine=m)
            if m.state == "created":
                self.log(pid, "created", "server restarted before analysis began; starting it now", "warn")
                self.fire(pid, "start", "system", "server restarted; starting analysis")
            elif m.state in ACTIVE_STAGES:
                self.log(pid, m.state, f"server restarted while {m.state}; the stage is queued again", "warn")
                m.facts = self.facts(pid)
                self._enter(pid, m.state)

    def summary(self, pid: str) -> dict:
        rt = self.rt[pid]
        rec = rt.rec
        p = self.proposal(pid)
        metric = None
        rep = self.report(pid) if rt.machine.state in ("reviewing", "packaging", "delivered", "no_fit", "errored",
                                                        "cancelled", "curating", "selecting", "training", "evaluating") else None
        if rep:
            ft = next((m for m in rep.models if m.name == "finetuned"), None)
            if ft:
                metric = {"name": rep.metric, "value": getattr(ft, rep.metric, ft.accuracy),
                          "threshold": rep.threshold, "gate_passed": gate_check(rep.model_dump())[0]}
        return {"id": pid, "name": rec["name"], "state": rt.machine.state, "created_at": rec.get("created_at"),
                "updated_at": rec.get("updated_at"), "repo": rec.get("repo"), "task": rec.get("task"),
                "provider": rec.get("provider"), "model": rec.get("model"),
                "verdict": p.verdict if p else None, "metric": metric}

    def detail(self, pid: str) -> dict:
        rt = self.rt[pid]
        d = self.summary(pid)
        d.update({
            "path": [list(x) for x in rt.machine.path],
            "log": rt.machine.log,
            "allowed_human_events": rt.machine.allowed_for("human"),
            "proposal": rt.rec.get("proposal"),
            "accepted_opportunity_id": rt.rec.get("accepted_opportunity_id"),
            "task_schema": rt.rec.get("task_schema"),
            "base_model": rt.rec.get("base_model"),
            "encoder": self._encoder_view(pid),
            "encoder_recommendations": self.encoder_recommendations(pid),
            "base_model_selection": rt.rec.get("base_model_selection"),
            "attempts": rt.rec.get("attempts", 0),
            "label_policy": self.label_policy(pid),
            "label_overrides": rt.rec.get("label_overrides") or [],
            "labels": self.label_queue(pid).counts(),
            "max_attempts": self.settings().max_attempts,
            "error": rt.rec.get("error") if rt.machine.state == "errored" else None,
            "failed_from": rt.machine.failed_from if rt.machine.state == "errored" else None,
            "chat": self.store.chat(pid),
        })
        return d

    # ---- human events ----------------------------------------------------------------------------
    def accept(self, pid: str, opportunity_id: str, overrides: dict | None = None) -> None:
        rt = self.rt[pid]
        p = self.proposal(pid)
        opp = analyst.opportunity(p, opportunity_id) if p else None
        overrides = overrides or {}
        if opp is not None:
            task = TaskSchema.model_validate(overrides["task"]) if overrides.get("task") else opp.task
            s = self.settings()
            rt.rec.update({
                "accepted_opportunity_id": opportunity_id,
                "task_schema": task.model_dump(),
                "target_examples": max(40, min(s.max_target_examples,
                                               int(overrides.get("target_examples") or opp.target_examples))),
                "success_threshold": float(overrides.get("success_threshold") or opp.success_threshold),
                "success_metric": overrides.get("success_metric") or opp.success_metric,
            })
            if overrides.get("label_policy") is not None:
                rt.rec["label_policy"] = self._check_policy(overrides["label_policy"])
        self.fire(pid, "accept", "human", f"accepted {opportunity_id}", opportunity_id=opportunity_id)

    def revise(self, pid: str, feedback: str) -> None:
        rt = self.rt[pid]
        fb = rt.rec.setdefault("feedback", [])
        fb.append(feedback)
        try:
            self.fire(pid, "revise", "human", short(feedback, 300), feedback=feedback)
        except Rejected:
            fb.pop()
            raise

    @staticmethod
    def _check_policy(policy: str) -> str:
        if policy not in POLICIES:
            raise ValueError(f"label policy must be one of {', '.join(POLICIES)}")
        return policy

    def set_label_policy(self, pid: str, policy: str) -> None:
        """Who labels real rows from the next curation / labelling stage on (agent | user | teacher | mixed)."""
        rt = self.rt[pid]
        rt.rec["label_policy"] = self._check_policy(policy)
        self.save(pid)
        self.log(pid, rt.machine.state, f"label policy set to {policy}")

    def set_encoder(self, pid: str, key: str) -> tuple[str | None, str]:
        """Train this experiment on registry encoder ``key`` from its next selecting / training stage on: sets
        ``base_model`` and emits ``encoder_changed {from_name, to_name}``. Refused (``ProjectBusy``) while a stage job or the teacher runs.
        -> (old spec, new spec)."""
        from forge import encoders
        enc = encoders.get(key)
        rt = self.rt[pid]
        busy = rt.machine.state in ACTIVE_STAGES or (rt.label_job is not None and not rt.label_job.done())
        if busy:
            raise ProjectBusy(f"the project is {rt.machine.state}; change its encoder once that job finishes")
        old = rt.rec.get("base_model")
        rt.rec["base_model"] = enc.spec
        rt.rec["encoder_source"] = {"from": "user", "at": now()}
        self.save(pid)
        if old != enc.spec:
            self.emit(pid, {"type": "encoder_changed", "from_name": encoders.name_of_spec(old), "to_name": enc.name,
                            "from": (encoders.by_spec(old).key if encoders.by_spec(old) else old), "to": enc.key})
        return old, enc.spec

    def review_done(self, pid: str, mode: str) -> None:
        """The human leaves the label_review gate (opt-in policies only): `continue` once every real row is
        labelled, or `exclude` to set the rows still pending aside (they stay queued and join the dataset when
        labelled). Re-enters `labelling`, whose job settles the labels."""
        if mode not in ("continue", "exclude"):
            raise ValueError("mode must be continue or exclude")
        rt = self.rt[pid]
        if rt.label_job and not rt.label_job.done():
            # leaving now would start jobs that read the dataset while the teacher still writes it
            raise LabelJobRunning("the teacher is still labelling this queue; wait for it to finish")
        q = self.label_queue(pid)
        n = q.counts()["needs_label"]
        reason = ("every real row is labelled" if not n else
                  f"continuing without {n} unlabelled real rows (excluded until labelled)")
        if n and mode == "exclude" and rt.machine.state == "label_review":
            q.exclude_pending()    # before the transition: the labelling job it starts must see them set aside
        self.fire(pid, "review_done", "human", reason, mode=mode)
        self.emit(pid, {"type": "labels", "counts": q.counts()})

    def label_override(self, pid: str, rid: str, label: str, via: dict | None = None) -> dict:
        """Record a user label for a real row in any non-terminal state; the next labelling stage applies it
        (a user label always wins over the agent). -> the pending overrides."""
        q = self.label_queue(pid)
        row = next((r for r in q.rows() if r.rid == rid), None)
        if row is None:
            raise LabelError(f"no row {rid!r} in this queue")
        keys = [o["key"] for o in q.options(row)]
        if keys and label not in keys:
            raise LabelError(f"label must be one of {keys}")
        rec = self.rt[pid].rec
        self.fire(pid, "label_override", "human", f"user label {label!r} for {rid} (applied at the next "
                                                  "labelling stage)", rid=rid, label=label)
        pending = [o for o in rec.get("label_overrides") or [] if o["rid"] != rid] + [
            {"rid": rid, "label": label, **({"via": via} if via else {})}]
        rec["label_overrides"] = pending
        self.save(pid)
        return {"pending": pending}

    def label_with_teacher(self, pid: str, mode: str = "label") -> int:
        """Override helper at the label_review gate: the project's teacher LLM labels (or, mode=suggest,
        suggests labels for) every pending real row. Runs as a background job in a model-job queue slot; the
        result goes into the queue, never into the state."""
        rt = self.rt[pid]
        if rt.machine.state != "label_review":
            # outside the gate a stage job may be reading or rewriting the dataset
            raise LabelJobRunning(f"send rows to the teacher at the label review gate, not while "
                                  f"{rt.machine.state}")
        if rt.label_job and not rt.label_job.done():
            raise LabelJobRunning("the teacher is already labelling this queue")
        if mode not in ("label", "suggest"):
            raise ValueError("mode must be label or suggest")
        task = self.task_schema(pid)
        if task is None:
            raise ValueError("no accepted task yet")
        pending = self.label_queue(pid).pending()
        if not pending:
            return 0

        async def job():
            from forge.curator import label_states
            state = rt.machine.state
            self.log(pid, state, f"teacher {'labelling' if mode == 'label' else 'suggesting labels for'} "
                                 f"{len(pending)} real rows")
            try:
                async with self.jobs.slot("teacher", f"project:{pid}", on_defer=self._on_defer(pid, state)):
                    got = await label_states(self.llm_for(pid), task, [r.state for r in pending],
                                             sem=asyncio.Semaphore(max(1, self.settings().llm_concurrency)))
                if rt.machine.state != "label_review":
                    # the gate was left while the teacher ran: its labels must not reach a locked dataset
                    self.log(pid, rt.machine.state, "teacher result dropped: the label review gate was left")
                    return
                q = self.label_queue(pid)
                stats = await asyncio.to_thread(q.apply_teacher, [
                    {"rid": r.rid, "label": lab, "conf": conf} for r, (lab, conf, _why) in zip(pending, got)], mode)
                self.log(pid, state, f"teacher result: {stats}")
                self.emit(pid, {"type": "labels", "counts": q.counts()})
            except Exception as e:  # noqa: BLE001 - a failed teacher job leaves the rows pending, visibly
                self.emit(pid, {"type": "error", "stage": state, "message": f"teacher labelling failed: "
                                f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]})

        rt.label_job = asyncio.get_event_loop().create_task(job())
        return len(pending)

    def retry(self, pid: str) -> None:
        self.rt[pid].rec["error"] = None
        self.fire(pid, "retry", "human", "retry requested")

    def cancel(self, pid: str) -> None:
        self.fire(pid, "cancel", "human", "cancelled by user")

    # ---- agent decision loop (LangGraph decide -> act -> route) --------------------------------
    async def decide_loop(self, pid: str, gen: int, context: str, rules_text: str) -> None:
        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, START, StateGraph

        rt = self.rt[pid]
        origin = rt.machine.state
        llm = self.llm_for(pid)

        async def decide(st: S) -> dict:
            allowed = rt.machine.allowed_for("agent")
            rej = st.get("rejections") or []
            prompt = (f"Project state: {rt.machine.state}\n\n{context}\n\nRules:\n{rules_text}\n\n"
                      f"You may choose exactly one of these events: {allowed}\n"
                      + (f"\nEarlier proposals were REJECTED by the rules layer:\n" + "\n".join(rej) if rej else "")
                      + "\nReturn {event, reason, guidance?}.")
            d = await llm.structured(prompt, Decision,
                                     system="You are the decision step of CLM Forge. You propose; deterministic rules decide.")
            self.log(pid, origin, f"agent proposes {d.event}: {d.reason}")
            return {"decision": d}

        async def act(st: S) -> dict:
            d = st["decision"]
            if not self._current(pid, gen):
                return {}
            if d.event == "recurate":
                rt.rec["guidance"] = d.guidance or d.reason
            try:
                self.fire(pid, d.event, "agent", d.reason)
            except Rejected as e:
                return {"rejections": (st.get("rejections") or []) + [str(e)]}
            return {}

        def route(st: S) -> str:
            if not self._current(pid, gen) or rt.machine.state != origin or not rt.machine.allowed_for("agent"):
                return END
            return "decide"

        g = StateGraph(S)
        g.add_node("decide", decide)
        g.add_node("act", act)
        g.add_edge(START, "decide")
        g.add_edge("decide", "act")
        g.add_conditional_edges("act", route, ["decide", END])
        graph = g.compile()
        try:
            async for _step in graph.astream({"decision": None, "rejections": []}, config={"recursion_limit": 8}):
                pass
        except GraphRecursionError:
            if self._current(pid, gen) and rt.machine.state == origin:
                raise RuntimeError("agent did not reach a legal decision within the step limit") from None

    # ---- stage jobs ------------------------------------------------------------------------------
    async def _emit_async(self, pid: str, ev: dict) -> None:
        self.emit(pid, ev)

    async def _job_analyze(self, pid: str, gen: int) -> None:
        rt = self.rt[pid]
        rec = rt.rec
        llm = self.llm_for(pid)
        repo_path = None
        if rec.get("repo"):
            from forge import repo as repo_mod
            self.log(pid, "analyzing", f"resolving repository {rec['repo']}")
            repo_path = str(await asyncio.to_thread(repo_mod.resolve, rec["repo"]))
            rec["repo_path"] = repo_path
            self.save(pid)
        previous = self.proposal(pid) if rec.get("feedback") else None
        self.log(pid, "analyzing", f"analysing with {llm.provider}/{llm.model}"
                 + (f" (revision; feedback: {short(rec['feedback'][-1], 200)})" if previous else ""))

        def on_event(ev: dict) -> None:
            if self._current(pid, gen):
                self.emit(pid, ev)

        p = await analyst.analyze(llm, repo_path, rec.get("task"), on_event, previous=previous,
                                  feedback=rec.get("feedback") if previous else None)
        if not self._current(pid, gen):
            return
        rec["proposal"] = p.model_dump()
        self.save(pid)
        self.emit(pid, {"type": "proposal", "proposal": p.model_dump()})
        await self.decide_loop(pid, gen, self._proposal_context(p), ANALYZE_RULES)

    def _proposal_context(self, p: Proposal) -> str:
        opps = "\n".join(f"- {o.id}: {o.title} (fit {o.fit_score:.2f}; {o.task.type} over "
                         f"{o.task.labels()}; {len(o.files)} file refs)" for o in p.opportunities)
        return (f"The analyst returned a proposal with verdict={p.verdict!r}.\nSummary: {p.summary}\n"
                f"Opportunities:\n{opps or '(none)'}\nNo-fit reasons: {p.no_fit_reasons or '(none)'}")

    async def _job_accepted(self, pid: str, gen: int) -> None:
        ts = self.task_schema(pid)
        if ts is not None:
            from forge.curator import write_task
            write_task(self.pdir(pid), ts)
        self._complete(pid, gen, "begin_curation", "opportunity accepted; curating training data")

    async def _job_curate(self, pid: str, gen: int) -> None:
        from forge.curator import Curator

        rt = self.rt[pid]
        rec, s = rt.rec, self.settings()
        task = self.task_schema(pid)
        previous = read_dataset(pid) if rec.get("attempts", 0) > 0 else []
        guidance = rec.get("guidance") if previous else None
        cur = Curator(self.llm_for(pid), task, lambda ev: self.emit(pid, ev) if self._current(pid, gen) else None,
                      target=int(rec.get("target_examples") or s.target_examples),
                      repo_path=rec.get("repo_path"), opportunity=self.accepted_opportunity(pid),
                      guidance=guidance, previous=previous, concurrency=s.llm_concurrency,
                      gen_concurrency=s.gen_concurrency, qa_min_pass=s.qa_min_pass,
                      dedup_threshold=s.dedup_cosine, encoder_spec=self.encoder_spec(pid),
                      label_policy=self.label_policy(pid),
                      queue=LabelQueue(self.pdir(pid), task), should_stop=rt.stop.is_set)
        rows = await cur.run()
        if not self._current(pid, gen):
            return
        write_dataset(pid, rows)
        c = dataset_counts(rows)
        pending = cur.stats.get("needs_label", 0)
        self._complete(pid, gen, "curated",
                       f"{len(rows)} rows; by label {c['by_label']}; dropped {cur.stats['disagreements']} "
                       f"teacher disagreements, {cur.stats['exact_dups'] + cur.stats['near_dups']} duplicates"
                       + (f"; {pending} real rows need a label" if pending else ""))

    async def _job_label(self, pid: str, gen: int) -> None:
        """The labelling stage: apply recorded user overrides, then dispose every undisposed real row per the
        label policy, then settle (`labels_settled`) or, under a human policy with rows left, `needs_review`."""
        from forge.curator import label_states

        rt = self.rt[pid]
        task = self.task_schema(pid)
        q = self.label_queue(pid)
        overrides = list(rt.rec.get("label_overrides") or [])
        for o in overrides:
            await asyncio.to_thread(q.set_user_label, o["rid"], o["label"], o.get("via"))
        if overrides:
            rt.rec["label_overrides"] = []
            self.save(pid)
            self.log(pid, "labelling", f"applied {len(overrides)} user label overrides")
        policy = self.label_policy(pid)
        pending = q.pending()
        sem = asyncio.Semaphore(max(1, self.settings().llm_concurrency))
        if pending and policy != "user":
            llm = self.llm_for(pid)
            async with self.jobs.slot("labelling", f"project:{pid}", on_defer=self._on_defer(pid, "labelling")):
                if not self._current(pid, gen):
                    return
                if policy == "agent":
                    self.log(pid, "labelling", f"labelling agent: {len(pending)} real rows")

                    async def teacher(rows):
                        got = await label_states(llm, task, [r.state for r in rows], sem=sem)
                        # (None, 0, "") is a failed call (asked again); (None, conf, why) an answer outside
                        # the options (the agent excludes the row)
                        return [[] if lab is None and not why else [{"pass": "a", "label": lab, "conf": conf,
                                                                     "reason": why}]
                                for lab, conf, why in got]
                    stats = await agent_label(q, teacher, None)
                else:
                    mode = "label" if policy == "teacher" else "suggest"
                    todo = [r for r in pending if mode == "label" or r.suggested_label is None]
                    got = await label_states(llm, task, [r.state for r in todo], sem=sem)
                    stats = await asyncio.to_thread(q.apply_teacher, [
                        {"rid": r.rid, "label": lab, "conf": conf} for r, (lab, conf, _w) in zip(todo, got)], mode)
            if not self._current(pid, gen):
                return
            self.log(pid, "labelling", f"{policy} labelling result: {stats}")
        counts = q.counts()
        self.emit(pid, {"type": "labels", "counts": counts})
        summary = (f"agent {counts['agent']} (adjudicated {counts['adjudicated']}), teacher {counts['teacher']}, "
                   f"user {counts['user']}, export {counts['export']}, excluded by agent "
                   f"{counts['excluded_by_agent']}, set aside {counts['excluded']}")
        if counts["needs_label"] and policy != "agent":
            self._complete(pid, gen, "needs_review", f"{counts['needs_label']} real rows wait for your review "
                                                     f"(policy {policy}); {summary}")
        else:
            self._complete(pid, gen, "labels_settled", f"every real row has a disposition: {summary}")

    async def _job_select(self, pid: str, gen: int) -> None:
        from forge.base_models import display_name, select_base_model

        rt = self.rt[pid]
        task, rows = self.task_schema(pid), read_dataset(pid)
        candidates = self._candidates(pid, strict=True)
        res = await asyncio.to_thread(select_base_model, task, rows, self.pdir(pid), candidates,
                                      self._thread_emit(pid), rt.stop.is_set)
        if not self._current(pid, gen):
            return
        rt.rec["base_model"] = res["chosen"]
        rt.rec["base_model_selection"] = res
        self.save(pid)
        done = [c for c in res["candidates"] if c.get("val_acc") is not None]
        self._complete(pid, gen, "selected", f"{display_name(res['chosen'])} selected"
                       + (": " + ", ".join(f"{c['name']} val acc {c['val_acc']:.3f}" for c in done) if len(done) > 1 else ""))

    async def _job_train(self, pid: str, gen: int) -> None:
        from forge.trainer import TrainConfig, train

        rt = self.rt[pid]
        task, rows = self.task_schema(pid), read_dataset(pid)
        cfg = TrainConfig(encoder=rt.rec.get("base_model"))
        res = await asyncio.to_thread(train, task, rows, self.pdir(pid), cfg, self._thread_emit(pid),
                                      rt.stop.is_set)
        if getattr(res, "cancelled", False):
            return
        self._complete(pid, gen, "trained",
                       f"best val acc {res.best_val_acc:.3f} at epoch {res.best_epoch} "
                       f"(init {res.init_val_acc:.3f}; {res.epochs_run} epochs, {res.seconds:.0f}s)")

    def _teacher(self, pid: str):
        """Sync teacher callable for the evaluator (runs in a worker thread; LLM on the main loop)."""
        from forge.curator import label_states

        loop, llm, task = self.loop, self.llm_for(pid), self.task_schema(pid)

        def teacher(states: list[str]) -> list[str]:
            fut = asyncio.run_coroutine_threadsafe(label_states(llm, task, states[:50]), loop)
            return [lab or "" for lab, _c, _r in fut.result(timeout=900)]
        return teacher

    async def _job_evaluate(self, pid: str, gen: int) -> None:
        from forge.evaluator import evaluate

        rt = self.rt[pid]
        rec, task, rows = rt.rec, self.task_schema(pid), read_dataset(pid)
        opp = self.accepted_opportunity(pid)
        metric = rec.get("success_metric") or (opp.success_metric if opp else "accuracy")
        thr = float(rec.get("success_threshold") or (opp.success_threshold if opp else self.settings().success_threshold))
        rep = await asyncio.to_thread(evaluate, task, rows, self.pdir(pid) / "best_head.pt", self.pdir(pid),
                                      self._thread_emit(pid), self._teacher(pid), metric=metric, threshold=thr)
        if not self._current(pid, gen):
            return
        (self.pdir(pid) / "eval_report.json").write_text(rep.model_dump_json(indent=1))  # evaluator emitted `eval`
        rec["attempts"] = int(rec.get("attempts", 0)) + 1
        self.save(pid)
        ok, why = gate_check(rep.model_dump())
        self._complete(pid, gen, "evaluated", f"attempt {rec['attempts']}: gate {'passed' if ok else 'failed'} — "
                       + "; ".join(why))

    async def _job_review(self, pid: str, gen: int) -> None:
        rep = self.report(pid)
        rec = self.rt[pid].rec
        ok, why = gate_check(rep.model_dump())
        scores = "\n".join(f"- {m.name}: accuracy {m.accuracy:.3f}, macro_f1 {m.macro_f1:.3f} (n={m.n})"
                           + (f" — {m.note}" if m.note else "") for m in rep.models)
        per_label = "\n".join(f"- {l.label}: P {l.precision:.2f} R {l.recall:.2f} F1 {l.f1:.2f} (n={l.support})"
                              for l in rep.per_label)
        rows = read_dataset(pid)
        c = dataset_counts(rows)
        ctx = (f"Evaluation report (test split, n={rep.n_test}); metric {rep.metric}, threshold {rep.threshold}.\n"
               f"{scores}\nDeterministic gate: {'PASSED' if ok else 'FAILED'} — {'; '.join(why)}\n"
               f"ECE {rep.ece:.3f}\nPer label:\n{per_label}\n{_review_evidence(rep)}\n"
               f"Dataset counts: {c}\nAttempts so far: {rec.get('attempts', 0)} of max {self.settings().max_attempts}.")
        await self.decide_loop(pid, gen, ctx, REVIEW_RULES)

    async def _job_package(self, pid: str, gen: int) -> None:
        from forge import packager

        rt = self.rt[pid]
        task, opp, rep = self.task_schema(pid), self.accepted_opportunity(pid), self.report(pid)
        repo_summary = await asyncio.to_thread(self._repo_summary, pid)
        manifest = await asyncio.to_thread(
            packager.build_package, self.pdir(pid), task, opp, rep, repo_summary, self._thread_emit(pid),
            ckpt=self.pdir(pid) / "best_head.pt", examples=read_dataset(pid), polish=self._polish(pid))
        if not self._current(pid, gen):
            return
        md = manifest.model_dump() if hasattr(manifest, "model_dump") else dict(manifest)
        (self.pdir(pid) / "manifest.json").write_text(json.dumps(md, indent=1))  # packager emitted `package`
        self._complete(pid, gen, "packaged", f"package {md.get('name')} {md.get('version')} built")

    def _repo_summary(self, pid: str) -> str:
        rec = self.rt[pid].rec
        parts = [f"Project: {rec['name']}"]
        if rec.get("repo"):
            parts.append(f"Repository: {rec['repo']}" + (f" (local path {rec['repo_path']})" if rec.get("repo_path") else ""))
        if rec.get("task"):
            parts.append(f"Task description: {rec['task']}")
        p = self.proposal(pid)
        if p:
            parts.append(f"Analysis summary: {p.summary}")
        if rec.get("repo_path"):
            try:
                from forge import repo as repo_mod
                sc = repo_mod.scan(Path(rec["repo_path"]))
                parts.append(f"Languages: {sc.languages}; files: {len(sc.files)}")
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(parts)

    def _polish(self, pid: str):
        """Sync `polish(markdown, instructions=None) -> markdown` for the packager (LLM on the main loop)."""
        loop, llm = self.loop, self.llm_for(pid)

        def polish(text: str, instructions: str | None = None) -> str:
            prompt = ((instructions or "Polish this integration guide for a coding agent: keep every file path, "
                       "line number, command and code block exactly; tighten prose; fill obvious gaps; keep "
                       "markdown. Return only the document.") + "\n\n" + text)
            fut = asyncio.run_coroutine_threadsafe(llm.chat([{"role": "user", "content": prompt}]), loop)
            try:
                out = fut.result(timeout=600)
            except Exception:  # noqa: BLE001 - polish is best-effort
                return text
            return out if len(out) > 0.5 * len(text) else text
        return polish

    # ---- chat ------------------------------------------------------------------------------------
    async def chat(self, pid: str, message: str) -> None:
        rt = self.rt[pid]
        uid = uuid.uuid4().hex[:12]
        self.store.add_chat(pid, uid, "user", message)
        self.emit(pid, {"type": "chat", "role": "user", "content": message, "id": uid})
        rt.chat_task = asyncio.get_event_loop().create_task(self._chat_reply(pid, message))

    def _chat_context(self, pid: str) -> str:
        rt = self.rt[pid]
        parts = [f"Project {rt.rec['name']!r}; current state: {rt.machine.state}; path: "
                 + " -> ".join(f"{e}" for _s, e, _d in rt.machine.path),
                 f"Human actions available now: {rt.machine.allowed_for('human')}"]
        if rt.rec.get("repo"):
            parts.append(f"Repository: {rt.rec['repo']}")
        if rt.rec.get("task"):
            parts.append(f"Task description: {rt.rec['task']}")
        p = self.proposal(pid)
        if p:
            parts.append("Current proposal:\n" + p.model_dump_json(indent=1)[:12000])
        ts = self.task_schema(pid)
        if ts:
            parts.append("Accepted task schema:\n" + ts.model_dump_json(indent=1))
        rep = self.report(pid)
        if rep:
            r = rep.model_dump()
            r["errors"] = r["errors"][:8]
            parts.append("Evaluation report:\n" + json.dumps(r)[:8000])
        if rt.machine.state == "errored":
            parts.append(f"Last error: {rt.rec.get('error')}")
        return "\n\n".join(parts)

    async def _chat_reply(self, pid: str, message: str) -> None:
        rt = self.rt[pid]
        llm = self.llm_for(pid)
        history = self.store.chat(pid)[-20:]
        aid = uuid.uuid4().hex[:12]
        system = (analyst.CLM_PRIMER + "\n\n## Your role now\nYou are the CLM Forge assistant chatting with the "
                  "user about this project. Be concise and concrete. You cannot press buttons: state changes "
                  "(accept, revise, retry, cancel) are the user's; you may explain them.\n\n## Project context\n"
                  + self._chat_context(pid))
        try:
            if rt.machine.state == "proposed":
                convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[:-1])
                prompt = ((f"Conversation so far:\n{convo}\n\n" if convo else "") + f"user: {message}\n\n"
                          "Reply to the user. If (and only if) they ask to change the proposal, also return the "
                          "complete revised proposal in `revised_proposal` (same schema; keep unchanged parts).")
                out = await llm.structured(prompt, ChatReply, system=system)
                reply = out.reply
                if out.revised_proposal is not None:
                    prev = self.proposal(pid)
                    revised, warnings = analyst._ground(out.revised_proposal, rt.rec.get("repo_path"))
                    revised.revision = (prev.revision + 1) if prev else 1
                    rt.rec["proposal"] = revised.model_dump()
                    self.save(pid)
                    self.emit(pid, {"type": "proposal", "proposal": revised.model_dump()})
                    for w in warnings:
                        self.log(pid, "proposed", w, "warn")
                    if revised.verdict == "no_fit":
                        gen = rt.gen
                        await self.decide_loop(pid, gen, self._proposal_context(revised), ANALYZE_RULES)
            else:
                async def on_delta(d: str) -> None:
                    self.emit(pid, {"type": "chat_delta", "id": aid, "delta": d})
                msgs = [{"role": m["role"], "content": m["content"]} for m in history]
                reply = await llm.chat(msgs, system=system, on_delta=on_delta)
        except Exception as e:  # noqa: BLE001
            reply = f"(chat failed: {type(e).__name__}: {short(str(e), 300)})"
        self.store.add_chat(pid, aid, "assistant", reply)
        self.emit(pid, {"type": "chat", "role": "assistant", "content": reply, "id": aid})


ANALYZE_RULES = """\
- `propose` moves the project to the user for review; it is only legal when the proposal verdict is "fit".
- `declare_no_fit` ends the project honestly; it is only legal when the proposal verdict is "no_fit".
- Choose the event that matches the analyst's verdict and give a one-sentence reason grounded in it."""

def _review_evidence(rep: EvalReport) -> str:
    """Per-row evidence for the review decision and its curation guidance.

    Guidance becomes generated training data, so it is built from VAL errors only: showing test rows here
    would leak the frozen test set into train. A report without ``val`` was computed on val already
    (no separate test split), so its own errors are val errors."""
    def lines(errs) -> str:
        return "\n".join(f"- gold={e.gold} pred={e.pred}: {short(e.state, 220)}" for e in errs[:12]) or "(none)"
    v = rep.val
    if v is None:
        return f"Worst errors:\n{lines(rep.errors)}"
    conf = "; ".join(f"{g}→{p} {n}" for g, row in zip(v.confusion.labels, v.confusion.matrix)
                     for p, n in zip(v.confusion.labels, row) if n and p != g)
    return (f"Val split (n={v.n}): accuracy {v.accuracy:.3f}, macro_f1 {v.macro_f1:.3f}; confusions "
            f"(gold→pred count): {conf or 'none'}\nWorst VAL errors (test rows are withheld from guidance):\n"
            f"{lines(v.errors)}")


REVIEW_RULES = """\
- `ship` packages the model. The rules layer only allows it when the deterministic gate PASSED
  (metric ≥ threshold AND beats the zero-shot reference head AND beats the majority baseline).
  Never propose ship when the gate FAILED.
- `recurate` goes back to curation with your `guidance` (be specific: which label confusions to fix,
  what hard negatives / phrasing / lengths to add). Only legal while attempts < max attempts.
  Use it when the errors look fixable with better or more data.
- `give_up` ends the project as no_fit. Use it when attempts are exhausted or the errors show the
  task is not learnable from the state text (labels depend on information the text lacks, or the
  teacher itself is inconsistent)."""
