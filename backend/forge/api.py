"""FastAPI app: REST + SSE per SPEC. `forge-api` runs it on 127.0.0.1:8765 and serves frontend/dist."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

from forge import config, encoders, labelling, web_guard
from forge.autopilot import AutopilotService, presets
from forge.autopilot.adapter import AdapterError
from forge.autopilot.targets import SERVER_FIELDS, Target, id_from_name
from forge.autopilot.validation import error_body, plain_errors
from forge.labelling import LabelError, LabelQueue
from forge.models import Example, TaskSchema
from forge.orchestrator import DATASET_LOCKED, LabelJobRunning, Orchestrator, ProjectBusy
from forge.rules import Rejected, all_transitions
from forge.store import dataset_counts, read_dataset, write_dataset


# ---- request bodies -------------------------------------------------------------------------------
class CreateProject(BaseModel):
    name: str | None = None
    repo: str | None = None
    task: str | None = None
    provider: str | None = None
    model: str | None = None
    encoder: str          # registry key; required (GET /api/encoders/recommendation preselects one)


class ChatBody(BaseModel):
    message: str


class AcceptBody(BaseModel):
    opportunity_id: str
    overrides: dict[str, Any] | None = None


class ReviseBody(BaseModel):
    feedback: str


class PatchExample(BaseModel):
    label: str | None = None
    split: str | None = None
    state: str | None = None


class LabelRowBody(BaseModel):
    label: str | None = None
    skip: bool = False


class PolicyBody(BaseModel):
    policy: str


class TeacherBody(BaseModel):
    mode: str = "label"     # label | suggest


class ReviewBody(BaseModel):
    mode: str               # continue | exclude


class OverrideBody(BaseModel):
    rid: str
    label: str


class ScheduleBody(BaseModel):
    cadence: str            # daily | weekly
    weekday: int | None = None
    time_local: str         # HH:MM
    enabled: bool = True


class EncoderBody(BaseModel):
    key: str


class EnabledBody(BaseModel):
    enabled: bool


class InferBody(BaseModel):
    state: str
    model: str | None = "finetuned"


class RankBody(BaseModel):
    state: str
    candidates: list[str]
    instructions: str | None = None


class PlaygroundBody(BaseModel):
    state: str
    task: TaskSchema


# ---- app ------------------------------------------------------------------------------------------
def create_app(orch: Orchestrator | None = None, autopilot: AutopilotService | None = None,
               guard: web_guard.SessionGuard | None = None) -> FastAPI:
    orch = orch or Orchestrator()
    auto = autopilot or AutopilotService(orch.store, orch.jobs, emit=orch.emit, orchestrator=orch)
    predictors: dict[tuple, Any] = {}
    pred_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        orch.loop = asyncio.get_running_loop()
        from forge import encoder_migration
        encoder_migration.migrate(orch.store, orch.emit)   # once: experiments without an encoder get theirs
        config.load_settings()               # strips the app-wide encoder fields (original kept beside it)
        orch.store.interrupt_model_jobs()   # rows left by the last server; their owners queue them again below
        orch.restore_all()
        auto.start()
        try:
            yield
        finally:
            await auto.stop()

    app = FastAPI(title="CLM Forge", lifespan=lifespan)
    app.state.orch = orch
    app.state.autopilot = auto

    @app.exception_handler(presets.PresetFileError)
    async def preset_file_error(_request: Request, e: presets.PresetFileError):
        # the operator's presets file is unusable: say which file and why, wherever a route reads it
        return JSONResponse({"detail": str(e)}, status_code=500)

    # no CORS grant: the UI is served same-origin (the built UI by this server; the Vite dev server on :5173
    # proxies /api, so the browser sees one origin), and no other page may read Forge's responses
    # every request addressed to Forge itself; changes only from the Forge page with this server's session
    web_guard.install(app, guard or web_guard.SessionGuard(port=int(os.environ.get("FORGE_PORT", "8765"))))

    def rt(pid: str):
        if pid not in orch.rt:
            raise HTTPException(404, f"no experiment {pid}")
        return orch.rt[pid]

    def rejected(e: Rejected) -> HTTPException:
        return HTTPException(409, str(e))

    # -- health / settings ------------------------------------------------------------------------
    @app.get("/api/health")
    async def health():
        from forge.llm import claude as claude_mod
        from forge.llm import ollama as ollama_mod
        s = config.load_settings()
        users = encoders.used_by(orch.store, auto.targets)
        from forge.encoder import get_encoder
        in_use = []
        for key in encoders.in_use(users):      # every encoder an experiment or a target trains on; none loads here
            e = encoders.get(key)
            row: dict = {"key": e.key, "name": e.name, "backend": e.backend, "cached": encoders.cached(e),
                         "used_by": users[key]}
            try:
                info = get_encoder(e.spec).info()
                row.update(model=info.get("model"), loaded=info.get("loaded", False), worker=info.get("worker"))
            except Exception as err:  # noqa: BLE001 - reported per encoder
                row.update(loaded=False, error=f"{type(err).__name__}: {err}")
            in_use.append(row)
        unknown = sorted({p.get("base_model") for p in orch.store.list_projects()
                          if p.get("base_model") and not encoders.by_spec(p.get("base_model"))})
        from forge.llm import openai as openai_mod
        models = await asyncio.to_thread(ollama_mod.models, s.ollama_url)
        return {"ok": True, "encoders": in_use, "unregistered_encoders": unknown,
                "providers": {"claude": claude_mod.available(), "ollama": bool(models), "ollama_models": models,
                              "openai": bool(openai_mod.api_key(s) or s.openai_base_url.strip())}}

    @app.get("/api/providers")
    async def providers(refresh: bool = False):
        """Each provider's availability and the models it lists right now."""
        from forge.llm import catalog
        return await catalog.all_providers(refresh=refresh)

    def public_settings(s: config.Settings) -> dict:
        d = s.model_dump()
        from forge.llm.openai import norm_url
        d["openai_api_key_set"] = bool(d.pop("openai_api_key"))
        # a saved key that belongs to a different URL is not used (see openai.api_key)
        d["openai_api_key_applies"] = d["openai_api_key_set"] and norm_url(d["openai_api_key_url"]) == norm_url(d["openai_base_url"])
        d["openai_api_key_env"] = bool(os.environ.get("OPENAI_API_KEY"))
        from forge.memory import resolve_policy
        try:   # the effective numbers (None fields resolved from this machine's RAM)
            d["memory_resolved"] = resolve_policy(s.memory, int(os.environ.get("FORGE_MAX_LEN", 2048))).summary()
        except ValueError as e:   # e.g. settings copied from a machine with more RAM; model jobs will refuse
            d["memory_resolved"] = {"error": str(e)}
        return d

    @app.get("/api/settings")
    async def get_settings():
        return public_settings(config.load_settings())

    @app.put("/api/settings")
    async def put_settings(body: dict):
        legacy = sorted(set(body) & set(config.LEGACY_ENCODER_FIELDS))
        if legacy:
            raise HTTPException(422, f"{', '.join(legacy)}: there is no app-wide encoder setting; each experiment "
                                     "and autopilot target picks its own encoder")
        cur = config.load_settings().model_dump()
        key = body.pop("openai_api_key", "")
        body.pop("openai_api_key_url", None)  # server-managed
        cur.update({k: ({**cur[k], **v} if k == "memory" and isinstance(v, dict) else v)
                    for k, v in body.items() if k in cur})
        # write-only key: "" keeps the stored one, null clears it, anything else replaces it
        if key is None:
            cur["openai_api_key"] = ""
        elif isinstance(key, str) and key.strip():
            cur["openai_api_key"] = key.strip()
            cur["openai_api_key_url"] = cur.get("openai_base_url", "")
        try:
            s = config.Settings.model_validate(cur)
            from forge.memory import resolve_policy
            resolve_policy(s.memory, int(os.environ.get("FORGE_MAX_LEN", 2048)))   # budget + reserve must fit this machine
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, str(e)) from e
        return public_settings(config.save_settings(s))

    @app.get("/api/encoders")
    async def list_encoders():
        """The encoder registry with what uses each entry and whether it is in the HF cache."""
        users = encoders.used_by(orch.store, auto.targets)
        return {"encoders": [encoders.view(e, users[e.key]) for e in encoders.all()]}

    @app.get("/api/encoders/recommendation")
    async def encoder_recommendation():
        """The encoder Forge recommends for a new experiment before its task is known (``forge.encoder_choice``);
        the proposal recommends one per opportunity (``ProjectDetail.encoder_recommendations``)."""
        from forge.encoder_choice import recommend
        try:
            return (await asyncio.to_thread(recommend)).model_dump()
        except ValueError as e:      # the memory settings do not fit this machine
            raise HTTPException(422, str(e)) from e

    @app.get("/api/rules")
    async def rules_graph():
        from forge.rules import AGENT_EVENTS, HUMAN_EVENTS, SYSTEM_EVENTS
        return {"transitions": [list(t) for t in all_transitions()],
                "actors": {"agent": sorted(AGENT_EVENTS), "human": sorted(HUMAN_EVENTS),
                           "system": sorted(SYSTEM_EVENTS)}}

    # -- projects ---------------------------------------------------------------------------------
    @app.get("/api/projects")
    async def list_projects():
        return sorted((orch.summary(pid) for pid in orch.rt), key=lambda s: s.get("created_at") or "", reverse=True)

    @app.post("/api/projects")
    async def create_project(body: CreateProject):
        repo = (body.repo or "").strip() or None
        if repo:
            from forge.repo import is_url
            if not is_url(repo) and not Path(repo).expanduser().is_absolute():
                raise HTTPException(422, "repo must be an https git URL or an absolute local path")
            if not is_url(repo) and not Path(repo).expanduser().is_dir():
                raise HTTPException(422, f"no such directory: {repo}")
        from forge.llm.catalog import PROVIDERS
        if body.provider and body.provider not in PROVIDERS:
            raise HTTPException(422, f"provider must be one of {', '.join(PROVIDERS)}")
        try:
            return await orch.create(name=body.name, repo=repo, task=(body.task or "").strip() or None,
                                     provider=body.provider, model=body.model, encoder=body.encoder)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/projects/{pid}")
    async def get_project(pid: str):
        rt(pid)
        return orch.detail(pid)

    @app.put("/api/projects/{pid}/encoder")
    async def put_project_encoder(pid: str, body: EncoderBody):
        """Train this project on registry encoder ``key`` from its next selecting / training stage on."""
        rt(pid)
        try:
            old, new = orch.set_encoder(pid, body.key)
        except encoders.UnknownEncoder as e:
            raise HTTPException(422, str(e)) from e
        except ProjectBusy as e:
            raise HTTPException(409, str(e)) from e
        if old != new:
            auto.project_encoder_changed(pid, old, new)
        return orch.detail(pid)

    @app.delete("/api/projects/{pid}")
    async def delete_project(pid: str):
        rt(pid)
        orch.delete(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/events")
    async def events(pid: str, request: Request, follow: bool = True):
        rt(pid)
        sub = orch.bus.subscribe(pid)

        async def gen():
            try:
                last = 0
                for ev in orch.store.events(pid):
                    last = ev["seq"]
                    yield {"data": json.dumps(ev, default=str)}
                while follow:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(sub.queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        continue
                    if ev["seq"] <= last:
                        continue
                    yield {"data": json.dumps(ev, default=str)}
            finally:
                orch.bus.unsubscribe(sub)

        return EventSourceResponse(gen(), ping=15)

    @app.post("/api/projects/{pid}/chat", status_code=202)
    async def chat(pid: str, body: ChatBody):
        rt(pid)
        if not body.message.strip():
            raise HTTPException(422, "empty message")
        await orch.chat(pid, body.message.strip())
        return {"ok": True}

    @app.post("/api/projects/{pid}/accept", status_code=202)
    async def accept(pid: str, body: AcceptBody):
        rt(pid)
        try:
            orch.accept(pid, body.opportunity_id, body.overrides)
        except Rejected as e:
            raise rejected(e) from e
        except Exception as e:  # noqa: BLE001 - bad overrides
            raise HTTPException(422, str(e)) from e
        return {"ok": True}

    @app.post("/api/projects/{pid}/revise", status_code=202)
    async def revise(pid: str, body: ReviseBody):
        rt(pid)
        try:
            orch.revise(pid, body.feedback)
        except Rejected as e:
            raise rejected(e) from e
        return {"ok": True}

    @app.post("/api/projects/{pid}/cancel", status_code=202)
    async def cancel(pid: str):
        rt(pid)
        try:
            orch.cancel(pid)
        except Rejected as e:
            raise rejected(e) from e
        return {"ok": True}

    @app.post("/api/projects/{pid}/retry", status_code=202)
    async def retry(pid: str):
        rt(pid)
        try:
            orch.retry(pid)
        except Rejected as e:
            raise rejected(e) from e
        return {"ok": True}

    # -- dataset ----------------------------------------------------------------------------------
    @app.get("/api/projects/{pid}/dataset")
    async def dataset(pid: str, split: str | None = None, label: str | None = None, source: str | None = None,
                      q: str | None = None, offset: int = 0, limit: int = 100):
        rt(pid)
        rows = read_dataset(pid)
        counts = dataset_counts(rows)
        if split:
            rows = [r for r in rows if r.split == split]
        if label:
            rows = [r for r in rows if r.label == label]
        if source:
            rows = [r for r in rows if r.source == source]
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r.state.lower() or ql in (r.rationale or "").lower()]
        return {"total": len(rows), "items": [r.model_dump() for r in rows[offset:offset + max(1, min(limit, 1000))]],
                "counts": counts}

    def _editable(pid: str):
        st = rt(pid).machine.state
        if st in DATASET_LOCKED:
            raise HTTPException(409, f"the dataset and its labels are locked while {st}; "
                                     "edit them once this stage finishes")

    @app.patch("/api/projects/{pid}/dataset/{eid}")
    async def patch_example(pid: str, eid: str, body: PatchExample, request: Request):
        _editable(pid)
        ts = orch.task_schema(pid)
        if body.label is not None and ts and body.label not in ts.labels():
            raise HTTPException(422, f"label must be one of {ts.labels()}")
        q = orch.label_queue(pid)
        if body.label is not None and any(r.rid == eid for r in q.rows()):
            q.set_user_label(eid, body.label, web_guard.browser_origin(request))   # a real row: the queue logs it
        rows = read_dataset(pid)
        for i, r in enumerate(rows):
            if r.id == eid:
                d = r.model_dump()
                if body.label is not None:
                    if ts and body.label not in ts.labels():
                        raise HTTPException(422, f"label must be one of {ts.labels()}")
                    d["label"] = body.label
                    d["probabilities"] = None
                if body.split is not None:
                    if body.split not in ("train", "val", "test"):
                        raise HTTPException(422, "split must be train|val|test")
                    d["split"] = body.split
                if body.state is not None:
                    d["state"] = body.state
                d["meta"] = {**d.get("meta", {}), "edited": True, "original_source": r.source}
                d["source"] = "user"
                d["label_source"] = "user"
                rows[i] = Example.model_validate(d)
                write_dataset(pid, rows)
                return rows[i].model_dump()
        raise HTTPException(404, f"no example {eid}")

    @app.delete("/api/projects/{pid}/dataset/{eid}")
    async def delete_example(pid: str, eid: str):
        _editable(pid)
        rows = read_dataset(pid)
        keep = [r for r in rows if r.id != eid]
        if len(keep) == len(rows):
            raise HTTPException(404, f"no example {eid}")
        write_dataset(pid, keep)
        return {"ok": True}

    # -- labelling queues (projects and registered head dirs) --------------------------------------
    def labelset(lsid: str) -> tuple[LabelQueue, str | None, str]:
        """-> (queue, pid for a project queue else None, display name). lsid = p:<pid> | h:<head_id>."""
        kind, _, key = lsid.partition(":")
        if kind == "p":
            r = rt(key)
            return orch.label_queue(key), key, r.rec.get("name", key)
        if kind == "h":
            try:
                d = labelling.head_dir(key)
                return LabelQueue(d, labelling.head_task(d)), None, key
            except LabelError as e:
                raise HTTPException(404, str(e)) from e
        raise HTTPException(404, f"label set id must be p:<experiment> or h:<head>, got {lsid!r}")

    def queue_editable(pid: str | None) -> None:
        if pid is not None:
            _editable(pid)

    @app.get("/api/labelsets")
    async def labelsets():
        out = []
        for pid in orch.rt:
            if (orch.pdir(pid) / "real_rows.jsonl").exists():
                r = orch.rt[pid]
                out.append({"id": f"p:{pid}", "kind": "project", "name": r.rec.get("name", pid),
                            "state": r.machine.state, "policy": r.rec.get("label_policy"),
                            "counts": orch.label_queue(pid).counts()})
        return out + await asyncio.to_thread(labelling.head_labelsets)

    @app.get("/api/labelsets/{lsid}/queue")
    async def label_queue(lsid: str, head: str | None = None, state: str = "needs_label", offset: int = 0,
                          limit: int = 50):
        q, pid, name = labelset(lsid)
        try:
            total, items = q.page(head or None, state, offset, limit)  # type: ignore[arg-type]
        except LabelError as e:
            raise HTTPException(422, str(e)) from e
        locked = pid is not None and orch.rt[pid].machine.state in DATASET_LOCKED
        return {"id": lsid, "name": name, "kind": "project" if pid else "head", "total": total,
                "items": [{**r.model_dump(), "options": q.options(r)} for r in items],
                "counts": q.counts(head or None), "task": q.task.model_dump() if q.task else None,
                "policy": orch.rt[pid].rec.get("label_policy") if pid else None,
                "state": orch.rt[pid].machine.state if pid else None, "editable": not locked}

    @app.post("/api/labelsets/{lsid}/rows/{rid}")
    async def label_row(lsid: str, rid: str, body: LabelRowBody, request: Request):
        q, pid, _ = labelset(lsid)
        queue_editable(pid)
        if q.task is None:
            raise HTTPException(409, "no accepted task yet")
        try:
            via = web_guard.browser_origin(request)
            if body.skip:
                row = await asyncio.to_thread(q.skip, rid, via)
            elif body.label is not None:
                row = await asyncio.to_thread(q.set_user_label, rid, body.label, via)
            else:
                raise HTTPException(422, "give a label or skip=true")
        except LabelError as e:
            raise HTTPException(404 if "no row" in str(e) else 422, str(e)) from e
        return {"row": {**row.model_dump(), "options": q.options(row)}, "counts": q.counts()}

    @app.post("/api/labelsets/{lsid}/undo")
    async def label_undo(lsid: str, request: Request):
        q, pid, _ = labelset(lsid)
        queue_editable(pid)
        row = await asyncio.to_thread(q.undo, web_guard.browser_origin(request))
        return {"row": {**row.model_dump(), "options": q.options(row)} if row else None, "counts": q.counts()}

    @app.put("/api/labelsets/{lsid}/policy")
    async def label_policy(lsid: str, body: PolicyBody):
        _q, pid, _ = labelset(lsid)
        if pid is None:
            raise HTTPException(422, "a head directory has no label policy: its rows are labelled here or by "
                                     "the head-set adapter's own tooling")
        try:
            orch.set_label_policy(pid, body.policy)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"policy": body.policy}

    @app.post("/api/labelsets/{lsid}/teacher", status_code=202)
    async def label_teacher(lsid: str, body: TeacherBody | None = None):
        """Override helper at a project's label_review gate (opt-in human policies)."""
        _q, pid, _ = labelset(lsid)
        if pid is None:
            # head queues hold real user data: only the local two-pass CLI pipeline may label them
            raise HTTPException(409, "teacher-label a head queue with the head-set adapter's own tooling (its "
                                     "rows hold real user data); here you can label and override its rows")
        queue_editable(pid)
        try:
            n = orch.label_with_teacher(pid, (body or TeacherBody()).mode)
        except LabelJobRunning as e:
            raise HTTPException(409, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"rows": n}

    @app.post("/api/projects/{pid}/labels/review", status_code=202)
    async def review_done(pid: str, body: ReviewBody):
        """Leave the label_review gate (review_done, human)."""
        rt(pid)
        try:
            orch.review_done(pid, body.mode)
        except Rejected as e:
            raise rejected(e) from e
        except LabelJobRunning as e:
            raise HTTPException(409, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"ok": True}

    @app.post("/api/projects/{pid}/labels/override", status_code=202)
    async def label_override(pid: str, body: OverrideBody, request: Request):
        """Record a user label (label_override, human) in any non-terminal state; the next labelling stage
        applies it."""
        rt(pid)
        try:
            return orch.label_override(pid, body.rid, body.label, web_guard.browser_origin(request))
        except Rejected as e:
            raise rejected(e) from e
        except LabelError as e:
            raise HTTPException(404 if "no row" in str(e) else 422, str(e)) from e

    # -- autopilot --------------------------------------------------------------------------------
    def target(tid: str) -> Target:
        t = auto.targets.get(tid)
        if t is None:
            raise HTTPException(404, f"no autopilot target {tid}")
        return t

    @app.get("/api/autopilot/targets")
    async def autopilot_targets():
        return [auto.target_view(t) for t in auto.targets.list()]

    @app.get("/api/autopilot/targets/{tid}")
    async def autopilot_target(tid: str):
        return auto.target_view(target(tid))

    def unprocessable(errors: list[dict]) -> JSONResponse:
        return JSONResponse(error_body(errors), status_code=422)

    def parse_target(body: dict, tid: str) -> Target | JSONResponse:
        """A target body -> ``Target``, or the 422 response naming each bad field. A head-set body naming a preset
        gets the preset's fields under its own (``presets.fill``)."""
        hs = body.get("headset")
        if isinstance(hs, dict):
            # what a head set runs comes only from its preset: a body naming a command or path field is refused
            bad = [{"field": f"headset.{k}", "message": "set by the head-set preset; a request cannot set it"}
                   for k in SERVER_FIELDS if k in hs]
            if not hs.get("preset"):
                bad.append({"field": "headset.preset", "message": "choose a head-set preset (what a head set runs "
                            "comes only from a preset)"})
            if bad:
                return unprocessable(bad)
            try:
                body = {**body, "headset": presets.fill(hs)}
            except KeyError as e:
                return unprocessable([{"field": "headset.preset", "message": str(e.args[0])}])
        try:
            t = Target.model_validate({**body, "id": tid})
        except ValidationError as e:
            return unprocessable(plain_errors(e))
        if t.kind == "project" and t.project and t.project.project_id not in orch.rt:
            return unprocessable([{"field": "project.project_id", "message": "that project no longer exists"}])
        return t

    @app.get("/api/autopilot/headsets")
    async def autopilot_headsets():
        return [p.view() for p in presets.all()]

    @app.get("/api/autopilot/headsets/{preset}/heads")
    async def autopilot_headset_heads(preset: str, encoder: str):
        """The preset's heads, from its adapter's read-only ``inspect`` (loads no model)."""
        try:
            p = presets.get(preset)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        if not encoders.known(encoder):
            return JSONResponse(error_body([{"field": "encoder", "message": f"{encoder or 'No encoder'} is not a "
                                             "registered encoder; pick one"}], fields=False), status_code=422)
        try:
            return await presets.heads(p, encoder, auto.runner.adapter_factory)
        except ValidationError as e:
            return JSONResponse(error_body(plain_errors(e), fields=False), status_code=422)
        except AdapterError as e:
            raise HTTPException(502, f"the {p.name} adapter could not list its heads: {e}") from e

    @app.post("/api/autopilot/targets", status_code=201)
    async def create_autopilot_target(body: dict):
        tid = str(body.get("id") or id_from_name(str(body.get("name") or "")))
        t = parse_target(body, tid)
        if isinstance(t, JSONResponse):
            return t
        try:
            auto.create(t)
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        return auto.target_view(t)

    @app.put("/api/autopilot/targets/{tid}")
    async def put_autopilot_target(tid: str, body: dict):
        """Full replace (creates the target when it does not exist yet)."""
        t = parse_target(body, tid)
        if isinstance(t, JSONResponse):
            return t
        old = auto.targets.get(tid)
        try:
            t = auto.replace(old, t) if old else auto.create(t)
        except ValidationError as e:        # the re-derived output root is not allowed either
            return unprocessable(plain_errors(e))
        return auto.target_view(t)

    @app.patch("/api/autopilot/targets/{tid}/enabled")
    async def patch_autopilot_target_enabled(tid: str, body: EnabledBody):
        return auto.target_view(auto.set_enabled(target(tid), body.enabled))

    @app.delete("/api/autopilot/targets/{tid}")
    async def delete_autopilot_target(tid: str):
        target(tid)
        auto.targets.delete(tid)
        return {"ok": True}

    @app.put("/api/autopilot/targets/{tid}/schedule")
    async def put_schedule(tid: str, body: ScheduleBody):
        target(tid)
        try:
            return auto.scheduler.put(tid, body.cadence, body.weekday, body.time_local, body.enabled)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/api/autopilot/targets/{tid}/run", status_code=202)
    async def run_now(tid: str):
        t = target(tid)
        if not t.enabled:
            raise HTTPException(409, f"target {tid} is disabled")
        return auto.enqueue(tid, "manual")

    @app.get("/api/autopilot/runs")
    async def autopilot_runs(target: str | None = None, limit: int = 50):
        return [auto.run_view(r) for r in orch.store.runs(target, limit=max(1, min(limit, 500)))]

    @app.get("/api/autopilot/runs/{run_id}")
    async def autopilot_run(run_id: str):
        r = orch.store.get_run(run_id)
        if r is None:
            raise HTTPException(404, f"no run {run_id}")
        return auto.run_view(r)

    @app.get("/api/autopilot/journal")
    async def autopilot_journal(target: str | None = None, head: str | None = None, action: str | None = None,
                                run: str | None = None, after: int = 0, limit: int = 200):
        return auto.journal.query(target_id=target, head=head, action=action, run_id=run, after=after,
                                  limit=max(1, min(limit, 2000)))

    @app.get("/api/autopilot/queue")
    async def autopilot_queue():
        return {**orch.jobs.snapshot(), "autopilot_current": auto.current}

    @app.get("/api/autopilot/groups")
    async def autopilot_groups():
        return auto.groups()

    @app.get("/api/autopilot/heads/{head}/provenance")
    async def head_provenance(head: str, target: str | None = None):
        try:
            return auto.provenance(head, target)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    # -- training / eval / package ----------------------------------------------------------------
    @app.get("/api/projects/{pid}/training")
    async def training(pid: str):
        rt(pid)
        evs = orch.store.events(pid, types=["train_epoch", "train_snapshot"])
        return {"history": [e for e in evs if e["type"] == "train_epoch"],
                "snapshots": [e for e in evs if e["type"] == "train_snapshot"]}

    @app.get("/api/projects/{pid}/eval")
    async def get_eval(pid: str):
        rt(pid)
        rep = orch.report(pid)
        if rep is None:
            raise HTTPException(404, "no evaluation yet")
        return rep.model_dump()

    @app.get("/api/projects/{pid}/package")
    async def get_package(pid: str):
        rt(pid)
        m = orch.manifest(pid)
        if m is None:
            raise HTTPException(404, "no package yet")
        return m

    def _pkg_path(pid: str, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (orch.pdir(pid) / pp)

    @app.get("/api/projects/{pid}/package/download")
    async def download_package(pid: str):
        rt(pid)
        m = orch.manifest(pid)
        if m is None:
            raise HTTPException(404, "no package yet")
        z = _pkg_path(pid, m["zip"])
        if not z.exists():
            raise HTTPException(404, "package zip missing")
        return FileResponse(z, filename=z.name, media_type="application/zip")

    @app.get("/api/projects/{pid}/package/file")
    async def package_file(pid: str, path: str):
        rt(pid)
        m = orch.manifest(pid)
        if m is None:
            raise HTTPException(404, "no package yet")
        base = _pkg_path(pid, m["dir"]).resolve()
        f = (base / path).resolve()
        if base not in f.parents and f != base:
            raise HTTPException(400, "path escapes the package directory")
        if not f.is_file():
            raise HTTPException(404, "no such file")
        if f.stat().st_size > 2_000_000:
            raise HTTPException(413, "file too large to preview")
        return PlainTextResponse(f.read_text(errors="replace"))

    # -- inference --------------------------------------------------------------------------------
    def predictor(pid: str | None, task: TaskSchema, which: str):
        from forge.inference import Predictor
        ckpt = None
        if which == "finetuned" and pid is not None:
            ckpt = orch.pdir(pid) / "best_head.pt"
            if not ckpt.exists():
                raise HTTPException(404, "no fine-tuned checkpoint yet")
        key = (pid, which, task.model_dump_json(), ckpt.stat().st_mtime if ckpt else None)
        with pred_lock:
            if key not in predictors:
                for k in [k for k in predictors if k[:2] == (pid, which)]:
                    predictors.pop(k)
                predictors[key] = Predictor(ckpt, task)
            return predictors[key]

    @app.post("/api/projects/{pid}/infer")
    async def infer(pid: str, body: InferBody):
        rt(pid)
        task = orch.task_schema(pid)
        if task is None:
            raise HTTPException(409, "no accepted task yet")
        which = body.model if body.model in ("finetuned", "reference") else "finetuned"
        p = await asyncio.to_thread(predictor, pid, task, which)
        ans, t = await asyncio.to_thread(p.answer, body.state)
        return {"answer": ans, **t}

    @app.post("/api/projects/{pid}/rank")
    async def rank(pid: str, body: RankBody):
        rt(pid)
        task = orch.task_schema(pid)
        if task is None:
            raise HTTPException(409, "no accepted task yet")
        which = "finetuned" if (orch.pdir(pid) / "best_head.pt").exists() else "reference"
        p = await asyncio.to_thread(predictor, pid, task, which)
        return await asyncio.to_thread(p.rank, body.state, body.candidates, body.instructions)

    @app.post("/api/playground/infer")
    async def playground(body: PlaygroundBody):
        p = await asyncio.to_thread(predictor, None, body.task, "reference")
        ans, t = await asyncio.to_thread(p.answer, body.state)
        return {"answer": ans, **t}

    # -- frontend ---------------------------------------------------------------------------------
    dist = config.FRONTEND_DIST
    if dist.is_dir() and (dist / "index.html").exists():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            f = (dist / full_path).resolve()
            if full_path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            # index.html names the hashed bundles; never let the browser reuse a stale copy after a rebuild
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def main() -> None:
    import uvicorn

    from forge import instance_lock
    config.ensure_dirs()
    port = int(os.environ.get("FORGE_PORT", "8765"))
    try:
        held = instance_lock.acquire(config.DATA_DIR, port)   # noqa: F841 - held open for the server's life
    except instance_lock.InstanceLockError as e:
        raise SystemExit(f"forge-api: {e}") from None
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
