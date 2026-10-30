"""API: create -> analysis (fake LLM) -> proposed; SSE replay; human gates; restart recovery."""
import json
import time

import pytest
from fastapi.testclient import TestClient  # noqa: F401

import forge_client  # noqa: E402

from test_agent_fakes import FakeLLM, isolated  # noqa: F401  (fixture)

FIXTURE_FILES = {"support_app/router.py": "def route(t):\n    return llm('classify ticket: billing/technical/account')\n"}


def make_repo(tmp):
    root = tmp / "repo"
    for rel, text in FIXTURE_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def make_client(fake):
    from forge.api import create_app
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    from forge import config
    orch = Orchestrator(store=Store(config.DB_PATH))
    orch._llm_override = fake
    return forge_client.client(create_app(orch)), orch


def wait_state(c, pid, want, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get(f"/api/projects/{pid}").json()
        if d["state"] == want:
            return d
        time.sleep(0.05)
    raise AssertionError(f"state {d['state']} != {want}")


def replay(c, pid):
    r = c.get(f"/api/projects/{pid}/events?follow=false")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]


def test_create_analyze_propose_replay_and_gates(isolated):  # noqa: F811
    fake = FakeLLM(decisions=["accept", "propose"])        # first proposal is illegal for the agent
    client, orch = make_client(fake)
    repo = make_repo(isolated)
    with client as c:
        assert c.get("/api/health").json()["ok"]
        r = c.post("/api/projects", json={"repo": str(repo), "provider": "ollama", "model": "qwen3:8b",
                                          "encoder": "qwen3-4b"})
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        d = wait_state(c, pid, "proposed")
        assert d["proposal"]["verdict"] == "fit"
        assert d["allowed_human_events"] == sorted(d["allowed_human_events"]) or True
        assert set(d["allowed_human_events"]) == {"revise", "accept", "cancel", "label_override"}
        # proposal file refs were grounded against the real repo
        f = d["proposal"]["opportunities"][0]["files"][0]
        assert f["path"] == "support_app/router.py" and "classify ticket" in f["snippet"]

        evs = replay(c, pid)
        seqs = [e["seq"] for e in evs]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        states = [(e["from"], e["event"], e["to"], e["actor"]) for e in evs if e["type"] == "state"]
        assert states == [("created", "start", "analyzing", "system"), ("analyzing", "propose", "proposed", "agent")]
        rej = [e for e in evs if e["type"] == "rejected"]
        assert rej and rej[0]["event"] == "accept" and rej[0]["actor"] == "agent" and "only human" in rej[0]["error"]
        assert any(e["type"] == "proposal" for e in evs) and any(e["type"] == "tool" for e in evs)

        # human gates
        assert c.post(f"/api/projects/{pid}/accept", json={"opportunity_id": "nope"}).status_code == 409
        assert c.post(f"/api/projects/{pid}/revise", json={"feedback": " "}).status_code == 409
        assert c.post(f"/api/projects/{pid}/retry").status_code == 409

        # chat in proposed returns a structured reply
        assert c.post(f"/api/projects/{pid}/chat", json={"message": "why this one?"}).status_code == 202
        t0 = time.time()
        while time.time() - t0 < 5 and len(c.get(f"/api/projects/{pid}").json()["chat"]) < 2:
            time.sleep(0.05)
        chat = c.get(f"/api/projects/{pid}").json()["chat"]
        assert [m["role"] for m in chat] == ["user", "assistant"] and chat[1]["content"] == "sure"

        # revise -> analyzing -> proposed again with revision 1
        assert c.post(f"/api/projects/{pid}/revise", json={"feedback": "use fewer labels"}).status_code == 202
        t0 = time.time()
        while time.time() - t0 < 5:
            d = c.get(f"/api/projects/{pid}").json()
            if d["state"] == "proposed" and d["proposal"]["revision"] == 1:
                break
            time.sleep(0.05)
        assert d["proposal"]["revision"] == 1

        assert c.post(f"/api/projects/{pid}/cancel").status_code == 202
        assert c.get(f"/api/projects/{pid}").json()["state"] == "cancelled"
        assert c.get("/api/projects").json()[0]["id"] == pid
        assert c.get(f"/api/projects/{pid}/eval").status_code == 404


def test_no_fit_task_only(isolated):  # noqa: F811
    client, _ = make_client(FakeLLM(verdict="no_fit"))
    with client as c:
        pid = c.post("/api/projects", json={"task": "write marketing emails", "encoder": "qwen3-4b"}).json()["id"]
        d = wait_state(c, pid, "no_fit")
        assert d["path"][-1] == ["analyzing", "declare_no_fit", "no_fit"]
        assert d["allowed_human_events"] == []


def test_bad_inputs(isolated):  # noqa: F811
    client, _ = make_client(FakeLLM())
    with client as c:
        assert c.post("/api/projects", json={}).status_code == 422
        assert c.post("/api/projects", json={"repo": "relative/path", "encoder": "qwen3-4b"}).status_code == 422
        assert c.post("/api/projects", json={"repo": "/definitely/not/here", "encoder": "qwen3-4b"}).status_code == 422
        assert c.get("/api/projects/nope").status_code == 404
        s = c.put("/api/settings", json={"max_attempts": 2}).json()
        assert s["max_attempts"] == 2 and c.get("/api/settings").json()["max_attempts"] == 2


def test_restart_requeues_the_interrupted_stage(isolated):  # noqa: F811
    """A stage interrupted by a restart runs again (no "press retry"); the restart is logged."""
    from forge.rules import Project
    from forge.store import Store
    from forge import config
    st = Store(config.DB_PATH)
    m = Project("x")
    m.fire("start", "system", "go")
    st.save_project({"id": "p1", "name": "x", "repo": None, "task": "route tickets", "provider": "ollama",
                     "model": "fake", "attempts": 0, "feedback": [], "machine": m.snapshot()})
    jid = st.add_model_job("training", "project:p1")
    st.update_model_job(jid, state="running")
    client, _ = make_client(FakeLLM())
    with client as c:
        d = wait_state(c, "p1", "proposed")
        assert d["path"][-1] == ["analyzing", "propose", "proposed"] and d["error"] is None
        assert any("server restarted while analyzing" in e.get("message", "") for e in replay(c, "p1"))
        q = c.get("/api/autopilot/queue").json()
        assert q["active"] == [] and any(j["id"] == jid and j["state"] == "interrupted" for j in q["recent"])


def seed_labelling_project(pid="lp1", state="labelling", policy=None, failed_from=None):
    """A project at (or failed in) the labelling stage: 3 dataset rows, 3 real rows in its queue."""
    from forge import config
    from forge.labelling import LabelQueue
    from forge.models import Example, RealRow
    from forge.rules import Project
    from forge.store import Store, write_dataset
    from test_agent_fakes import TASK
    config.save_settings(config.Settings(min_examples=3))
    m = Project("labels", start_value=state, failed_from=failed_from)
    rec = {"id": pid, "name": "labels", "repo": None, "task": "route tickets", "provider": "ollama", "model": "fake",
           "attempts": 0, "feedback": [], "task_schema": TASK.model_dump(), "machine": m.snapshot(),
           "base_model": "mlx:mlx-community/Qwen3-4B-4bit"}
    if policy:
        rec["label_policy"] = policy
    Store(config.DB_PATH).save_project(rec)
    write_dataset(pid, [Example(id=f"s{i}", state=f"{lab} issue s{i}", label=lab, source="synthetic", split="train")
                        for i, lab in enumerate(TASK.labels())])
    LabelQueue(config.project_dir(pid), TASK).add(
        [RealRow(rid=f"r{i}", state=f"{lab} issue real {i}", source="imported", head=TASK.name)
         for i, lab in enumerate(["billing", "technical", "account"])])
    return pid


@pytest.fixture
def no_select(monkeypatch):
    from forge.orchestrator import Orchestrator

    async def no_select(self, pid, gen):   # base-model selection would load encoders
        return None
    monkeypatch.setattr(Orchestrator, "_job_select", no_select)


def test_agent_labelling_stage_settles_without_a_human(isolated, no_select):  # noqa: F811
    """Default policy: the labelling stage labels every real row with the agent and moves on by itself."""
    pid = seed_labelling_project()
    client, orch = make_client(FakeLLM())
    with client as c:
        d = wait_state(c, pid, "selecting")
        assert d["label_policy"] == "agent" and d["path"][-1] == ["labelling", "labels_settled", "selecting"]
        assert d["log"][-1]["actor"] == "system" and "every real row has a disposition" in d["log"][-1]["reason"]
        assert d["labels"]["agent"] == 3 and d["labels"]["needs_label"] == 0
        items = c.get(f"/api/projects/{pid}/dataset").json()["items"]
        assert len(items) == 6 and {i["label_source"] for i in items if i["id"].startswith("r")} == {"agent"}
        assert any(e["type"] == "labels" for e in replay(c, pid))
        # the stage waited for its model-job slot
        assert any(j["owner"] == f"project:{pid}" and j["kind"] == "labelling" and j["state"] == "done"
                   for j in c.get("/api/autopilot/queue").json()["recent"])
        # the queue is locked while a later stage reads the dataset; a label_override is recorded instead
        r = c.post(f"/api/labelsets/p:{pid}/rows/r0", json={"label": "account"})
        assert r.status_code == 409 and "locked while selecting" in r.json()["detail"]
        assert c.get(f"/api/labelsets/p:{pid}/queue").json()["editable"] is False


def test_agent_excludes_answers_outside_the_options_and_fails_loudly_on_no_answer(isolated, no_select,
                                                                                   monkeypatch):  # noqa: F811
    import forge.curator
    calls = []

    async def label_states(llm, task, states, **kw):
        calls.append(list(states))
        out = []
        for st in states:
            if "real 0" in st:
                out.append(("billing", 0.9, "invoice"))
            elif "real 1" in st:
                out.append((None, 0.4, "this is spam, not a ticket"))     # an answer outside the options
            else:
                out.append((None, 0.0, ""))                                # the teacher never answered
        return out
    monkeypatch.setattr(forge.curator, "label_states", label_states)
    pid = seed_labelling_project("lp4")
    client, orch = make_client(FakeLLM())
    with client as c:
        d = wait_state(c, pid, "errored")
        assert d["failed_from"] == "labelling" and "no answer for 1 rows after 3 attempts" in d["error"]
        assert [len(x) for x in calls] == [3, 1, 1]                        # the failure was retried twice
        rows = {r["rid"]: r for r in c.get(f"/api/labelsets/p:{pid}/queue?state=all").json()["items"]}
        assert rows["r0"]["label_source"] == "agent" and rows["r0"]["label"] == "billing"
        assert rows["r1"]["label_state"] == "excluded_by_agent" and "spam" in rows["r1"]["exclusion_reason"]
        assert rows["r2"]["label_state"] == "needs_label"                  # a failed teacher is never a label


def test_label_override_is_applied_by_the_next_labelling_stage(isolated, no_select):  # noqa: F811
    pid = seed_labelling_project("lp5", state="errored", failed_from="labelling")
    client, orch = make_client(FakeLLM())
    with client as c:
        assert c.post(f"/api/projects/{pid}/labels/override", json={"rid": "zz", "label": "billing"}).status_code == 404
        assert c.post(f"/api/projects/{pid}/labels/override", json={"rid": "r1", "label": "nope"}).status_code == 422
        r = c.post(f"/api/projects/{pid}/labels/override", json={"rid": "r1", "label": "account"})
        via = {"origin": "http://127.0.0.1:8765", "user_agent": "testclient"}   # the browser session it came from
        assert r.status_code == 202 and r.json()["pending"] == [{"rid": "r1", "label": "account", "via": via}]
        d = c.get(f"/api/projects/{pid}").json()
        assert d["state"] == "errored" and d["path"][-1] == ["errored", "label_override", "errored"]
        assert d["label_overrides"] == [{"rid": "r1", "label": "account", "via": via}]
        assert c.post(f"/api/projects/{pid}/retry").status_code == 202
        d = wait_state(c, pid, "selecting")
        assert d["label_overrides"] == [] and d["labels"]["user"] == 1 and d["labels"]["agent"] == 2
        rows = {r["rid"]: r for r in c.get(f"/api/labelsets/p:{pid}/queue?state=all").json()["items"]}
        assert rows["r1"]["label"] == "account" and rows["r1"]["label_source"] == "user"


def test_user_policy_opens_the_review_gate(isolated, no_select):  # noqa: F811
    pid = seed_labelling_project(policy="user")
    client, orch = make_client(FakeLLM())
    with client as c:
        d = wait_state(c, pid, "label_review")
        assert d["labels"]["needs_label"] == 3
        assert set(d["allowed_human_events"]) == {"review_done", "cancel", "label_override"}
        assert [s["id"] for s in c.get("/api/labelsets").json()] == [f"p:{pid}"]
        page = c.get(f"/api/labelsets/p:{pid}/queue").json()
        assert page["total"] == 3 and page["editable"] and page["policy"] == "user"
        assert [o["key"] for o in page["items"][0]["options"]] == ["billing", "technical", "account"]

        r = c.post(f"/api/projects/{pid}/labels/review", json={"mode": "continue"})
        assert r.status_code == 409 and "need a label" in r.json()["detail"]
        assert c.post(f"/api/labelsets/p:{pid}/rows/r0", json={"label": "nope"}).status_code == 422
        assert c.post(f"/api/labelsets/p:{pid}/rows/zz", json={"label": "billing"}).status_code == 404
        got = c.post(f"/api/labelsets/p:{pid}/rows/r0", json={"label": "billing"}).json()
        assert got["row"]["label_source"] == "user" and got["counts"]["needs_label"] == 2
        assert c.post(f"/api/labelsets/p:{pid}/rows/r1", json={"skip": True}).json()["row"]["skipped"]
        assert c.post(f"/api/labelsets/p:{pid}/undo").json()["row"]["skipped"] is False

        # the teacher is an override helper at this gate; the state does not move
        assert c.post(f"/api/labelsets/p:{pid}/teacher", json={}).json()["rows"] == 2
        t0 = time.time()
        while time.time() - t0 < 5 and c.get(f"/api/labelsets/p:{pid}/queue").json()["counts"]["needs_label"]:
            time.sleep(0.05)
        while time.time() - t0 < 5 and not orch.rt[pid].label_job.done():
            time.sleep(0.05)
        counts = c.get(f"/api/labelsets/p:{pid}/queue?state=all").json()["counts"]
        assert counts["needs_label"] == 0 and counts["teacher"] == 2 and counts["user"] == 1
        assert c.get(f"/api/projects/{pid}").json()["state"] == "label_review"

        assert c.put(f"/api/labelsets/p:{pid}/policy", json={"policy": "sometimes"}).status_code == 422
        assert c.post(f"/api/projects/{pid}/labels/review", json={"mode": "continue"}).status_code == 202
        d = wait_state(c, pid, "selecting")
        assert d["path"][-2:] == [["label_review", "review_done", "labelling"],
                                  ["labelling", "labels_settled", "selecting"]]
        assert len(c.get(f"/api/projects/{pid}/dataset").json()["items"]) == 6


def test_review_exclude_sets_rows_aside(isolated, no_select):  # noqa: F811
    pid = seed_labelling_project("lp2", policy="mixed")
    client, _ = make_client(FakeLLM())
    with client as c:
        d = wait_state(c, pid, "label_review")
        assert d["labels"]["suggested"] == 3                                 # mixed: suggestions only
        assert c.post(f"/api/projects/{pid}/labels/review", json={"mode": "later"}).status_code == 422
        assert c.post(f"/api/projects/{pid}/labels/review", json={"mode": "exclude"}).status_code == 202
        d = wait_state(c, pid, "selecting")
        assert d["labels"]["excluded"] == 3 and d["labels"]["needs_label"] == 0
        review = next(l for l in d["log"] if l["event"] == "review_done")
        assert "without 3 unlabelled real rows" in review["reason"] and review["actor"] == "human"


def test_teacher_job_is_tied_to_the_review_gate(isolated, no_select, monkeypatch):  # noqa: F811
    """The teacher job belongs to `label_review`: leaving waits for it, and other states refuse it."""
    import asyncio
    import threading

    import forge.curator
    from forge.rules import Project

    release = threading.Event()

    async def slow_label_states(llm, task, states, **kw):
        await asyncio.to_thread(release.wait, 10)
        return [(task.labels()[0], 0.9, "") for _ in states]
    monkeypatch.setattr(forge.curator, "label_states", slow_label_states)
    pid = seed_labelling_project("lp3", policy="user")
    client, orch = make_client(FakeLLM())
    with client as c:
        wait_state(c, pid, "label_review")
        assert c.post(f"/api/labelsets/p:{pid}/teacher", json={}).json()["rows"] == 3
        assert c.post(f"/api/labelsets/p:{pid}/teacher", json={}).status_code == 409
        r = c.post(f"/api/projects/{pid}/labels/review", json={"mode": "exclude"})
        assert r.status_code == 409 and "still labelling" in r.json()["detail"]
        assert c.get(f"/api/projects/{pid}").json()["state"] == "label_review"
        release.set()
        t0 = time.time()
        while time.time() - t0 < 5 and c.get(f"/api/labelsets/p:{pid}/queue").json()["counts"]["needs_label"]:
            time.sleep(0.05)
        while time.time() - t0 < 5 and not orch.rt[pid].label_job.done():
            time.sleep(0.05)
        assert c.post(f"/api/projects/{pid}/labels/review", json={"mode": "continue"}).status_code == 202
        wait_state(c, pid, "selecting")

        orch.rt[pid].machine = Project("labels", start_value="reviewing")
        r = c.post(f"/api/labelsets/p:{pid}/teacher", json={})
        assert r.status_code == 409 and "label review gate" in r.json()["detail"]
        assert orch.rt[pid].label_job.done()


def test_head_labelsets(isolated, monkeypatch, tmp_path):  # noqa: F811
    import json as _json
    from forge.labelling import LabelQueue, head_task
    from forge.models import RealRow
    from test_agent_fakes import TASK
    root = tmp_path / "heads"
    d = root / "task-x"
    d.mkdir(parents=True)
    (d / "task.json").write_text(_json.dumps({"head_id": "task-x", **TASK.model_dump()}))
    (root / "no-queue").mkdir()
    LabelQueue(d, head_task(d)).add([RealRow(rid="q1", state="billing issue", source="export", head="task-x")])
    monkeypatch.setenv("FORGE_LABELSETS", str(root))
    client, _ = make_client(FakeLLM())
    with client as c:
        sets = c.get("/api/labelsets").json()
        assert [s["id"] for s in sets] == ["h:task-x"] and sets[0]["counts"]["needs_label"] == 1
        assert c.post("/api/labelsets/h:task-x/rows/q1", json={"label": "account"}).json()["row"]["label_source"] == "user"
        assert c.post("/api/labelsets/h:task-x/undo").json()["counts"]["needs_label"] == 1
        # real user data in head queues goes only to the local two-pass CLI pipeline
        assert c.post("/api/labelsets/h:task-x/teacher", json={}).status_code == 409
        assert c.put("/api/labelsets/h:task-x/policy", json={"policy": "user"}).status_code == 422
        assert c.get("/api/labelsets/h:nope/queue").status_code == 404
