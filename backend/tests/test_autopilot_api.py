"""Autopilot REST: targets, schedule, run now, runs, journal, queue, groups, provenance."""
import json
import time

from fastapi.testclient import TestClient  # noqa: F401

import forge_client  # noqa: E402

import headset_presets
from test_agent_fakes import FakeLLM, isolated  # noqa: F401  (fixture)
from test_autopilot_runner import FakeAdapter


def use_preset(root, monkeypatch, **kw):
    """A presets file with one preset whose folders are under ``root``."""
    return headset_presets.use(monkeypatch, root / "presets.json", headset_presets.preset(root, **kw))


def make(tmp, adapter):
    from forge import config
    from forge.api import create_app
    from forge.autopilot import AutopilotService
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    orch = Orchestrator(store=Store(config.DB_PATH))
    orch._llm_override = FakeLLM()
    auto = AutopilotService(orch.store, orch.jobs, emit=orch.emit, orchestrator=orch, adapter_factory=lambda t: adapter)
    return forge_client.client(create_app(orch, auto)), auto


def test_autopilot_routes(isolated, monkeypatch):  # noqa: F811
    use_preset(isolated, monkeypatch)
    data, out = isolated / "data", isolated / "heads-qwen3-0.6b-bf16"
    (data / "h").mkdir(parents=True)
    (data / "h" / "rows.jsonl").write_text("{}\n")
    (out / "h").mkdir(parents=True)
    (out / "h" / "provenance.json").write_text(json.dumps({"class": "includes_user_data"}))
    ad = FakeAdapter(data)
    client, auto = make(isolated, ad)
    body = {"kind": "headset", "headset": {"preset": "test-heads", "encoder": "qwen3-0.6b-bf16", "output_root": str(out)}}
    with client as c:
        assert c.put("/api/autopilot/targets/Bad Id", json=body).status_code == 422
        assert c.put("/api/autopilot/targets/x", json={**body, "headset": {**body["headset"], "cwd": "/tmp"}}
                     ).status_code == 422
        t = c.put("/api/autopilot/targets/heads", json=body).json()
        assert t["id"] == "heads" and t["headset"]["policy"]["min_count"] == 150 and t["schedule"] is None
        assert c.put("/api/autopilot/targets/heads/schedule", json={"cadence": "weekly", "time_local": "03:00"}
                     ).status_code == 422
        s = c.put("/api/autopilot/targets/heads/schedule",
                  json={"cadence": "weekly", "weekday": 6, "time_local": "03:30"}).json()
        assert s["next_run"] and s["weekday"] == 6
        assert c.get("/api/autopilot/targets").json()[0]["schedule"]["time_local"] == "03:30"
        assert c.post("/api/autopilot/targets/nope/run").status_code == 404
        run = c.post("/api/autopilot/targets/heads/run").json()
        assert run["state"] == "queued" and run["trigger"] == "manual"
        t0 = time.time()
        while time.time() - t0 < 10 and c.get(f"/api/autopilot/runs/{run['id']}").json()["state"] in ("queued", "running"):
            time.sleep(0.05)
        r = c.get(f"/api/autopilot/runs/{run['id']}").json()
        assert r["state"] == "done", r["result"].get("error")
        assert r["result"]["heads"]["h"]["outcome"] == "promoted"
        assert c.get("/api/autopilot/runs?target=heads").json()[0]["id"] == run["id"]
        j = c.get("/api/autopilot/journal?target=heads&action=gate").json()
        assert len(j) == 1 and j[0]["head"] == "h" and "test macro-F1" in j[0]["reason"]
        assert c.get("/api/autopilot/journal?target=heads&head=h").json()
        q = c.get("/api/autopilot/queue").json()
        assert q["active"] == [] and {x["kind"] for x in q["recent"]} >= {"train", "gate"}
        g = c.get("/api/autopilot/groups").json()
        assert g[0]["target"] == "heads" and g[0]["report"]["groups"][0]["name"] == "classify"
        assert c.get("/api/autopilot/heads/h/provenance").json()["provenance"]["class"] == "includes_user_data"
        assert c.get("/api/autopilot/heads/zz/provenance").status_code == 404
        assert c.get("/api/autopilot/heads/..%2Fetc/provenance").status_code in (404, 422)
        assert c.delete("/api/autopilot/targets/heads").json() == {"ok": True}
        assert c.get("/api/autopilot/targets").json() == []


def test_project_target_inspects_and_journals_into_the_project_stream(isolated):  # noqa: F811
    from test_api_projects import seed_labelling_project
    pid = seed_labelling_project("ap1", policy="user")
    client, auto = make(isolated, None)
    with client as c:
        t0 = time.time()
        while time.time() - t0 < 5 and c.get(f"/api/projects/{pid}").json()["state"] != "label_review":
            time.sleep(0.05)
        assert c.put("/api/autopilot/targets/proj", json={"kind": "project", "project": {"project_id": "nope"}}
                     ).status_code == 422
        assert c.put("/api/autopilot/targets/proj", json={"kind": "project", "project": {"project_id": pid}}
                     ).status_code == 200
        rid = c.post("/api/autopilot/targets/proj/run").json()["id"]
        while time.time() - t0 < 10 and c.get(f"/api/autopilot/runs/{rid}").json()["state"] in ("queued", "running"):
            time.sleep(0.05)
        r = c.get(f"/api/autopilot/runs/{rid}").json()
        assert r["state"] == "done", r["result"]
        needs = {e["counts"]["kind"] for e in c.get("/api/autopilot/journal?target=proj&action=need").json()}
        assert {"unlabelled", "small_data"} <= needs and "no_control" not in needs
        evs = c.get(f"/api/projects/{pid}/events?follow=false").text
        assert '"type": "autopilot"' in evs


def _headset(root, **kw):
    data, out = root / "data", root / "heads"
    (data / "h").mkdir(parents=True, exist_ok=True)
    (data / "h" / "rows.jsonl").write_text("{}\n")
    return {"kind": "headset", "name": "Test heads 1", "headset": {
        "preset": "test-heads", "encoder": "qwen3-0.6b-bf16", "output_root": f"{out}-qwen3-0.6b-bf16",
        "sources": [{"kind": "event_log", "enabled": True, "settings": {"stage": "qa"}}], **kw}}


def _errors(r):
    assert r.status_code == 422, r.text
    return {e["field"]: e["message"] for e in r.json()["errors"]}


def test_target_crud_field_errors_and_journal(isolated, monkeypatch):  # noqa: F811
    from forge.autopilot.runner import default_adapter
    use_preset(isolated, monkeypatch)
    ad = FakeAdapter(isolated / "data")
    client, auto = make(isolated, ad)
    body = _headset(isolated)
    with client as c:
        # create: the id comes from the name when not given; a second create of the id is a conflict
        t = c.post("/api/autopilot/targets", json=body)
        assert t.status_code == 201, t.text
        t = t.json()
        assert t["id"] == "test-heads-1" and t["name"] == "Test heads 1" and t["encoder_name"] == "Qwen3-0.6B (MLX bf16)"
        assert t["output_roots"]["qwen3-0.6b"] == f"{isolated / 'heads'}-qwen3-0.6b"     # where an encoder change goes
        assert c.post("/api/autopilot/targets", json={**body, "id": "test-heads-1"}).status_code == 409
        assert c.get("/api/autopilot/journal?target=test-heads-1&action=target_created").json()
        # field errors, in words, per field
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "output_root": str(isolated / "heads" / "sub")}}))
        assert e["headset.output_root"].startswith("Output folder is inside a protected folder")
        # what a head set runs comes only from its preset: a body naming a command or path is refused
        for k, v in (("adapter", ["sh", "-c", "touch /tmp/pwned"]), ("cwd", "/"), ("env", {"PYTHONPATH": "/x"}),
                     ("dataset_root", "/"), ("protected_roots", []), ("protected_files", [])):
            e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {**body["headset"], k: v}}))
            assert e == {f"headset.{k}": "set by the head-set preset; a request cannot set it"}, k
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **{k: v for k, v in body["headset"].items() if k != "preset"}}}))
        assert "choose a head-set preset" in e["headset.preset"]
        for key in ("adapter_src", "python", "data_root", "sources"):
            e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
                **body["headset"], "adapter_config": {key: "/tmp/evil"}}}))
            assert list(e) == ["headset.adapter_config"], key
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "output_root": "/srv/shared/anything"}}))
        assert e["headset.output_root"].startswith("Output folder must be a heads-<name> folder")
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "policy": {"min_count": 0, "max_rounds": 11}}}))
        assert e == {"headset.policy.min_count": "must be at least 1", "headset.policy.max_rounds": "must be at most 10"}
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "encoder": "nope"}}))
        assert "not a registered encoder" in e["headset.encoder"]
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "encoder": "qwen3-4b-gguf-q4km"}}))
        assert "Qwen3-4B (GGUF Q4_K_M) is a llamacpp build" in e["headset.encoder"]
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "sources": [{"kind": "rss", "settings": {}}]}}))
        assert "headset.sources.0.kind" in e
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "sources": [{"kind": "shared_export", "settings": {"stage": "qa"}}]}}))
        assert list(e) == ["headset.sources.0.settings.stage"] and "has no setting" in e["headset.sources.0.settings.stage"]
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "sources": [{"kind": "shared_export", "settings": {"stages": ["dev"]}}]}}))
        assert e == {"headset.sources.0.settings.stages": "dev is not one of qa, prod"}
        e = _errors(c.post("/api/autopilot/targets", json={**body, "id": "x2", "headset": {
            **body["headset"], "sources": [{"kind": "event_log", "settings": {"stage": "staging"}}]}}))
        assert e == {"headset.sources.0.settings.stage": "must be one of dev, qa, prod"}
        # a data-only adapter key the preset allows is accepted; settings left out get the preset's defaults
        ok = c.post("/api/autopilot/targets", json={**body, "id": "x3", "headset": {
            **body["headset"], "adapter_config": {"epochs": 3},
            "sources": [{"kind": "shared_export", "enabled": True}]}})
        assert ok.status_code == 201, ok.text
        assert ok.json()["headset"]["sources"] == [{"kind": "shared_export", "enabled": True,
                                                    "settings": {"stages": ["qa", "prod"]}}]
        assert c.delete("/api/autopilot/targets/x3").json() == {"ok": True}
        e = _errors(c.post("/api/autopilot/targets", json={"id": "p", "kind": "project",
                                                           "project": {"project_id": "nope"}}))
        assert e == {"project.project_id": "that project no longer exists"}
        assert _errors(c.post("/api/autopilot/targets", json={**body, "id": "Bad Id"}))["id"]
        # replace: an encoder change re-derives the derived output root and is journalled with names
        new = {**body, "headset": {**body["headset"], "encoder": "qwen3-0.6b"}}
        t = c.put("/api/autopilot/targets/test-heads-1", json=new).json()
        assert t["headset"]["output_root"] == f"{isolated / 'heads'}-qwen3-0.6b"
        j = c.get("/api/autopilot/journal?target=test-heads-1&action=encoder_changed").json()
        assert len(j) == 1 and j[0]["reason"].startswith(
            "encoder changed from Qwen3-0.6B (MLX bf16) to Qwen3-0.6B (MLX 4-bit); the next run trains fresh controls")
        assert j[0]["counts"]["from"] == "qwen3-0.6b-bf16" and j[0]["counts"]["to"] == "qwen3-0.6b"
        assert not c.get("/api/autopilot/journal?target=test-heads-1&action=target_edited").json()
        # the adapter now gets the new encoder: its control fingerprints (encoder + hparams + rows) change with it
        ad_ = default_adapter(auto.targets.get("test-heads-1"))
        assert ad_.target_config["encoder"] == "qwen3-0.6b"
        assert ad_.argv == ["x"] and ad_.cwd == str(isolated) and ad_.env == {}     # the preset's command, only
        # any other change: target_edited with the fields
        new["headset"] = {**new["headset"], "encoder": "qwen3-0.6b", "output_root": t["headset"]["output_root"],
                          "policy": {"gen_cap": 300}}
        c.put("/api/autopilot/targets/test-heads-1", json={**new, "name": "Test heads 1 (0.6B)"})
        j = c.get("/api/autopilot/journal?target=test-heads-1&action=target_edited").json()
        assert j[0]["counts"]["fields"] == ["headset.policy.gen_cap", "name"]
        # enable / disable
        assert c.patch("/api/autopilot/targets/test-heads-1/enabled", json={"enabled": False}).json()["enabled"] is False
        assert c.post("/api/autopilot/targets/test-heads-1/run").status_code == 409
        assert c.patch("/api/autopilot/targets/test-heads-1/enabled", json={"enabled": True}).json()["enabled"] is True
        assert c.patch("/api/autopilot/targets/zz/enabled", json={"enabled": True}).status_code == 404
        # a run, then delete: runs and journal stay, the run says its target was removed
        rid = c.post("/api/autopilot/targets/test-heads-1/run").json()["id"]
        t0 = time.time()
        while time.time() - t0 < 10 and c.get(f"/api/autopilot/runs/{rid}").json()["state"] in ("queued", "running"):
            time.sleep(0.05)
        r = c.get(f"/api/autopilot/runs/{rid}").json()
        assert r["state"] == "done" and r["encoder"] == "Qwen3-0.6B (MLX 4-bit)" and r["target_name"] == "Test heads 1 (0.6B)"
        assert r["progress"]["fraction"] == 1.0 and r["progress"]["heads_done"] == 1
        assert [x["state"] for x in r["timeline"]] and all(x["state"] == "done" for x in r["timeline"])
        assert r["counts"]["total"]["generated"] > 0
        last = c.get("/api/autopilot/targets/test-heads-1").json()["last_run"]
        assert last["id"] == rid and last["progress"]["heads_total"] == 1
        assert c.delete("/api/autopilot/targets/test-heads-1").json() == {"ok": True}
        gone = c.get(f"/api/autopilot/runs/{rid}").json()
        assert gone["target_removed"] is True and gone["target_name"] is None
        assert c.get(f"/api/autopilot/journal?run={rid}").json()


def test_headset_presets_and_their_heads(isolated, monkeypatch):  # noqa: F811
    use_preset(isolated, monkeypatch)
    pinned = headset_presets.preset(isolated, key="pinned", name="Pinned", fixed_heads=["only"],
                                    env={"MY_SETTING": "1"}, sources=headset_presets.SOURCES[:1])
    headset_presets.use(monkeypatch, isolated / "presets.json", headset_presets.preset(isolated), pinned)
    ad = FakeAdapter(isolated / "data", heads=("h", "g"))
    client, auto = make(isolated, ad)
    with client as c:
        hs = {p["key"]: p for p in c.get("/api/autopilot/headsets").json()}
        assert set(hs) == {"test-heads", "pinned"}
        th = hs["test-heads"]
        assert set(th) == {"key", "name", "description", "dataset_root", "output_base", "protected_roots", "sources",
                           "fixed_heads", "encoders", "output_roots", "recommendation"}
        assert th["sources"] == [
            {"kind": "event_log", "label": "Event log", "enabled_by_default": True,
             "settings": {"stage": {"values": ["dev", "qa", "prod"], "default": "qa", "multiple": False}}},
            {"kind": "shared_export", "label": "Shared export", "enabled_by_default": False,
             "settings": {"stages": {"values": ["qa", "prod"], "default": ["qa", "prod"], "multiple": True}}}]
        assert th["fixed_heads"] is None and hs["pinned"]["fixed_heads"] == ["only"]
        assert "qwen3-0.6b-bf16" in th["encoders"] and "qwen3-4b-gguf-q4km" not in th["encoders"]
        assert th["output_roots"]["qwen3-0.6b"] == th["output_base"] + "-qwen3-0.6b"
        assert th["output_roots"]["qwen3-4b"] == th["output_base"] + "-qwen3-4b"   # never the protected base
        got = c.get("/api/autopilot/headsets/test-heads/heads?encoder=qwen3-0.6b-bf16").json()
        assert [x["head"] for x in got] == ["h", "g"] and got[0]["group"] == "classify"
        assert ad.calls[-1][0] == "inspect"
        assert c.get("/api/autopilot/headsets/nope/heads?encoder=qwen3-0.6b").status_code == 404
        # a target made from a preset: the form sends only what it edits, the preset fills the rest
        t = c.post("/api/autopilot/targets", json={"name": "Pinned nightly", "kind": "headset", "headset": {
            "preset": "pinned", "encoder": "qwen3-0.6b", "heads": ["only"], "policy": {"gen_cap": 200}}})
        assert t.status_code == 201, t.text
        h = t.json()["headset"]
        assert h["adapter"] == ["x"] and h["cwd"] == str(isolated) and h["env"] == {"MY_SETTING": "1"}
        assert h["output_root"] == hs["pinned"]["output_roots"]["qwen3-0.6b"] and h["policy"]["gen_cap"] == 200
        assert h["protected_roots"] == [str(isolated / "heads")]
        assert h["sources"] == [{"kind": "event_log", "enabled": True, "settings": {"stage": "qa"}}]
        assert _errors(c.post("/api/autopilot/targets", json={"name": "x", "kind": "headset", "headset": {
            "preset": "nope", "encoder": "qwen3-0.6b"}}))["headset.preset"].startswith("no head-set preset")
        # the 4B encoder lists heads (and saves) like any other
        got = c.get("/api/autopilot/headsets/test-heads/heads?encoder=qwen3-4b")
        assert got.status_code == 200 and [x["head"] for x in got.json()] == ["h", "g"]
        # an unknown encoder: a 422 whose one line reads as plain words, no field path
        r = c.get("/api/autopilot/headsets/test-heads/heads?encoder=nope")
        assert r.status_code == 422 and r.json()["detail"] == "nope is not a registered encoder; pick one"


def test_without_a_presets_file_there_are_no_presets_and_a_broken_one_is_named(isolated, monkeypatch):  # noqa: F811
    client, auto = make(isolated, FakeAdapter(isolated / "data"))
    with client as c:
        assert c.get("/api/autopilot/headsets").json() == []
        e = _errors(c.post("/api/autopilot/targets", json={"name": "x", "kind": "headset", "headset": {
            "preset": "test-heads", "encoder": "qwen3-0.6b"}}))
        assert "no head-set preset 'test-heads' (known: none" in e["headset.preset"]
        bad = isolated / "broken.json"
        bad.write_text("{not json")
        monkeypatch.setenv("FORGE_HEADSET_PRESETS", str(bad))
        r = c.get("/api/autopilot/headsets")
        assert r.status_code == 500 and str(bad) in r.json()["detail"]


def test_every_preset_encoder_makes_a_target_that_validates(isolated, monkeypatch):  # noqa: F811
    """Whatever encoder the target form starts on (the preset's recommendation) or the user picks, the preset's
    derived output folder saves; a target without an encoder is refused."""
    use_preset(isolated, monkeypatch)
    ad = FakeAdapter(isolated / "data", heads=("h",))
    client, auto = make(isolated, ad)
    with client as c:
        for p in c.get("/api/autopilot/headsets").json():
            rec = p["recommendation"]
            assert rec["key"] in p["encoders"] and rec["reason"], p["key"]
            r = c.post("/api/autopilot/targets", json={"name": f"{p['key']} none", "kind": "headset",
                                                       "headset": {"preset": p["key"]}})
            assert r.status_code == 422 and any(e["field"] == "headset.encoder" for e in r.json()["errors"])
            for i, enc in enumerate([rec["key"], *p["encoders"]]):
                assert not any(p["output_roots"][enc] == r or p["output_roots"][enc].startswith(r + "/")
                               for r in p["protected_roots"]), (p["key"], enc)
                r = c.post("/api/autopilot/targets", json={"name": f"{p['key']} {enc} {i}", "kind": "headset",
                                                           "headset": {"preset": p["key"], "encoder": enc}})
                assert r.status_code == 201, (p["key"], enc, r.text)
                assert r.json()["headset"]["output_root"] == p["output_roots"][enc]


def test_inspect_does_not_check_the_output_folder(tmp_path, monkeypatch):
    """``presets.heads`` validates with ``inspect_only``: a read-only inspect never refuses on the output folder,
    a save still does."""
    import pytest
    from pydantic import ValidationError
    from forge.autopilot import presets
    from forge.autopilot.targets import Target
    use_preset(tmp_path, monkeypatch)
    p = presets.get("test-heads")
    protected = str(tmp_path / "heads")
    body = {"id": "x", "kind": "headset", "headset": {**p.headset("qwen3-0.6b"), "output_root": protected}}
    assert Target.model_validate(body, context={"inspect_only": True}).headset.output_root == protected
    with pytest.raises(ValidationError):
        Target.model_validate(body)
