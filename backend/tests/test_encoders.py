"""The encoder registry (backend/encoders.json, forge.encoders): its entries, the HF-cache check, spec lookup, who uses what, no app-wide encoder (settings, health), the per-experiment encoder migration,
GET /api/encoders and the per-project encoder. No model is loaded."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient  # noqa: F401

import forge_client  # noqa: E402

from forge import encoders
from test_agent_fakes import FakeLLM, isolated  # noqa: F401  (fixture)


def test_registry_entries_are_pinned_and_unique():
    mlx = {e.key for e in encoders.all() if e.backend == "mlx"}
    assert {"qwen3-4b", "qwen3-0.6b", "qwen3-0.6b-8bit", "qwen3-0.6b-bf16"} <= mlx
    keys = [e.key for e in encoders.all()]
    assert {"qwen3-4b-gguf-q4km", "qwen3-8b-mlx-4bit"} <= set(keys) and len(keys) == len(set(keys))
    for e in encoders.all():
        assert len(e.revision) == 40 and e.hidden in (1024, 2560, 3840, 4096)
        assert e.approx_gb is None or (e.approx_gb > 0 and e.approx_gb_source)   # measured, never guessed


def test_spec_grammar_lookup_and_output_root():
    q4 = encoders.get("qwen3-4b-gguf-q4km")
    assert q4.spec == "llamacpp:Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"
    assert encoders.get("qwen3-0.6b-bf16").spec == "mlx:mlx-community/Qwen3-0.6B-bf16"
    assert encoders.by_spec(q4.spec) is q4 or encoders.by_spec(q4.spec) == q4
    assert encoders.by_spec("hf:Qwen/Qwen3-8B") is None and encoders.by_spec(None) is None
    with pytest.raises(encoders.UnknownEncoder, match="not a registered encoder"):
        encoders.get("nope")
    assert encoders.output_root("/x/heads", "qwen3-4b") == "/x/heads-qwen3-4b"      # never the base itself
    assert encoders.output_root("/x/heads", "qwen3-0.6b") == "/x/heads-qwen3-0.6b"
    assert encoders.name_of_spec("hf:Qwen/Qwen3-8B") == "Qwen3-8B"


def test_cached_checks_the_pinned_snapshot_in_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    mlx, gguf = encoders.get("qwen3-0.6b"), encoders.get("qwen3-4b-gguf-q4km")
    assert not encoders.cached(mlx) and not encoders.cached(gguf)
    snap = tmp_path / "hub" / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / mlx.revision
    snap.mkdir(parents=True)
    assert not encoders.cached(mlx)                         # an empty snapshot dir is not a download
    (snap / "config.json").write_text("{}")
    assert encoders.cached(mlx)
    other = tmp_path / "hub" / "models--Qwen--Qwen3-4B-GGUF" / "snapshots" / ("0" * 40)
    other.mkdir(parents=True)
    (other / gguf.file).write_text("x")
    assert not encoders.cached(gguf)                        # another revision is not the pinned one
    pinned = tmp_path / "hub" / "models--Qwen--Qwen3-4B-GGUF" / "snapshots" / gguf.revision
    pinned.mkdir(parents=True)
    assert not encoders.cached(gguf)                        # the GGUF file itself must be there
    (pinned / gguf.file).write_text("x")
    assert encoders.cached(gguf)


def _orch():
    from forge import config
    from forge.autopilot import AutopilotService
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    orch = Orchestrator(store=Store(config.DB_PATH))
    orch._llm_override = FakeLLM()
    return orch, AutopilotService(orch.store, orch.jobs, emit=orch.emit, orchestrator=orch,
                                  adapter_factory=lambda t: None)


def test_used_by_groups_projects_and_targets(isolated):  # noqa: F811
    from forge.autopilot.targets import Target
    orch, auto = _orch()
    q4 = encoders.get("qwen3-4b-gguf-q4km")
    orch.store.save_project({"id": "p1", "name": "Tickets", "base_model": q4.spec})
    orch.store.save_project({"id": "p2", "name": "Not chosen yet", "base_model": None})
    auto.targets.put(Target.model_validate({"id": "heads", "name": "Test heads", "kind": "headset", "headset": {
        "adapter": ["x"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b-bf16", "output_root": "/o"}}))
    auto.targets.put(Target.model_validate({"id": "tp", "name": "Tickets nightly", "kind": "project",
                                            "project": {"project_id": "p1"}}))
    u = encoders.used_by(orch.store, auto.targets)
    assert u["qwen3-4b-gguf-q4km"] == {"projects": [{"id": "p1", "name": "Tickets"}],
                                       "targets": [{"id": "tp", "name": "Tickets nightly"}]}
    assert u["qwen3-0.6b-bf16"] == {"projects": [], "targets": [{"id": "heads", "name": "Test heads"}]}
    assert u["qwen3-4b"] == {"projects": [], "targets": []}


def test_settings_have_no_app_wide_encoder(isolated):  # noqa: F811
    """The app-wide encoder fields leave settings.json (kept in the backup the migration reads); a PUT naming one is
    refused; GET /api/encoders has no default."""
    from forge import config
    from forge.api import create_app
    legacy = {"provider": "ollama", "default_encoder": "qwen3-4b-gguf-q4km", "encoder": "mlx:x",
              "base_models": ["qwen3-4b-gguf-q4km", "qwen3-8b-mlx-4bit"], "unregistered_base_models": []}
    config.SETTINGS_PATH.write_text(json.dumps(legacy))
    s = config.load_settings()
    assert not hasattr(s, "default_encoder") and not hasattr(s, "base_models") and s.provider == "ollama"
    raw = json.loads(config.SETTINGS_PATH.read_text())
    assert not set(raw) & set(config.LEGACY_ENCODER_FIELDS)
    assert json.loads(config.legacy_backup().read_text()) == legacy       # what the encoder migration reads
    assert config.legacy_settings() == legacy
    orch, auto = _orch()
    with forge_client.client(create_app(orch, auto)) as c:
        got = c.get("/api/settings").json()
        assert not {"default_encoder", "base_models", "encoder_error"} & set(got)
        r = c.put("/api/settings", json={"default_encoder": "qwen3-4b"})
        assert r.status_code == 422 and "no app-wide encoder" in r.json()["detail"]
        enc = c.get("/api/encoders").json()
        assert set(enc) == {"encoders"}
        q4 = next(e for e in enc["encoders"] if e["key"] == "qwen3-4b")
        assert q4["spec"] == "mlx:mlx-community/Qwen3-4B-4bit" and q4["name"] == "Qwen3-4B (MLX 4-bit)"
        assert set(q4) >= {"backend", "quant", "approx_gb", "cached", "used_by", "revision", "hidden"}


def test_create_requires_an_encoder_key(isolated):  # noqa: F811
    from forge.api import create_app
    orch, auto = _orch()
    with forge_client.client(create_app(orch, auto)) as c:
        assert c.post("/api/projects", json={"task": "route support tickets to teams"}).status_code == 422
        r = c.post("/api/projects", json={"task": "route support tickets to teams", "encoder": "nope"})
        assert r.status_code == 422 and "not a registered encoder" in r.json()["detail"]
        p = c.post("/api/projects", json={"task": "route support tickets to teams", "provider": "ollama",
                                          "model": "fake", "encoder": "qwen3-4b-gguf-q4km"}).json()
        d = c.get(f"/api/projects/{p['id']}").json()
        assert d["base_model"] == encoders.get("qwen3-4b-gguf-q4km").spec
        assert d["encoder"]["key"] == "qwen3-4b-gguf-q4km" and d["encoder"]["source"]["from"] == "create"


def test_health_reports_every_encoder_in_use(isolated):  # noqa: F811
    from forge.api import create_app
    from forge.autopilot.targets import Target
    orch, auto = _orch()
    orch.store.save_project({"id": "p1", "name": "Tickets", "base_model": encoders.get("qwen3-4b-gguf-q4km").spec})
    orch.store.save_project({"id": "p2", "name": "Odd", "base_model": "hf:Qwen/Qwen3-8B"})
    auto.targets.put(Target.model_validate({"id": "heads", "name": "Test heads", "kind": "headset", "headset": {
        "adapter": ["x"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b-bf16", "output_root": "/o"}}))
    with forge_client.client(create_app(orch, auto)) as c:
        h = c.get("/api/health").json()
    assert "encoder" not in h
    assert [e["key"] for e in h["encoders"]] == ["qwen3-0.6b-bf16", "qwen3-4b-gguf-q4km"]
    e = h["encoders"][1]
    assert e["loaded"] is False and e["used_by"]["projects"] == [{"id": "p1", "name": "Tickets"}]
    assert h["unregistered_encoders"] == ["hf:Qwen/Qwen3-8B"]


def _ckpt(path: Path, cfg: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_head": {}, "action_head": {}, "logit_scale": torch.tensor(1.0), "cfg": cfg}, path)


def _cfg(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)["cfg"]


def test_migration_gives_every_experiment_its_encoder_once(isolated):  # noqa: F811
    """Experiments without an encoder get the one their checkpoint was trained on, else the app-wide one the
    settings named at migration time; checkpoints without embed_model get theirs backfilled; recorded, once."""
    from forge import config, encoder_migration
    pytest.importorskip("torch")
    config.SETTINGS_PATH.write_text(json.dumps({"default_encoder": "qwen3-4b-gguf-q4km", "base_models": ["x"]}))
    config.load_settings()                                   # strips the file first: the backup still names it
    orch, _auto = _orch()
    q06, q4g = encoders.get("qwen3-0.6b"), encoders.get("qwen3-4b-gguf-q4km")
    orch.store.save_project({"id": "ck", "name": "From ckpt", "base_model": None, "encoder_pinned": False})
    _ckpt(config.project_dir("ck") / "best_head.pt", {"embed_model": q06.spec})
    orch.store.save_project({"id": "st", "name": "From settings", "base_model": None})
    orch.store.save_project({"id": "old", "name": "Old ckpt", "base_model": q06.spec, "encoder_pinned": True})
    _ckpt(config.project_dir("old") / "best_head.pt", {"width": 8})
    _ckpt(config.project_dir("old") / "package" / "clm_x" / "head.pt", {"width": 8})
    _ckpt(config.project_dir("old") / "selection" / q4g.name / "best_head.pt", {"width": 8})
    _ckpt(config.project_dir("old") / "selection" / "Mystery 9B" / "best_head.pt", {"width": 8})
    out = encoder_migration.migrate(orch.store, orch.emit)
    assert out["projects"] == {"ck": {"from": "checkpoint", "spec": q06.spec},
                               "st": {"from": "settings", "spec": q4g.spec}}
    recs = {r["id"]: r for r in orch.store.list_projects()}
    assert recs["ck"]["base_model"] == q06.spec and recs["st"]["base_model"] == q4g.spec
    assert recs["st"]["encoder_source"]["detail"] == "settings default_encoder qwen3-4b-gguf-q4km"
    assert not any("encoder_pinned" in r for r in recs.values())
    old = config.project_dir("old")
    assert _cfg(old / "best_head.pt")["embed_model"] == q06.spec
    assert _cfg(old / "package" / "clm_x" / "head.pt")["embed_model"] == q06.spec
    assert _cfg(old / "selection" / q4g.name / "best_head.pt")["embed_model"] == q4g.spec
    assert "embed_model" not in _cfg(old / "selection" / "Mystery 9B" / "best_head.pt")
    assert any("Mystery 9B" in k for k in out["skipped"])
    logs = [e["message"] for e in orch.store.events("st") if e["type"] == "log" and e.get("stage") == "migration"]
    assert logs and "from the settings" in logs[0]
    assert encoder_migration.migrate(orch.store, orch.emit) is None          # once per database
    # a checkpoint's own encoder is what inference reads; one without any is refused, never defaulted
    assert encoders.head_spec({"embed_model": q06.spec}, reference=False) == q06.spec
    with pytest.raises(encoders.CheckpointEncoderMissing, match="records no embed_model"):
        encoders.head_spec({}, reference=False, path="/x/best_head.pt")


def test_migration_without_any_source_leaves_the_pick_to_the_user(isolated):  # noqa: F811
    from forge import encoder_migration
    from forge.orchestrator import NoEncoder
    orch, _auto = _orch()
    orch.store.save_project({"id": "none", "name": "Nothing known", "base_model": None})
    out = encoder_migration.migrate(orch.store, orch.emit)
    assert out["projects"] == {"none": {"error": "the settings named no encoder"}}
    warn = [e for e in orch.store.events("none") if e["type"] == "log" and e.get("level") == "warn"]
    assert warn and "pick this experiment's encoder" in warn[0]["message"]
    from forge.orchestrator import Runtime
    from forge.rules import Project
    orch.rt["none"] = Runtime(rec=orch.store.get_project("none"), machine=Project("none"))
    with pytest.raises(NoEncoder, match="pick one"):
        orch.encoder_spec("none")


def test_project_encoder_endpoint(isolated):  # noqa: F811
    from forge.api import create_app
    from forge.autopilot.targets import Target
    from test_api_projects import seed_labelling_project
    from forge.orchestrator import Runtime
    from forge.rules import Project
    pid = seed_labelling_project("pe1", state="label_review", policy="user")
    orch, auto = _orch()
    with forge_client.client(create_app(orch, auto)) as c:
        auto.targets.put(Target.model_validate({"id": "tp", "name": "Labels nightly", "kind": "project",
                                                "project": {"project_id": pid}}))
        # a stage job running on the project: the encoder can't change under it
        orch.rt["busy"] = Runtime(rec={"id": "busy", "name": "busy", "base_model": None},
                                  machine=Project("busy", start_value="training"))
        r = c.put("/api/projects/busy/encoder", json={"key": "qwen3-4b"})
        assert r.status_code == 409 and "training" in r.json()["detail"]
        del orch.rt["busy"]
        assert c.put(f"/api/projects/{pid}/encoder", json={"key": "nope"}).status_code == 422
        d = c.put(f"/api/projects/{pid}/encoder", json={"key": "qwen3-0.6b"}).json()
        assert d["base_model"] == "mlx:mlx-community/Qwen3-0.6B-4bit"
        assert {k: v for k, v in d["encoder"].items() if k != "source"} == {
            "key": "qwen3-0.6b", "name": "Qwen3-0.6B (MLX 4-bit)", "spec": "mlx:mlx-community/Qwen3-0.6B-4bit"}
        assert d["encoder"]["source"]["from"] == "user"
        evs = [e for e in orch.store.events(pid) if e["type"] == "encoder_changed"]
        assert evs and evs[-1]["to_name"] == "Qwen3-0.6B (MLX 4-bit)" and evs[-1]["from_name"] == "Qwen3-4B (MLX 4-bit)"
        j = c.get("/api/autopilot/journal?target=tp&action=encoder_changed").json()
        assert len(j) == 1 and "to Qwen3-0.6B (MLX 4-bit)" in j[0]["reason"] and j[0]["counts"]["to"] == "qwen3-0.6b"
        used = {e["key"]: e["used_by"] for e in c.get("/api/encoders").json()["encoders"]}
        assert {"id": pid, "name": "labels"} in used["qwen3-0.6b"]["projects"]
        assert used["qwen3-0.6b"]["targets"] == [{"id": "tp", "name": "Labels nightly"}]
        # base-model selection trains on the experiment's own encoder
        assert orch._candidates(pid) == ["mlx:mlx-community/Qwen3-0.6B-4bit"]
