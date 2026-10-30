"""The autopilot run loop against a fake head-set adapter: bounded rounds, val-only diagnosis, the test split read
once per run (also across a restart), promotion, the writer guard, and the subprocess contract client."""
import asyncio
import sys
import textwrap

import pytest

from forge.autopilot import AutopilotService
from forge.autopilot.adapter import Adapter, AdapterError
from forge.autopilot.targets import Target
from forge.jobqueue import JobQueue, Preflight
from forge.store import Store


def val(f1):
    return {"n": 80, "macro_f1": f1, "accuracy": f1, "per_label": {"a": f1 + 0.05, "b": f1 - 0.05},
            "confusion": {"labels": ["a", "b"], "matrix": [[40, 2], [10, 30]]}}


class FakeAdapter:
    """Speaks the adapter contract (SPEC "Autopilot adapter contract") in-process. ``vals`` is the candidate val
    macro-F1 per training round."""

    def __init__(self, root, vals=(0.7, 0.75, 0.85), test=0.84, incumbent=0.845, interrupt=None, write_rows=False,
                 gate_passed=True, threshold=0.8, heads=("h",), fail_generate=(), generate=None, check=None, same_fp=False, label_undisposed=0, cleanup=None):
        self.root, self.calls, self.check, self.same_fp = root, [], check, same_fp
        self.label_undisposed, self.cleanup = label_undisposed, cleanup
        self.heads, self.fail_generate = list(heads), set(fail_generate)
        self.generate = generate or {"available": True, "reason": None,
                                     "groups": {"a": {"label": "a"}, "b": {"label": "b"}}}
        self.vals, self.test, self.incumbent, self.gate_passed, self.threshold = list(vals), test, incumbent, gate_passed, threshold
        self.synthetic, self.shaped, self.controls = {"a": 300, "b": 60}, {"a": 0, "b": 0}, set()
        self.interrupt, self.write_rows, self.round = interrupt, write_rows, 0

    def fps(self):
        fp = f"fp-{self.shaped['b']}"
        return {"synthetic": "fp-syn", **({"synthetic+shaped": fp} if self.shaped["b"] else {}),
                "all": fp if self.same_fp else fp + "-all"}

    def status(self, head="h"):
        counts = {l: {"synthetic": self.synthetic[l], "shaped": self.shaped[l], "real_labelled": 40 if l == "a" else 10,
                      "real_agent": 40 if l == "a" else 10} for l in ("a", "b")}
        counts["(undisposed)"] = {"real_needs_label": 2}
        fps = self.fps()
        return {"head": head, "group": "classify", "retired": False, "kind": "decision", "counts": counts,
                "teacher_disagreement_rate": 0.05, "excluded_rate": 0.02, "shape": None, "last_run": None,
                "incumbent": {"macro_f1": self.incumbent, "path": "/out/h"} if self.incumbent is not None else None,
                "fingerprints": fps, "controls": {m: (m, fp) in self.controls for m, fp in fps.items() if m != "all"},
                "stale": {"new_real_rows": 3}, "tol": 0.01, "gate_threshold": self.threshold,
                "generate": self.generate, **({"cleanup": self.cleanup} if self.cleanup else {})}

    async def call(self, verb, *, run_id=None, on_progress=None, config_overrides=None, **kw):
        self.calls.append((verb, {**kw, **({"config_overrides": config_overrides} if config_overrides else {})}))
        if self.interrupt and self.interrupt(verb, kw, self):
            raise asyncio.CancelledError()
        if on_progress:
            on_progress({"verb": verb, "message": "working"})
        if verb == "pull-sources":
            return {"decisions": {"file": "x", "rows": 50, "new_rows": 3, "labelled": 0}}
        if verb == "ingest":
            return {"heads": {"h": {"added": 3, "had": 47, "dropped": {"leak": 1}, "counts": {}, "pruned_deleted": 2,
                                    "artifacts_removed": 1,
                                    "same_content": 1, "format_excluded": {"state keys this head build has never seen": 4},
                                    "check": {"stages": {"prod": {"check": 5, "check_labelled": 4}}}}}}
        if verb == "inspect":
            return {"heads": [self.status(h) for h in (kw.get("heads") or self.heads)]}
        if verb == "clean":
            self.cleanup = {"smoke": 0, "stale_guard": 0, "smoke_stages": 0, "guard_version": 2}
            return {"head": kw["head"], "smoke": ["shaped:s1", "shaped:s2"], "reshape": {"real_text": ["shaped:x"]},
                    "checked": 10, "kept": 9, "guard_version": 2, "smoke_stages": ["smoke4-true"]}
        if verb == "label":
            return {"heads": {"h": {"agent_labelled": 1, "adjudicated": 1, "excluded_by_agent": 0,
                                    "undisposed": self.label_undisposed, "teacher_failed": self.label_undisposed,
                                    "export_contradictions": [{"rid": "e1", "export_label": "a", "teacher_label": "b"}]}}}
        if verb == "generate":
            if kw["head"] in self.fail_generate:
                raise AdapterError(f"adapter generate exited 1: ERROR VerbError: {kw['head']}: no real rows",
                                   verb="generate", stderr=f"ERROR VerbError: {kw['head']}: no real rows")
            if self.write_rows:
                (self.root / "h" / "rows.jsonl").write_text("tampered\n")
            self.shaped[kw["label"]] += kw["n"] - 2
            return {"head": "h", "label": kw["label"], "requested": kw["n"], "accepted": kw["n"] - 2,
                    "drops": {"shared_8gram": 1, "teacher": 1}, "batch": kw["batch"]}
        if verb == "train":
            fp = self.fps()[kw["mix"]]
            if kw["mix"] != "all":
                reused = (kw["mix"], fp) in self.controls
                self.controls.add((kw["mix"], fp))
                return {"head": "h", "mix": kw["mix"], "path": f"/out/_controls/{fp}/h", "fingerprint": fp,
                        "val": val(0.72), "reused": reused}
            f1 = self.vals[self.round]
            self.round += 1
            return {"head": "h", "mix": "all", "path": f"/out/_staging/{run_id}/h-r{self.round}", "fingerprint": fp,
                    "attempts": [{"seed": 1, "val": val(f1)}], "chosen": 0, "val": val(f1), "reused": False}
        if verb == "gate":
            return {"passed": self.gate_passed, "test_macro_f1": self.test,
                    "gate_check": {"passed": self.gate_passed, "reasons": ["PASS: fine-tuned macro_f1 beats baselines"]},
                    "incumbent": {"macro_f1": self.incumbent, "path": "/out/h"} if self.incumbent is not None else None,
                    "reference": {"macro_f1": 0.85}, "tol": 0.01, "cached": False,
                    **({"check": self.check} if self.check is not None else {})}
        if verb == "promote":
            return {"head": "h", "path": "/out/h", "replaced": "/out/_history/t/h", "provenance": {"class": "includes_user_data"}}
        if verb == "groups":
            return {"encoder": "qwen3-0.6b-bf16", "groups": {"classify": [
                        {"head": "h", "retired": False, "promoted": True, "encoder": "qwen3-0.6b-bf16", "gate_passed": True,
                         "macro_f1": self.test, "reference_macro_f1": 0.85, "tol": 0.01, "provenance": "includes_user_data"},
                        {"head": "old", "retired": "superseded", "promoted": False}]},
                    "call_cooccurrence": {"h": 3}, "cross_group_calls": {},
                    "shared_encoder": {"setting": "app.heads-encoder", "note": "one shipping decision"}}
        raise AssertionError(verb)


def setup(tmp_path, adapter):
    data = tmp_path / "data"
    (data / "h").mkdir(parents=True)
    (data / "h" / "rows.jsonl").write_text('{"split": "test"}\n')
    store = Store(tmp_path / "f.db")
    svc = AutopilotService(store, JobQueue(store, preflight=lambda: Preflight(True)),
                           adapter_factory=lambda t: adapter)
    svc.targets.put(Target.model_validate({"id": "heads", "kind": "headset", "headset": {
        "adapter": ["true"], "cwd": str(tmp_path), "dataset_root": str(data), "encoder": "qwen3-0.6b-bf16",
        "output_root": str(tmp_path / "out"), "protected_roots": [str(tmp_path / "heads-4b")]}}))
    adapter.root = data
    return svc, store


def verbs(ad, verb):
    return [kw for v, kw in ad.calls if v == verb]


async def test_full_run_rounds_val_only_test_once_and_promotes(tmp_path):
    ad = FakeAdapter(tmp_path)
    svc, store = setup(tmp_path, ad)
    run = svc.enqueue("heads", "manual")
    out = await svc.execute(run["id"])
    assert out["state"] == "done", out["result"].get("error")
    h = out["result"]["heads"]["h"]
    assert h["outcome"] == "promoted" and [r["val_gate"] for r in h["rounds"]] == [False, False, True]
    assert len(verbs(ad, "gate")) == 1 and verbs(ad, "gate")[0]["staging"].endswith("/h-r3")   # test read once
    assert len([kw for kw in verbs(ad, "train") if kw["mix"] == "all"]) == 3            # bounded by max_rounds
    assert verbs(ad, "label")[0] == {"head": "h", "adjudicate": True}
    gens = verbs(ad, "generate")
    assert sum(g["n"] for g in gens if g["batch"].endswith("-1-0")) <= 600                # the round cap
    assert len({g["batch"] for g in gens}) == len(gens)                                   # one batch id per ask
    remedies = [e["counts"] for e in store.journal(target_id="heads", action="remedy")]
    assert any(c["kind"] == "hard_negatives" and c["label"] == "b" and c["focus"] == "a" for c in remedies)
    actions = [e["action"] for e in reversed(store.journal(target_id="heads"))]
    for a in ("queued", "run_started", "pull", "ingest", "need", "label", "export_contradicted", "generate",
              "control", "round", "remedy", "gate", "promote", "groups", "run_done"):
        assert a in actions, a
    gate = store.journal(target_id="heads", action="gate")[0]
    assert gate["counts"]["incumbent"] == 0.845 and gate["counts"]["reference"] == 0.85
    rnd = out["result"]["heads"]["h"]["rounds"][-1]
    assert rnd["control_val_deltas"] == {"synthetic": 0.13, "synthetic+shaped": 0.13}           # vs each control
    controls = [kw for kw in verbs(ad, "train") if kw["mix"] != "all"]
    assert {c["mix"] for c in controls} == {"synthetic", "synthetic+shaped"}
    # like for like: every control trains the candidate's seed set and is chosen on val the same way
    assert {c["attempts"] for c in controls} == {kw["attempts"] for kw in verbs(ad, "train") if kw["mix"] == "all"} == {3}
    groups = out["result"]["groups"]
    assert groups["groups"][0]["ready"] and "app.heads-encoder" in groups["notice"]       # retired head: no block
    assert verbs(ad, "inspect")[0] == {"heads": None} and verbs(ad, "pull-sources")[0]["out"].endswith("_raw/real")
    assert all(e["reason"] for e in store.journal(target_id="heads"))
    # every model step took a slot in the global queue
    kinds = sorted({j["kind"] for j in store.model_jobs()})
    assert kinds == ["gate", "generate", "label", "train"]


async def test_regression_against_incumbent_stays_in_staging(tmp_path):
    ad = FakeAdapter(tmp_path, vals=(0.9,), test=0.82, incumbent=0.86)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    h = out["result"]["heads"]["h"]
    assert h["outcome"] == "rejected" and "regression" in h["why"] and verbs(ad, "promote") == []
    assert "stays in staging (" in store.journal(target_id="heads", action="reject")[0]["reason"]


async def test_a_real_check_regression_blocks_promotion_and_is_journalled(tmp_path):
    check = {"deciding": "prod", "passed": False, "reason": "prod check set (40 rows): macro-F1 0.700 vs incumbent "
             "0.750: regression", "check_min": 30,
             "stages": {"prod": {"n": 40, "candidate": {"macro_f1": 0.7}, "incumbent": {"macro_f1": 0.75}},
                        "qa": {"n": 12, "candidate": {"macro_f1": 0.8}, "incumbent": {"macro_f1": 0.7}}}}
    ad = FakeAdapter(tmp_path, vals=(0.9,), test=0.9, incumbent=0.85, check=check)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    h = out["result"]["heads"]["h"]
    assert h["outcome"] == "rejected" and "real check: prod check set" in h["why"] and verbs(ad, "promote") == []
    gate = store.journal(target_id="heads", action="gate")[0]
    assert gate["counts"]["check"] == {"deciding": "prod", "passed": False, "stages": {
        "prod": {"n": 40, "candidate": 0.7, "incumbent": 0.75}, "qa": {"n": 12, "candidate": 0.8, "incumbent": 0.7}}}
    ingest = store.journal(target_id="heads", action="ingest")[0]
    assert ingest["counts"]["format_excluded"] == {"state keys this head build has never seen": 4}
    assert "excluded by state format 4" in ingest["reason"] and ingest["counts"]["pruned_deleted"] == 2
    assert ingest["counts"]["artifacts_removed"] == 1 and "1 unpromoted heads trained on them" in ingest["reason"]


async def test_no_round_passes_val_gate_test_is_never_read(tmp_path):
    ad = FakeAdapter(tmp_path, vals=(0.5, 0.6, 0.7))
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["result"]["heads"]["h"]["outcome"] == "rejected" and verbs(ad, "gate") == []
    assert "test not read" in store.journal(target_id="heads", action="reject")[0]["reason"]


async def test_restart_resumes_at_cursor_and_never_rereads_test(tmp_path):
    hits = []

    def once_in_round_2(verb, kw, a):        # the server stops while round 2 trains
        if verb == "train" and kw["mix"] == "all" and a.round == 1 and not hits:
            hits.append(1)
            return True
        return False
    ad = FakeAdapter(tmp_path, interrupt=once_in_round_2)
    svc, store = setup(tmp_path, ad)
    rid = svc.enqueue("heads", "manual")["id"]
    with pytest.raises(asyncio.CancelledError):
        await svc.execute(rid)
    assert store.get_run(rid)["state"] == "interrupted"
    before = list(ad.calls)
    svc.restore()                                       # what the next server start does
    assert store.get_run(rid)["state"] == "queued"
    out = await svc.execute(rid)
    assert out["state"] == "done"
    resumed = ad.calls[len(before):]
    assert ("pull-sources", {"out": str(tmp_path / "data" / "_raw" / "real")}) not in resumed   # done steps replayed
    assert [v for v, _ in resumed].count("label") == 0
    assert len(verbs(ad, "gate")) == 1

    # a gate call interrupted mid-way is not repeated in that run
    ad2 = FakeAdapter(tmp_path / "2", interrupt=lambda v, kw, a: v == "gate")
    (tmp_path / "2").mkdir()
    svc2, store2 = setup(tmp_path / "2", ad2)
    rid2 = svc2.enqueue("heads", "manual")["id"]
    with pytest.raises(asyncio.CancelledError):
        await svc2.execute(rid2)
    ad2.interrupt = None
    svc2.restore()
    out2 = await svc2.execute(rid2)
    assert len(verbs(ad2, "gate")) == 1 and out2["result"]["heads"]["h"]["outcome"] == "rejected"
    assert "read once per run" in out2["result"]["heads"]["h"]["why"]


async def test_writer_guard_fails_the_run_when_a_protected_file_changes(tmp_path):
    ad = FakeAdapter(tmp_path, write_rows=True)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "failed" and "WriterGuardError" in out["result"]["error"]
    assert "rows.jsonl" in store.journal(target_id="heads", action="run_failed")[0]["reason"]


async def test_enqueue_coalesces_and_targets_are_validated(tmp_path):
    svc, store = setup(tmp_path, FakeAdapter(tmp_path))
    a = svc.enqueue("heads", "manual")
    assert svc.enqueue("heads", "schedule")["id"] == a["id"]
    with pytest.raises(ValueError, match="inside a protected folder"):
        Target.model_validate({"id": "x", "kind": "headset", "headset": {
            "adapter": ["a"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b", "output_root": "/heads/sub",
            "protected_roots": ["/heads"]}})
    with pytest.raises(ValueError, match="may not set"):
        Target.model_validate({"id": "x", "kind": "headset", "headset": {
            "adapter": ["a"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b", "output_root": "/o",
            "adapter_config": {"encoder": "other"}}})
    with pytest.raises(ValueError, match="stage is a source setting"):
        Target.model_validate({"id": "x", "kind": "headset", "headset": {
            "adapter": ["a"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b", "output_root": "/o",
            "adapter_config": {"stage": "qa"}}})
    with pytest.raises(ValueError, match="not a registered encoder"):
        Target.model_validate({"id": "x", "kind": "headset", "headset": {
            "adapter": ["a"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "e", "output_root": "/o"}})
    t = Target.model_validate({"id": "x", "kind": "headset", "headset": {
        "adapter": ["a"], "cwd": "/tmp", "dataset_root": "/d", "encoder": "qwen3-0.6b-bf16", "output_root": "/o",
        "sources": [{"kind": "event_log", "settings": {"stage": "prod"}}],
        "adapter_config": {"adapter_python": "/py"}, "policy": {"tol_per_head": {"h": 0.02}}}})
    assert t.name == "x"
    assert t.headset.target_config() == {"encoder": "qwen3-0.6b-bf16", "data_root": "/d", "output_root": "/o",
                                         "heads": [], "attempts": 3, "tol": {"h": 0.02}, "adapter_python": "/py",
                                         "sources": [{"kind": "event_log", "enabled": True,
                                                      "settings": {"stage": "prod"}}]}
    with pytest.raises(ValueError, match="headset"):
        Target.model_validate({"id": "x", "kind": "headset"})


FAKE_CLI = textwrap.dedent('''
    import json, sys
    verb = sys.argv[1]
    cfg = json.loads(sys.argv[sys.argv.index("--target-config") + 1])
    if verb == "inspect":
        print(json.dumps({"event": "progress", "verb": verb, "message": "m"}))
        print(json.dumps({"event": "result", "verb": verb, "data": {"heads": [], "encoder": cfg["encoder"],
                                                                     "argv": sys.argv[2:]}}))
    elif verb == "boom":
        print("[claude] attempt 1 failed: the teacher quoted: Hi Alex, my mom is sick", file=sys.stderr)
        print("ERROR VerbError: cannot open dataset", file=sys.stderr); sys.exit(3)
    elif verb == "garbage":
        print("not json")
    elif verb == "twice":
        print(json.dumps({"event": "result", "data": {}})); print(json.dumps({"event": "result", "data": {}}))
    elif verb == "nodata":
        print(json.dumps({"event": "result", "verb": verb}))
    elif verb == "silent":
        pass
''')


async def test_subprocess_adapter_contract(tmp_path, monkeypatch):
    from forge.autopilot import adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "log_dir", lambda: tmp_path / "logs")
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI)
    ad = Adapter([sys.executable, str(cli)], str(tmp_path), {}, {"encoder": "e1"})
    seen = []
    out = await ad.call("inspect", run_id="r1", on_progress=seen.append, heads=["a", "b"], adjudicate=True,
                        skip=False, n=3, focus_template=("t1", "t2"), focus_channel=())
    assert out["encoder"] == "e1" and seen == [{"verb": "inspect", "message": "m"}]
    assert out["argv"][2:] == ["--run", "r1", "--heads", "a,b", "--adjudicate", "--n", "3",
                               "--focus-template", "t1", "--focus-template", "t2"]
    out = await ad.call("inspect", config_overrides={"real_cap": 40})
    assert out["encoder"] == "e1" and ad.target_config == {"encoder": "e1"}
    with pytest.raises(AdapterError, match="no data object"):
        await ad.call("nodata")
    with pytest.raises(AdapterError, match="exited 3: ERROR VerbError: cannot open dataset") as boom:
        await ad.call("boom")
    # only the adapter's ERROR line reaches the error (and so the journal); the rest stays in a local 0600 log
    assert "my mom is sick" not in str(boom.value) and "my mom is sick" not in boom.value.stderr
    log = next((tmp_path / "logs").glob("*-boom.stderr.log"))
    assert "my mom is sick" in log.read_text() and oct(log.stat().st_mode & 0o777) == "0o600"
    assert str(log) in str(boom.value)
    with pytest.raises(AdapterError, match=r"not JSON \(8 characters\)") as garbage:
        await ad.call("garbage")
    assert "'not json'" not in str(garbage.value)
    with pytest.raises(AdapterError, match="second result"):
        await ad.call("twice")
    with pytest.raises(AdapterError, match="without a result"):
        await ad.call("silent")


async def test_real_rows_that_hurt_get_capped_next_round(tmp_path):
    ad = FakeAdapter(tmp_path, vals=(0.6, 0.85))           # round 1 trails the synthetic control (0.72) by > 0.02
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    trains = [kw for kw in verbs(ad, "train") if kw["mix"] == "all"]
    assert "config_overrides" not in trains[0] and trains[1]["config_overrides"]["real_cap"] > 0
    assert any(e["counts"]["kind"] == "cap_real" for e in store.journal(target_id="heads", action="remedy"))
    assert out["result"]["heads"]["h"]["rounds"][1]["real_cap"] == trains[1]["config_overrides"]["real_cap"]


async def test_missing_gate_threshold_fails_the_run_loudly(tmp_path):
    ad = FakeAdapter(tmp_path, threshold=None)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "failed" and "no gate_threshold" in out["result"]["error"]


async def test_one_head_failing_does_not_stop_the_others(tmp_path):
    """A verb that exits non-zero fails that head only: the other heads are labelled, trained and promoted, the
    groups report is built and the run ends done_with_failures with the failure journalled."""
    ad = FakeAdapter(tmp_path, heads=("bad", "h"), fail_generate={"bad"})
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done_with_failures", out["result"].get("error")
    heads = out["result"]["heads"]
    assert heads["bad"]["outcome"] == "failed" and "no real rows" in heads["bad"]["error"]
    assert heads["h"]["outcome"] == "promoted" and "groups" in out["result"]
    failed = store.journal(target_id="heads", action="head_failed")
    assert len(failed) == 1 and failed[0]["head"] == "bad" and "generate failed" in failed[0]["reason"]
    assert "no real rows" in failed[0]["reason"]
    assert store.journal(target_id="heads", action="run_done_with_failures")[0]["counts"]["failed_heads"] == ["bad"]
    assert not [kw for v, kw in ad.calls if v == "train" and kw["head"] == "bad"]


async def test_contract_breaks_still_fail_the_whole_run(tmp_path):
    from forge.autopilot.adapter import AdapterContractError

    class Broken(FakeAdapter):
        async def call(self, verb, **kw):
            if verb == "generate":
                raise AdapterContractError("adapter generate broke the output contract: line 1 is not JSON")
            return await super().call(verb, **kw)
    ad = Broken(tmp_path, heads=("bad", "h"))
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "failed" and "output contract" in out["result"]["error"]


async def test_generation_asks_only_for_groups_inspect_says_it_can_fill(tmp_path):
    ad = FakeAdapter(tmp_path, generate={"available": True, "reason": None, "groups": {
        "a": {"label": "a"}, "b": {"label": None, "reason": "shaped generation cannot place the answer at b"}}})
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done", out["result"].get("error")
    assert {g["label"] for g in verbs(ad, "generate")} <= {"a"}
    skips = [e for e in store.journal(target_id="heads", action="skip") if "no generate label" in e["reason"]]
    assert skips and "cannot place the answer at b" in skips[0]["reason"] and skips[0]["counts"]["b"] > 0
    needs = [e["counts"] for e in store.journal(target_id="heads", action="need") if e["counts"]["kind"] == "small_data"]
    assert any(c["remedy"] == "none" for c in needs)
    assert all(e["counts"].get("b") is None for e in store.journal(target_id="heads", action="carry"))


async def test_no_generation_for_a_head_generate_cannot_serve(tmp_path):
    ad = FakeAdapter(tmp_path, generate={"available": False, "reason": "h has no real rows to take the shape from",
                                         "groups": {"a": {"label": "a"}, "b": {"label": "b"}}})
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done", out["result"].get("error")
    assert verbs(ad, "generate") == [] and out["result"]["heads"]["h"]["outcome"] == "promoted"
    skip = [e for e in store.journal(target_id="heads", action="skip") if "no generation for this head" in e["reason"]]
    assert skip and "no real rows to take the shape from" in skip[0]["reason"]


async def test_plan_timings_and_first_started_survive_a_resume(tmp_path):
    """The plan is written once (after the first inspect), every step records {s, e}, heads record started /
    ended, the run records its encoder, and a resumed run keeps first_started while started moves on."""
    from forge.autopilot.runner import PHASES
    ad = FakeAdapter(tmp_path, heads=("h", "g"), vals=(0.85, 0.85, 0.85),
                     interrupt=lambda v, kw, a: v == "train" and kw.get("head") == "g")
    svc, store = setup(tmp_path, ad)
    rid = svc.enqueue("heads", "manual")["id"]
    with pytest.raises(asyncio.CancelledError):
        await svc.execute(rid)
    mid = store.get_run(rid)
    plan = mid["result"]["plan"]
    assert plan["heads"] == ["h", "g"] and plan["phases"] == PHASES and plan["started_at"]
    assert mid["first_started"] == mid["started"] and mid["result"]["encoder"] == "qwen3-0.6b-bf16"
    t = mid["result"]["timings"]
    assert t["pull"]["s"] <= t["pull"]["e"] and t["inspect"]["e"]
    open_steps = [k for k, v in t.items() if v["e"] is None]
    assert len(open_steps) == 1 and open_steps[0].startswith("control:g:1")      # interrupted mid-step
    h = mid["result"]["heads"]["h"]
    assert h["outcome"] == "promoted" and h["started"] <= h["ended"]
    assert mid["result"]["heads"]["g"]["started"] and "ended" not in mid["result"]["heads"]["g"]
    ad.interrupt = None
    svc.restore()
    await asyncio.sleep(0.01)
    out = await svc.execute(rid)
    assert out["state"] == "done", out["result"].get("error")
    end = store.get_run(rid)
    assert end["first_started"] == mid["first_started"] and end["started"] > mid["started"]
    assert end["result"]["plan"] == plan                                            # written once
    assert all(v["e"] for v in end["result"]["timings"].values())
    assert end["result"]["heads"]["g"]["ended"] and end["result"]["heads"]["h"]["ended"] == h["ended"]


async def test_a_control_with_the_candidates_training_set_reports_delta_zero(tmp_path):
    """No real row reaches train: the candidate's fingerprint is the synthetic+shaped control's; the delta to it is
    0 by definition (the adapter copies the control), never seed noise."""
    ad = FakeAdapter(tmp_path, vals=(0.85,), same_fp=True)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    rnd = out["result"]["heads"]["h"]["rounds"][0]
    assert rnd["same_as_controls"] == ["synthetic+shaped"]
    assert rnd["control_val_deltas"] == {"synthetic": 0.13, "synthetic+shaped": 0.0}
    j = store.journal(target_id="heads", action="round")[0]
    assert "same training set as the synthetic+shaped control" in j["reason"]


async def test_a_label_step_that_leaves_rows_undisposed_fails_the_head(tmp_path):
    """Undisposed rows after labelling are a failure: the head is not trained and the journal says why."""
    ad = FakeAdapter(tmp_path, label_undisposed=7)
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done_with_failures"
    assert out["result"]["heads"]["h"]["outcome"] == "failed" and not verbs(ad, "train")
    j = store.journal(target_id="heads", action="head_failed")[0]
    assert "label left 7 rows undisposed" in j["reason"] and j["counts"]["verb"] == "label"


async def test_smoke_rows_and_a_guard_bump_are_cleaned_before_training(tmp_path):
    ad = FakeAdapter(tmp_path, vals=(0.85,), cleanup={"smoke": 2, "stale_guard": 10, "guard_version": 2})
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done", out["result"].get("error")
    names = [v for v, _ in ad.calls]
    assert names.index("clean") < names.index("train") and verbs(ad, "clean") == [{"head": "h"}]
    needs = {e["counts"]["kind"] for e in store.journal(target_id="heads", action="need")}
    assert {"smoke_rows", "reshape"} <= needs
    j = store.journal(target_id="heads", action="clean")[0]
    assert j["counts"] == {"smoke": 2, "smoke_stages": 1, "reshape": {"real_text": 1}, "checked": 10, "kept": 9}
    assert "removed 2 smoke rows, 1 smoke stage folders and 1 rows" in j["reason"]


async def test_smoke_stage_folders_alone_are_a_need_and_get_cleaned(tmp_path):
    """Smoke rows cleaned by an earlier version can leave their generation stage folders behind: still a need."""
    ad = FakeAdapter(tmp_path, vals=(0.85,), cleanup={"smoke": 0, "stale_guard": 0, "smoke_stages": 3,
                                                      "guard_version": 2})
    svc, store = setup(tmp_path, ad)
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done", out["result"].get("error")
    need = [e for e in store.journal(target_id="heads", action="need") if e["counts"]["kind"] == "smoke_rows"][0]
    assert need["counts"]["smoke_stages"] == 3 and "3 generation stage folders" in need["reason"]
    assert verbs(ad, "clean") == [{"head": "h"}]


def test_the_journal_cuts_any_long_text():
    from forge.autopilot.journal import REASON_MAX, VALUE_MAX, bounded
    assert bounded("x" * (VALUE_MAX + 5)).endswith(f"({VALUE_MAX + 5} chars)")
    assert bounded({"a": ["y" * 999, 3]})["a"][1] == 3 and len(bounded({"a": ["y" * 999]})["a"][0]) < 260
    assert len(bounded("z" * 5000, REASON_MAX)) < REASON_MAX + 30


async def test_the_failing_incumbent_policy_is_journalled_and_asks_promote_to_recheck(tmp_path):
    ad = FakeAdapter(tmp_path, vals=(0.85,), test=0.7, incumbent=0.6, gate_passed=False)
    orig = ad.call

    async def call(verb, **kw):
        out = await orig(verb, **kw)
        if verb == "gate":
            out["incumbent"]["gate_passed"] = False
        return out
    ad.call = call
    svc, store = setup(tmp_path, ad)
    t = svc.targets.get("heads")
    svc.targets.put(t.model_copy(update={"headset": t.headset.model_copy(update={
        "policy": t.headset.policy.model_copy(update={"promote_over_failing_incumbent": True})})}))
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["result"]["heads"]["h"]["outcome"] == "promoted"
    assert verbs(ad, "promote")[0]["over_failing_incumbent"] is True
    j = store.journal(target_id="heads", action="gate_policy")[0]
    assert j["counts"]["policy"] == "promote_over_failing_incumbent" and "0.700 > 0.600" in j["reason"]
