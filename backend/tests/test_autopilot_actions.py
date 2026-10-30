"""One-time autopilot actions (forge.autopilot.actions) and the runner's resplit_val action: scheduled or manual runs
only, journalled per head, the writer guard lets it create only the new val files, failures keep it queued. No model,
no LLM call (a fake adapter)."""
import json

import pytest

from forge.autopilot import AutopilotService
from forge.autopilot.adapter import AdapterError
from forge.autopilot.targets import Target
from forge.jobqueue import JobQueue, Preflight
from forge.store import Store
from test_autopilot_runner import FakeAdapter, verbs

#: dataset files no step may change: the assembled rows, and a redrawn val with its manifest and the archived old val
FROZEN_GLOBS = ("*/rows.jsonl", "*/val_redrawn.jsonl", "*/val_split.json", "*/_val_archive/*")


class ResplitAdapter(FakeAdapter):
    """The fake adapter plus ``resplit-val``: writes the new val files the way an adapter's ``resplit-val``
    does (``also`` names another file it would wrongly change; ``fail`` makes it exit non-zero)."""

    def __init__(self, root, also=None, fail=False, **kw):
        super().__init__(root, **kw)
        self.also, self.fail = also, fail

    async def call(self, verb, *, run_id=None, on_progress=None, config_overrides=None, **kw):
        if verb != "resplit-val":
            return await super().call(verb, run_id=run_id, on_progress=on_progress, config_overrides=config_overrides,
                                      **kw)
        self.calls.append((verb, kw))
        if self.fail:
            raise AdapterError("adapter resplit-val exited 1: ERROR VerbError: 3 generation batches failed",
                               verb="resplit-val", stderr="ERROR VerbError: 3 generation batches failed")
        out = {}
        for h in kw["heads"]:
            d = self.root / h
            (d / "_val_archive").mkdir(parents=True, exist_ok=True)
            (d / "_val_archive" / f"val-{run_id}.jsonl").write_text('{"split": "val", "id": "old"}\n')
            (d / "val_redrawn.jsonl").write_text('{"split": "val", "id": "new"}\n')
            (d / "val_split.json").write_text(json.dumps({"sha256": "x"}))
            out[h] = {"applicable": True, "cached": False, "procedure": "gen.py --pool valtest (Codex, TEST_SYSTEM)",
                      "val": {"file": "val_redrawn.jsonl", "rows": 1, "labels": {"a": 1}, "sha256": "x"},
                      "archived": {"file": f"_val_archive/val-{run_id}.jsonl", "rows": 1, "sha256": "y"},
                      "leak_check": {"near_test": 2}, "test_sha256": "abc123"}
        if self.also:
            (self.root / self.also).write_text("changed\n")
        return {"heads": out}


def _svc(tmp_path, adapter, tid="heads", policy=None):
    data = tmp_path / "data"
    (data / "h").mkdir(parents=True, exist_ok=True)
    (data / "h" / "rows.jsonl").write_text('{"split": "test"}\n')
    store = Store(tmp_path / "f.db")
    svc = AutopilotService(store, JobQueue(store, preflight=lambda: Preflight(True)), adapter_factory=lambda t: adapter)
    svc.targets.put(Target.model_validate({"id": tid, "kind": "headset", **_cfg(tmp_path, data, policy)}))
    adapter.root = data
    return svc, store


def _cfg(tmp_path, data, policy):
    return {"name": "Test heads", "headset": {
        "adapter": ["true"], "cwd": str(tmp_path), "dataset_root": str(data), "encoder": "qwen3-0.6b-bf16",
        "output_root": str(tmp_path / "out"), "protected_roots": [str(tmp_path / "heads-4b")],
        "protected_globs": list(FROZEN_GLOBS), **({"policy": policy} if policy else {})}}


def test_nothing_is_queued_or_changed_unless_asked(tmp_path):
    svc, _ = _svc(tmp_path, ResplitAdapter(tmp_path))
    assert svc.targets.get("heads").headset.policy.promote_over_failing_incumbent is False
    assert svc.actions.all() == []


def test_a_queued_action_is_journalled_and_shown_on_the_target(tmp_path):
    svc, store = _svc(tmp_path, ResplitAdapter(tmp_path))
    svc.actions.queue("heads", "resplit_val", "approved by test", "val drawn from another distribution")
    [q] = svc.journal.query(target_id="heads", action="action_queued")
    assert q["counts"]["action"] == "resplit_val" and q["counts"]["runs_in"] == ["schedule", "manual"]
    assert svc.target_view(svc.targets.get("heads"))["pending_actions"][0]["approved"] == "approved by test"
    again = AutopilotService(store, JobQueue(store, preflight=lambda: Preflight(True)))   # a restart keeps it queued
    assert [a["state"] for a in again.actions.all("heads")] == ["queued"]


async def test_resplit_val_waits_out_a_data_run_then_runs_once(tmp_path):
    ad = ResplitAdapter(tmp_path)
    svc, _ = _svc(tmp_path, ad)
    a = svc.actions.queue("heads", "resplit_val", "approved by test", "val drawn from another distribution")
    out = await svc.execute(svc.enqueue("heads", "data")["id"])
    assert out["state"] == "done" and verbs(ad, "resplit-val") == []
    waiting = svc.journal.query(target_id="heads", action="action_waiting")
    assert len(waiting) == 1 and "scheduled or manual run" in waiting[0]["reason"]
    ad.round = 0
    out = await svc.execute(svc.enqueue("heads", "manual")["id"])
    assert out["state"] == "done", out["result"].get("error")
    assert verbs(ad, "resplit-val") == [{"heads": ["h"]}]
    steps = list(out["result"]["steps"])
    assert steps.index(f"action:{a['id']}") < steps.index("inspect:after_actions") < steps.index("needs:h")
    per_head = svc.journal.query(target_id="heads", action="resplit_val")
    assert len(per_head) == 1 and per_head[0]["head"] == "h" and "archived" in per_head[0]["reason"]
    assert per_head[0]["counts"]["leak_check"] == {"near_test": 2}
    assert svc.journal.query(target_id="heads", action="action_done")
    done = svc.actions.all("heads")[0]
    assert done["state"] == "done" and done["run_id"] == out["id"] and done["result"] == {"heads": ["h"]}
    # the new files are protected from then on
    assert str(ad.root / "h" / "val_redrawn.jsonl") in out["result"]["protected"]
    ad.round = 0
    await svc.execute(svc.enqueue("heads", "schedule")["id"])
    assert len(verbs(ad, "resplit-val")) == 1                       # once


async def test_resplit_val_may_create_only_its_files(tmp_path):
    ad = ResplitAdapter(tmp_path, also="h/rows.jsonl")
    svc, _ = _svc(tmp_path, ad)
    svc.actions.queue("heads", "resplit_val", "approved by test", "why")
    out = await svc.execute(svc.enqueue("heads", "schedule")["id"])
    assert out["state"] == "failed" and "WriterGuardError" in out["result"]["error"]
    assert "rows.jsonl" in out["result"]["error"] and "val_redrawn" not in out["result"]["error"]
    assert svc.actions.pending("heads")                              # not done: the next scheduled run retries


async def test_a_failed_resplit_fails_the_run_and_stays_queued(tmp_path):
    ad = ResplitAdapter(tmp_path, fail=True)
    svc, _ = _svc(tmp_path, ad)
    svc.actions.queue("heads", "resplit_val", "approved by test", "why")
    out = await svc.execute(svc.enqueue("heads", "schedule")["id"])
    assert out["state"] == "failed" and verbs(ad, "train") == []     # never trained on the val it replaces
    [a] = svc.actions.pending("heads")
    assert "generation batches failed" in a["last_error"]
    assert svc.journal.query(target_id="heads", action="action_failed")


def test_unknown_actions_are_refused(tmp_path):
    svc, _ = _svc(tmp_path, ResplitAdapter(tmp_path))
    with pytest.raises(ValueError, match="unknown action"):
        svc.actions.queue("heads", "retrain_everything", "x", "y")
