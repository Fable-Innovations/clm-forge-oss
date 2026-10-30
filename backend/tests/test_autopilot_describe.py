"""forge.autopilot.describe: step labels, progress fraction and ETA, timeline states, row counts. Pure functions over
a stored run record (the shape the runner writes, copied from a live head-set run)."""
from datetime import datetime, timedelta, timezone

from forge.autopilot import describe
from forge.autopilot.runner import PHASES


def iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def run(state="running", **res):
    """Three heads: a promoted one (40 min), a skipped one, and ticket-escalation training round 1 right now."""
    steps = {"pull": {"decisions": {"new_rows": 3}}, "ingest": {"heads": {}}, "inspect": {"heads": []},
             "needs:a": [], "label:a": {"heads": {"a": {"agent_labelled": 12, "excluded_by_agent": 2}}},
             "inspect:a:1": {}, "gen:a:1:0": {"accepted": 90}, "inspect:a:1:post": {}, "carry:a:1": {"x": 4},
             "control:a:1:synthetic": {}, "train:a:1": {}, "gate:a": {}, "decision:a": {}, "promote:a": {},
             "needs:b": [],
             "needs:ticket-escalation": [], "label:ticket-escalation": {"heads": {"ticket-escalation": {
                 "agent_labelled": 5, "excluded_by_agent": 1}}},
             "inspect:ticket-escalation:1": {}, "gen:ticket-escalation:1:0": {"accepted": 600},
             "inspect:ticket-escalation:1:post": {}, "control:ticket-escalation:1:synthetic": {}}
    timings = {k: {"s": iso(70), "e": iso(69)} for k in steps if k.split(":")[0] not in ("needs", "carry")}
    timings["train:ticket-escalation:1"] = {"s": iso(2), "e": None}
    heads = {"a": {"outcome": "promoted", "started": iso(68), "ended": iso(28)},
             "b": {"outcome": "skipped", "started": iso(28), "ended": iso(27.5)},
             "ticket-escalation": {"started": iso(27), "rounds": []}}
    plan = {"heads": ["a", "b", "ticket-escalation", "d"], "phases": PHASES, "started_at": iso(70)}
    return {"id": "r1", "state": state, "first_started": iso(70), "started": iso(10), "ended": None,
            "result": {"steps": steps, "timings": timings, "heads": heads, "plan": plan, **res}}


def test_step_labels():
    assert describe.step_label("train:ticket-escalation:1") == "Training ticket-escalation (round 1)"
    assert describe.step_label("gen:h:1:0") == "Generating rows for h (round 1)"
    assert describe.step_label("control:h:2:synthetic+shaped") == "Training the synthetic+shaped control for h (round 2)"
    assert describe.step_label("inspect:h:1:post") == "Re-inspecting h after generation (round 1)"
    assert describe.step_label("gate:h") == "Gating h on the test split"
    assert describe.step_label("pull") == "Pulling real data" and describe.step_label("groups") == "Checking call-group readiness"
    assert describe.phase_of("control:h:1:synthetic") == "controls" and describe.phase_of("pull") is None


def test_progress_fraction_current_step_and_eta():
    p = describe.progress(run(), detail="training seed 2 of 3")
    # units: 3 prelude + 4 heads + 1 groups = 8; done: 3 prelude + 2 heads; ticket-escalation reached "train" (4/8)
    assert p["heads_done"] == 2 and p["heads_total"] == 4
    assert abs(p["fraction"] - (3 + 2 + 4 / 8) / 8) < 1e-3
    assert p["current"] == {"label": "Training ticket-escalation (round 1)", "head": "ticket-escalation"}
    assert p["detail"] == "training seed 2 of 3"
    assert 69 * 60 <= p["elapsed_s"] <= 71 * 60                       # from first_started, not this execution
    assert abs(p["eta_s"] - 40 * 60 * 2) < 5 and p["eta_note"] is None   # mean of non-skipped heads x 2 remaining


def test_eta_waits_for_the_first_head_and_finished_runs_are_whole():
    r = run()
    r["result"]["heads"]["a"].pop("outcome")
    r["result"]["heads"]["b"]["outcome"] = "skipped"
    p = describe.progress(r)
    assert p["eta_s"] is None and p["eta_note"] == "ETA after the first head"
    done = run(state="done")
    done["ended"] = iso(0)
    for h in ("ticket-escalation", "d"):
        done["result"]["heads"][h] = {"outcome": "rejected", "started": iso(20), "ended": iso(1)}
    done["result"]["steps"]["groups"] = {}
    p = describe.progress(done)
    assert p["fraction"] == 1.0 and p["current"] is None and p["eta_s"] is None and p["detail"] is None


def test_a_run_without_a_plan_has_no_progress():
    r = run()
    r["result"].pop("plan")
    assert describe.progress(r) is None


def test_timeline_states():
    tl = describe.timeline(run())
    by = {x["key"]: x for x in tl}
    assert by["train:a:1"]["state"] == "done" and by["train:a:1"]["label"] == "Training a (round 1)"
    assert by["train:ticket-escalation:1"]["state"] == "running" and by["train:ticket-escalation:1"]["ended"] is None
    assert "carry:a:1" not in by                                    # bookkeeping, not work
    pending = [x for x in tl if x["state"] == "pending"]
    assert {x["phase"] for x in pending if x["head"] == "ticket-escalation"} == {"gate", "decision", "promote"}
    assert {x["phase"] for x in pending if x["head"] == "d"} == set(PHASES)
    assert pending[-1]["key"] == "groups"
    order = [x["key"] for x in tl]
    assert order.index("pull") < order.index("train:a:1") < order.index("train:ticket-escalation:1")
    finished = run(state="done")
    assert all(x["state"] != "pending" for x in describe.timeline(finished))


def test_counts_per_head_and_total():
    c = describe.counts(run())
    assert c["heads"]["a"] == {"generated": 90, "labelled": 12, "excluded": 2}
    assert c["heads"]["ticket-escalation"] == {"generated": 600, "labelled": 5, "excluded": 1}
    assert c["total"] == {"generated": 690, "labelled": 17, "excluded": 3}


def test_results_rows_per_head():
    r = run()
    r["result"]["steps"]["control:a:1:synthetic"] = {"val": {"macro_f1": 0.71}}
    r["result"]["heads"]["a"].update({"rounds": [{"round": 1, "val_macro_f1": 0.83}],
                                      "chosen": {"round": 1, "val_macro_f1": 0.83},
                                      "gate": {"test_macro_f1": 0.85, "reference": {"macro_f1": 0.84},
                                               "decision": {"gate_passed": True, "incumbent_test_macro_f1": 0.8}}})
    rows = {x["head"]: x for x in describe.results(r)}
    assert rows["a"] == {"head": "a", "control_f1": 0.71, "val_f1": 0.83, "final_f1": 0.85, "reference_f1": 0.84,
                         "incumbent_f1": 0.8, "gate_passed": True, "outcome": "promoted", "decision": "promoted",
                         "rounds": 1}
    assert rows["b"]["decision"] == "skipped" and rows["ticket-escalation"]["outcome"] is None
    assert list(rows) == ["a", "b", "ticket-escalation"]
