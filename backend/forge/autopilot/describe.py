"""What a run is doing, in words: step labels, progress, the step timeline and row counts (SPEC "Run progress").

One owner for every derived view of a run; the frontend only renders these. All functions are pure over the
stored run record (``forge.autopilot.runner`` writes ``result.plan``, ``result.timings``, per-head ``started`` /
``ended`` and ``run.first_started``) plus, for a live run, the adapter's last progress message.

* ``step_label(key)``: ``train:ticket-escalation:1`` -> "Training ticket-escalation (round 1)".
* ``phase_of(key)``: the head phase a step belongs to (``runner.PHASES``), or None for run-level steps.
* ``progress(run, detail)`` -> ``{fraction, heads_done, heads_total, current: {label, head}, detail, elapsed_s,
  eta_s, eta_note}``, or None for a run started before plans were recorded. Units = 3 prelude steps (pull,
  ingest, inspect) + one per head + 1 (call groups); the head in progress adds its phase index / 8. A head is done
  when it has an ``outcome``. ETA = mean duration of the completed non-skipped heads x heads remaining; None
  (``eta_note`` "ETA after the first head") until one completes.
* ``timeline(run)`` -> ``[{key, label, head, phase, state: done|running|pending, started, ended}]`` in run order;
  pending entries come from the plan (the phases a head has not reached, the prelude, the call groups).
* ``counts(run)`` -> generated / labelled / excluded rows per head and in total, summed from the step outputs.
* ``results(run)`` -> one row per head: control val macro-F1 (the synthetic control of the chosen round), the
  candidate's val and test macro-F1, the reference head's and the incumbent's test macro-F1, the gate, the
  outcome (promoted / kept in staging / skipped / ...), the number of rounds.
"""
from __future__ import annotations

from datetime import datetime, timezone

from forge.autopilot.runner import PHASES

PRELUDE = ["pull", "ingest", "inspect"]
_PHASE_LABEL = {"needs": "Detecting what {h} needs", "label": "Labelling real rows for {h}",
                "generate": "Generating rows for {h}", "controls": "Training controls for {h}",
                "train": "Training {h}", "gate": "Gating {h} on the test split", "decision": "Deciding on {h}",
                "promote": "Promoting {h}"}
_RUN_LABEL = {"pull": "Pulling real data", "ingest": "Ingesting real rows", "inspect": "Inspecting the heads",
              "groups": "Checking call-group readiness"}
_PREFIX_PHASE = {"needs": "needs", "inspect": "needs", "clean": "needs", "label": "label", "gen": "generate", "control": "controls",
                 "train": "train", "gate": "gate", "decision": "decision", "rejected": "decision",
                 "promote": "promote"}
#: step keys that are bookkeeping, not work: kept in the run record, left out of the timeline
_BOOKKEEPING = ("nogen", "carry", "rejected")
SKIPPED = ("skipped", "retired")


def _parts(key: str) -> list[str]:
    return key.split(":")


def head_of(key: str) -> str | None:
    p = _parts(key)
    return p[1] if len(p) > 1 else None


def phase_of(key: str) -> str | None:
    p = _parts(key)
    return _PREFIX_PHASE.get(p[0]) if len(p) > 1 else None


def step_label(key: str) -> str:
    p = _parts(key)
    if len(p) == 1:
        return _RUN_LABEL.get(key, key)
    kind, h = p[0], p[1]
    rnd = f" (round {p[2]})" if len(p) > 2 and p[2].isdigit() else ""
    if kind == "inspect":
        return f"Re-inspecting {h} after generation{rnd}" if p[-1] == "post" else f"Inspecting {h}{rnd}"
    if kind == "needs":
        return f"Detecting what {h} needs"
    if kind == "label":
        return f"Labelling real rows for {h}"
    if kind == "clean":
        return f"Removing smoke and re-guarded rows of {h}"
    if kind == "gen":
        return f"Generating rows for {h}{rnd}"
    if kind == "control":
        return f"Training the {p[3]} control for {h}{rnd}" if len(p) > 3 else f"Training a control for {h}{rnd}"
    if kind == "train":
        return f"Training {h}{rnd}"
    if kind == "gate":
        return f"Gating {h} on the test split"
    if kind == "decision":
        return f"Deciding on {h}"
    if kind == "promote":
        return f"Promoting {h}"
    if kind == "rejected":
        return f"Keeping {h} in staging"
    if kind == "nogen":
        return f"No generation for {h}{rnd}"
    if kind == "carry":
        return f"Carrying rows over for {h}{rnd}"
    return key


def _ts(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _running_key(run: dict) -> str | None:
    if run.get("state") != "running":
        return None
    live = [k for k, t in (run["result"].get("timings") or {}).items() if t.get("s") and not t.get("e")]
    return live[-1] if live else None


def _phase_index(run: dict, head: str) -> int:
    """How many of the head's phases it has reached (0..8), from its finished and running steps."""
    res = run["result"]
    keys = [k for k in (res.get("steps") or {}) if head_of(k) == head]
    rk = _running_key(run)
    if rk and head_of(rk) == head:
        keys.append(rk)
    idx = [PHASES.index(ph) for k in keys if (ph := phase_of(k)) in PHASES]
    return max(idx) if idx else 0


def progress(run: dict, detail: str | None = None) -> dict | None:
    res = run.get("result") or {}
    plan = res.get("plan")
    if not plan:
        return None
    heads = res.get("heads") or {}
    steps = res.get("steps") or {}
    names = list(plan.get("heads") or [])
    total = len(names)
    done = [h for h in names if (heads.get(h) or {}).get("outcome")]
    finished = run.get("state") not in ("queued", "running")
    current_head = next((h for h in names if h not in done and (heads.get(h) or {}).get("started")), None)
    units = len(PRELUDE) + total + 1
    got = sum(k in steps for k in PRELUDE) + len(done) + (1 if "groups" in steps else 0)
    if current_head and not finished:
        got += _phase_index(run, current_head) / len(PHASES)
    fraction = 1.0 if run.get("state") in ("done", "done_with_failures") else min(1.0, got / units)
    rk = _running_key(run)
    start = _ts(run.get("first_started") or run.get("started"))
    end = _ts(run.get("ended")) if finished else _now()
    elapsed = (end - start).total_seconds() if start and end else None
    durations = [(_ts(v["ended"]) - _ts(v["started"])).total_seconds() for h, v in heads.items()
                 if v.get("outcome") and v["outcome"] not in SKIPPED and v.get("started") and v.get("ended")]
    remaining = total - len(done)
    eta = None if finished or not durations else round(sum(durations) / len(durations) * remaining)
    note = None if finished or durations or remaining == 0 else "ETA after the first head"
    return {"fraction": round(fraction, 4), "heads_done": len(done), "heads_total": total,
            "current": {"label": step_label(rk), "head": head_of(rk)} if rk else None,
            "detail": detail if not finished else None, "elapsed_s": round(elapsed) if elapsed is not None else None,
            "eta_s": eta, "eta_note": note}


def _entry(key: str, state: str, t: dict | None) -> dict:
    return {"key": key, "label": step_label(key), "head": head_of(key), "phase": phase_of(key), "state": state,
            "started": (t or {}).get("s"), "ended": (t or {}).get("e")}


def timeline(run: dict) -> list[dict]:
    res = run.get("result") or {}
    steps, timings = res.get("steps") or {}, res.get("timings") or {}
    out: list[dict] = []
    seen: set[str] = set()
    for k in steps:
        if k.split(":")[0] in _BOOKKEEPING:
            continue
        out.append(_entry(k, "done", timings.get(k)))
        seen.add(k)
    rk = _running_key(run)
    if rk and rk not in seen:
        out.append(_entry(rk, "running", timings.get(rk)))
        seen.add(rk)
    plan = res.get("plan")
    if not plan or run.get("state") not in ("queued", "running"):
        return out
    for k in PRELUDE:
        if k not in seen:
            out.append(_entry(k, "pending", None))
    heads = res.get("heads") or {}
    for h in plan.get("heads") or []:
        if (heads.get(h) or {}).get("outcome"):
            continue
        reached = _phase_index(run, h) if any(head_of(k) == h for k in seen) else -1
        for i, ph in enumerate(plan.get("phases") or PHASES):
            if i > reached:
                out.append({"key": f"{ph}:{h}", "label": _PHASE_LABEL.get(ph, ph).format(h=h), "head": h,
                            "phase": ph, "state": "pending", "started": None, "ended": None})
    if "groups" not in seen:
        out.append(_entry("groups", "pending", None))
    return out


def counts(run: dict) -> dict:
    steps = (run.get("result") or {}).get("steps") or {}
    per: dict[str, dict[str, int]] = {}

    def add(h: str, k: str, n) -> None:
        if isinstance(n, (int, float)):
            per.setdefault(h, {"generated": 0, "labelled": 0, "excluded": 0})[k] += int(n)

    for key, out in steps.items():
        p = _parts(key)
        if not isinstance(out, dict) or len(p) < 2:
            continue
        if p[0] == "gen":
            add(p[1], "generated", out.get("accepted", 0))
        elif p[0] == "label":
            got = (out.get("heads") or {}).get(p[1]) or {}
            add(p[1], "labelled", got.get("agent_labelled", 0))
            add(p[1], "excluded", got.get("excluded_by_agent", 0))
    total = {k: sum(v[k] for v in per.values()) for k in ("generated", "labelled", "excluded")}
    return {"heads": per, "total": total}


def results(run: dict) -> list[dict]:
    res = run.get("result") or {}
    steps, heads = res.get("steps") or {}, res.get("heads") or {}
    order = list((res.get("plan") or {}).get("heads") or []) + sorted(h for h in heads if h not in
                                                                      ((res.get("plan") or {}).get("heads") or []))
    out = []
    for h in order:
        rec = heads.get(h)
        if rec is None:
            continue
        rounds = rec.get("rounds") or []
        chosen = rec.get("chosen") or (rounds[-1] if rounds else None)
        control = None
        if chosen:
            got = steps.get(f"control:{h}:{chosen['round']}:synthetic")
            control = ((got or {}).get("val") or {}).get("macro_f1") if isinstance(got, dict) else None
        gate = rec.get("gate") or {}
        d = gate.get("decision") or {}
        outcome = rec.get("outcome")
        out.append({"head": h, "control_f1": control, "val_f1": chosen.get("val_macro_f1") if chosen else None,
                    "final_f1": gate.get("test_macro_f1"), "reference_f1": (gate.get("reference") or {}).get("macro_f1"),
                    "incumbent_f1": d.get("incumbent_test_macro_f1"), "gate_passed": d.get("gate_passed"),
                    "outcome": outcome, "decision": ("promoted" if outcome == "promoted" else
                                                     "kept" if outcome == "rejected" and gate else outcome),
                    "rounds": len(rounds)})
    return out
