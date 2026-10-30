"""End-to-end ML smoke test with the real encoder (MLX Qwen3-8B 4-bit) and reference head.

A 3-way support-ticket router (~90 templated tickets): embed -> train a few epochs -> evaluate ->
Predictor -> build the package -> pip-install the wheel into a throwaway venv -> call
``classify()`` from a fresh interpreter. Plus noul / score shape checks of the trainer.

Run: ``cd backend && uv run pytest tests/test_ml_e2e.py -x -s``
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

SCRATCH = Path(os.environ.get("FORGE_TEST_SCRATCH", Path(tempfile.gettempdir()) / "forge-test-scratch"))

from forge.base_models import QWEN  # noqa: E402  (the reference head's encoder)

pytestmark = pytest.mark.model_loading
from forge.models import Example, FileRef, Opportunity, TaskSchema  # noqa: E402

ROUTER = TaskSchema(
    name="ticket_router", question_id="department", type="choice",
    instructions="Which team should handle this support ticket?",
    criteria={"billing": "Billing: charges, invoices, refunds, payment methods and subscription pricing",
              "technical": "Technical support: bugs, errors, crashes, outages and integration problems",
              "account": "Account management: login, password, profile, user seats and account settings"},
    state_description="Subject and body of an inbound customer support ticket",
    state_example="Subject: Charged twice\n\nI was billed twice for my March invoice.")

_TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month. Please refund the duplicate payment.",
        "Our invoice #{n} shows the wrong amount, we were billed for {k} seats instead of {j}.",
        "How do I update the credit card used for {plan} billing? The old one expired.",
        "Can I get a refund? I cancelled the {plan} plan on day {j} but was still charged.",
        "Please send a VAT invoice for order {n}, our finance team needs it for the audit.",
        "Why did the price of the {plan} plan go up on my last bill without notice?",
        "The payment failed with my company card but money was still taken from the account, order {n}.",
        "I'd like to switch from monthly to annual billing on {plan} to get the discount.",
        "We were overcharged ${k}0 on the latest statement, please explain the extra line items.",
        "Is there a way to pay invoice {n} by bank transfer instead of card?",
    ],
    "technical": [
        "The app crashes every time I open the {feature} page on {os}.",
        "Getting a 500 error when calling the /v{j}/{feature} API endpoint since this morning.",
        "Your {feature} integration stopped syncing data after the latest update, logs show a timeout.",
        "The dashboard is extremely slow, {feature} takes over {k} seconds to load.",
        "Webhook deliveries for {feature} are failing with SSL handshake errors.",
        "Export to CSV produces a corrupted file when there are more than {n} rows.",
        "After upgrading to version {j}.{k} the {feature} button does nothing on {os}.",
        "Is the service down? We see connection refused errors from all of our {os} servers.",
        "The mobile app on {os} freezes when uploading attachments larger than {k} MB.",
        "Search in {feature} returns no results even though the records exist, looks like an indexing bug.",
    ],
    "account": [
        "I forgot my password and the reset email never arrives at my {domain} address.",
        "Please add {k} more user seats for new hires on our team workspace.",
        "How do I change the email address on my profile to my new {domain} one?",
        "I'm locked out of my account after enabling two-factor authentication on a lost phone.",
        "Can you transfer ownership of our workspace to my colleague at {domain}?",
        "Please delete my account and all associated personal data.",
        "I can't log in with single sign-on, it says my user is not provisioned for {domain}.",
        "How do I remove a former employee from our organisation and revoke their access?",
        "I want to rename our company workspace and update the profile picture.",
        "My username shows the wrong name, please update it to match my legal name.",
    ],
}
_FILL = dict(plan=["Pro", "Team", "Enterprise", "Starter"], feature=["reports", "calendar", "analytics", "sync"],
             os=["Windows 11", "macOS", "Android", "iOS"], domain=["acme.com", "gmail.com", "globex.io"])


def router_examples(per_template: int = 3, seed: int = 7) -> list[Example]:
    """~90 rows (3 labels x 10 templates x 3 fills), stratified 60/15/15-ish split."""
    rng = random.Random(seed)
    rows: list[Example] = []
    for label, temps in _TEMPLATES.items():
        lab_rows = []
        for ti, t in enumerate(temps):
            for r in range(per_template):
                text = t.format(n=rng.randint(1000, 9999), k=rng.randint(2, 40), j=rng.randint(1, 9),
                                **{k: rng.choice(v) for k, v in _FILL.items()})
                lab_rows.append(Example(id=f"{label}-{ti}-{r}", state=f"Subject: ticket {ti}{r}\n\n{text}",
                                        label=label, source="synthetic", split="train"))
        rng.shuffle(lab_rows)
        for i, ex in enumerate(lab_rows):
            ex.split = "test" if i < 6 else "val" if i < 11 else "train"
        rows += lab_rows
    return rows


def _events(sink: list[dict]):
    def emit(ev: dict) -> None:
        assert "ts" not in ev and "seq" not in ev
        sink.append(ev)
    return emit


@pytest.fixture(scope="module")
def workdir() -> Path:
    d = SCRATCH / "ml_e2e"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def test_router_end_to_end(workdir: Path) -> None:
    from forge.encoder import get_encoder
    from forge.evaluator import evaluate
    from forge.inference import Predictor
    from forge.packager import build_package
    from forge.trainer import TrainConfig, train

    examples = router_examples()
    assert len(examples) == 90
    events: list[dict] = []
    emit = _events(events)

    # -- embed throughput (uncached) --------------------------------------------------------
    enc = get_encoder(QWEN)
    enc.load()
    probe = [f"{e.state}\n\nthroughput probe {time.time_ns()}" for e in examples[:32]]
    t0 = time.perf_counter()
    v = enc.embed(probe, use_cache=False)
    tput = len(probe) / (time.perf_counter() - t0)
    assert v.shape == (32, 4096)
    assert abs(float((v ** 2).sum(1).mean()) - 1.0) < 1e-4
    print(f"\n[e2e] embed throughput {tput:.1f} texts/s (uncached, batch path)")
    single = enc.embed(probe[:4][::-1], use_cache=False)[::-1]   # different batch composition
    one = enc.embed(probe[:1], use_cache=False)
    assert float((single * v[:4]).sum(1).min()) > 0.999 and float(one[0] @ v[0]) > 0.999

    # -- train ---------------------------------------------------------------------------------
    proj_dir = workdir / "project"
    (proj_dir / "train").mkdir(parents=True)
    (proj_dir / "dataset.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples))
    (proj_dir / "task.json").write_text(ROUTER.model_dump_json(indent=1))
    res = train(ROUTER, examples, proj_dir / "train", TrainConfig(epochs=8, batch=32, patience=4, encoder=QWEN), emit)
    print(f"[e2e] train {res.seconds:.1f}s, epochs {res.epochs_run}, best epoch {res.best_epoch}, "
          f"val acc init {res.init_val_acc:.3f} -> best {res.best_val_acc:.3f}")
    assert res.ckpt.exists() and (proj_dir / "train" / "history.json").exists()
    types = {e["type"] for e in events}
    assert {"embed", "log", "train_step", "train_epoch", "train_snapshot"} <= types
    snaps = [e for e in events if e["type"] == "train_snapshot"]
    assert snaps[0]["epoch"] == 0 and len(snaps) == res.epochs_run + 1
    s0 = snaps[0]
    assert s0["similarity"]["col_labels"] == list(ROUTER.criteria)
    assert len(s0["points"]) <= 400 and {p["kind"] for p in s0["points"]} == {"state", "option"}
    assert all(len(p["text"]) <= 160 for p in s0["points"])
    assert res.best_val_acc >= res.init_val_acc

    # -- evaluate ------------------------------------------------------------------------------
    def teacher(states: list[str]) -> list[str]:  # stand-in "LLM": keyword rules
        out = []
        for s in states:
            s = s.lower()
            out.append("billing" if any(w in s for w in ("invoice", "charge", "refund", "bill", "pay", "price"))
                       else "account" if any(w in s for w in ("password", "seat", "log in", "account", "profile"))
                       else "technical")
        return out

    report = evaluate(ROUTER, examples, res.ckpt, proj_dir / "eval", emit, teacher=teacher,
                      metric="accuracy", threshold=0.8)
    scores = {m.name: m for m in report.models}
    print("[e2e] eval " + " | ".join(f"{m.name} acc {m.accuracy:.3f} f1 {m.macro_f1:.3f} n={m.n}"
                                     for m in report.models))
    print(f"[e2e] gate {report.gate_passed}: {report.gate_reasons}")
    print(f"[e2e] latency {report.latency.model_dump()} | ece {report.ece:.3f}")
    assert set(scores) == {"finetuned", "reference", "majority", "teacher"}
    assert scores["finetuned"].accuracy >= 0.8
    assert report.n_test == 18 and len(report.reliability) == 10
    assert sum(map(sum, report.confusion.matrix)) == 18
    assert (proj_dir / "eval" / "eval_report.json").exists()

    # -- predictor -------------------------------------------------------------------------------
    pred = Predictor(res.ckpt, ROUTER)
    ans, tm = pred.answer("Subject: help\n\nMy card was charged three times for one invoice, I want my money back.")
    print(f"[e2e] predictor {ans['choice']} conf {ans['confidence']:.2f} timings {tm}")
    assert ans["type"] == "choice" and ans["choice"] == "billing"
    assert set(tm) == {"embed_ms", "head_ms", "latency_ms"}
    ranked = pred.rank("The website shows a blank page and a JS error", ["a software bug", "a refund request"])
    assert ranked[0]["candidate"] == "a software bug" and ranked[0]["rank"] == 1
    ref_ans, _ = Predictor(None, ROUTER).answer("Subject: x\n\nI can't reset my password")
    assert ref_ans["choice"] in ROUTER.criteria

    # -- package ---------------------------------------------------------------------------------
    opp = Opportunity(
        id="opp-1", title="Replace the LLM ticket router", current_approach="GPT call in route_ticket()",
        files=[FileRef(path="app/support/router.py", start_line=40, end_line=72,
                       snippet="def route_ticket(ticket):\n    resp = llm.chat(...)\n    return resp.department")],
        task=ROUTER, fit_score=0.9, rationale="closed label set, high volume", data_strategy="synthetic",
        integration_sketch="call classify() in route_ticket, fall back to the LLM below 0.6 confidence",
        expected_benefits=["p50 latency ~1.2s -> ~60ms"], risks=["new departments need retraining"],
        success_threshold=0.8)
    polished: list[str] = []
    manifest = build_package(proj_dir, ROUTER, opp, report, "A Flask helpdesk app.", emit,
                             ckpt=res.ckpt, polish=lambda md: polished.append(md) or md)
    print(f"[e2e] package {manifest.name} {manifest.version} -> {manifest.zip}")
    pkg_dir = Path(manifest.dir)
    names = {f.path for f in manifest.files}
    for must in ("INTEGRATION.md", "EVAL_REPORT.md", "eval_report.json", "examples/golden.jsonl", "pyproject.toml",
                 "README.md", "src/clm_ticket_router/head.pt", "src/clm_ticket_router/task.json"):
        assert must in names, must
    assert polished, "polish hook not called"
    wheels = list((pkg_dir / "dist").glob("*.whl"))
    assert len(wheels) == 1 and Path(manifest.zip).exists()
    integ = (pkg_dir / manifest.integration_md).read_text()
    assert "app/support/router.py" in integ and "Prompt for your coding agent" in integ
    assert len((pkg_dir / "examples/golden.jsonl").read_text().splitlines()) == 18

    # -- the package's HTTP server, in-process (reuses the already-loaded encoder) ----------------------
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(pkg_dir / "src"))
    try:
        import clm_ticket_router.model as pkg_model
        from clm_ticket_router.serve import create_app
        pkg_model._MODEL = pkg_model.Model(encoder=enc)
        client = TestClient(create_app())
        r = client.post("/v1/systemone", json={"state": "Subject: x\n\nThe API returns 500 errors since the deploy."})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answers"]["department"]["choice"] == "technical", body
        r = client.post("/v1/systemone", json={"state": "my invoice is wrong", "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"}}})
        assert r.status_code == 200 and 0 <= r.json()["answers"]["urgent"]["noul"] <= 1
        assert client.post("/v1/predict", json={"state": "reset my password"}).json()["label"] in ROUTER.criteria
        assert client.get("/health").json()["ok"] is True
    finally:
        sys.path.remove(str(pkg_dir / "src"))

    # -- install the wheel in a throwaway venv and call classify() ------------------------------------
    venv = workdir / "venv"
    subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True, capture_output=True)
    py = venv / "bin" / "python"
    t0 = time.perf_counter()
    subprocess.run(["uv", "pip", "install", "--python", str(py), str(wheels[0])], check=True, capture_output=True)
    print(f"[e2e] wheel installed in {time.perf_counter() - t0:.1f}s")
    code = ("import json; from clm_ticket_router import classify, task; "
            "r = classify('Subject: refund\\n\\nPlease refund my duplicate charge on invoice 4411.'); "
            "f = classify('Subject: ?\\n\\nhello', threshold=0.99, fallback=lambda s: 'billing'); "
            "print(json.dumps({'r': r, 'f': f, 'q': task()['question_id']}))")
    out = subprocess.run([str(py), "-c", code], check=True, capture_output=True, text=True,
                         cwd=str(workdir), env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"})
    got = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"[e2e] installed classify -> {got['r']}")
    assert got["r"]["label"] == "billing" and set(got["r"]) >= {"label", "probabilities", "confidence"}
    assert got["f"]["label"] == "billing" and got["f"]["source"] == "fallback" and got["q"] == "department"
    # the installed package must agree with the in-process predictor
    p_in = pred.answer("Subject: refund\n\nPlease refund my duplicate charge on invoice 4411.")[0]["probabilities"]
    for k, p in p_in.items():
        assert abs(p - got["r"]["probabilities"][k]) < 2e-3, (k, p, got["r"]["probabilities"][k])
    assert sys.version_info >= (3, 12)


@pytest.mark.parametrize("qtype", ["noul", "score"])
def test_trainer_shapes_noul_score(qtype: str, workdir: Path) -> None:
    """Tiny noul / score tasks: training runs, checkpoint loads, Predictor answers the right type."""
    from forge.inference import Predictor
    from forge.trainer import TrainConfig, train

    if qtype == "noul":
        task = TaskSchema(name="is_refund", question_id="is_refund", type="noul",
                          instructions="Is the customer asking for a refund?", criteria=None,
                          state_description="ticket", state_example="I want my money back")
        rows = [(s, "true") for s in ("I want my money back", "Please refund my order", "Refund the duplicate charge",
                                      "Give me a refund now", "Return my payment please", "I need a refund for March")]
        rows += [(s, "false") for s in ("The app crashes", "How do I reset my password", "Great product, thanks",
                                        "Add two seats", "The API returns 500", "Change my email address")]
    else:
        task = TaskSchema(name="urgency", question_id="urgency", type="score",
                          instructions="How urgent is this ticket?", criteria=["low", "medium", "high"],
                          state_description="ticket", state_example="site is down")
        rows = [(s, "0") for s in ("Just a feature idea for later", "Minor typo on the pricing page",
                                   "No rush, curious about the roadmap", "Small cosmetic issue")]
        rows += [(s, "1") for s in ("Export is slow today", "A report shows wrong totals",
                                    "Login is flaky sometimes", "Sync is delayed by an hour")]
        rows += [(s, "2") for s in ("Production is down for all users", "Data loss after the update, urgent",
                                    "Security breach, accounts compromised", "Checkout broken, losing sales now")]
    exs = [Example(id=f"{qtype}{i}", state=s, label=lab, source="user",
                   split="val" if i % 4 == 3 else "train",
                   probabilities={lab: 0.9, **({"0": 0.1} if qtype == "score" and lab != "0" else {})})
           for i, (s, lab) in enumerate(rows)]
    events: list[dict] = []
    res = train(task, exs, workdir / f"train_{qtype}", TrainConfig(epochs=3, batch=8, patience=3, encoder=QWEN), _events(events))
    import torch
    ck = torch.load(res.ckpt, map_location="cpu", weights_only=True)
    assert ck["cfg"]["question_type"] == qtype and ck["cfg"]["labels"] == task.labels()
    assert ck["state_head"]["out.weight"].shape == (512, 1536)
    snap = [e for e in events if e["type"] == "train_snapshot"][-1]
    assert snap["similarity"]["col_labels"] == task.labels()
    assert len(snap["similarity"]["values"][0]) == len(task.labels())
    ans, _ = Predictor(res.ckpt, task).answer(rows[0][0])
    assert ans["type"] == qtype
    if qtype == "score":
        assert 0.0 <= ans["score"] <= 2.0 and set(ans["probabilities"]) == {"0", "1", "2"}
    else:
        assert 0.0 <= ans["noul"] <= 1.0


def test_trainer_should_stop(workdir: Path) -> None:
    """Cancellation mid-training keeps the best-so-far checkpoint and reports it."""
    from forge.trainer import TrainConfig, train

    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 3          # 1 pre-check + 2 steps, then cancel

    res = train(ROUTER, router_examples(), workdir / "train_cancel", TrainConfig(epochs=5, batch=16, encoder=QWEN), lambda e: None, stop)
    assert res.cancelled and res.ckpt.exists() and res.epochs_run == 0
