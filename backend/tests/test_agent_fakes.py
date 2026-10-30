"""Shared test doubles for the agent/API tests (no tests here). Real-LLM runs live in the smoke scripts."""
from __future__ import annotations

import re

import pytest

from forge.models import FileRef, Opportunity, Proposal, TaskSchema

TASK = TaskSchema(name="ticket_router", question_id="queue", type="choice",
                  instructions="Which support queue should handle this ticket?",
                  criteria={"billing": "invoices, charges, refunds", "technical": "bugs, errors, outages",
                            "account": "login, password, 2FA, profile"},
                  state_description="ticket subject + body", state_example="Subject: Refund\n\nI was double charged.")


def proposal(verdict: str = "fit") -> Proposal:
    if verdict == "no_fit":
        return Proposal(verdict="no_fit", summary="nothing to classify", no_fit_reasons=["only generation"])
    return Proposal(verdict="fit", summary="route tickets with a CLM", recommended_id="opp1", opportunities=[
        Opportunity(id="opp1", title="Ticket routing", files=[FileRef(path="support_app/router.py", start_line=1, end_line=5)],
                    current_approach="LLM call", task=TASK, fit_score=0.9, rationale="3 labels",
                    data_strategy="mine tests + synthetic", target_examples=150, integration_sketch="call head")])


class FakeLLM:
    """Deterministic stand-in. `decisions` is consumed in order by Decision requests."""

    provider, model = "fake", "fake-1"

    def __init__(self, verdict="fit", decisions=None, qa_fail=()):
        self.verdict = verdict
        self.qa_fail, self.qa_seen = set(qa_fail), set()
        self.prompts: list[str] = []
        self.decisions = list(decisions or [])
        self.calls: list[str] = []

    async def structured(self, prompt, schema, system=None, **kw):
        name = schema.__name__
        self.calls.append(name)
        self.prompts.append(prompt)
        if name == "Proposal":
            return proposal(self.verdict)
        if name == "Decision":
            ev = self.decisions.pop(0) if self.decisions else ("propose" if self.verdict == "fit" else "declare_no_fit")
            return schema(event=ev, reason=f"fake chose {ev}")
        if name == "ChatReply":
            return schema(reply="sure")
        if name == "TableMapping":
            if "`clm_data/tickets.csv`" not in prompt:
                return schema(relevant=False, reason="not tickets")
            return schema(relevant=True, reason="exported tickets", template="Subject: {subject}\n\n{body}",
                          label_column="queue", label_map={"Billing": "billing", "Tech": "technical", "Acct": "account"})
        if name == "MinedBatch":
            return schema(items=[])
        if name == "Voices":
            return schema(personas=["a terse power user", "a polite first-time user", "an admin"],
                          styles=["one short sentence", "two sentences", "a paragraph"])
        if name == "GenBatch":
            m = re.search(r"correct label is (\w+)", prompt)
            lab = m.group(1) if m else "x"
            n = int(re.search(r"Write (\d+)", prompt).group(1))
            import random
            return schema(states=[f"{lab} issue #{random.random():.8f} needs help" for _ in range(n)])
        if name == "QAReview":
            n = len(re.findall(r"^\[(\d+)\] \(", prompt, re.M))
            # `qa_fail` labels fail their first review (every item rejected) until regenerated with feedback
            m = re.search(r"to have the label (\w+)", prompt)
            fail = m and m.group(1) in self.qa_fail and m.group(1) not in self.qa_seen
            if m:
                self.qa_seen.add(m.group(1)) if fail else None
            return schema(items=[{"i": i, "ok": not fail, "issue": "off-label" if fail else ""} for i in range(n)],
                          feedback="make every text clearly about this queue" if fail else "")
        if name == "LabelBatch":
            states = re.findall(r"\[(\d+)\]\n(\w+) issue", prompt)
            # disagree on every 10th to exercise the drop path
            return schema(labels=[{"i": int(i), "label": (lab if int(i) % 10 else "account" if lab != "account" else "billing"),
                                   "confidence": 0.9, "reason": "fake"} for i, lab in states])
        raise AssertionError(f"unexpected schema {name}")

    async def chat(self, messages, system=None, on_delta=None):
        if on_delta:
            await on_delta("hel")
            await on_delta("lo")
        return "hello"

    async def explore(self, task, repo_path, schema, on_event, system=None):
        on_event({"type": "tool", "name": "Read", "input": "support_app/router.py", "output": "..."})
        return await self.structured(task, schema, system)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every runtime path at tmp so tests never touch data/."""
    from forge import config

    def pdir(pid):
        p = tmp_path / "projects" / pid
        p.mkdir(parents=True, exist_ok=True)
        return p
    monkeypatch.setattr(config, "project_dir", pdir)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "forge.db")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "REPOS_DIR", tmp_path / "repos")
    monkeypatch.setattr(config, "FRONTEND_DIST", tmp_path / "nodist")
    monkeypatch.setattr(config, "_cached", None)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    # the model-job queue's preflight reads this machine's memory pressure and process table: tests use fakes
    from forge import jobqueue
    monkeypatch.setattr(jobqueue, "system_preflight", lambda: jobqueue.Preflight(True))
    return tmp_path
