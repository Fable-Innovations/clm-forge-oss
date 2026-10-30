"""The guardrail: the project state machine. Deterministic, knows nothing about AI.

Every transition goes through `Project.fire(event, actor, reason, **kw)`, which
1. refuses events the actor is not entitled to (the agent can never `accept`),
2. lets python-statemachine refuse structurally illegal events and failed guards,
3. records (src, event, dst) in `path` and a dict with actor + reason in `log`.

Guards read `Project.facts`, which the orchestrator fills from the store / disk before firing.
The facts are data, not decisions: e.g. `ship` checks the EvalReport numbers itself
(`gate_passed`) rather than trusting anyone's claim that the gate passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed

Actor = str  # "agent" | "human" | "system"

AGENT_EVENTS = frozenset({"propose", "declare_no_fit", "ship", "recurate", "give_up"})
HUMAN_EVENTS = frozenset({"revise", "accept", "retry", "cancel", "review_done", "label_override"})
SYSTEM_EVENTS = frozenset({"start", "begin_curation", "curated", "labels_settled", "needs_review", "selected",
                           "trained", "evaluated", "packaged", "fail"})
EVENTS_BY_ACTOR = {"agent": AGENT_EVENTS, "human": HUMAN_EVENTS, "system": SYSTEM_EVENTS}
# Events that record a decision without changing the state (and so start no job).
STAY_EVENTS = frozenset({"label_override"})

FINAL_STATES = frozenset({"delivered", "no_fit", "cancelled"})
# States a project can fail in (and so `retry` back into).
RETRYABLE = ("created", "analyzing", "proposed", "accepted", "curating", "labelling", "selecting", "training",
             "evaluating", "reviewing", "packaging")


class Rejected(TransitionNotAllowed):
    """A proposal the rules layer refused. Carries a readable reason."""

    def __init__(self, event: str, state: str, why: str):
        self.event_name, self.state_name, self.why = event, state, why
        Exception.__init__(self, f"REJECTED {event!r} in {state}: {why}")

    def __str__(self) -> str:
        return f"REJECTED {self.event_name!r} in {self.state_name}: {self.why}"


@dataclass
class Facts:
    """What the guards are allowed to look at. Filled by the orchestrator, never by the model."""

    proposal_verdict: str | None = None           # "fit" | "no_fit"
    opportunity_ids: list[str] = field(default_factory=list)
    accepted_opportunity_id: str | None = None
    dataset_size: int = 0
    labels: list[str] = field(default_factory=list)          # TaskSchema labels
    train_labels: list[str] = field(default_factory=list)    # labels present in train split
    min_examples: int = 120
    undisposed_rows: int = 0                       # real rows with no final disposition (labelled / excluded)
    excluded_by_agent: int = 0                     # real rows the labelling agent dropped (with a reason)
    label_policy: str = "agent"                    # agent | user | teacher | mixed (forge.labelling.POLICIES)
    base_model: str | None = None                  # encoder spec chosen by base-model selection
    base_model_candidates: list[str] = field(default_factory=list)
    checkpoint_exists: bool = False
    report: dict | None = None                     # EvalReport as dict
    manifest_exists: bool = False
    attempts: int = 0
    max_attempts: int = 3


def gate_check(report: dict | None) -> tuple[bool, list[str]]:
    """Deterministic ship gate. Recomputed from the numbers; `report.gate_passed` is not trusted."""
    if not report:
        return False, ["no evaluation report"]
    metric = report.get("metric", "accuracy")
    thr = float(report.get("threshold", 1.0))
    models = {m["name"]: m for m in report.get("models", [])}
    ft = models.get("finetuned")
    if ft is None:
        return False, ["no finetuned score in report"]
    v = float(ft.get(metric, 0.0))
    reasons, ok = [], True
    if v < thr:
        ok = False
        reasons.append(f"finetuned {metric} {v:.3f} < threshold {thr:.3f}")
    for base in ("reference", "majority"):
        b = models.get(base)
        if b is None:
            ok = False
            reasons.append(f"missing {base} baseline")
        elif v <= float(b.get(metric, 0.0)):
            ok = False
            reasons.append(f"finetuned {metric} {v:.3f} does not beat {base} {float(b.get(metric, 0.0)):.3f}")
    if ok:
        reasons.append(f"finetuned {metric} {v:.3f} ≥ {thr:.3f} and beats reference and majority")
    return ok, reasons


def _from(state_id: str):
    def cond(machine) -> bool:
        return machine.failed_from == state_id
    cond.__name__ = f"failed_in_{state_id}"
    return cond


class Project(StateMachine):
    created = State(initial=True)
    analyzing = State()
    proposed = State()
    accepted = State()
    curating = State()
    labelling = State()     # system stage: the labelling agent disposes every real row (forge.labelling)
    label_review = State()  # opt-in human gate (policy user | teacher | mixed) while rows remain undisposed
    selecting = State()
    training = State()
    evaluating = State()
    reviewing = State()
    packaging = State()
    delivered = State(final=True)
    no_fit = State(final=True)
    cancelled = State(final=True)
    errored = State()

    start = created.to(analyzing)
    propose = analyzing.to(proposed, cond="verdict_is_fit")
    declare_no_fit = analyzing.to(no_fit, cond="verdict_is_no_fit") | proposed.to(no_fit, cond="verdict_is_no_fit")
    revise = proposed.to(analyzing, cond="has_feedback")
    accept = proposed.to(accepted, cond="opportunity_chosen")
    begin_curation = accepted.to(curating)
    curated = curating.to(labelling)
    labels_settled = labelling.to(selecting, cond=["all_disposed", "dataset_ready"])
    needs_review = labelling.to(label_review, cond=["human_policy", "has_undisposed"])
    review_done = label_review.to(labelling, cond="review_settled")
    selected = selecting.to(training, cond="base_model_chosen")
    trained = training.to(evaluating, cond="checkpoint_ready")
    evaluated = evaluating.to(reviewing, cond="report_ready")
    ship = reviewing.to(packaging, cond="gate_passed")
    recurate = reviewing.to(curating, cond="attempts_left")
    give_up = reviewing.to(no_fit)
    packaged = packaging.to(delivered, cond="manifest_ready")
    fail = (created.to(errored) | analyzing.to(errored) | proposed.to(errored) | accepted.to(errored)
            | curating.to(errored) | labelling.to(errored) | selecting.to(errored) | training.to(errored)
            | evaluating.to(errored) | reviewing.to(errored) | packaging.to(errored))
    retry = (errored.to(created, cond=_from("created")) | errored.to(analyzing, cond=_from("analyzing"))
             | errored.to(proposed, cond=_from("proposed")) | errored.to(accepted, cond=_from("accepted"))
             | errored.to(curating, cond=_from("curating")) | errored.to(labelling, cond=_from("labelling"))
             | errored.to(selecting, cond=_from("selecting"))
             | errored.to(training, cond=_from("training"))
             | errored.to(evaluating, cond=_from("evaluating")) | errored.to(reviewing, cond=_from("reviewing"))
             | errored.to(packaging, cond=_from("packaging")))
    cancel = (created.to(cancelled) | analyzing.to(cancelled) | proposed.to(cancelled)
              | accepted.to(cancelled) | curating.to(cancelled) | labelling.to(cancelled)
              | label_review.to(cancelled) | selecting.to(cancelled) | training.to(cancelled)
              | evaluating.to(cancelled) | reviewing.to(cancelled) | packaging.to(cancelled)
              | errored.to(cancelled))
    # a user label for a real row, recorded in any non-terminal state; applied by the next labelling stage
    label_override = (created.to.itself(internal=True) | analyzing.to.itself(internal=True)
                      | proposed.to.itself(internal=True) | accepted.to.itself(internal=True)
                      | curating.to.itself(internal=True) | labelling.to.itself(internal=True)
                      | label_review.to.itself(internal=True) | selecting.to.itself(internal=True)
                      | training.to.itself(internal=True) | evaluating.to.itself(internal=True)
                      | reviewing.to.itself(internal=True) | packaging.to.itself(internal=True)
                      | errored.to.itself(internal=True))

    def __init__(self, name: str = "", facts: Facts | None = None, start_value: str | None = None,
                 path: list | None = None, log: list | None = None, failed_from: str | None = None):
        self.name_ = name
        self.facts = facts or Facts()
        self.path: list[tuple[str, str, str]] = [tuple(p) for p in (path or [])]
        self.log: list[dict] = list(log or [])
        self.failed_from: str | None = failed_from
        self._pending: dict[str, Any] = {}
        super().__init__(start_value=start_value)

    # ---- guards (pure functions of facts + event kwargs) ---------------------------------------
    def verdict_is_fit(self) -> bool:
        return self.facts.proposal_verdict == "fit"

    def verdict_is_no_fit(self) -> bool:
        return self.facts.proposal_verdict == "no_fit"

    def has_feedback(self, feedback: str = "") -> bool:
        return bool((feedback or "").strip())

    def opportunity_chosen(self, opportunity_id: str | None = None) -> bool:
        return bool(opportunity_id) and opportunity_id in self.facts.opportunity_ids

    def dataset_ready(self) -> bool:
        f = self.facts
        return f.dataset_size >= f.min_examples and bool(f.labels) and set(f.labels) <= set(f.train_labels)

    def all_disposed(self) -> bool:
        """Every real row has a final disposition: labelled, excluded by the agent, or set aside by the user."""
        return self.facts.undisposed_rows == 0

    def has_undisposed(self) -> bool:
        return self.facts.undisposed_rows > 0

    def human_policy(self) -> bool:
        """The human review gate exists only when the project opted out of the labelling agent."""
        return self.facts.label_policy != "agent"

    def review_settled(self, mode: str = "continue") -> bool:
        """The reviewer leaves the gate once every row is labelled, or sets the rest aside (mode=exclude)."""
        return self.facts.undisposed_rows == 0 or mode == "exclude"

    def base_model_chosen(self) -> bool:
        f = self.facts
        return bool(f.base_model) and (not f.base_model_candidates or f.base_model in f.base_model_candidates)

    def checkpoint_ready(self) -> bool:
        return self.facts.checkpoint_exists

    def report_ready(self) -> bool:
        return self.facts.report is not None

    def gate_passed(self) -> bool:
        return gate_check(self.facts.report)[0]

    def attempts_left(self) -> bool:
        return self.facts.attempts < self.facts.max_attempts

    def manifest_ready(self) -> bool:
        return self.facts.manifest_exists

    # ---- history --------------------------------------------------------------------------------
    def on_transition(self, event, source, target):
        ev = getattr(event, "id", str(event))
        self.path.append((source.id, ev, target.id))
        self.log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": source.id, "event": ev, "to": target.id,
            "actor": self._pending.get("actor", "system"),
            "reason": self._pending.get("reason", ""),
        })
        if ev == "fail":
            self.failed_from = source.id

    # ---- the only way in ------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self.current_state_value

    def allowed_for(self, actor: Actor) -> list[str]:
        """Events structurally available now to `actor` (guards are the machine's call on send)."""
        evs = EVENTS_BY_ACTOR.get(actor, frozenset())
        return [e.id for e in self.allowed_events if e.id in evs]

    def explain_guard(self, event: str, **kw) -> str:
        f = self.facts
        if event == "ship":
            return "; ".join(gate_check(f.report)[1])
        if event == "recurate":
            return f"attempts {f.attempts} ≥ max_attempts {f.max_attempts}"
        if event == "curated":
            missing = sorted(set(f.labels) - set(f.train_labels))
            return (f"dataset {f.dataset_size} rows (min {f.min_examples})"
                    + (f"; labels missing from train: {missing}" if missing else ""))
        if event == "labels_settled":
            if f.undisposed_rows:
                return f"{f.undisposed_rows} real rows have no final disposition yet"
            missing = sorted(set(f.labels) - set(f.train_labels))
            return (f"dataset {f.dataset_size} rows (min {f.min_examples})"
                    + (f"; labels missing from train: {missing}" if missing else ""))
        if event == "needs_review":
            return ("the labelling agent owns this project's labels (policy agent)" if f.label_policy == "agent"
                    else "no real row is waiting for a label")
        if event == "review_done":
            return (f"{f.undisposed_rows} real rows still need a label: label them, send them to the teacher, "
                    "or continue with mode=exclude")
        if event in ("propose", "declare_no_fit"):
            return f"proposal verdict is {f.proposal_verdict!r}"
        if event == "accept":
            return f"opportunity {kw.get('opportunity_id')!r} not in {f.opportunity_ids}"
        if event == "revise":
            return "feedback is empty"
        if event == "selected":
            return (f"base model {f.base_model!r} not chosen from candidates {f.base_model_candidates}"
                    if f.base_model else "no base model chosen")
        if event == "trained":
            return "checkpoint missing"
        if event == "evaluated":
            return "report missing"
        if event == "packaged":
            return "manifest missing"
        return "guard failed"

    def fire(self, event: str, actor: Actor, reason: str = "", **kw) -> tuple[str, str]:
        """Attempt `event` as `actor`. Returns (src, dst). Raises `Rejected` on any refusal."""
        src = self.state
        if event not in EVENTS_BY_ACTOR.get(actor, frozenset()):
            owner = next((a for a, evs in EVENTS_BY_ACTOR.items() if event in evs), None)
            why = f"{actor} may not fire {event!r}" + (f" (only {owner})" if owner else " (unknown event)")
            raise Rejected(event, src, why)
        if event not in [e.id for e in self.allowed_events]:
            raise Rejected(event, src, f"not a transition out of {src}")
        self._pending = {"actor": actor, "reason": reason}
        try:
            self.send(event, **kw)
        except TransitionNotAllowed:
            raise Rejected(event, src, self.explain_guard(event, **kw)) from None
        finally:
            self._pending = {}
        if self.state == src and event not in STAY_EVENTS:  # guard rejected silently
            raise Rejected(event, src, self.explain_guard(event, **kw))
        return src, self.state

    def snapshot(self) -> dict:
        return {"state": self.state, "path": [list(p) for p in self.path], "log": self.log,
                "failed_from": self.failed_from}

    @classmethod
    def restore(cls, snap: dict, facts: Facts | None = None, name: str = "") -> "Project":
        return cls(name=name, facts=facts, start_value=snap.get("state") or "created",
                   path=snap.get("path"), log=snap.get("log"), failed_from=snap.get("failed_from"))

    def __str__(self) -> str:
        return f"[{self.state}] {self.name_}"


def all_transitions() -> list[tuple[str, str, str]]:
    """(src, event, dst) for every edge — used by the API/README to draw the diagram."""
    out = []
    for s in Project.states:
        for t in s.transitions:
            for ev in str(t.event).split():
                out.append((t.source.id, ev, t.target.id))
    return out
