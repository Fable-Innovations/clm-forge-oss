"""Pure detectors: a head's status (the adapter's ``inspect`` output) -> the needs the autopilot acts on.

``HeadStatus`` parses one ``inspect`` entry as documented in SPEC "Autopilot adapter contract" ("HeadStatus"):
``counts`` per label group (``synthetic``, ``shaped``, ``real_labelled``, ``real_<source>``) plus the pseudo
groups ``(undisposed)`` and ``(excluded)``; ``teacher_disagreement_rate``, ``excluded_rate``, ``shape``,
``last_run``, ``incumbent``, ``fingerprints`` / ``controls``, ``stale.new_real_rows``, ``gate_threshold`` and
``generate`` (``available`` / ``reason``, and per balance group the generate label that fills it).
Every ``Need`` carries its kind, a reason string and the counts behind it; the runner journals each one.

| need | fires when | remedy |
|---|---|---|
| unlabelled | a real row has no final disposition | agent labelling |
| imbalance | a label group is below its target share (``balance``) | shaped generation for it (``none``,
  with the reason, when ``generate`` reports the group or head cannot be generated for) |
| small_data | a label group has fewer than ``min_count`` train rows | shaped generation (same) |
| shape_gap | >= 20% of real key templates absent from synthetic, length KS D > 0.25, or a real channel
  holds less than half its real share in synthetic | shaped generation from the uncovered shapes |
| label_noise | teacher disagreement > 15% or excluded > 10% of real rows | adjudication re-run |
| confusable | one off-diagonal pair holds > 25% of a label's VAL errors | hard negatives for the pair |
| real_hurts | candidate val macro-F1 < control val macro-F1 - 0.02 | cap real rows (``real_cap``) |
| no_control | a control mix has no run for the current train fingerprint | control run |
| stale | new real rows since the last run, or the schedule is due | full run |
| smoke_rows | shaped rows or generation stage folders a smoke or test run wrote into the real dataset (``cleanup.smoke``, ``cleanup.smoke_stages``) | ``clean`` |
| reshape | shaped rows accepted under an older real-source guard (``cleanup.stale_guard``) | ``clean`` (recheck,
  remove the failures; balance generation replaces them) |
| retrain | the promoted head learned from rows removed since (deleted at the source, smoke, reshape: ``retrain``) |
  full run |

``real_hurts`` needs a control's val score, which ``inspect`` does not carry: the runner passes it in once the
round's control runs have reported (``control_val_macro_f1``).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge.autopilot import balance
from forge.autopilot.targets import Policy

UNDISPOSED, EXCLUDED = "(undisposed)", "(excluded)"


class Confusion(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class SplitMetrics(BaseModel):
    """Scores of one split. The autopilot only ever passes VAL metrics to detectors and diagnosis."""
    model_config = ConfigDict(extra="allow")
    macro_f1: float
    per_label: dict[str, float] = Field(default_factory=dict)
    confusion: Confusion | None = None


class ShapeStats(BaseModel):
    model_config = ConfigDict(extra="allow")
    uncovered_template_share: float = 0.0
    top_uncovered: dict[str, int] = Field(default_factory=dict)
    length_ks_d: float = 0.0
    channel_gaps: list[str] = Field(default_factory=list)


class LastRun(BaseModel):
    model_config = ConfigDict(extra="allow")
    run: str | None = None
    staging: str | None = None
    val: SplitMetrics


class Incumbent(BaseModel):
    model_config = ConfigDict(extra="allow")
    macro_f1: float | None = None
    path: str | None = None


class GroupGeneration(BaseModel):
    """The ``generate`` label that fills one balance group, or None with the reason no label does."""
    label: str | None = None
    reason: str | None = None


class Generation(BaseModel):
    """``HeadStatus.generate``: can the adapter's ``generate`` serve this head now (``available``, else
    ``reason``: no real rows to take the shape from, the adapter has no source for the head's family, no
    generator), and which generate label fills each balance group."""
    available: bool
    reason: str | None = None
    groups: dict[str, GroupGeneration] = Field(default_factory=dict)

    def label_for(self, group: str) -> tuple[str | None, str | None]:
        """(generate label, None) or (None, why not)."""
        if not self.available:
            return None, self.reason or "generation unavailable"
        g = self.groups.get(group)
        if g is None:
            return None, f"inspect reported no generate label for group {group!r}"
        return g.label, (None if g.label else (g.reason or "no generate label"))


class HeadStatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    head: str
    retired: bool | str | None = None
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    teacher_disagreement_rate: float | None = None     # None: no row has two teacher votes yet
    excluded_rate: float | None = None
    shape: ShapeStats | None = None
    last_run: LastRun | None = None
    incumbent: Incumbent | None = None
    fingerprints: dict[str, str] = Field(default_factory=dict)
    controls: dict[str, bool] = Field(default_factory=dict)
    stale: dict[str, int] = Field(default_factory=dict)
    gate_threshold: float | None = None                 # the head's test gate threshold (the val gate uses it too)
    # a Forge project's queue cannot attribute undisposed rows to a label group: their count directly
    undisposed: int | None = None
    # head sets: what ``generate`` can do for this head (None for Forge projects, which generate in their own
    # pipeline; the runner requires it before planning generation for a head set)
    generate: Generation | None = None
    # head sets: shaped rows the next ``clean`` acts on ({smoke, stale_guard, guard_version}) and, when the promoted
    # head learned from rows removed since, ``{n, kinds, reasons}`` (the adapter's removal record)
    cleanup: dict = Field(default_factory=dict)
    retrain: dict | None = None

    @property
    def labels(self) -> list[str]:
        """The label groups balance works over (the pseudo groups excluded)."""
        return sorted(g for g in self.counts if not g.startswith("("))

    def per(self, field: str) -> dict[str, int]:
        return {l: int(self.counts.get(l, {}).get(field, 0)) for l in self.labels}

    def have(self) -> dict[str, int]:
        """Train rows per label group, every source."""
        return {l: sum(int(self.counts[l].get(k, 0)) for k in ("synthetic", "shaped", "real_labelled"))
                for l in self.labels}

    def n_undisposed(self) -> int:
        if self.undisposed is not None:
            return self.undisposed
        return int(self.counts.get(UNDISPOSED, {}).get("real_needs_label", 0))

    def required_controls(self) -> list[str]:
        """Control mixes a fair comparison needs: every mix the adapter fingerprints except ``all``."""
        return sorted(m for m in self.fingerprints if m != "all")


class Need(BaseModel):
    kind: str
    head: str
    reason: str
    counts: dict = Field(default_factory=dict)
    label: str | None = None
    focus: str | None = None
    remedy: str


def balance_plan(st: HeadStatus, policy: Policy) -> balance.BalancePlan:
    return balance.plan(st.labels, st.have(), st.per("real_labelled"), st.per("synthetic"),
                        min_count=policy.min_count, gen_cap=policy.gen_cap, real_prior_min=policy.real_prior_min)


def _gen_remedy(st: HeadStatus, group: str) -> tuple[str, str]:
    """(remedy, reason suffix) for a group short of rows: shaped generation, or ``none`` with why the adapter
    cannot generate for it (the need is still reported)."""
    if st.generate is None:
        return "shaped_generation", ""
    label, why = st.generate.label_for(group)
    return ("shaped_generation", "") if label else ("none", f"; no generation: {why}")


def detect(st: HeadStatus, policy: Policy, *, schedule_due: bool = False, fair_baselines: bool = True,
           control_val_macro_f1: float | None = None) -> list[Need]:
    """``fair_baselines=False`` (Forge projects, which have no control runs) skips ``no_control``."""
    h, out = st.head, []
    undisposed = st.n_undisposed()
    if undisposed:
        out.append(Need(kind="unlabelled", head=h, remedy="agent_label",
                        reason=f"{undisposed} real rows have no final disposition", counts={"undisposed": undisposed}))

    if st.labels:
        plan = balance_plan(st, policy)
        have = st.have()
        total = sum(have.values()) or 1
        for l in st.labels:
            n = have[l]
            remedy, why_not = _gen_remedy(st, l)
            if n / total < plan.shares[l] - 1e-9:
                out.append(Need(kind="imbalance", head=h, label=l, remedy=remedy,
                                reason=f"{l} holds {n / total:.1%} of train, target {plan.shares[l]:.1%}" + why_not,
                                counts={"have": n, "share": round(n / total, 4),
                                        "target_share": round(plan.shares[l], 4), "deficit": plan.deficits[l]}))
            if n < policy.min_count:
                out.append(Need(kind="small_data", head=h, label=l, remedy=remedy,
                                reason=f"{l} has {n} train rows (< {policy.min_count})" + why_not,
                                counts={"have": n, "min_count": policy.min_count}))

    if st.shape is not None:
        sh, why = st.shape, []
        if sh.uncovered_template_share >= policy.shape_template_share:
            why.append(f"{sh.uncovered_template_share:.0%} of real key templates are absent from synthetic")
        if sh.length_ks_d > policy.shape_ks:
            why.append(f"length KS D {sh.length_ks_d:.2f} > {policy.shape_ks}")
        if sh.channel_gaps:
            why.append(f"channels under half their real share in synthetic: {', '.join(sh.channel_gaps)}")
        if why:
            out.append(Need(kind="shape_gap", head=h, remedy="shaped_generation", reason="; ".join(why),
                            counts={"uncovered_template_share": sh.uncovered_template_share,
                                    "length_ks_d": sh.length_ks_d, "channel_gaps": sh.channel_gaps,
                                    "templates": list(sh.top_uncovered)[:10]}))

    dis, exc = st.teacher_disagreement_rate, st.excluded_rate
    if (dis is not None and dis > policy.noise_disagreement) or (exc is not None and exc > policy.noise_excluded):
        out.append(Need(kind="label_noise", head=h, remedy="adjudicate",
                        reason=f"teacher disagreement {dis if dis is not None else 'n/a'}, excluded rate "
                               f"{exc if exc is not None else 'n/a'}",
                        counts={"disagreement": dis, "excluded_rate": exc}))

    if st.last_run is not None:
        out += confusable_needs(h, st.last_run.val, policy)
        c = control_val_macro_f1
        if c is not None and st.last_run.val.macro_f1 < c - policy.real_hurts_margin:
            out.append(Need(kind="real_hurts", head=h, remedy="cap_real",
                            reason=f"val macro-F1 {st.last_run.val.macro_f1:.3f} < control {c:.3f} - "
                                   f"{policy.real_hurts_margin}",
                            counts={"val_macro_f1": st.last_run.val.macro_f1, "control_val_macro_f1": c}))

    missing_ctl = [m for m in st.required_controls() if not st.controls.get(m)]
    if missing_ctl and fair_baselines:
        out.append(Need(kind="no_control", head=h, remedy="control_run",
                        reason=f"no control run for the current train fingerprint: {', '.join(missing_ctl)}",
                        counts={"mixes": missing_ctl}))

    smoke, old = int(st.cleanup.get("smoke") or 0), int(st.cleanup.get("stale_guard") or 0)
    smoke_stages = int(st.cleanup.get("smoke_stages") or 0)
    if smoke or smoke_stages:
        what = [f"{smoke} shaped rows"] if smoke else []
        what += [f"{smoke_stages} generation stage folders"] if smoke_stages else []
        out.append(Need(kind="smoke_rows", head=h, remedy="clean",
                        reason=f"{' and '.join(what)} were written by a smoke or test run into the real dataset",
                        counts={"smoke": smoke, **({"smoke_stages": smoke_stages} if smoke_stages else {})}))
    if old:
        out.append(Need(kind="reshape", head=h, remedy="clean",
                        reason=f"{old} shaped rows were accepted under an older real-source guard (now "
                               f"v{st.cleanup.get('guard_version')}): recheck, remove the failures, replace them",
                        counts={"stale_guard": old, "guard_version": st.cleanup.get("guard_version")}))
    if st.retrain:
        out.append(Need(kind="retrain", head=h, remedy="full_run",
                        reason=f"the promoted head learned from {st.retrain.get('n')} rows removed since: "
                               + "; ".join(st.retrain.get("reasons") or []),
                        counts={"n": st.retrain.get("n"), "kinds": st.retrain.get("kinds") or {}}))

    new = int(st.stale.get("new_real_rows", 0))
    if new or schedule_due:
        out.append(Need(kind="stale", head=h, remedy="full_run",
                        reason=(f"{new} new real rows since the last run" if new else "the schedule is due"),
                        counts={"new_real_rows": new, "schedule_due": schedule_due}))
    return out


def confusable_needs(head: str, val: SplitMetrics, policy: Policy) -> list[Need]:
    """Pairs holding > ``confusable_share`` of a label's VAL errors (once it has ``confusable_min_errors``)."""
    out = []
    cm = val.confusion
    if cm is None:
        return out
    for i, gold in enumerate(cm.labels):
        if i >= len(cm.matrix):
            break
        row = cm.matrix[i]
        errors = sum(n for j, n in enumerate(row) if j != i)
        if errors < policy.confusable_min_errors:
            continue
        for j, n in enumerate(row):
            if j != i and n / errors > policy.confusable_share:
                out.append(Need(kind="confusable", head=head, label=gold, focus=cm.labels[j], remedy="hard_negatives",
                                reason=f"{gold}->{cm.labels[j]} holds {n}/{errors} of {gold}'s val errors",
                                counts={"pair_errors": n, "label_errors": errors}))
    return out
