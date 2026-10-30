"""Gate-failure diagnosis: a failed round's VAL metrics -> the remedies for the next round.

Only validation evidence is used: the function takes the round's val ``SplitMetrics`` and the control's val
macro-F1, never test numbers (test is read once per run, for the final candidate, by the ``gate`` verb).

Remedies, in this order:

- ``hard_negatives`` for every confusable pair (``detect.confusable_needs``), n sized to the pair's errors
  (the head-set adapter generates more rows of the confused label; it has no pair focus yet);
- ``shaped_generation`` for every label whose val F1 is below the round's gate threshold;
- ``cap_real`` when the candidate trails the synthetic control (real_hurts): real rows per label are capped
  at the majority label's target share of the train set (the adapter's ``real_cap``).

Generation sizes are later capped by ``Policy.gen_cap`` for the whole round (the runner journals the carry).
"""
from __future__ import annotations

from pydantic import BaseModel

from forge.autopilot.detect import SplitMetrics, confusable_needs
from forge.autopilot.targets import Policy


class Remedy(BaseModel):
    kind: str                      # hard_negatives | shaped_generation | cap_real
    reason: str
    label: str | None = None
    focus: str | None = None
    n: int = 0
    share: float | None = None


def diagnose(head: str, val: SplitMetrics, *, threshold: float, policy: Policy,
             control_val_macro_f1: float | None = None, real_counts: dict[str, int] | None = None,
             target_shares: dict[str, float] | None = None, train_total: int = 0) -> list[Remedy]:
    out: list[Remedy] = []
    for need in confusable_needs(head, val, policy):
        n = min(policy.gen_cap, max(40, 10 * int(need.counts["pair_errors"])))
        out.append(Remedy(kind="hard_negatives", label=need.label, focus=need.focus, n=n,
                          reason=f"val: {need.reason}"))
    weak = sorted((l, f) for l, f in val.per_label.items() if f < threshold)
    for l, f in weak:
        out.append(Remedy(kind="shaped_generation", label=l, n=min(policy.gen_cap, 100),
                          reason=f"val F1 for {l} is {f:.3f} < {threshold:.3f}"))
    if control_val_macro_f1 is not None and val.macro_f1 < control_val_macro_f1 - policy.real_hurts_margin \
            and real_counts:
        major = max(real_counts, key=lambda l: (real_counts[l], l))
        share = (target_shares or {}).get(major)
        cap = int(share * train_total) if share is not None and train_total else None
        out.append(Remedy(kind="cap_real", label=major, share=share, n=cap or 0,
                          reason=f"val macro-F1 {val.macro_f1:.3f} trails the synthetic control "
                                 f"{control_val_macro_f1:.3f}: cap real rows per label"
                                 + (f" at {cap} ({major}'s {share:.1%} target share)" if cap else "")))
    return out
