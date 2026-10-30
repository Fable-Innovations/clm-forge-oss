"""The promotion rule (SPEC "Autopilot" > "Selection, rounds, promotion").

A candidate is promoted when

1. its TEST gate passed: the ``gate`` verb's ``passed`` (``forge.rules.gate_check`` on the evaluator report,
   plus any extra non-regression gate the adapter has for that head),
2. and its test macro-F1 >= the incumbent's test macro-F1 - ``tol`` (default 0.01 per head). The incumbent
   score comes from its stored ``eval_report.json`` (the adapter reports it with the gate).
3. and it does not regress on the rolling real check set (the gate's ``check`` record, the adapter's
   ``real_check.py``): the deciding stage is prod once its check set is big enough, QA standing in before that.
   ``check.passed`` false blocks; ``null`` (no stage has enough labelled check rows yet) leaves the decision to
   1 and 2, and its reason is recorded. Every stage's check scores are reported beside the frozen test.

With no incumbent the gate alone decides (and the check set has nothing to regress from). A rejected candidate
stays in ``_staging`` and is recorded.

Target policy ``promote_over_failing_incumbent`` (default off, ``targets.Policy``): when the candidate fails its
test gate but the incumbent fails its own (``incumbent_gate_passed`` false) and the candidate's test macro-F1 is
strictly higher (and the check set does not block), the candidate is promoted anyway: a better failing head
replaces a worse failing one. ``Decision.policy`` says so and the runner journals it (``gate_policy``).
"""
from __future__ import annotations

from pydantic import BaseModel


class Decision(BaseModel):
    promote: bool
    gate_passed: bool
    reasons: list[str]
    test_macro_f1: float
    incumbent_test_macro_f1: float | None = None
    delta: float | None = None
    check_passed: bool | None = None
    check_stage: str | None = None
    policy: str | None = None          # a target policy that changed the outcome (promote_over_failing_incumbent)


def decide(gate_passed: bool, gate_reasons: list[str], test_macro_f1: float,
           incumbent_test_macro_f1: float | None, tol: float, check: dict | None = None, *,
           incumbent_gate_passed: bool | None = None, over_failing_incumbent: bool = False) -> Decision:
    reasons = list(gate_reasons) or [f"test gate {'passed' if gate_passed else 'failed'}"]
    delta = None
    within = True
    if incumbent_test_macro_f1 is not None:
        delta = test_macro_f1 - incumbent_test_macro_f1
        within = test_macro_f1 >= incumbent_test_macro_f1 - tol - 1e-12
        reasons.append(f"test macro-F1 {test_macro_f1:.3f} vs incumbent {incumbent_test_macro_f1:.3f} "
                       f"(delta {delta:+.3f}, tolerance {tol:.3f}): {'within' if within else 'regression'}")
    else:
        reasons.append("no incumbent: the gate alone decides")
    check_passed = (check or {}).get("passed")
    if check is not None:
        reasons.append(f"real check: {check.get('reason')}")
    promote, policy = gate_passed and within and check_passed is not False, None
    if (not gate_passed and over_failing_incumbent and incumbent_test_macro_f1 is not None
            and incumbent_gate_passed is False and test_macro_f1 > incumbent_test_macro_f1
            and check_passed is not False):
        promote, policy = True, "promote_over_failing_incumbent"
        reasons.append(f"policy promote_over_failing_incumbent: both miss the gate and the candidate beats the "
                       f"failing incumbent on test ({test_macro_f1:.3f} > {incumbent_test_macro_f1:.3f})")
    return Decision(promote=promote, gate_passed=gate_passed,
                    reasons=reasons, test_macro_f1=test_macro_f1, incumbent_test_macro_f1=incumbent_test_macro_f1,
                    delta=delta, check_passed=check_passed, check_stage=(check or {}).get("deciding"),
                    policy=policy)
