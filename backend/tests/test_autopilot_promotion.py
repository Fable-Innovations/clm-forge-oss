"""Promotion: the test gate plus a tolerance against the incumbent's test macro-F1; no incumbent = gate alone."""
from forge.autopilot.promotion import decide

WHY = ["PASS: fine-tuned macro_f1 0.860 >= threshold 0.800"]


def test_within_tolerance_promotes_and_a_regression_does_not():
    assert decide(True, WHY, 0.86, 0.87, 0.01).promote
    d = decide(True, WHY, 0.85, 0.87, 0.01)
    assert not d.promote and d.gate_passed and any("regression" in r for r in d.reasons)
    assert round(d.delta, 3) == -0.02 and d.reasons[0] == WHY[0]


def test_no_incumbent_gate_alone_and_failed_gate_never_promotes():
    d = decide(True, WHY, 0.86, None, 0.01)
    assert d.promote and any("no incumbent" in r for r in d.reasons)
    assert not decide(False, [], 0.99, 0.5, 0.01).promote
    assert decide(False, [], 0.99, 0.5, 0.01).reasons[0] == "test gate failed"


def test_policy_promotes_a_better_failing_candidate_over_a_failing_incumbent_only_when_on():
    fail = ["FAIL: fine-tuned macro_f1 0.700 < threshold 0.750"]
    assert not decide(False, fail, 0.70, 0.60, 0.01, incumbent_gate_passed=False).promote          # off: never
    d = decide(False, fail, 0.70, 0.60, 0.01, incumbent_gate_passed=False, over_failing_incumbent=True)
    assert d.promote and d.policy == "promote_over_failing_incumbent" and "beats the failing incumbent" in d.reasons[-1]
    on = {"over_failing_incumbent": True}
    assert not decide(False, fail, 0.70, 0.60, 0.01, incumbent_gate_passed=True, **on).promote     # incumbent passes
    assert not decide(False, fail, 0.60, 0.60, 0.01, incumbent_gate_passed=False, **on).promote    # not strictly better
    assert not decide(False, fail, 0.70, None, 0.01, **on).promote                                 # no incumbent
    assert not decide(False, fail, 0.70, 0.60, 0.01, {"passed": False, "reason": "regressed"},
                      incumbent_gate_passed=False, **on).promote                                   # the check blocks
    assert decide(True, WHY, 0.86, 0.87, 0.01, incumbent_gate_passed=False, **on).policy is None   # gate passed: rule 1-3
