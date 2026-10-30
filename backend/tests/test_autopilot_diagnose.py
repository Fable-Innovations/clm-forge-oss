"""Gate-failure diagnosis uses validation evidence only."""
import inspect

from forge.autopilot.detect import SplitMetrics
from forge.autopilot.diagnose import diagnose
from forge.autopilot.targets import Policy


def val(**kw):
    return SplitMetrics.model_validate({"macro_f1": 0.7, "per_label": {"a": 0.9, "b": 0.6},
                                        "confusion": {"labels": ["a", "b"], "matrix": [[90, 2], [12, 30]]}, **kw})


def test_remedies_from_val_confusions_weak_labels_and_real_hurts():
    rem = diagnose("h", val(), threshold=0.8, policy=Policy(), control_val_macro_f1=0.8,
                   real_counts={"a": 300, "b": 20}, target_shares={"a": 0.6, "b": 0.4}, train_total=1000)
    assert [(r.kind, r.label, r.focus) for r in rem] == [("hard_negatives", "b", "a"),
                                                          ("shaped_generation", "b", None), ("cap_real", "a", None)]
    assert rem[0].n == 120 and all(r.reason.startswith(("val", "val F1", "val macro")) for r in rem)
    assert rem[2].share == 0.6 and rem[2].n == 600            # real rows per label capped at a's target share


def test_diagnosis_cannot_be_given_test_evidence():
    params = inspect.signature(diagnose).parameters
    assert not any("test" in p for p in params)
    assert diagnose("h", SplitMetrics(macro_f1=0.95, per_label={"a": 0.95}), threshold=0.8, policy=Policy()) == []
