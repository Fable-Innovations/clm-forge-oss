"""Pure detectors over a head's status (the adapter's inspect shape)."""
from forge.autopilot.detect import HeadStatus, detect
from forge.autopilot.targets import Policy

POL = Policy()


def status(**kw):
    base = {"head": "h", "counts": {"a": {"real_labelled": 100, "synthetic": 200}, "b": {"real_labelled": 100, "synthetic": 200}},
            "fingerprints": {"synthetic": "f1", "all": "f2"}, "controls": {"synthetic": True}, "gate_threshold": 0.8}
    base.update(kw)
    return HeadStatus.model_validate(base)


def kinds(needs):
    return sorted({n.kind for n in needs})


def test_balanced_head_with_controls_has_no_needs():
    assert detect(status(), POL) == []
    assert status().labels == ["a", "b"]


def test_every_need_carries_reason_and_counts():
    st = status(counts={"a": {"real_labelled": 400, "synthetic": 300}, "b": {"real_labelled": 10, "synthetic": 60, "shaped": 10},
                        "(undisposed)": {"real_needs_label": 3}, "(excluded)": {"excluded_by_agent": 50}},
                teacher_disagreement_rate=0.2, excluded_rate=0.05, stale={"new_real_rows": 7},
                fingerprints={"synthetic": "f1", "synthetic+shaped": "f3", "all": "f2"}, controls={"synthetic": True},
                last_run={"run": "r0", "val": {"macro_f1": 0.6, "per_label": {"a": 0.9, "b": 0.3},
                                               "confusion": {"labels": ["a", "b"], "matrix": [[90, 2], [8, 2]]}}})
    assert st.labels == ["a", "b"] and st.n_undisposed() == 3
    needs = detect(st, POL, control_val_macro_f1=0.7)
    assert kinds(needs) == ["confusable", "imbalance", "label_noise", "no_control", "real_hurts", "small_data",
                            "stale", "unlabelled"]
    assert all(n.reason and n.counts is not None and n.remedy for n in needs)
    conf = next(n for n in needs if n.kind == "confusable")
    assert (conf.label, conf.focus) == ("b", "a")
    assert next(n for n in needs if n.kind == "no_control").counts["mixes"] == ["synthetic+shaped"]
    assert next(n for n in needs if n.kind == "imbalance").label == "b"
    # without a control's val score there is no real_hurts evidence
    assert "real_hurts" not in kinds(detect(st, POL))


def test_label_noise_from_the_excluded_rate_and_unknown_disagreement():
    assert "label_noise" in kinds(detect(status(excluded_rate=0.2), POL))
    assert "label_noise" not in kinds(detect(status(teacher_disagreement_rate=None, excluded_rate=None), POL))


def test_confusable_needs_enough_val_errors():
    st = status(last_run={"val": {"macro_f1": 0.9, "confusion": {"labels": ["a", "b"], "matrix": [[50, 3], [1, 50]]}}})
    assert "confusable" not in kinds(detect(st, POL))      # 3 errors < confusable_min_errors


def test_shape_gap_templates_length_and_channels():
    need = next(n for n in detect(status(shape={"uncovered_template_share": 0.4, "top_uncovered": {"subject|from": 9},
                                                "length_ks_d": 0.31, "channel_gaps": ["slack"]}), POL)
                if n.kind == "shape_gap")
    assert "40%" in need.reason and "KS D 0.31" in need.reason and need.counts["channel_gaps"] == ["slack"]
    assert need.counts["templates"] == ["subject|from"]
    ok = {"uncovered_template_share": 0.1, "top_uncovered": {}, "length_ks_d": 0.1, "channel_gaps": []}
    assert "shape_gap" not in kinds(detect(status(shape=ok), POL))


def test_schedule_due_makes_a_head_stale_and_projects_skip_controls():
    assert kinds(detect(status(), POL, schedule_due=True)) == ["stale"]
    st = status(controls={}, undisposed=5)
    assert kinds(detect(st, POL, fair_baselines=False)) == ["unlabelled"]


def test_smoke_rows_guard_bumps_and_removed_training_rows_are_needs():
    st = status(cleanup={"smoke": 4, "stale_guard": 120, "guard_version": 2},
                retrain={"n": 4, "kinds": {"smoke": 4}, "reasons": ["rows a smoke or test run wrote"]})
    needs = {n.kind: n for n in detect(st, POL)}
    assert set(needs) == {"smoke_rows", "reshape", "retrain"}
    assert needs["smoke_rows"].remedy == needs["reshape"].remedy == "clean"
    assert needs["reshape"].counts == {"stale_guard": 120, "guard_version": 2}
    assert needs["retrain"].remedy == "full_run" and "smoke" in needs["retrain"].reason
    assert detect(status(cleanup={"smoke": 0, "stale_guard": 0}), POL) == []
