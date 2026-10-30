"""Rules layer: legal paths, illegal events raise, actor permissions, guards."""
import pytest

from forge.rules import (AGENT_EVENTS, HUMAN_EVENTS, SYSTEM_EVENTS, Facts, Project, Rejected, all_transitions,
                         gate_check)

PASS_REPORT = {"metric": "accuracy", "threshold": 0.8, "models": [
    {"name": "finetuned", "accuracy": 0.9, "macro_f1": 0.88, "n": 60},
    {"name": "reference", "accuracy": 0.6, "macro_f1": 0.55, "n": 60},
    {"name": "majority", "accuracy": 0.34, "macro_f1": 0.17, "n": 60}]}
FAIL_REPORT = {**PASS_REPORT, "models": [
    {"name": "finetuned", "accuracy": 0.7, "macro_f1": 0.7, "n": 60},
    {"name": "reference", "accuracy": 0.6, "macro_f1": 0.55, "n": 60},
    {"name": "majority", "accuracy": 0.34, "macro_f1": 0.17, "n": 60}]}


def ready_facts(**kw) -> Facts:
    f = Facts(proposal_verdict="fit", opportunity_ids=["opp1"], dataset_size=200, labels=["a", "b"],
              train_labels=["a", "b"], min_examples=120, checkpoint_exists=True, report=PASS_REPORT,
              manifest_exists=True, attempts=1, max_attempts=3, base_model="mlx:qwen",
              base_model_candidates=["mlx:qwen", "llamacpp:winnow"])
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def to_reviewing(p: Project) -> None:
    p.fire("start", "system")
    p.fire("propose", "agent", "fit")
    p.fire("accept", "human", "", opportunity_id="opp1")
    p.fire("begin_curation", "system")
    p.fire("curated", "system")
    p.fire("labels_settled", "system")
    p.fire("selected", "system")
    p.fire("trained", "system")
    p.fire("evaluated", "system")


def test_happy_path_records_actor_and_reason():
    p = Project("t", facts=ready_facts())
    to_reviewing(p)
    p.fire("ship", "agent", "gate passed")
    p.fire("packaged", "system", "built")
    assert p.state == "delivered"
    assert [e for _, e, _ in p.path] == ["start", "propose", "accept", "begin_curation", "curated", "labels_settled",
                                         "selected", "trained", "evaluated", "ship", "packaged"]
    ship = next(l for l in p.log if l["event"] == "ship")
    assert ship["actor"] == "agent" and ship["reason"] == "gate passed"
    assert p.allowed_events == [] or p.is_terminated


def test_agent_cannot_accept_and_is_never_offered_human_events():
    p = Project("t", facts=ready_facts())
    p.fire("start", "system")
    p.fire("propose", "agent", "fit")
    assert "accept" not in p.allowed_for("agent")
    assert set(p.allowed_for("agent")) <= AGENT_EVENTS
    with pytest.raises(Rejected, match="only human"):
        p.fire("accept", "agent", "I like it", opportunity_id="opp1")
    assert p.state == "proposed"


def test_cannot_ship_failed_gate():
    p = Project("t", facts=ready_facts(report=FAIL_REPORT))
    to_reviewing(p)
    with pytest.raises(Rejected, match="threshold"):
        p.fire("ship", "agent", "looks fine")
    assert p.state == "reviewing"
    assert gate_check(FAIL_REPORT)[0] is False


def test_gate_requires_beating_reference_even_above_threshold():
    rep = {**PASS_REPORT, "models": [{"name": "finetuned", "accuracy": 0.9, "macro_f1": 0.9, "n": 9},
                                     {"name": "reference", "accuracy": 0.92, "macro_f1": 0.9, "n": 9},
                                     {"name": "majority", "accuracy": 0.3, "macro_f1": 0.1, "n": 9}]}
    ok, why = gate_check(rep)
    assert not ok and any("reference" in w for w in why)


def test_illegal_transition_raises():
    p = Project("t", facts=ready_facts())
    with pytest.raises(Rejected, match="not a transition"):
        p.fire("ship", "agent", "skip ahead")
    with pytest.raises(Rejected):
        p.fire("nonsense", "agent")
    assert p.state == "created"


def test_propose_guard_follows_verdict():
    p = Project("t", facts=ready_facts(proposal_verdict="no_fit"))
    p.fire("start", "system")
    with pytest.raises(Rejected, match="verdict"):
        p.fire("propose", "agent", "fit anyway")
    p.fire("declare_no_fit", "agent", "honest")
    assert p.state == "no_fit"


def test_accept_requires_known_opportunity_and_revise_requires_feedback():
    p = Project("t", facts=ready_facts())
    p.fire("start", "system")
    p.fire("propose", "agent", "fit")
    with pytest.raises(Rejected):
        p.fire("accept", "human", opportunity_id="nope")
    with pytest.raises(Rejected, match="feedback"):
        p.fire("revise", "human", feedback="   ")
    p.fire("revise", "human", feedback="use 4 labels")
    assert p.state == "analyzing"


def test_labels_settled_requires_size_and_all_labels_in_train():
    """The dataset guard sits on leaving `labelling`: the real rows the agent labels join the dataset first."""
    p = Project("t", facts=ready_facts(dataset_size=50))
    p.fire("start", "system"); p.fire("propose", "agent"); p.fire("accept", "human", opportunity_id="opp1")
    p.fire("begin_curation", "system")
    p.fire("curated", "system")
    assert p.state == "labelling"
    with pytest.raises(Rejected, match="min 120"):
        p.fire("labels_settled", "system")
    p.facts = ready_facts(train_labels=["a"])
    with pytest.raises(Rejected, match="missing from train"):
        p.fire("labels_settled", "system")
    p.facts = ready_facts()
    p.fire("labels_settled", "system")
    assert p.state == "selecting"


def test_selected_requires_a_candidate_base_model():
    p = Project("t", facts=ready_facts(base_model=None))
    p.fire("start", "system"); p.fire("propose", "agent"); p.fire("accept", "human", opportunity_id="opp1")
    p.fire("begin_curation", "system"); p.fire("curated", "system"); p.fire("labels_settled", "system")
    with pytest.raises(Rejected, match="no base model"):
        p.fire("selected", "system")
    p.facts = ready_facts(base_model="hf:other")
    with pytest.raises(Rejected, match="not chosen from candidates"):
        p.fire("selected", "system")
    p.facts = ready_facts()
    p.fire("selected", "system")
    assert p.state == "training"


def test_recurate_bounded_by_attempts_and_give_up():
    p = Project("t", facts=ready_facts(report=FAIL_REPORT, attempts=3))
    to_reviewing(p)
    with pytest.raises(Rejected, match="attempts"):
        p.fire("recurate", "agent", "more data")
    p.fire("give_up", "agent", "exhausted")
    assert p.state == "no_fit"


def test_recurate_loops_back_to_curating():
    p = Project("t", facts=ready_facts(report=FAIL_REPORT, attempts=1))
    to_reviewing(p)
    p.fire("recurate", "agent", "hard negatives billing vs account")
    assert p.state == "curating"


def test_fail_then_retry_returns_to_failed_state():
    p = Project("t", facts=ready_facts())
    p.fire("start", "system"); p.fire("propose", "agent"); p.fire("accept", "human", opportunity_id="opp1")
    p.fire("begin_curation", "system"); p.fire("curated", "system"); p.fire("labels_settled", "system")
    p.fire("selected", "system")
    p.fire("fail", "system", "OOM")
    assert p.state == "errored" and p.failed_from == "training"
    with pytest.raises(Rejected):
        p.fire("retry", "agent")
    p.fire("retry", "human", "try again")
    assert p.state == "training"


def test_cancel_from_any_non_final_state_and_final_is_final():
    for n_steps in range(0, 4):
        p = Project("t", facts=ready_facts())
        steps = [("start", "system", {}), ("propose", "agent", {}), ("accept", "human", {"opportunity_id": "opp1"})]
        for ev, actor, kw in steps[:n_steps]:
            p.fire(ev, actor, **kw)
        p.fire("cancel", "human")
        assert p.state == "cancelled"
        with pytest.raises(Rejected):
            p.fire("retry", "human")


def test_snapshot_restore_roundtrip():
    p = Project("t", facts=ready_facts())
    to_reviewing(p)
    p.fire("fail", "system", "crash")
    q = Project.restore(p.snapshot(), facts=ready_facts())
    assert q.state == "errored" and q.failed_from == "reviewing" and q.path == p.path and q.log == p.log
    q.fire("retry", "human")
    assert q.state == "reviewing"


def test_actor_partition_and_transition_table():
    assert not (AGENT_EVENTS & HUMAN_EVENTS) and not (AGENT_EVENTS & SYSTEM_EVENTS) and not (HUMAN_EVENTS & SYSTEM_EVENTS)
    evs = {e for _, e, _ in all_transitions()}
    assert evs == AGENT_EVENTS | HUMAN_EVENTS | SYSTEM_EVENTS
    assert ("reviewing", "ship", "packaging") in all_transitions()
    assert ("proposed", "declare_no_fit", "no_fit") in all_transitions()
