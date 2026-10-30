"""Rules for the automatic labelling stage: no human gate under the `agent` policy, the `labels_settled` guard,
the opt-in `label_review` gate, and `label_override` recorded in any non-terminal state."""
import pytest

from forge.rules import FINAL_STATES, HUMAN_EVENTS, SYSTEM_EVENTS, Project, Rejected, all_transitions

from test_agent_rules import ready_facts


def to_labelling(p: Project) -> None:
    p.fire("start", "system"); p.fire("propose", "agent"); p.fire("accept", "human", opportunity_id="opp1")
    p.fire("begin_curation", "system")
    p.fire("curated", "system")
    assert p.state == "labelling"


def test_labelling_is_a_system_stage_with_no_human_gate_under_agent_policy():
    assert "labels_resolved" not in HUMAN_EVENTS | SYSTEM_EVENTS
    p = Project("t", facts=ready_facts(undisposed_rows=4))
    to_labelling(p)
    # only cancel / label_override for the human; nothing for the agent
    assert set(p.allowed_for("human")) == {"cancel", "label_override"} and p.allowed_for("agent") == []
    # the agent policy never reaches the review gate, and rows without a disposition block settling
    with pytest.raises(Rejected, match="labelling agent owns"):
        p.fire("needs_review", "system")
    with pytest.raises(Rejected, match="4 real rows have no final disposition"):
        p.fire("labels_settled", "system")
    for actor in ("human", "agent"):
        with pytest.raises(Rejected, match="only system"):
            p.fire("labels_settled", actor)
    p.facts = ready_facts(undisposed_rows=0, excluded_by_agent=3)
    p.fire("labels_settled", "system", "every row disposed")
    assert p.state == "selecting" and p.log[-1]["actor"] == "system"


def test_human_policy_opens_the_review_gate_and_review_done_resettles():
    p = Project("t", facts=ready_facts(undisposed_rows=2, label_policy="user"))
    to_labelling(p)
    p.fire("needs_review", "system", "2 rows wait")
    assert p.state == "label_review"
    assert set(p.allowed_for("human")) == {"review_done", "cancel", "label_override"}
    with pytest.raises(Rejected, match="2 real rows still need a label"):
        p.fire("review_done", "human", mode="continue")
    p.fire("review_done", "human", "set the rest aside", mode="exclude")
    assert p.state == "labelling"                     # labelling settles; review_done never skips it
    p.facts = ready_facts(undisposed_rows=0, label_policy="user")
    with pytest.raises(Rejected, match="no real row is waiting"):
        p.fire("needs_review", "system")
    p.fire("labels_settled", "system")
    assert p.state == "selecting"
    assert ("label_review", "review_done", "labelling") in all_transitions()


def test_label_override_records_without_changing_state_in_every_non_terminal_state():
    p = Project("t", facts=ready_facts())
    to_labelling(p)
    n = len(p.path)
    src, dst = p.fire("label_override", "human", "user label billing for r1", rid="r1", label="billing")
    assert src == dst == "labelling" and len(p.path) == n + 1 and p.log[-1]["event"] == "label_override"
    for actor in ("agent", "system"):
        with pytest.raises(Rejected, match="only human"):
            p.fire("label_override", actor)
    states = {s for s, e, _ in all_transitions() if e == "label_override"}
    non_terminal = {s.id for s in Project.states} - FINAL_STATES
    assert states == non_terminal
    assert all(src == dst for src, e, dst in all_transitions() if e == "label_override")
    p.fire("cancel", "human")
    with pytest.raises(Rejected):
        p.fire("label_override", "human")


def test_labelling_fails_and_retries_like_any_stage():
    p = Project("t", facts=ready_facts(undisposed_rows=1))
    to_labelling(p)
    p.fire("fail", "system", "teacher gave no answer for 1 rows")
    assert p.state == "errored" and p.failed_from == "labelling"
    p.fire("retry", "human")
    assert p.state == "labelling"
