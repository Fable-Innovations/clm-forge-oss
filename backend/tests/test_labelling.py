"""The labelling queue (forge.labelling): pending rows, user labels win over the teacher, export labels are
ground truth, undo, project dataset upsert, legacy JSONL backfill of label_source."""
import json

import pytest

from forge.labelling import LabelError, LabelQueue, from_example, row_options
from forge.models import Example, RealRow
from forge.store import read_dataset_file

from test_agent_fakes import TASK


def rr(rid, **kw):
    return RealRow(rid=rid, state=f"state {rid}", source=kw.pop("source", "imported"), head="ticket_router", **kw)


def queue(tmp_path, sync=False):
    q = LabelQueue(tmp_path, TASK, dataset_sync=sync)
    q.add([rr("a"), rr("b"), rr("c"),
           rr("e", source="export", label="billing", label_state="labelled", label_source="export")])
    return q


def test_pending_and_counts(tmp_path):
    q = queue(tmp_path)
    assert [r.rid for r in q.pending()] == ["a", "b", "c"]
    assert q.counts() == {"needs_label": 3, "user": 0, "teacher": 0, "agent": 0, "adjudicated": 0, "export": 1,
                          "excluded": 0, "excluded_by_agent": 0, "suggested": 0, "total": 4}
    # re-adding keeps the existing state (a recurate re-imports the same export)
    q.set_user_label("a", "technical")
    assert q.add([rr("a"), rr("d")]) == 1
    assert next(r for r in q.rows() if r.rid == "a").label == "technical"


def test_user_label_wins_over_teacher(tmp_path):
    q = queue(tmp_path)
    q.set_user_label("a", "account")
    stats = q.apply_teacher([{"rid": "a", "label": "billing", "conf": 0.9},
                             {"rid": "b", "label": "technical", "conf": 0.8},
                             {"rid": "e", "label": "technical", "conf": 0.7}])
    rows = {r.rid: r for r in q.rows()}
    assert rows["a"].label == "account" and rows["a"].label_source == "user" and rows["a"].suggested_label is None
    assert rows["b"].label == "technical" and rows["b"].label_source == "teacher" and rows["b"].label_state == "labelled"
    # an export label is ground truth: the teacher only adds a suggestion
    assert rows["e"].label == "billing" and rows["e"].label_source == "export" and rows["e"].suggested_label == "technical"
    assert stats == {"kept_user": 1, "labelled": 1, "export_suggested": 1}
    # and the user can still override the teacher
    q.set_user_label("b", "billing")
    assert next(r for r in q.rows() if r.rid == "b").label_source == "user"


def test_suggest_mode_and_two_pass_disagreement_stay_pending(tmp_path):
    q = queue(tmp_path)
    q.apply_teacher([{"rid": "a", "label": "billing", "conf": 0.6}], mode="suggest")
    q.apply_teacher([{"rid": "b", "label": None, "suggested": "account",
                      "votes": [{"pass": "a", "label": "account"}, {"pass": "b", "label": "technical"}]}])
    rows = {r.rid: r for r in q.rows()}
    assert rows["a"].label_state == "needs_label" and rows["a"].suggested_label == "billing" and rows["a"].label is None
    assert rows["b"].label_state == "needs_label" and rows["b"].suggested_label == "account" and len(rows["b"].teacher_votes) == 2
    assert q.counts()["suggested"] == 2 and q.counts()["needs_label"] == 3


def test_label_must_be_an_option(tmp_path):
    q = queue(tmp_path)
    with pytest.raises(LabelError):
        q.set_user_label("a", "nope")
    with pytest.raises(LabelError):
        q.set_user_label("zzz", "billing")
    # a teacher label outside the options labels nothing
    assert q.apply_teacher([{"rid": "a", "label": "nope"}]) == {"invalid": 1, "disagreed": 1}


def test_undo_walks_back_user_actions(tmp_path):
    q = queue(tmp_path)
    q.set_user_label("a", "billing")
    q.skip("b")
    q.set_user_label("a", "account")
    assert q.undo().label == "billing"            # last action: the relabel
    assert q.undo().skipped is False              # then the skip
    r = q.undo()                                  # then the first label
    assert r.rid == "a" and r.label is None and r.label_state == "needs_label"
    assert q.undo() is None
    log = [json.loads(x) for x in (tmp_path / "labelling_log.jsonl").read_text().splitlines()]
    assert [e["action"] for e in log][-3:] == ["undo", "undo", "undo"]


def test_undo_goes_past_the_entries_of_a_row_deleted_at_its_source(tmp_path):
    """A deletion at the row's source (the adapter's) removes the row and keeps its log entries' seq without their
    content (``redacted``): nothing to restore, so undo reverts the user's previous action instead of failing."""
    q = queue(tmp_path)
    q.set_user_label("a", "billing")
    q.set_user_label("b", "account")
    rows = [json.loads(x) for x in (tmp_path / "real_rows.jsonl").read_text().splitlines()]
    (tmp_path / "real_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["rid"] != "b"))
    log = tmp_path / "labelling_log.jsonl"
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    for e in entries:
        if e.get("rid") == "b":
            e.update(prev=None, next=None, redacted="deleted at the source")
    log.write_text("".join(json.dumps(e) + "\n" for e in entries))
    r = q.undo()
    assert r.rid == "a" and r.label is None
    assert q.undo() is None


def test_skip_orders_skipped_rows_last(tmp_path):
    q = queue(tmp_path)
    q.skip("a")
    total, items = q.page(state="needs_label")
    assert total == 3 and [r.rid for r in items] == ["b", "c", "a"]
    assert [r.rid for r in q.page(state="export")[1]] == ["e"]
    assert q.page(state="all")[0] == 4


def test_exclude_pending_unblocks_and_labelling_rejoins(tmp_path):
    q = queue(tmp_path)
    assert q.exclude_pending() == 3
    assert q.pending() == [] and q.counts()["excluded"] == 3
    q.set_user_label("c", "technical")
    assert q.counts()["excluded"] == 2 and q.counts()["user"] == 1


def test_project_queue_upserts_into_dataset(tmp_path):
    q = queue(tmp_path, sync=True)
    ds = tmp_path / "dataset.jsonl"
    q.set_user_label("a", "technical")
    q.apply_teacher([{"rid": "b", "label": "billing", "conf": 0.9}])
    rows = {e.id: e for e in read_dataset_file(ds)}
    assert set(rows) == {"a", "b"}
    assert rows["a"].label == "technical" and rows["a"].label_source == "user" and rows["a"].source == "imported"
    assert rows["b"].label_source == "teacher" and rows["b"].split == "train"
    q.set_user_label("a", "account")
    assert {e.id: e.label for e in read_dataset_file(ds)}["a"] == "account"
    q.undo()
    q.undo()                                      # back to needs_label: leaves the dataset
    assert {e.id for e in read_dataset_file(ds)} == {"b"}


def test_dynamic_question_options():
    row = RealRow(rid="x", state="s", source="export",
                  question_meta={"question": {"type": "noul", "instructions": "same task?", "criteria": None}})
    noul = TASK.model_copy(update={"type": "noul", "criteria": None})
    assert [o["key"] for o in row_options(noul, row)] == ["true", "false"]
    choice = RealRow(rid="y", state="s", source="export",
                     question_meta={"question": {"type": "choice", "criteria": {"t1": "first", "t2": "second"}}})
    assert row_options(TASK, choice) == [{"key": "t1", "description": "first"}, {"key": "t2", "description": "second"}]


def test_legacy_rows_backfill_label_source():
    def ex(**kw):
        base = {"id": "1", "state": "s", "label": "billing", "split": "train"}
        return Example.model_validate_json(json.dumps({**base, **kw}))
    assert ex(source="imported", meta={"label_origin": "export"}).label_source == "export"
    assert ex(source="imported", meta={"label_origin": "teacher"}).label_source == "teacher"
    assert ex(source="user", meta={"edited": True, "original_source": "synthetic"}).label_source == "user"
    assert ex(source="synthetic", meta={"edited": True}).label_source == "user"
    assert ex(source="mined", rationale="label stated in source").label_source == "export"
    assert ex(source="mined").label_source == "teacher"
    assert ex(source="synthetic").label_source is None
    assert ex(source="hard_negative", label_source="teacher").label_source == "teacher"
    # round trip keeps it
    e = ex(source="imported", meta={"label_origin": "export"})
    assert Example.model_validate_json(e.model_dump_json()).label_source == "export"


def test_from_example_round_trip():
    e = Example(id="i1", state="s", label="billing", source="imported", split="test", label_source="export",
                meta={"frozen_test": True})
    r = from_example(e)
    assert r.label_state == "labelled" and r.split == "test" and r.label_source == "export"
    u = from_example(e, label_state="needs_label")
    assert u.label is None and u.pending


# ---- the labelling agent (agent policy) --------------------------------------------------------------------
def two_pass(answers):
    """A two-pass teacher stub: answers[rid] = [label_a, label_b] (None -> answer outside the options),
    or [] for a failed call; consumed per call so retries can change the answer."""
    calls = []

    async def teacher(rows):
        calls.append([r.rid for r in rows])
        out = []
        for r in rows:
            got = answers[r.rid]
            got = got.pop(0) if got and isinstance(got[0], list) else got
            out.append([{"pass": p, "label": lab, "conf": 0.8, "reason": "off-topic" if lab is None else "fits"}
                        for p, lab in zip("ab", got)])
        return out
    teacher.calls = calls
    return teacher


def adjudicator(verdicts):
    seen = []

    async def adj(disputed):
        seen.extend((r.rid, [v["label"] for v in votes]) for r, votes in disputed)
        return [verdicts[r.rid] for r, _ in disputed]
    adj.seen = seen
    return adj


async def test_agent_labels_agreements_adjudicates_disagreements_and_excludes(tmp_path):
    from forge.labelling import agent_label
    q = queue(tmp_path, sync=True)
    q.add([rr("d"), rr("f")])
    q.set_user_label("c", "billing")                    # user labels always win: the agent never sees them
    teacher = two_pass({"a": ["billing", "billing"], "b": ["billing", "technical"], "d": ["account", "technical"],
                        "f": ["account", None]})
    adj = adjudicator({"b": ("technical", "the error message decides"), "d": (None, "cannot tell")})
    stats = await agent_label(q, teacher, adj)
    rows = {r.rid: r for r in q.rows()}
    assert rows["a"].label == "billing" and rows["a"].label_source == "agent" and not rows["a"].adjudicated
    assert rows["b"].label == "technical" and rows["b"].adjudicated and len(rows["b"].teacher_votes) == 2
    assert rows["d"].label_state == "excluded_by_agent" and "abstained" in rows["d"].exclusion_reason
    assert rows["f"].label_state == "excluded_by_agent" and "not among the options" in rows["f"].exclusion_reason
    assert rows["c"].label_source == "user" and rows["e"].label_source == "export"
    assert adj.seen == [("b", ["billing", "technical"]), ("d", ["account", "technical"])]
    assert stats == {"labelled": 1, "adjudicated": 1, "excluded_by_agent": 2}
    assert q.pending() == [] and q.counts()["agent"] == 2 and q.counts()["excluded_by_agent"] == 2
    # labelled rows join the dataset, excluded ones never do
    assert {e.id: e.label_source for e in read_dataset_file(tmp_path / "dataset.jsonl")} == {
        "a": "agent", "b": "agent", "c": "user"}
    log = [json.loads(x) for x in (tmp_path / "labelling_log.jsonl").read_text().splitlines()]
    assert {e["actor"] for e in log if e["action"] == "agent"} == {"agent"}


async def test_adjudicator_third_answer_excludes_and_missing_adjudicator_refuses(tmp_path):
    from forge.labelling import AgentLabelError, agent_label
    q = queue(tmp_path)
    teacher = two_pass({"a": ["billing", "technical"], "b": ["account", "account"], "c": ["account", "account"]})
    await agent_label(q, teacher, adjudicator({"a": ("account", "neither")}))
    assert next(r for r in q.rows() if r.rid == "a").exclusion_reason.startswith("passes disagreed")
    q2 = queue(tmp_path / "q2")
    with pytest.raises(AgentLabelError, match="no adjudicator"):
        await agent_label(q2, two_pass({"a": ["billing", "account"], "b": ["account"] * 2, "c": ["account"] * 2}))


async def test_agent_retries_failures_then_fails_loudly(tmp_path):
    from forge.labelling import AgentLabelError, agent_label
    q = queue(tmp_path)
    teacher = two_pass({"a": [[], ["billing", "billing"]], "b": [[], [], []], "c": ["account", "account"]})
    with pytest.raises(AgentLabelError, match="no answer for 1 rows after 3 attempts"):
        await agent_label(q, teacher)
    assert teacher.calls == [["a", "b", "c"], ["a", "b"], ["b"]]
    rows = {r.rid: r for r in q.rows()}
    assert rows["a"].label == "billing" and rows["c"].label == "account"     # the answered rows were written
    assert rows["b"].pending                                                 # a failure is never a disposition


async def test_agent_keeps_export_labels_and_counts_contradictions(tmp_path):
    from forge.labelling import agent_label
    q = queue(tmp_path)
    export = [r for r in q.rows() if r.rid == "e"]
    stats = await agent_label(q, two_pass({"e": ["technical", "technical"]}), rows=export)
    e = next(r for r in q.rows() if r.rid == "e")
    assert e.label == "billing" and e.label_source == "export" and e.suggested_label == "technical"
    assert stats == {"export_contradicted": 1}


ADAPTER_WRITER = """
import fcntl, json, sys, time
from pathlib import Path
head, data = Path(sys.argv[1]), Path(sys.argv[2])
path = head / "real_rows.jsonl"
with open(head / ".real_rows.lock", "a") as lock:   # the adapter's read-modify-write (its labelling commit, ingest)
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    (data / "ready").write_text("1")
    time.sleep(1.0)                                  # the UI labels a row meanwhile
    for r in rows:
        if r["rid"] == "b":
            r.update(label="billing", label_state="labelled", label_source="agent", adjudicated=True,
                     adjudication={"model": "adj", "decision": "billing", "reason": "pass a"})
    rows.append({**rows[0], "rid": "n1", "label": None, "label_state": "needs_label", "label_source": None})
    path.write_text("".join(json.dumps(r) + "\\n" for r in rows))
"""


def test_queue_and_adapter_share_one_lock_so_no_write_is_lost(tmp_path):
    """The adapter's processes and the Forge queue lock the same ``.real_rows.lock``: a user label made while an
    adapter commit or ingest holds it waits, re-reads, and both the agent's rows and the user's label survive."""
    import subprocess
    import sys
    import time

    data = tmp_path / "data"
    head = data / "ticket_router"
    head.mkdir(parents=True)
    q = LabelQueue(head, TASK)
    q.add([rr("a"), rr("b"), rr("c")])
    child = subprocess.Popen([sys.executable, "-c", ADAPTER_WRITER, str(head), str(data)])
    deadline = time.time() + 30
    while not (data / "ready").exists():
        assert child.poll() is None and time.time() < deadline, "the adapter writer did not start"
        time.sleep(0.02)
    q.set_user_label("a", "technical")                 # blocks until the adapter's write is done
    assert child.wait(30) == 0
    rows = {r.rid: r for r in q.rows()}
    assert rows["a"].label == "technical" and rows["a"].label_source == "user"
    assert rows["b"].label_source == "agent" and "n1" in rows                 # the adapter's write survived too
    # the queue's rewrite kept the adjudicator's verdict (and any key the adapter adds)
    assert rows["b"].adjudication == {"model": "adj", "decision": "billing", "reason": "pass a"}
    raw = {json.loads(l)["rid"]: json.loads(l) for l in (head / "real_rows.jsonl").read_text().splitlines()}
    assert raw["b"]["adjudication"]["model"] == "adj"


def test_queue_keeps_keys_the_adapter_wrote(tmp_path):
    (tmp_path / "real_rows.jsonl").write_text(json.dumps({
        "rid": "x", "state": "s", "source": "export", "label_state": "needs_label", "adjudication": {"model": "m"},
        "future_key": [1, 2]}) + "\n")
    q = LabelQueue(tmp_path, TASK)
    q.skip("x")
    row = json.loads((tmp_path / "real_rows.jsonl").read_text())
    assert row["adjudication"] == {"model": "m"} and row["future_key"] == [1, 2] and row["skipped"] is True
