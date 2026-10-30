"""The labelling queue: every real (not generated) row and who labelled it. See SPEC "Labelling real rows".

One queue per directory:

- a project's queue lives in ``output/projects/<pid>/`` beside ``dataset.jsonl``; a row that becomes
  ``labelled`` is upserted into that dataset (with its ``label_source``) and a row that goes back to
  ``needs_label`` is removed from it, so the dataset never holds an unlabelled row;
- a head directory (``<root>/<head_id>/`` with ``task.json``, written by a head-set adapter) keeps its queue in
  ``real_rows.jsonl`` there; the adapter's training reads the ``labelled`` rows as train-only data.

Files: ``real_rows.jsonl`` is the current state of every row (``store.read/write_real_rows``);
``labelling_log.jsonl`` is an append-only action log ``{seq, rid, action, prev, next, actor, ts}`` that
``undo`` walks back.

Every read-modify-write holds ``forge.filelock.real_rows_lock`` (``<dir>/.real_rows.lock``), the same lock the
head-set adapter's processes take, and re-reads the file inside it.

The rules live here and only here:

- a user label is final for the teacher and the agent: ``apply_teacher`` / ``apply_agent`` never touch a
  row labelled by the user;
- an export label is ground truth: the teacher may only add ``suggested_label`` / ``teacher_conf``, and an
  agent answer that contradicts it is counted (``export_contradicted``) for the journal, never applied;
- a teacher result that is a suggestion (``mixed`` policy, or the two passes disagree) never labels a row;
- the labelling agent (``agent_label``, the default ``agent`` policy) gives every row a final disposition:
  agreeing teacher passes label it (``label_source=agent``), a disagreement goes to the adjudicator, which
  either picks one of the pass labels (``adjudicated``) or the row is ``excluded_by_agent`` with a reason.
  Rows the teacher never answered (a failure, not a disagreement) are retried, then fail the job loudly.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal

from forge import config
from forge.filelock import real_rows_lock
from forge.models import Example, RealRow, TaskSchema
from forge.store import read_dataset_file, read_real_rows, write_dataset_file, write_real_rows

LOG_FILE = "labelling_log.jsonl"
DATASET_FILE = "dataset.jsonl"
POLICIES = ("agent", "user", "teacher", "mixed")
DEFAULT_POLICY = "agent"
QueueFilter = Literal["needs_label", "teacher", "user", "agent", "export", "labelled", "excluded_by_agent", "all"]
_HEAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Example.source for a queue row once it is labelled (Example has no "export" source: those rows are imported)
_EXAMPLE_SOURCE = {"imported": "imported", "mined": "mined", "user": "user", "export": "imported"}

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LabelError(ValueError):
    """A queue action the rules refuse (unknown row, label not among the row's options)."""


def row_task(task: TaskSchema, row: RealRow) -> TaskSchema:
    """The question this row is asked: the task's, or the row's own (heads whose options differ per row)."""
    q = (row.question_meta or {}).get("question")
    if not isinstance(q, dict):
        return task
    upd: dict[str, Any] = {}
    if q.get("instructions"):
        upd["instructions"] = q["instructions"]
    if "criteria" in q:
        upd["criteria"] = q["criteria"]
    return task.model_copy(update=upd)


def row_options(task: TaskSchema, row: RealRow) -> list[dict[str, str]]:
    """[{key, description}] in the order the labeller sees them (number keys 1..n)."""
    t = row_task(task, row)
    crit = t.criteria
    if t.type == "choice":
        return [{"key": k, "description": str(v)} for k, v in (crit or {}).items()]
    if t.type == "score":
        return [{"key": str(i), "description": str(v)} for i, v in enumerate(crit or [])]
    c = crit if isinstance(crit, dict) else {}
    return [{"key": "true", "description": str(c.get("true", "yes"))},
            {"key": "false", "description": str(c.get("false", "no"))}]


class LabelQueue:
    """The queue in one directory. ``task`` validates labels (None: any label is accepted, used by the
    curator, which only ever writes labels it already normalised). ``dataset_sync`` mirrors labelled rows
    into ``<dir>/dataset.jsonl`` (projects)."""

    def __init__(self, d: Path | str, task: TaskSchema | None = None, *, dataset_sync: bool = False):
        self.dir = Path(d)
        self.task = task
        self.dataset_sync = dataset_sync

    def _locked(self):
        """``forge.filelock.real_rows_lock``: the lock the head-set adapter's processes take too (every writer
        re-reads ``real_rows.jsonl`` inside it), so a user label and an agent commit or ingest never overwrite
        each other."""
        return real_rows_lock(self.dir)

    # ---- files -----------------------------------------------------------------------------------
    def rows(self) -> list[RealRow]:
        return read_real_rows(self.dir)

    def _log_path(self) -> Path:
        return self.dir / LOG_FILE

    def _read_log(self) -> list[dict]:
        p = self._log_path()
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def _append_log(self, entries: list[dict]) -> None:
        if not entries:
            return
        seq = len(self._read_log())
        with self._log_path().open("a") as f:
            for e in entries:
                seq += 1
                f.write(json.dumps({"seq": seq, "ts": now(), **e}, default=str) + "\n")

    def _commit(self, rows: list[RealRow], changed: list[tuple[RealRow | None, RealRow, str, str]],
                via: dict | None = None) -> None:
        """Write rows, log each (prev, next, action, actor), and keep a project's dataset in step. ``via``: where
        a user action came from (``forge.web_guard.browser_origin``: the Forge page's origin and user agent),
        logged with it so an ``actor=user`` entry says which browser session made it."""
        write_real_rows(self.dir, rows)
        self._append_log([{"rid": nxt.rid, "action": action, "actor": actor,
                           "prev": prev.model_dump() if prev else None, "next": nxt.model_dump(),
                           **({"via": via} if via else {})}
                          for prev, nxt, action, actor in changed])
        if self.dataset_sync:
            self._sync_dataset([nxt for _p, nxt, _a, _actor in changed])

    def _sync_dataset(self, rows: list[RealRow]) -> None:
        if not rows:
            return
        p = self.dir / DATASET_FILE
        ds = read_dataset_file(p)
        idx = {e.id: i for i, e in enumerate(ds)}
        drop: set[str] = set()
        for r in rows:
            if r.label_state == "labelled" and r.label is not None:
                ex = to_example(r)
                if r.rid in idx:
                    ds[idx[r.rid]] = ex
                else:
                    idx[r.rid] = len(ds)
                    ds.append(ex)
            elif r.rid in idx:
                drop.add(r.rid)
        write_dataset_file(p, [e for e in ds if e.id not in drop])

    # ---- reading ---------------------------------------------------------------------------------
    def pending(self) -> list[RealRow]:
        return [r for r in self.rows() if r.pending]

    def counts(self, head: str | None = None) -> dict[str, int]:
        rows = [r for r in self.rows() if head is None or r.head == head]
        src = Counter(r.label_source for r in rows if r.label_state == "labelled")
        return {"needs_label": sum(r.pending for r in rows), "user": src["user"], "teacher": src["teacher"],
                "agent": src["agent"], "adjudicated": sum(r.adjudicated and r.label_state == "labelled" for r in rows),
                "export": src["export"], "excluded": sum(r.excluded and r.label_state == "needs_label" for r in rows),
                "excluded_by_agent": sum(r.label_state == "excluded_by_agent" for r in rows),
                "suggested": sum(r.pending and r.suggested_label is not None for r in rows), "total": len(rows)}

    def page(self, head: str | None = None, state: QueueFilter = "needs_label", offset: int = 0,
             limit: int = 50) -> tuple[int, list[RealRow]]:
        rows = [r for r in self.rows() if head is None or r.head == head]
        if state == "needs_label":
            rows = [r for r in rows if r.label_state == "needs_label"]
            rows.sort(key=lambda r: (r.excluded, r.skipped))   # stable: queue order within each group
        elif state in ("teacher", "user", "agent", "export"):
            rows = [r for r in rows if r.label_state == "labelled" and r.label_source == state]
        elif state in ("labelled", "excluded_by_agent"):
            rows = [r for r in rows if r.label_state == state]
        elif state != "all":
            raise LabelError("state must be needs_label|teacher|user|agent|export|labelled|excluded_by_agent|all, "
                             f"got {state!r}")
        return len(rows), rows[max(0, offset):max(0, offset) + max(1, min(limit, 500))]

    def options(self, row: RealRow) -> list[dict[str, str]]:
        return row_options(self.task, row) if self.task else []

    # ---- writing ---------------------------------------------------------------------------------
    def add(self, new: Iterable[RealRow], actor: str = "system") -> int:
        """Insert rows not in the queue yet. Rows already queued keep their state (a recurate re-imports the
        same export; a user's label must survive that)."""
        with self._locked():
            rows = self.rows()
            have = {r.rid for r in rows}
            added = []
            for r in new:
                if r.rid in have:
                    continue
                have.add(r.rid)
                r = r.model_copy(update={"updated_at": now()})
                rows.append(r)
                added.append((None, r, "add", actor))
            if added:
                write_real_rows(self.dir, rows)
                self._append_log([{"rid": nxt.rid, "action": "add", "actor": actor, "prev": None, "next": None}
                                  for _p, nxt, _a, _actor in added])
            return len(added)

    def _find(self, rows: list[RealRow], rid: str) -> int:
        for i, r in enumerate(rows):
            if r.rid == rid:
                return i
        raise LabelError(f"no row {rid!r} in this queue")

    def set_user_label(self, rid: str, label: str, via: dict | None = None) -> RealRow:
        with self._locked():
            rows = self.rows()
            i = self._find(rows, rid)
            prev = rows[i]
            keys = [o["key"] for o in self.options(prev)]
            if keys and label not in keys:
                raise LabelError(f"label must be one of {keys}")
            nxt = prev.model_copy(update={"label": label, "label_state": "labelled", "label_source": "user",
                                          "skipped": False, "excluded": False, "adjudicated": False,
                                          "adjudication": None, "exclusion_reason": None, "updated_at": now()})
            rows[i] = nxt
            self._commit(rows, [(prev, nxt, "label", "user")], via)
            return nxt

    def skip(self, rid: str, via: dict | None = None) -> RealRow:
        with self._locked():
            rows = self.rows()
            i = self._find(rows, rid)
            prev = rows[i]
            nxt = prev.model_copy(update={"skipped": True, "updated_at": now()})
            rows[i] = nxt
            self._commit(rows, [(prev, nxt, "skip", "user")], via)
            return nxt

    def exclude_pending(self) -> int:
        """The user chose to go on without the rows still pending: they stop blocking the gate, stay in the
        queue, and join the dataset as soon as someone labels them."""
        with self._locked():
            rows = self.rows()
            changed = []
            for i, r in enumerate(rows):
                if r.pending:
                    rows[i] = r.model_copy(update={"excluded": True, "updated_at": now()})
                    changed.append((r, rows[i], "exclude", "user"))
            if changed:
                self._commit(rows, changed)
            return len(changed)

    def undo(self, via: dict | None = None) -> RealRow | None:
        """Revert the most recent user action not undone yet (label, skip). -> the restored row, or None."""
        with self._locked():
            log = self._read_log()
            undone = {e["undo_of"] for e in log if e.get("undo_of")}
            # an entry of a row deleted at its source keeps its seq but not its content (``redacted``): there is
            # nothing to restore, so undo goes past it to the user's previous action
            target = next((e for e in reversed(log) if e.get("actor") == "user" and e.get("action") in ("label", "skip")
                           and e["seq"] not in undone and not e.get("redacted")), None)
            if target is None:
                return None
            rows = self.rows()
            i = self._find(rows, target["rid"])
            cur = rows[i]
            restored = RealRow.model_validate({**target["prev"], "updated_at": now()})
            rows[i] = restored
            write_real_rows(self.dir, rows)
            self._append_log([{"rid": restored.rid, "action": "undo", "actor": "user", "undo_of": target["seq"],
                               "prev": cur.model_dump(), "next": restored.model_dump(), **({"via": via} if via else {})}])
            if self.dataset_sync:
                self._sync_dataset([restored])
            return restored

    def apply_teacher(self, results: Iterable[dict], mode: Literal["label", "suggest"] = "label") -> dict[str, int]:
        """Feed teacher results back. Each result: ``{rid, label (None when the passes disagree), conf?,
        votes?: [{pass, label, conf}], suggested?}``.

        ``mode="label"`` labels rows that nobody labelled yet (teacher-labelled rows may be relabelled);
        ``mode="suggest"`` (the ``mixed`` policy) only fills ``suggested_label``. User rows are never touched;
        export rows only get the suggestion."""
        stats = Counter()
        with self._locked():
            rows = self.rows()
            idx = {r.rid: i for i, r in enumerate(rows)}
            changed = []
            for res in results:
                i = idx.get(res.get("rid"))
                if i is None:
                    stats["unknown"] += 1
                    continue
                prev = rows[i]
                if prev.label_source == "user":
                    stats["kept_user"] += 1
                    continue
                label, conf, votes = res.get("label"), res.get("conf"), list(res.get("votes") or [])
                keys = [o["key"] for o in self.options(prev)]
                if label is not None and keys and label not in keys:
                    label = None
                    stats["invalid"] += 1
                suggestion = label if label is not None else res.get("suggested")
                upd: dict[str, Any] = {"suggested_label": suggestion, "teacher_conf": conf, "teacher_votes": votes,
                                       "updated_at": now()}
                if prev.label_source == "export":
                    stats["export_suggested"] += 1
                elif mode == "label" and label is not None:
                    upd.update({"label": label, "label_state": "labelled", "label_source": "teacher",
                                "skipped": False, "excluded": False})
                    stats["labelled"] += 1
                else:
                    stats["disagreed" if label is None and mode == "label" else "suggested"] += 1
                rows[i] = prev.model_copy(update=upd)
                changed.append((prev, rows[i], "teacher", "teacher"))
            if changed:
                self._commit(rows, changed)
        return dict(stats)


    def apply_agent(self, results: Iterable[dict]) -> dict[str, int]:
        """Feed the labelling agent's dispositions back. Each result: ``{rid, label (None = exclude),
        votes, adjudicated, reason}``. User rows are never touched; an export row keeps its label and a
        contradicting answer is only counted (``export_contradicted``) and kept as ``suggested_label``."""
        stats = Counter()
        with self._locked():
            rows = self.rows()
            idx = {r.rid: i for i, r in enumerate(rows)}
            changed = []
            for res in results:
                i = idx.get(res.get("rid"))
                if i is None:
                    stats["unknown"] += 1
                    continue
                prev = rows[i]
                label, votes = res.get("label"), list(res.get("votes") or [])
                if prev.label_source == "user":
                    stats["kept_user"] += 1
                    continue
                keys = [o["key"] for o in self.options(prev)]
                if label is not None and keys and label not in keys:
                    raise LabelError(f"agent label {label!r} for {prev.rid} is not one of {keys}")
                upd: dict[str, Any] = {"teacher_votes": votes, "updated_at": now()}
                if prev.label_source == "export":
                    if label is not None and label != prev.label:
                        upd["suggested_label"] = label
                        stats["export_contradicted"] += 1
                    else:
                        stats["export_confirmed"] += 1
                elif label is None:
                    upd.update({"label": None, "label_state": "excluded_by_agent", "label_source": None,
                                "exclusion_reason": res.get("reason") or "excluded by the labelling agent",
                                "adjudicated": bool(res.get("adjudicated")), "skipped": False})
                    stats["excluded_by_agent"] += 1
                else:
                    upd.update({"label": label, "label_state": "labelled", "label_source": "agent",
                                "adjudicated": bool(res.get("adjudicated")), "exclusion_reason": None,
                                "skipped": False, "excluded": False})
                    stats["adjudicated" if res.get("adjudicated") else "labelled"] += 1
                rows[i] = prev.model_copy(update=upd)
                changed.append((prev, rows[i], "agent", "agent"))
            if changed:
                self._commit(rows, changed)
        return dict(stats)


# ---- the labelling agent ----------------------------------------------------------------------------
# One teacher vote: {"pass": str, "label": str | None, "conf": float, "reason": str}. A vote whose label is
# None but has a reason is an answer outside the options; a row with no votes at all is a teacher failure.
Vote = dict
Teacher = Callable[[list[RealRow]], Awaitable[list[list[Vote]]]]
# Adjudicator: sees each disputed row with both passes' votes and rationales -> [(label | None, reason)].
Adjudicator = Callable[[list[tuple[RealRow, list[Vote]]]], Awaitable[list[tuple[str | None, str]]]]


class AgentLabelError(RuntimeError):
    """The labelling agent could not give every row a disposition (teacher failures after the retries)."""


async def agent_label(queue: LabelQueue, teacher: Teacher, adjudicator: Adjudicator | None = None, *,
                      retries: int = 2, rows: list[RealRow] | None = None) -> dict[str, int]:
    """Give every undisposed row in ``queue`` (or ``rows``) a final disposition (SPEC "Labelling agent").

    1. The teacher answers each row (one vote per pass). Rows with no vote at all are failures, not
       disagreements: they are asked again, up to ``retries`` more times.
    2. Agreeing votes label the row (``label_source=agent``).
    3. Disagreeing votes go to ``adjudicator``: a pass label -> labelled, ``adjudicated``; an abstention or
       a third answer -> ``excluded_by_agent`` with the reason. A vote outside the options excludes the row.
    4. Rows the teacher still never answered stay pending and raise ``AgentLabelError`` after every other
       row was written: a failed teacher is never turned into a label or an exclusion."""
    todo = [r for r in (rows if rows is not None else queue.pending())]
    stats: Counter = Counter()
    if not todo:
        return {}
    votes: dict[str, list[Vote]] = {r.rid: [] for r in todo}
    ask = list(todo)
    for attempt in range(retries + 1):
        got = await teacher(ask)
        if len(got) != len(ask):
            raise AgentLabelError(f"teacher returned {len(got)} answers for {len(ask)} rows")
        for r, v in zip(ask, got):
            votes[r.rid] = list(v or [])
        ask = [r for r in ask if not votes[r.rid]]
        if not ask:
            break
        if attempt < retries:
            stats["retried"] += len(ask)
    results, disputed = [], []
    for r in todo:
        v = votes[r.rid]
        if not v:
            continue
        labels = [x.get("label") for x in v]
        if any(lab is None for lab in labels):
            why = next((x.get("reason") for x in v if x.get("label") is None), "") or ""
            results.append({"rid": r.rid, "label": None, "votes": v,
                            "reason": f"teacher answer not among the options: {why[:200]}"})
        elif len(set(labels)) == 1:
            results.append({"rid": r.rid, "label": labels[0], "votes": v})
        else:
            disputed.append((r, v))
    if disputed:
        if adjudicator is None:
            raise AgentLabelError(f"{len(disputed)} rows have disagreeing teacher passes and no adjudicator "
                                  "is configured")
        verdicts = await adjudicator(disputed)
        if len(verdicts) != len(disputed):
            raise AgentLabelError(f"adjudicator returned {len(verdicts)} verdicts for {len(disputed)} rows")
        for (r, v), (lab, why) in zip(disputed, verdicts):
            pass_labels = {x.get("label") for x in v}
            if lab is not None and lab in pass_labels:
                results.append({"rid": r.rid, "label": lab, "votes": v, "adjudicated": True, "reason": why})
            else:
                what = "abstained" if lab is None else f"gave a third answer {lab!r}"
                results.append({"rid": r.rid, "label": None, "votes": v, "adjudicated": True,
                                "reason": f"passes disagreed ({', '.join(sorted(map(str, pass_labels)))}); "
                                          f"adjudicator {what}: {why[:200]}"})
    stats.update(queue.apply_agent(results))
    failed = [r.rid for r in todo if not votes[r.rid]]
    if failed:
        raise AgentLabelError(f"the teacher gave no answer for {len(failed)} rows after {retries + 1} attempts "
                              f"(first: {failed[:5]}); they stay pending")
    return dict(stats)


def to_example(r: RealRow) -> Example:
    """A labelled queue row as a dataset Example (same id, its label_source kept)."""
    meta = {k: v for k, v in (r.question_meta or {}).items()}
    meta.update({"label_origin": r.label_source, "queue": True})
    if r.teacher_conf is not None:
        meta["teacher_confidence"] = r.teacher_conf
    if r.adjudicated:
        meta["adjudicated"] = True
    return Example(id=r.rid, state=r.state, label=r.label or "", source=_EXAMPLE_SOURCE[r.source], split=r.split,
                   rationale=f"labelled by {r.label_source}", meta=meta, label_source=r.label_source)


def from_example(e: Example, *, label_state: str | None = None, head: str | None = None) -> RealRow:
    """An Example (curator output) as a queue row."""
    labelled = (label_state or ("labelled" if e.label else "needs_label")) == "labelled"
    return RealRow(rid=e.id, state=e.state, question_meta=dict(e.meta), source=e.source if e.source in
                   ("imported", "mined", "user") else "imported", split=e.split,
                   label=e.label if labelled else None, label_state="labelled" if labelled else "needs_label",
                   label_source=e.label_source if labelled else None, head=head)


# ---- label sets: projects and registered head directories -------------------------------------------
def labelset_roots() -> list[Path]:
    """Directories holding head dirs (``<root>/<head_id>/task.json``): Settings.labelset_roots plus the
    ``FORGE_LABELSETS`` environment variable (os.pathsep-separated)."""
    import os
    roots = [*config.load_settings().labelset_roots, *os.environ.get("FORGE_LABELSETS", "").split(os.pathsep)]
    out: list[Path] = []
    for r in roots:
        if r and Path(r).expanduser().is_dir() and Path(r).expanduser() not in out:
            out.append(Path(r).expanduser())
    return out


def head_dir(head_id: str) -> Path:
    if not _HEAD_ID.match(head_id or ""):
        raise LabelError(f"bad head id {head_id!r}")
    for root in labelset_roots():
        d = root / head_id
        if (d / "task.json").is_file():
            return d
    raise LabelError(f"no registered head {head_id!r} (roots: {[str(r) for r in labelset_roots()]})")


def head_task(d: Path) -> TaskSchema:
    meta = json.loads((d / "task.json").read_text())
    return TaskSchema(**{k: meta[k] for k in ("name", "question_id", "type", "instructions", "criteria",
                                              "state_description", "state_example")})


def head_labelsets() -> list[dict]:
    """Registered head dirs that have a queue (``real_rows.jsonl``)."""
    out = []
    for root in labelset_roots():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / "task.json").is_file() and (d / "real_rows.jsonl").is_file() and _HEAD_ID.match(d.name):
                out.append({"id": f"h:{d.name}", "kind": "head", "name": d.name, "dir": str(d),
                            "counts": LabelQueue(d).counts()})
    return out
