"""SQLite persistence (output/forge.db): projects, events (seq per project), chat, the global model-job queue
(``model_jobs``, rules in ``forge.jobqueue``) and the autopilot's targets, schedules, runs and journal
(rules in ``forge.autopilot``).

Dataset rows live in output/projects/<pid>/dataset.jsonl (see curator / dataset helpers below).
One connection per call keeps this safe across the event loop and worker threads.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from forge import config
from forge.models import Example, RealRow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  state TEXT NOT NULL,
  data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  pid TEXT NOT NULL,
  seq INTEGER NOT NULL,
  ts TEXT NOT NULL,
  type TEXT NOT NULL,
  data TEXT NOT NULL,
  PRIMARY KEY (pid, seq)
);
CREATE TABLE IF NOT EXISTS chat (
  pid TEXT NOT NULL,
  id TEXT NOT NULL,
  ts TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  PRIMARY KEY (pid, id)
);
CREATE TABLE IF NOT EXISTS model_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  owner TEXT NOT NULL,
  state TEXT NOT NULL,            -- queued | deferred | running | done | failed | cancelled | interrupted
  enqueued TEXT NOT NULL,
  started TEXT,
  ended TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS autopilot_targets (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,             -- project | headset
  config_json TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS autopilot_schedules (
  target_id TEXT PRIMARY KEY,
  cadence TEXT NOT NULL,          -- daily | weekly
  weekday INTEGER,                -- 0 = Monday (weekly only)
  time_local TEXT NOT NULL,       -- HH:MM, machine local time
  next_run TEXT,
  last_run TEXT,
  enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS autopilot_runs (
  id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL,
  trigger TEXT NOT NULL,          -- schedule | manual | data
  state TEXT NOT NULL,            -- queued | running | done | done_with_failures | failed | interrupted
  step_cursor TEXT,
  round INTEGER NOT NULL DEFAULT 0,
  result_json TEXT NOT NULL DEFAULT '{}',
  started TEXT,
  ended TEXT,
  created TEXT NOT NULL,
  first_started TEXT              -- set once, when the run first starts; a resumed run keeps it
);
CREATE TABLE IF NOT EXISTS autopilot_journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  target_id TEXT NOT NULL,
  head TEXT,
  round INTEGER,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  counts_json TEXT NOT NULL DEFAULT '{}',
  ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS autopilot_journal_target ON autopilot_journal(target_id, id);
CREATE TABLE IF NOT EXISTS autopilot_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_id TEXT NOT NULL,
  action TEXT NOT NULL,           -- a one-time action the user approved (forge.autopilot.actions)
  approved TEXT NOT NULL,         -- who / when approved it, in words
  reason TEXT NOT NULL,
  state TEXT NOT NULL,            -- queued | done
  queued TEXT NOT NULL,
  run_id TEXT,                    -- the run that completed it
  done TEXT,
  last_error TEXT,
  result_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS migrations (
  name TEXT PRIMARY KEY,          -- one-off data migrations (Store.migrate_once)
  applied TEXT NOT NULL
);
"""
# columns added after their table first shipped: (table, column, type)
_ADDED_COLUMNS = [("autopilot_runs", "first_started", "TEXT")]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            for table, col, typ in _ADDED_COLUMNS:
                if col not in {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    def migrate_once(self, name: str, fn) -> bool:
        """Run the data migration ``fn()`` once per database (recorded in ``migrations``); -> whether it ran now.
        The owner module supplies ``fn``."""
        with self._lock:
            with self._conn() as c:
                if c.execute("SELECT 1 FROM migrations WHERE name=?", (name,)).fetchone():
                    return False
            fn()
            with self._conn() as c:
                c.execute("INSERT OR IGNORE INTO migrations(name, applied) VALUES(?,?)", (name, now()))
            return True

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            try:
                c.execute("PRAGMA journal_mode=WAL")
                yield c
                c.commit()
            finally:
                c.close()

    # ---- projects -------------------------------------------------------------------------------
    def save_project(self, rec: dict) -> dict:
        rec = dict(rec)
        rec.setdefault("created_at", now())
        rec["updated_at"] = now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO projects(id,name,created_at,updated_at,state,data) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at, "
                "state=excluded.state, data=excluded.data",
                (rec["id"], rec.get("name", rec["id"]), rec["created_at"], rec["updated_at"],
                 rec.get("machine", {}).get("state", "created"), json.dumps(rec, default=str)),
            )
        return rec

    def get_project(self, pid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT data FROM projects WHERE id=?", (pid,)).fetchone()
        return json.loads(row["data"]) if row else None

    def list_projects(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM projects ORDER BY created_at DESC").fetchall()
        return [json.loads(r["data"]) for r in rows]

    def delete_project(self, pid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM projects WHERE id=?", (pid,))
            c.execute("DELETE FROM events WHERE pid=?", (pid,))
            c.execute("DELETE FROM chat WHERE pid=?", (pid,))

    # ---- events ---------------------------------------------------------------------------------
    def max_seq(self, pid: str) -> int:
        with self._conn() as c:
            row = c.execute("SELECT MAX(seq) AS m FROM events WHERE pid=?", (pid,)).fetchone()
        return int(row["m"] or 0)

    def add_event(self, pid: str, ev: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO events(pid,seq,ts,type,data) VALUES(?,?,?,?,?)",
                      (pid, ev["seq"], ev["ts"], ev["type"], json.dumps(ev, default=str)))

    def events(self, pid: str, after: int = 0, types: list[str] | None = None) -> list[dict]:
        q, args = "SELECT data FROM events WHERE pid=? AND seq>?", [pid, after]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        q += " ORDER BY seq"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ---- chat -----------------------------------------------------------------------------------
    def add_chat(self, pid: str, mid: str, role: str, content: str) -> dict:
        msg = {"id": mid, "ts": now(), "role": role, "content": content}
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO chat(pid,id,ts,role,content) VALUES(?,?,?,?,?)",
                      (pid, mid, msg["ts"], role, content))
        return msg

    def chat(self, pid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id,ts,role,content FROM chat WHERE pid=? ORDER BY ts", (pid,)).fetchall()
        return [dict(r) for r in rows]


    # ---- model-job queue (rules in forge.jobqueue) ------------------------------------------------
    def add_model_job(self, kind: str, owner: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO model_jobs(kind,owner,state,enqueued) VALUES(?,?,?,?)",
                            (kind, owner, "queued", now()))
            return int(cur.lastrowid)

    def update_model_job(self, jid: int, **cols: Any) -> None:
        if not cols:
            return
        sets = ",".join(f"{k}=?" for k in cols)
        with self._conn() as c:
            c.execute(f"UPDATE model_jobs SET {sets} WHERE id=?", (*cols.values(), jid))

    def model_jobs(self, states: tuple[str, ...] | None = None, limit: int = 200) -> list[dict]:
        q, args = "SELECT * FROM model_jobs", []
        if states:
            q += f" WHERE state IN ({','.join('?' * len(states))})"
            args += list(states)
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, (*args, limit)).fetchall()]

    def interrupt_model_jobs(self) -> int:
        """Jobs left queued, deferred or running by a previous server: their owners re-queue them."""
        with self._conn() as c:
            cur = c.execute("UPDATE model_jobs SET state='interrupted', ended=? "
                            "WHERE state IN ('queued','deferred','running')", (now(),))
            return cur.rowcount

    # ---- autopilot (rules in forge.autopilot) ----------------------------------------------------
    def save_target(self, tid: str, kind: str, config: dict, enabled: bool) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO autopilot_targets(id,kind,config_json,enabled) VALUES(?,?,?,?) "
                      "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, config_json=excluded.config_json, "
                      "enabled=excluded.enabled", (tid, kind, json.dumps(config, default=str), int(enabled)))

    def targets(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM autopilot_targets ORDER BY id").fetchall()
        return [{"id": r["id"], "kind": r["kind"], "config": json.loads(r["config_json"]),
                 "enabled": bool(r["enabled"])} for r in rows]

    def delete_target(self, tid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM autopilot_targets WHERE id=?", (tid,))
            c.execute("DELETE FROM autopilot_schedules WHERE target_id=?", (tid,))

    def save_schedule(self, sched: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO autopilot_schedules(target_id,cadence,weekday,time_local,next_run,last_run,enabled) "
                      "VALUES(:target_id,:cadence,:weekday,:time_local,:next_run,:last_run,:enabled) "
                      "ON CONFLICT(target_id) DO UPDATE SET cadence=excluded.cadence, weekday=excluded.weekday, "
                      "time_local=excluded.time_local, next_run=excluded.next_run, last_run=excluded.last_run, "
                      "enabled=excluded.enabled", {**sched, "enabled": int(bool(sched.get("enabled", True)))})

    def schedules(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM autopilot_schedules ORDER BY target_id").fetchall()
        return [{**dict(r), "enabled": bool(r["enabled"])} for r in rows]

    def save_run(self, run: dict) -> None:
        d = {"first_started": None, **run, "result_json": json.dumps(run.get("result") or {}, default=str)}
        d.pop("result", None)
        with self._conn() as c:
            c.execute("INSERT INTO autopilot_runs(id,target_id,trigger,state,step_cursor,round,result_json,started,"
                      "ended,created,first_started) VALUES(:id,:target_id,:trigger,:state,:step_cursor,:round,"
                      ":result_json,:started,:ended,:created,:first_started) ON CONFLICT(id) DO UPDATE SET "
                      "state=excluded.state, step_cursor=excluded.step_cursor, round=excluded.round, "
                      "result_json=excluded.result_json, started=excluded.started, ended=excluded.ended, "
                      "first_started=COALESCE(autopilot_runs.first_started, excluded.first_started)", d)

    @staticmethod
    def _run(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["result"] = json.loads(d.pop("result_json") or "{}")
        return d

    def get_run(self, rid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM autopilot_runs WHERE id=?", (rid,)).fetchone()
        return self._run(r) if r else None

    def runs(self, target_id: str | None = None, states: tuple[str, ...] | None = None,
             limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM autopilot_runs WHERE 1=1", []
        if target_id:
            q += " AND target_id=?"
            args.append(target_id)
        if states:
            q += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        q += " ORDER BY created DESC, rowid DESC LIMIT ?"
        with self._conn() as c:
            return [self._run(r) for r in c.execute(q, (*args, limit)).fetchall()]

    def add_action(self, target_id: str, action: str, approved: str, reason: str) -> dict:
        with self._conn() as c:
            cur = c.execute("INSERT INTO autopilot_actions(target_id,action,approved,reason,state,queued) "
                            "VALUES(?,?,?,?,'queued',?)", (target_id, action, approved, reason, now()))
            rid = int(cur.lastrowid)
        return self.actions(ids=(rid,))[0]

    def save_action(self, a: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE autopilot_actions SET state=?, run_id=?, done=?, last_error=?, result_json=? WHERE id=?",
                      (a["state"], a.get("run_id"), a.get("done"), a.get("last_error"),
                       json.dumps(a.get("result") or {}, default=str), a["id"]))

    def actions(self, target_id: str | None = None, states: tuple[str, ...] | None = None,
                ids: tuple[int, ...] | None = None) -> list[dict]:
        q, args = "SELECT * FROM autopilot_actions WHERE 1=1", []
        if target_id:
            q += " AND target_id=?"
            args.append(target_id)
        for col, vals in (("state", states), ("id", ids)):
            if vals:
                q += f" AND {col} IN ({','.join('?' * len(vals))})"
                args += list(vals)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d.pop("result_json") or "{}")
            out.append(d)
        return out

    def add_journal(self, entry: dict) -> dict:
        e = {"run_id": None, "head": None, "round": None, "counts": {}, "ts": now(), **entry}
        with self._conn() as c:
            cur = c.execute("INSERT INTO autopilot_journal(run_id,target_id,head,round,action,reason,counts_json,ts) "
                            "VALUES(?,?,?,?,?,?,?,?)", (e["run_id"], e["target_id"], e["head"], e["round"],
                                                        e["action"], e["reason"], json.dumps(e["counts"], default=str),
                                                        e["ts"]))
            e["id"] = int(cur.lastrowid)
        return e

    def journal(self, *, target_id: str | None = None, run_id: str | None = None, head: str | None = None,
                action: str | None = None, after: int = 0, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM autopilot_journal WHERE id>?", [after]
        for col, v in (("target_id", target_id), ("run_id", run_id), ("head", head), ("action", action)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, (*args, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["counts"] = json.loads(d.pop("counts_json") or "{}")
            out.append(d)
        return out


# ---- dataset files ------------------------------------------------------------------------------
_ds_lock = threading.Lock()


def dataset_path(pid: str) -> Path:
    return config.project_dir(pid) / "dataset.jsonl"


def read_dataset(pid: str) -> list[Example]:
    return read_dataset_file(dataset_path(pid))


def write_dataset(pid: str, rows: list[Example]) -> None:
    write_dataset_file(dataset_path(pid), rows)


def read_dataset_file(p: Path) -> list[Example]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(Example.model_validate_json(line))
    return out


def write_dataset_file(p: Path, rows: list[Example]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with _ds_lock:
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        tmp.replace(p)


def dataset_counts(rows: list[Example]) -> dict[str, Any]:
    c: dict[str, dict[str, int]] = {"by_label": {}, "by_source": {}, "by_split": {}}
    for r in rows:
        c["by_label"][r.label] = c["by_label"].get(r.label, 0) + 1
        c["by_source"][r.source] = c["by_source"].get(r.source, 0) + 1
        c["by_split"][r.split] = c["by_split"].get(r.split, 0) + 1
    return c


# ---- labelling queue files (rules in forge.labelling) ---------------------------------------------
REAL_ROWS_FILE = "real_rows.jsonl"
_rr_lock = threading.Lock()


def queue_dir(pid: str) -> Path:
    """A project's labelling queue lives beside its dataset."""
    return config.project_dir(pid)


def read_real_rows(d: Path) -> list[RealRow]:
    p = Path(d) / REAL_ROWS_FILE
    if not p.exists():
        return []
    return [RealRow.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()]


def write_real_rows(d: Path, rows: list[RealRow]) -> None:
    p = Path(d) / REAL_ROWS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with _rr_lock:
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        tmp.replace(p)
