"""Scheduler: next slot maths, one run for any number of missed slots, and runs resumed at their cursor."""
from datetime import datetime, timedelta

import pytest

from forge.autopilot.scheduler import Scheduler, next_occurrence
from forge.store import Store

TZ = datetime.now().astimezone().tzinfo


def at(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


def test_next_occurrence_daily_and_weekly():
    assert next_occurrence("daily", None, "03:00", at(2026, 10, 14, 2, 0)) == at(2026, 10, 14, 3, 0)
    assert next_occurrence("daily", None, "03:00", at(2026, 10, 14, 3, 0)) == at(2026, 10, 15, 3, 0)
    # 2026-10-14 is a Wednesday; weekday 0 = Monday
    assert next_occurrence("weekly", 0, "04:30", at(2026, 10, 14, 9, 0)) == at(2026, 10, 19, 4, 30)
    assert next_occurrence("weekly", 2, "10:00", at(2026, 10, 14, 9, 0)) == at(2026, 10, 14, 10, 0)
    for bad in (("hourly", None, "03:00"), ("daily", None, "3am"), ("weekly", None, "03:00")):
        with pytest.raises(ValueError):
            next_occurrence(*bad, at(2026, 10, 14, 0, 0))


def test_missed_runs_enqueue_once_without_a_catch_up_storm(tmp_path):
    store, fired = Store(tmp_path / "f.db"), []
    s = Scheduler(store, lambda tid, trig: fired.append((tid, trig)), lambda tid: True)
    s.put("heads", "daily", None, "03:00", now=at(2026, 10, 1, 12, 0))
    # the server was down for ~2 weeks: one run, and the next slot is in the future
    assert s.tick(now=at(2026, 10, 14, 9, 0)) == ["heads"]
    assert fired == [("heads", "schedule")]
    sched = s.get("heads")
    assert datetime.fromisoformat(sched["next_run"]) == at(2026, 10, 15, 3, 0) and sched["last_run"]
    assert s.tick(now=at(2026, 10, 14, 9, 1)) == []
    assert s.tick(now=at(2026, 10, 15, 3, 0) + timedelta(seconds=5)) == ["heads"]


def test_disabled_schedules_and_targets_do_not_fire(tmp_path):
    store, fired = Store(tmp_path / "f.db"), []
    s = Scheduler(store, lambda tid, trig: fired.append(tid), lambda tid: tid != "off")
    s.put("paused", "daily", None, "03:00", enabled=False, now=at(2026, 10, 1, 0, 0))
    s.put("off", "daily", None, "03:00", now=at(2026, 10, 1, 0, 0))
    assert s.tick(now=at(2026, 10, 14, 9, 0)) == [] and fired == []


async def test_interrupted_run_is_requeued_and_resumes_at_its_cursor(tmp_path):
    from forge.autopilot import AutopilotService
    from forge.jobqueue import JobQueue, Preflight
    store = Store(tmp_path / "f.db")
    svc = AutopilotService(store, JobQueue(store, preflight=lambda: Preflight(True)))
    run = {"id": "r1", "target_id": "t", "trigger": "schedule", "state": "running", "step_cursor": "train:h:1",
           "round": 1, "result": {"steps": {"pull": {}, "train:h:1": {"val": {}}}}, "started": "x", "ended": None,
           "created": "2026-10-14T00:00:00"}
    store.save_run(run)
    svc.restore()
    r = store.get_run("r1")
    assert r["state"] == "queued" and r["step_cursor"] == "train:h:1" and "train:h:1" in r["result"]["steps"]
    j = store.journal(target_id="t", action="interrupted")
    assert j and "re-queued at step train:h:1" in j[0]["reason"]
