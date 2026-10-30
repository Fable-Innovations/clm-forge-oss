"""The global model-job queue: concurrency 1, FIFO, persisted states, preflight deferral, restart."""
import asyncio

import pytest

from forge.jobqueue import JobQueue, Preflight, foreign_model_processes
from forge.store import Store


def q(tmp_path, preflight=lambda: Preflight(True), defer_s=0.01):
    return JobQueue(Store(tmp_path / "f.db"), preflight=preflight, defer_s=defer_s)


async def test_one_job_at_a_time_in_fifo_order(tmp_path):
    jobs, order, running = q(tmp_path), [], []

    async def job(name, hold):
        async with jobs.slot("train", name):
            running.append(name)
            assert len(running) == 1, f"{running} ran at once"
            order.append(name)
            await asyncio.sleep(hold)
            running.remove(name)

    await asyncio.gather(job("a", 0.05), job("b", 0.01), job("c", 0.0))
    assert order == ["a", "b", "c"]
    rows = sorted(jobs.store.model_jobs(), key=lambda j: j["id"])
    assert [(j["owner"], j["state"]) for j in rows] == [("a", "done"), ("b", "done"), ("c", "done")]
    assert all(j["started"] and j["ended"] for j in rows)


async def test_waiting_jobs_are_visible_as_queued(tmp_path):
    jobs, gate = q(tmp_path), asyncio.Event()

    async def holder():
        async with jobs.slot("train", "holder"):
            await gate.wait()

    async def waiter():
        async with jobs.slot("label", "waiter"):
            pass

    t1 = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    snap = jobs.snapshot()
    assert [(j["owner"], j["state"]) for j in snap["active"]] == [("holder", "running"), ("waiter", "queued")]
    gate.set()
    await asyncio.gather(t1, t2)
    assert jobs.snapshot()["active"] == []


async def test_preflight_defers_with_a_reason_and_kills_nothing(tmp_path):
    answers = [Preflight(False, "memory pressure is warn (level 2); waiting for normal")] * 3 + [Preflight(True)]
    jobs, reasons, states = q(tmp_path, preflight=lambda: answers.pop(0)), [], []

    def on_defer(why):
        reasons.append(why)
        states.append(jobs.store.model_jobs()[0]["state"])

    async with jobs.slot("gate", "autopilot:t:r", on_defer=on_defer):
        pass
    assert reasons == ["memory pressure is warn (level 2); waiting for normal"]   # journalled once per reason
    assert states == ["deferred"]
    j = jobs.store.model_jobs()[0]
    assert j["state"] == "done" and j["error"] is None


async def test_failure_and_cancellation_are_recorded(tmp_path):
    jobs = q(tmp_path)
    with pytest.raises(RuntimeError):
        async with jobs.slot("train", "boom"):
            raise RuntimeError("OOM")
    gate = asyncio.Event()

    async def holder():
        async with jobs.slot("train", "holder"):
            await gate.wait()

    async def waiter():
        async with jobs.slot("train", "waiter"):
            pass

    t1 = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    gate.set()
    await t1
    states = {j["owner"]: (j["state"], j["error"]) for j in jobs.store.model_jobs()}
    assert states["boom"] == ("failed", "RuntimeError: OOM")
    assert states["waiter"][0] == "cancelled" and states["holder"][0] == "done"


def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    st = Store(tmp_path / "f.db")
    a, b, c = (st.add_model_job("train", o) for o in "abc")
    st.update_model_job(a, state="running")
    st.update_model_job(c, state="done")
    assert st.interrupt_model_jobs() == 2
    assert {j["owner"]: j["state"] for j in st.model_jobs()} == {"a": "interrupted", "b": "interrupted", "c": "done"}


def test_foreign_process_scan_ignores_our_children_and_marked_processes():
    import subprocess
    import sys
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", "mlx_lm-fake-marker"])
    try:
        found = foreign_model_processes([])
        assert not any(f"pid {p.pid}:" in f for f in found)     # our own child is not foreign
    finally:
        p.kill()
