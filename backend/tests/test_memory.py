"""Memory governance units: policy, buckets, batch sizing, admission, model lock, watchdog kill.

Fast and model-free (the watchdog and lock tests use tiny helper subprocesses).
Run: ``cd backend && uv run pytest tests/test_memory.py -q``
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from forge.memory import (
    BatchPlanner,
    GiB,
    MemoryAdmissionError,
    MemorySettings,
    ModelShape,
    Watchdog,
    admit,
    bucket_for,
    resolve_policy,
    tree_footprint,
)
from forge.model_lock import ModelJobLock, ModelLockTimeout

TOTAL = 48 * GiB
QWEN8B = ModelShape(weights_bytes=int(4.6 * GiB), hidden=4096, layers=36, intermediate=12288, heads=32)


def policy(**kw):
    return resolve_policy(MemorySettings(**kw), 2048, total_bytes=TOTAL)


# ---- policy -------------------------------------------------------------------------------------
def test_defaults_derive_from_ram():
    p = policy()
    assert p.budget_bytes == int(0.6 * TOTAL)
    assert p.reserve_bytes == int(0.15 * TOTAL)
    assert p.buckets == (32, 64, 128, 256, 512, 1024, 2048)
    assert p.cache_limit_bytes == 2 * GiB
    small = resolve_policy(MemorySettings(), 2048, total_bytes=16 * GiB)
    assert small.reserve_bytes == 4 * GiB          # floor on small machines


def test_explicit_settings_and_bucket_cap():
    p = resolve_policy(MemorySettings(budget_gb=20, reserve_gb=6, buckets=[100, 300, 5000]), 1000, total_bytes=TOTAL)
    assert (p.budget_bytes, p.reserve_bytes) == (20 * GiB, 6 * GiB)
    assert p.buckets == (100, 300, 1000)            # max_len is always the last bucket; larger ones dropped


def test_budget_plus_reserve_must_fit_machine():
    with pytest.raises(ValueError, match="exceed"):
        policy(budget_gb=45, reserve_gb=8)


# ---- buckets / batch sizing -------------------------------------------------------------------------
def test_bucket_for():
    b = (32, 64, 128, 2048)
    assert [bucket_for(n, b) for n in (1, 32, 33, 64, 65, 2047)] == [32, 32, 64, 64, 128, 2048]
    with pytest.raises(ValueError):
        bucket_for(2049, b)


def test_per_token_bytes_matches_calibration():
    s = ModelShape(weights_bytes=0, hidden=1024, layers=28, intermediate=3072, heads=16)
    assert s.per_token_bytes(2048) == 2 * (10 * 1024 + 3 * 3072)          # 38912 B, measured 38916 on 0.6B
    hf = ModelShape(weights_bytes=0, hidden=1024, layers=28, intermediate=3072, heads=16, attention_materialised=True)
    assert hf.per_token_bytes(2048) > s.per_token_bytes(2048)


def test_planner_rows_scale_with_headroom_and_width():
    p = policy(budget_gb=12, reserve_gb=4)
    plan = BatchPlanner(QWEN8B, p, baseline_bytes=6 * GiB)
    assert plan.rows(32) == p.max_rows                 # short rows: capped by max_rows
    r2048 = plan.rows(2048)
    assert 1 <= r2048 < p.max_rows
    assert QWEN8B.batch_bytes(r2048, 2048) <= plan.headroom
    assert QWEN8B.batch_bytes(r2048 + 1, 2048) > plan.headroom
    tight = BatchPlanner(QWEN8B, p, baseline_bytes=int(p.budget_bytes * p.soft_fraction) + 1)
    assert tight.rows(2048) == 1                       # never zero rows


def test_planner_backoff_is_sticky_per_bucket():
    plan = BatchPlanner(QWEN8B, policy(), baseline_bytes=6 * GiB)
    before = plan.rows(512)
    assert plan.backoff(512) == max(1, before // 2)
    assert plan.rows(512) == max(1, before // 2)
    assert plan.rows(256) == plan.policy.max_rows      # other buckets untouched
    for _ in range(10):
        plan.backoff(512)
    assert plan.rows(512) == 1


def test_plan_groups_by_bucket_shortest_first():
    plan = BatchPlanner(QWEN8B, policy(max_rows=2), baseline_bytes=6 * GiB)
    lengths = {0: 500, 1: 10, 2: 40, 3: 20, 4: 30, 5: 600}
    out = plan.plan(lengths)
    assert out == [(32, [1, 3]), (32, [4]), (64, [2]), (512, [0]), (1024, [5])]


# ---- admission -----------------------------------------------------------------------------------------
def test_admission_refuses_over_budget_immediately():
    p = policy(budget_gb=8, reserve_gb=4)
    with pytest.raises(MemoryAdmissionError, match="budget is 8.0 GB"):
        admit(9 * GiB, "winnow", p, lambda *_: None, available=lambda: 40 * GiB)


def test_admission_waits_then_admits():
    p = policy(budget_gb=20, reserve_gb=4)
    free = iter([5 * GiB, 6 * GiB, 30 * GiB])
    t = [0.0]
    notes = []
    admit(10 * GiB, "qwen", p, lambda lvl, m: notes.append(m), available=lambda: next(free),
          sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert notes and "waiting for memory" in notes[0]


def test_admission_wait_is_bounded():
    p = policy(budget_gb=20, reserve_gb=4, admission_wait_s=10)
    t = [0.0]
    with pytest.raises(MemoryAdmissionError, match="waited 10s"):
        admit(10 * GiB, "qwen", p, lambda *_: None, available=lambda: 5 * GiB,
              sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])


# ---- machine-wide model lock --------------------------------------------------------------------------------
HOLDER = textwrap.dedent("""
    import sys, time
    from forge.model_lock import ModelJobLock
    lk = ModelJobLock(sys.argv[1]); lk.acquire("holder-job", 5, lambda *a: None)
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
""")


def _holder(tmp_path, seconds: float) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(tmp_path), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_lock_second_process_waits_and_reports(tmp_path):
    p = _holder(tmp_path, 1.5)
    notes = []
    lk = ModelJobLock(tmp_path)
    t0 = time.monotonic()
    lk.acquire("me", 30, lambda lvl, m: notes.append(m), report_every_s=0.2)
    try:
        assert time.monotonic() - t0 >= 1.0
        assert any("waiting for model lock" in m and str(p.pid) in m and "holder-job" in m for m in notes)
        assert lk.holder()["owner"] == "me"
    finally:
        lk.release()
        p.wait()


def test_lock_wait_times_out_explicitly(tmp_path):
    p = _holder(tmp_path, 5)
    try:
        with pytest.raises(ModelLockTimeout, match="holder-job"):
            ModelJobLock(tmp_path).acquire("me", 0.5, lambda *a: None)
    finally:
        p.kill()
        p.wait()


def test_lock_released_when_holder_dies(tmp_path):
    p = _holder(tmp_path, 60)
    p.kill()
    p.wait()
    lk = ModelJobLock(tmp_path)
    lk.acquire("me", 2, lambda *a: None)
    lk.release()


def test_waiters_are_visible_to_the_holder(tmp_path):
    p = _holder(tmp_path, 3)
    seen = []

    def wait():
        lk = ModelJobLock(tmp_path)
        lk.acquire("waiter", 10, lambda *a: None)
        lk.release()

    th = threading.Thread(target=wait)
    th.start()
    deadline = time.monotonic() + 2
    observer = ModelJobLock(tmp_path)
    while time.monotonic() < deadline and not seen:
        seen = [pid for pid in (int(f.name) for f in observer.waiters.glob("*"))]
        time.sleep(0.05)
    p.wait()
    th.join()
    assert seen          # the waiting process announced itself


# ---- hard budget watchdog -----------------------------------------------------------------------------------
GROW = textwrap.dedent("""
    import time
    hog = []
    while True:
        hog.append(bytearray(50 * 1024 * 1024))
        for i in range(0, len(hog[-1]), 4096):
            hog[-1][i] = 1
        time.sleep(0.05)
""")


def test_watchdog_kills_a_runaway_process_tree():
    p = subprocess.Popen([sys.executable, "-c", GROW])
    hit = []
    w = Watchdog(p.pid, 400 * 1024 * 1024, hit.append, interval=0.05).start()
    try:
        p.wait(timeout=20)
    finally:
        w.stop()
        if p.poll() is None:
            p.kill()
    assert w.tripped and hit and hit[0] > 400 * 1024 * 1024
    assert p.poll() is not None                 # (psutil reaped it, so Popen reports 0, not -SIGTERM)
    assert tree_footprint(p.pid) == 0


def test_a_tripped_watchdog_already_knows_why():
    """``tripped`` turns true only after ``exceeded_bytes`` is recorded, so a caller that sees the flag while the kill
    is still in progress reads the real reason (the encoder's ``trip_message``), never an empty one."""
    from forge.encoder import _WorkerClient
    from forge.memory import resolve_policy, MemorySettings
    p = subprocess.Popen([sys.executable, "-c", GROW])
    seen: list[int] = []
    w = Watchdog(p.pid, 300 * 1024 * 1024, lambda used: None, interval=0.05).start()
    try:
        deadline = time.monotonic() + 20
        while not w.tripped and time.monotonic() < deadline:
            pass                                # spin: catch the flag the instant it turns
        seen.append(w.exceeded_bytes)
        client = _WorkerClient.__new__(_WorkerClient)
        client.spec, client.watchdog = "mlx:test/model", w
        client.policy = resolve_policy(MemorySettings(budget_gb=0.3, reserve_gb=0), 512, total_bytes=8 * 1024 ** 3)
        msg = client.trip_message
    finally:
        w.stop()
        if p.poll() is None:
            p.kill()
            p.wait(timeout=20)
    assert w.tripped and seen[0] > 300 * 1024 * 1024
    assert "over the" in msg and "memory budget; it was terminated" in msg
