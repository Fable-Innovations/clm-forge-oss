"""Embedding worker integration on the smallest cached model (mlx-community/Qwen3-0.6B-4bit).

Asserts the worker's peak memory stays inside an explicit budget, that bucketed batching gives the
same vectors as one-at-a-time, that the memmap output matches, and that the parent's watchdog kill
and admission refusal surface as explicit errors. Loads only the 0.6B model, one worker at a time.
Run: ``cd backend && uv run pytest tests/test_encoder_worker.py -q -s``
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from forge.encoder import Encoder
from forge.memory import GiB, MemoryAdmissionError, MemoryBudgetExceeded, MemorySettings, resolve_policy

SPEC = "mlx:mlx-community/Qwen3-0.6B-4bit"
BUDGET_GB = 2.5
MAX_LEN = 512

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.model_loading


def _policy(budget_gb: float = BUDGET_GB):
    return lambda max_len: resolve_policy(MemorySettings(budget_gb=budget_gb, reserve_gb=2, max_rows=8,
                                                         admission_wait_s=5, lock_wait_s=120), max_len)


def _texts() -> list[str]:
    words = ["invoice", "refund", "crash", "login", "password", "latency", "export", "webhook", "seat", "billing", "outage", "token"]
    out = []
    for i in range(48):
        n = [3, 20, 60, 150, 400][i % 5]          # spans the 32..512 buckets
        out.append(f"ticket {i}: " + " ".join(words[(i + j) % len(words)] for j in range(n)))
    return out


@pytest.fixture
def enc(tmp_path):
    e = Encoder(SPEC, MAX_LEN, cache_dir=tmp_path, policy=_policy())
    yield e
    e.unload()


def test_peak_memory_stays_within_budget(enc, tmp_path):
    notes: list[str] = []
    texts = _texts()
    E = enc.embed(texts, use_cache=False, on_status=lambda lvl, m: notes.append(m))
    assert E.shape == (len(texts), 1024)
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)
    peak = max(enc.last_peak_bytes, enc._client.watchdog.peak_bytes)
    print(f"\n0.6B worker peak {peak / GiB:.2f} GB (worker estimate {enc.last_peak_bytes / GiB:.2f}, "
          f"watchdog {enc._client.watchdog.peak_bytes / GiB:.2f}) vs budget {BUDGET_GB:.2f} GB")
    assert peak <= BUDGET_GB * GiB
    assert any("loaded" in m for m in notes) and any("worker peak" in m for m in notes)

    # fixed-width buckets are exact: batched == one at a time
    for i in (0, 3, 4):
        one = enc.embed([texts[i]], use_cache=False)[0]
        assert float(one @ E[i]) > 0.999

    # streamed memmap output == in-memory output (served from the cache written per batch)
    M = enc.embed_to_file(texts + texts[:2], tmp_path / "t.npy")
    assert isinstance(M, np.memmap) and M.shape == (len(texts) + 2, 1024)
    assert np.allclose(np.asarray(M[:len(texts)]), enc.embed(texts), atol=1e-6)
    assert np.array_equal(M[len(texts)], M[0])


def test_budget_breach_kills_worker_with_explicit_error(enc):
    enc.load()
    pid = enc._client.proc.pid
    enc._client.watchdog.limit_bytes = 64 * 1024 * 1024     # far below what the loaded model holds
    deadline = time.monotonic() + 10
    while not enc._client.watchdog.tripped and time.monotonic() < deadline:
        time.sleep(0.1)
    with pytest.raises(MemoryBudgetExceeded, match="over the .* memory budget"):
        enc.embed(["after the kill"], use_cache=False)
    assert not enc.loaded()
    import psutil
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_admission_refuses_a_model_that_cannot_fit(tmp_path):
    e = Encoder(SPEC, MAX_LEN, cache_dir=tmp_path, policy=_policy(budget_gb=0.5))
    with pytest.raises(MemoryAdmissionError, match="model memory budget is 0.5 GB"):
        e.embed(["hello"], use_cache=False)
    assert not e.loaded()


def test_after_a_kill_the_next_call_starts_a_fresh_worker(enc):
    enc.load()
    enc._client.watchdog.limit_bytes = 64 * 1024 * 1024
    deadline = time.monotonic() + 10
    while not enc._client.watchdog.tripped and time.monotonic() < deadline:
        time.sleep(0.1)
    with pytest.raises(MemoryBudgetExceeded):
        enc.embed(["x " * 300], use_cache=False)     # reported once, to the next caller
    v = enc.embed(["fresh start"], use_cache=False)  # not reported again: a new worker serves this
    assert v.shape == (1, 1024) and enc.loaded()


def test_idle_worker_yields_the_model_lock_to_another_process(enc):
    import subprocess
    import sys
    enc.load()
    probe = ("import time; from forge.model_lock import ModelJobLock; lk = ModelJobLock(); t = time.monotonic(); "
             "lk.acquire('probe', 60, lambda *a: None); print(round(time.monotonic() - t, 1)); lk.release()")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=90, check=False)
    assert out.returncode == 0, out.stderr
    assert float(out.stdout) < 10          # the idle holder noticed the waiter and exited
    deadline = time.monotonic() + 5
    while enc.loaded() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not enc.loaded()
    assert enc.embed(["back again"], use_cache=False).shape == (1, 1024)   # reloads on demand


def test_worker_that_left_before_reading_the_request_is_replaced(enc):
    """The idle/yield exit can race a request the parent just sent; the request is resent to a fresh worker."""
    enc.load()
    old = enc._client
    pid = old.proc.pid
    from forge.embed_worker import write_frame
    write_frame(old.proc.stdin.fileno(), ("stop",))       # the worker leaves on its own...
    old.proc.wait(15)
    old.alive = lambda: True                              # ...but the parent has not noticed yet (the race window)
    notes: list[str] = []
    v = enc.embed(["sent into the gap"], use_cache=False, on_status=lambda lvl, m: notes.append(m))
    assert v.shape == (1, 1024) and enc.loaded() and enc._client.proc.pid != pid
    assert any("starting a new one" in m for m in notes)


def test_cancel_while_waiting_for_the_model_lock_kills_the_worker(tmp_path, monkeypatch):
    """No model loads here: the lock is held by this test, the worker waits, the caller cancels."""
    from forge.model_lock import ModelJobLock
    monkeypatch.setenv("FORGE_MODEL_LOCK_DIR", str(tmp_path / "lock"))
    held = ModelJobLock()
    held.acquire("test holder", 5, lambda *a: None)
    try:
        e = Encoder(SPEC, MAX_LEN, cache_dir=tmp_path, policy=_policy())
        t0 = time.monotonic()
        with pytest.raises(InterruptedError, match="cancelled"):
            e.embed(["never embedded"], use_cache=False, should_stop=lambda: time.monotonic() - t0 > 1.5)
        assert time.monotonic() - t0 < 10 and not e.loaded()
        assert held.others_waiting() == []            # the killed waiter's marker is gone or reaped
    finally:
        held.release()
