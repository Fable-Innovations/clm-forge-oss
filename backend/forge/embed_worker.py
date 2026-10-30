"""The embedding worker: one subprocess that hosts one base model under a hard memory budget.

``forge.encoder.Encoder`` starts it with ``python -m forge.embed_worker`` and talks to it over
stdin/stdout with length-prefixed pickle frames (``write_frame``/``read_frame``). The worker's
real stdout is kept for frames and fd 1 is pointed at stderr, so a library that prints cannot
corrupt the stream. The parent polls this process tree's footprint and kills it past the budget
(``memory.Watchdog``), so a runaway model can take down only the worker, never the server.

Life of a worker:
    1. ``init`` frame: spec, max_len, resolved ``MemoryPolicy``.
    2. Take the machine-wide model lock (``model_lock``), reporting while it waits.
    3. ``prepare()`` the backend (files, shape) and run admission control (``memory.admit``):
       weights + KV + one max-width row + runtime overhead must fit the budget and the free RAM.
    4. Load (framework caps applied first), warm up one max-width row to measure the real fixed
       cost, size the ``BatchPlanner`` from that, send ``ready``.
    5. Serve ``embed`` requests: tokenise, tail-truncate, bucket, forward batch by batch, streaming
       each batch's vectors back as soon as it is done. A batch that hits an allocation failure or
       runs past the soft limit halves that bucket's rows; free RAM below the reserve does too.
    6. Exit on ``stop``, after ``idle_unload_s`` without work, or when idle while another process
       waits for the model lock — releasing the weights and the lock.

Frames to the parent: ``("status", level, msg)``, ``("ready", info)``, ``("batch", idx, vecs, stats)``,
``("done", stats)``, ``("error", kind, msg)``, ``("bye", reason)``.
"""
from __future__ import annotations

import os
import pickle
import select
import struct
import sys
import time
import traceback

import numpy as np

GiB = 1024 ** 3
RUNTIME_OVERHEAD = GiB // 2   # interpreter + framework + tokenizer, on top of weights


# Frames are pickled: both ends are this package (the server and the worker it spawned) over a
# private pipe, and they carry numpy arrays; nothing from outside ever reaches read_frame.
def write_frame(fd: int, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    buf = struct.pack("<Q", len(data)) + data
    view = memoryview(buf)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _read_exact(fd: int, n: int) -> bytes | None:
    chunks, got = [], 0
    while got < n:
        b = os.read(fd, min(1 << 20, n - got))
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_frame(fd: int):
    """Next frame, or ``None`` at EOF."""
    head = _read_exact(fd, 8)
    if head is None:
        return None
    body = _read_exact(fd, struct.unpack("<Q", head)[0])
    return None if body is None else pickle.loads(body)


def _is_oom(e: BaseException) -> bool:
    m = str(e).lower()
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in m or "malloc" in m or "allocate" in m


class Worker:
    def __init__(self, out_fd: int, init: dict):
        from .memory import MemoryPolicy
        self.out = out_fd
        self.spec, self.max_len = init["spec"], init["max_len"]
        self.policy = MemoryPolicy(**{**init["policy"], "buckets": tuple(init["policy"]["buckets"])})
        self.backend_name, _, self.model_name = self.spec.partition(":")

    def send(self, *frame) -> None:
        write_frame(self.out, frame)

    def status(self, level: str, msg: str) -> None:
        self.send("status", level, msg)

    # -- startup -------------------------------------------------------------------------------
    def start(self) -> None:
        from .encoder_backends import make_backend
        from .memory import BatchPlanner, admit, describe, tree_footprint
        from .model_lock import ModelJobLock

        self.lock = ModelJobLock()
        self.lock.acquire(f"{self.spec} (embedding)", self.policy.lock_wait_s, self.status)
        t0 = time.perf_counter()
        self.be = make_backend(self.backend_name, self.model_name, self.max_len, self.policy)
        self.shape = self.be.prepare()
        kv = self.shape.kv_bytes_per_token * self.max_len * self.policy.llama_parallel
        need = (self.shape.weights_bytes + kv + self.shape.batch_bytes(1, self.max_len) + RUNTIME_OVERHEAD)
        admit(need, f"{self.spec}", self.policy, self.status)
        self.status("info", f"loading {self.spec}: weights {describe(self.shape.weights_bytes)}, "
                            f"budget {describe(self.policy.budget_bytes)}")
        self.be.load()
        # warm-up: one max-width row measures the real fixed cost of a forward pass on this model
        pad = (self.be.tokenize(" ") or [0])[:1]
        self.be.reset_peak()
        self.be.forward([pad * (self.max_len - 1)], self.max_len)
        warm = self.be.peak_delta()
        self.be.after_batch()
        self.baseline = tree_footprint(os.getpid())
        if self.baseline + warm > self.policy.budget_bytes:
            raise MemoryError(f"{self.spec} needs {describe(self.baseline + warm)} for a single {self.max_len}-token "
                              f"row, over the {describe(self.policy.budget_bytes)} budget")
        self.planner = BatchPlanner(self.shape, self.policy, self.baseline + warm)
        self.dim = self.shape.hidden
        self.send("ready", {"pid": os.getpid(), "dim": self.dim, "load_s": round(time.perf_counter() - t0, 2),
                            "weights_bytes": self.shape.weights_bytes, "baseline_bytes": self.baseline,
                            "warmup_bytes": warm, "rows_max_bucket": self.planner.rows(self.max_len)})

    # -- embedding -----------------------------------------------------------------------------
    def embed(self, texts: list[str]) -> None:
        from .memory import available_bytes, bucket_for, describe

        ids = []
        for t in texts:
            row = self.be.tokenize(t) or self.be.tokenize(" ")
            ids.append(row[-(self.max_len - 1):])
        queues: dict[int, list[int]] = {}
        for i in sorted(range(len(ids)), key=lambda i: len(ids[i])):
            queues.setdefault(bucket_for(len(ids[i]), self.policy.buckets), []).append(i)
        soft = int(self.policy.budget_bytes * self.policy.soft_fraction)
        peak_total, backoffs, batches = self.baseline, 0, 0
        for bucket in sorted(queues):
            q = queues[bucket]
            while q:
                rows = self.planner.rows(bucket)
                if available_bytes() < self.policy.reserve_bytes and rows > 1:
                    rows = self.planner.backoff(bucket)
                    backoffs += 1
                    self.status("warn", f"system memory below the {describe(self.policy.reserve_bytes)} reserve; "
                                        f"{bucket}-token batches now {rows} rows")
                take = q[:rows]
                self.be.reset_peak()
                try:
                    vecs = self.be.forward([ids[i] for i in take], bucket)
                except Exception as e:
                    self.be.after_batch()
                    if not _is_oom(e):
                        raise
                    if rows == 1:
                        raise MemoryError(f"one {bucket}-token row does not fit in memory: {e}") from e
                    new = self.planner.backoff(bucket)
                    backoffs += 1
                    self.status("warn", f"allocation failed at {rows}x{bucket} tokens; retrying with {new} rows")
                    continue
                used = self.baseline + self.be.peak_delta()
                self.be.after_batch()
                peak_total = max(peak_total, used)
                if used > soft and rows > 1:
                    new = self.planner.backoff(bucket)
                    backoffs += 1
                    self.status("warn", f"batch {rows}x{bucket} peaked at {describe(used)} (> {describe(soft)}); "
                                        f"{bucket}-token batches now {new} rows")
                del q[:len(take)]
                batches += 1
                vecs = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)
                self.send("batch", take, vecs.astype(np.float32), {"rows": len(take), "bucket": bucket,
                                                                    "peak_bytes": used})
        self.send("done", {"peak_bytes": peak_total, "backoffs": backoffs, "batches": batches})

    # -- loop ----------------------------------------------------------------------------------
    def serve(self, in_fd: int) -> str:
        last = time.monotonic()
        while True:
            ready, _, _ = select.select([in_fd], [], [], 1.0)
            if not ready:
                if time.monotonic() - last > self.policy.idle_unload_s:
                    return f"idle for {self.policy.idle_unload_s}s"
                waiting = self.lock.others_waiting()
                if waiting:
                    return f"released the model for waiting pid {waiting[0]}"
                continue
            msg = read_frame(in_fd)
            if msg is None or msg[0] == "stop":
                return "stopped"
            if msg[0] == "embed":
                self.embed(msg[1])
            last = time.monotonic()

    def close(self) -> None:
        be = getattr(self, "be", None)
        if be is not None:
            be.close()
        lock = getattr(self, "lock", None)
        if lock is not None:
            lock.release()


def main() -> int:
    out_fd = os.dup(1)
    os.dup2(2, 1)                 # stray prints from libraries go to the log, not the frame stream
    sys.stdout = sys.stderr
    in_fd = 0
    init = read_frame(in_fd)
    if init is None:
        return 0
    w = Worker(out_fd, init)
    code = 0
    try:
        w.start()
        reason = w.serve(in_fd)
        w.send("bye", reason)
    except BaseException as e:  # noqa: BLE001 - every failure goes to the parent as an explicit error
        kind = "MemoryBudgetExceeded" if isinstance(e, MemoryError) else type(e).__name__
        traceback.print_exc()
        try:
            w.send("error", kind, str(e))
        except OSError:
            pass
        code = 1
    finally:
        w.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
