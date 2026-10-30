"""Memory governance for model jobs: budget policy, measurement, padding buckets, batch sizing,
admission control and the hard-budget watchdog.

Why this exists: on 2026-10-09 two processes each embedded with Qwen3-8B-4bit through MLX and each
grew to ~37 GB until the 48 GB Mac kernel-panicked. MLX's default memory limit is 1.5x the
device's recommended working set and its buffer cache limit defaults to the memory limit, so a
long run whose padded batch widths keep changing parks one freed buffer per distinct size in the
cache and never gives them back. The fix is layered; each layer lives in one place:

    policy        ``MemorySettings`` (Settings.memory) -> ``MemoryPolicy`` with defaults from total RAM
    framework     ``encoder_backends``: mx.set_memory_limit / set_cache_limit + clear_cache per batch;
                  torch.mps.set_per_process_memory_fraction + empty_cache
    shapes        ``bucket_for``: padded widths come from a fixed set, so buffers are reused
    batch size    ``BatchPlanner``: rows from the model shape, the bucket and the measured headroom;
                  halves on memory pressure or an allocation failure
    admission     ``admit``: weights + minimum activations must fit the budget (else refuse) and
                  the currently available RAM minus the reserve (else wait, bounded, then refuse)
    one at a time ``forge.model_lock``: machine-wide lock, one loaded model across all processes
    hard stop     ``Watchdog``: the parent polls the embedding worker's process tree footprint and
                  kills it when it crosses the budget (``forge.embed_worker`` hosts the model)

Footprint is ``max(phys_footprint, rss)`` per process. On macOS ``phys_footprint`` is what Activity
Monitor shows and, unlike RSS, includes Metal/unified-memory buffers (verified: a 1 GB MLX array
moves phys_footprint by ~1 GB and RSS by ~0.1 GB); RSS covers mmapped file pages (llama.cpp
weights) that phys_footprint may not count.
"""
from __future__ import annotations

import ctypes
import math
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

GiB = 1024 ** 3
Notify = Callable[[str, str], None]          # (level, message)
DEFAULT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)


class MemoryAdmissionError(RuntimeError):
    """A model job was refused before loading: it cannot fit the budget or the machine right now."""


class MemoryBudgetExceeded(RuntimeError):
    """A running model job crossed its hard memory budget and was terminated."""


# ---- settings / policy -------------------------------------------------------------------------
class MemorySettings(BaseModel):
    """``Settings.memory``. ``None`` means "derive from this machine's RAM" (see ``resolve_policy``)."""
    budget_gb: float | None = Field(default=None, description="hard cap for one model job (weights + activations "
                                    "+ caches, whole worker process tree); default 60% of RAM")
    reserve_gb: float | None = Field(default=None, description="RAM always left free for the OS and other apps; "
                                     "default max(4 GB, 15% of RAM)")
    buckets: list[int] = Field(default_factory=lambda: list(DEFAULT_BUCKETS),
                               description="padded sequence widths; max_len is always the last bucket")
    max_rows: int = Field(default=16, description="upper bound on rows per forward pass")
    admission_wait_s: int = Field(default=300, description="how long a job waits for enough free RAM before refusing")
    lock_wait_s: int = Field(default=3600, description="how long a job waits for the machine-wide model lock")
    idle_unload_s: int = Field(default=600, description="an idle embedding worker exits after this many seconds")
    llama_parallel: int = Field(default=2, description="llama-server slots (-np) for embeddings")
    llama_kv_type: str = Field(default="q8_0", description="llama-server KV cache type (-ctk/-ctv)")


@dataclass(frozen=True)
class MemoryPolicy:
    """Resolved, absolute numbers (bytes). Picklable; handed to the embedding worker at start."""
    total_bytes: int
    budget_bytes: int
    reserve_bytes: int
    buckets: tuple[int, ...]
    max_rows: int
    admission_wait_s: int
    lock_wait_s: int
    idle_unload_s: int
    llama_parallel: int
    llama_kv_type: str
    soft_fraction: float = 0.85      # the batch planner aims below this share of the budget
    cache_limit_bytes: int = 0       # MLX free-buffer cache cap

    def summary(self) -> dict:
        return {"budget_gb": round(self.budget_bytes / GiB, 1), "reserve_gb": round(self.reserve_bytes / GiB, 1),
                "total_gb": round(self.total_bytes / GiB, 1), "buckets": list(self.buckets)}

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_policy(ms: MemorySettings, max_len: int, total_bytes: int | None = None) -> MemoryPolicy:
    total = int(total_bytes or psutil.virtual_memory().total)
    reserve = int(ms.reserve_gb * GiB) if ms.reserve_gb is not None else max(4 * GiB, int(0.15 * total))
    budget = int(ms.budget_gb * GiB) if ms.budget_gb is not None else int(0.60 * total)
    if budget <= 0 or reserve < 0:
        raise ValueError(f"memory budget must be > 0 and reserve >= 0 (got budget {budget}, reserve {reserve})")
    if budget + reserve > total:
        raise ValueError(f"memory budget {budget / GiB:.1f} GB + reserve {reserve / GiB:.1f} GB exceed this machine's "
                         f"{total / GiB:.1f} GB of RAM")
    buckets = tuple(sorted({b for b in ms.buckets if 0 < b < max_len} | {max_len}))
    return MemoryPolicy(total_bytes=total, budget_bytes=budget, reserve_bytes=reserve, buckets=buckets,
                        max_rows=max(1, ms.max_rows), admission_wait_s=ms.admission_wait_s, lock_wait_s=ms.lock_wait_s,
                        idle_unload_s=ms.idle_unload_s, llama_parallel=max(1, ms.llama_parallel),
                        llama_kv_type=ms.llama_kv_type, cache_limit_bytes=min(2 * GiB, budget // 10))


def current_policy(max_len: int) -> MemoryPolicy:
    """Policy from the saved Forge settings (``Settings.memory``)."""
    from .config import load_settings
    return resolve_policy(load_settings().memory, max_len)


# ---- measurement ---------------------------------------------------------------------------------
_RUSAGE_INFO_V2 = 2
_PHYS_FOOTPRINT_OFFSET = 72          # rusage_info_v2: uuid[16] + 7 x uint64 precede ri_phys_footprint
_libproc = None


def _phys_footprint(pid: int) -> int | None:
    global _libproc
    if sys.platform != "darwin":
        return None
    if _libproc is None:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    buf = ctypes.create_string_buffer(256)
    if _libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), buf) != 0:
        return None
    return int.from_bytes(buf.raw[_PHYS_FOOTPRINT_OFFSET:_PHYS_FOOTPRINT_OFFSET + 8], "little")


def process_footprint(pid: int) -> int:
    """Bytes one process holds: ``max(phys_footprint, rss)`` (see module docstring); 0 if it is gone."""
    try:
        rss = psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    return max(rss, _phys_footprint(pid) or 0)


def tree_footprint(pid: int) -> int:
    """Footprint of ``pid`` plus all its descendants (the worker and its llama-server)."""
    try:
        kids = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    return process_footprint(pid) + sum(process_footprint(k.pid) for k in kids)


def available_bytes() -> int:
    return int(psutil.virtual_memory().available)


def kill_tree(pid: int, grace: float = 3.0) -> None:
    """Terminate ``pid`` and its descendants (children first), escalating to SIGKILL after ``grace``."""
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace)


# ---- model shape -----------------------------------------------------------------------------------
@dataclass
class ModelShape:
    """What a model costs: weights and, per padded token of one forward pass, activation bytes."""
    weights_bytes: int
    hidden: int
    layers: int
    intermediate: int
    heads: int
    act_bytes: int = 2                     # fp16/bf16 activations
    attention_materialised: bool = False   # True when the backend may build [heads, w, w] scores
    kv_bytes_per_token: int = 0            # llama.cpp: KV cache bytes per context token (all layers)

    def per_token_bytes(self, width: int) -> int:
        """Activation bytes per padded token of one forward pass (live at the peak, not summed over layers).

        Calibrated on Qwen3-0.6B-4bit under MLX 0.32 (2026-10-09): above the fixed overhead the peak
        grows by ~38.9 KB per padded token = 2 bytes x (10 x hidden + 3 x intermediate): the residual
        stream, q/k/v/o and norm temporaries, and the gate/up/act MLP tensors of the layer in flight.
        MLX's fused SDPA never materialises the attention matrix; a torch backend may, so that term is
        added when ``attention_materialised``.
        """
        b = self.act_bytes * (10 * self.hidden + 3 * self.intermediate)
        if self.attention_materialised:
            b += self.heads * width * 4
        return b

    def batch_bytes(self, rows: int, width: int) -> int:
        return rows * width * self.per_token_bytes(width)


def shape_from_config(cfg: dict, weights_bytes: int, **kw) -> ModelShape:
    """``config.json`` (HF/MLX; Gemma-style ``text_config`` nesting handled) -> ``ModelShape``."""
    c = cfg.get("text_config") or cfg
    hidden = int(c["hidden_size"])
    return ModelShape(weights_bytes=weights_bytes, hidden=hidden, layers=int(c.get("num_hidden_layers", 0)),
                      intermediate=int(c.get("intermediate_size") or 4 * hidden),
                      heads=int(c.get("num_attention_heads", 0)), **kw)


def weights_size(files: list[Path]) -> int:
    return sum(Path(f).stat().st_size for f in files)


# ---- GGUF metadata (llama.cpp models) ----------------------------------------------------------------
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_metadata(path: Path) -> dict:
    """Scalar and small-array key/values from a GGUF header (large arrays such as the vocab are skipped)."""
    out: dict = {}
    with open(path, "rb") as f:
        def rd(fmt: str):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rstr() -> str:
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def value(t: int):
            if t in _GGUF_SCALARS:
                return rd(_GGUF_SCALARS[t])
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                if et in _GGUF_SCALARS and n > 4096:
                    f.seek(n * struct.calcsize(_GGUF_SCALARS[et]), 1)
                    return None
                items = [value(et) for _ in range(n)]
                return items if n <= 4096 else None
            raise ValueError(f"unknown GGUF value type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        rd("<I")
        rd("<Q")
        for _ in range(rd("<Q")):
            k = rstr()
            out[k] = value(rd("<I"))
    return out


_KV_TYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
                  "iq4_nl": 18 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32}


def shape_from_gguf(path: Path, kv_type: str) -> ModelShape:
    md = gguf_metadata(path)
    arch = md.get("general.architecture", "llama")

    def g(key: str, default=0):
        v = md.get(f"{arch}.{key}", default)
        return max(v) if isinstance(v, list) else (v if v is not None else default)

    hidden = int(g("embedding_length"))
    heads = int(g("attention.head_count"))
    layers = int(g("block_count"))
    key_len = int(g("attention.key_length", hidden // max(1, heads)))
    val_len = int(g("attention.value_length", key_len))
    kv_heads = md.get(f"{arch}.attention.head_count_kv", heads)
    swa = md.get(f"{arch}.attention.sliding_window_pattern")
    if not isinstance(kv_heads, list):
        kv_heads = [int(kv_heads)] * layers
    if not isinstance(swa, list) or len(swa) != layers:
        swa = [False] * layers
    k_swa, v_swa = int(g("attention.key_length_swa", key_len)), int(g("attention.value_length_swa", val_len))
    # per layer: sliding-window layers use their own (smaller) head size; counted at full context (upper bound)
    elems = sum(h * ((k_swa + v_swa) if s else (key_len + val_len)) for h, s in zip(kv_heads, swa))
    kv_per_token = int(elems * _KV_TYPE_BYTES.get(kv_type, 2.0))
    return ModelShape(weights_bytes=Path(path).stat().st_size, hidden=hidden, layers=layers,
                      intermediate=int(g("feed_forward_length", 4 * hidden)), heads=heads,
                      kv_bytes_per_token=kv_per_token)


# ---- padding buckets / batch planning -----------------------------------------------------------------
def bucket_for(n_tokens: int, buckets: tuple[int, ...] | list[int]) -> int:
    """Smallest bucket >= ``n_tokens`` (the last bucket is max_len, which bounds every input)."""
    for b in buckets:
        if n_tokens <= b:
            return b
    raise ValueError(f"{n_tokens} tokens exceed the largest bucket {buckets[-1]}")


class BatchPlanner:
    """Rows per forward pass for a bucket, from the model shape and the measured headroom.

    ``headroom`` = soft share of the budget minus what the worker already holds after load and
    warm-up (weights, runtime, fixed per-pass overhead). ``rows(bucket)`` = how many rows of that
    width fit, capped by ``max_rows``. ``backoff`` halves a bucket's rows (sticky for the run).
    """

    def __init__(self, shape: ModelShape, policy: MemoryPolicy, baseline_bytes: int):
        self.shape, self.policy = shape, policy
        self.headroom = int(policy.budget_bytes * policy.soft_fraction) - baseline_bytes
        self.cap: dict[int, int] = {}

    def rows(self, bucket: int) -> int:
        if self.headroom <= 0:
            return 1
        fit = self.headroom // max(1, self.shape.batch_bytes(1, bucket))
        return int(max(1, min(self.policy.max_rows, self.cap.get(bucket, self.policy.max_rows), fit)))

    def backoff(self, bucket: int) -> int:
        self.cap[bucket] = max(1, self.rows(bucket) // 2)
        return self.cap[bucket]

    def plan(self, lengths: dict[int, int]) -> list[tuple[int, list[int]]]:
        """``{item: n_tokens}`` -> ``[(bucket, [items])]``, shortest first, sized by the current caps."""
        by_bucket: dict[int, list[int]] = {}
        for i in sorted(lengths, key=lambda i: lengths[i]):
            by_bucket.setdefault(bucket_for(lengths[i], self.policy.buckets), []).append(i)
        out = []
        for b in sorted(by_bucket):
            items, r = by_bucket[b], self.rows(b)
            out += [(b, items[k:k + r]) for k in range(0, len(items), r)]
        return out


# ---- admission control ------------------------------------------------------------------------------
def admit(required_bytes: int, what: str, policy: MemoryPolicy, notify: Notify, *,
          available: Callable[[], int] = available_bytes, sleep: Callable[[float], None] = time.sleep,
          clock: Callable[[], float] = time.monotonic, poll_s: float = 2.0, report_every_s: float = 15.0) -> None:
    """Refuse ``what`` if it can never fit the budget; wait (bounded) until free RAM minus the reserve fits it."""
    gb = lambda b: f"{b / GiB:.1f} GB"
    if required_bytes > policy.budget_bytes:
        raise MemoryAdmissionError(f"{what} needs ~{gb(required_bytes)} but the model memory budget is "
                                   f"{gb(policy.budget_bytes)} (Settings > memory.budget_gb)")
    deadline, last = clock() + policy.admission_wait_s, -math.inf
    while True:
        free = available() - policy.reserve_bytes
        if required_bytes <= free:
            return
        now = clock()
        if now >= deadline:
            raise MemoryAdmissionError(f"{what} needs ~{gb(required_bytes)} but only {gb(max(0, free))} is free after "
                                       f"the {gb(policy.reserve_bytes)} reserve; waited {policy.admission_wait_s}s "
                                       f"for memory to free up")
        if now - last >= report_every_s:
            notify("warn", f"waiting for memory: {what} needs ~{gb(required_bytes)}, {gb(max(0, free))} free after "
                           f"the {gb(policy.reserve_bytes)} reserve")
            last = now
        sleep(poll_s)


# ---- hard budget watchdog -----------------------------------------------------------------------------
@dataclass
class Watchdog:
    """Polls a process tree's footprint; above ``limit_bytes`` it records ``exceeded_bytes``, sets ``tripped``, kills
    the tree and calls ``on_exceed``.

    Runs in the parent, so a runaway worker can never take the server (or the machine) with it.
    """
    pid: int
    limit_bytes: int
    on_exceed: Callable[[int], None]
    interval: float = 0.5
    peak_bytes: int = 0
    exceeded_bytes: int = 0      # the footprint that tripped it; set before ``tripped`` turns true
    tripped: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> Watchdog:
        self._thread = threading.Thread(target=self._run, name=f"mem-watchdog-{self.pid}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            used = tree_footprint(self.pid)
            if used == 0 and not psutil.pid_exists(self.pid):
                return
            self.peak_bytes = max(self.peak_bytes, used)
            if used > self.limit_bytes:
                # whoever sees ``tripped`` can already say why (the kill below takes up to its grace period)
                self.exceeded_bytes = used
                self.tripped = True
                kill_tree(self.pid)
                self.on_exceed(used)
                return
            self._stop.wait(self.interval)


def describe(b: int) -> str:
    return f"{b / GiB:.2f} GB"


__all__ = [
    "DEFAULT_BUCKETS",
    "BatchPlanner",
    "GiB",
    "MemoryAdmissionError",
    "MemoryBudgetExceeded",
    "MemoryPolicy",
    "MemorySettings",
    "ModelShape",
    "Notify",
    "Watchdog",
    "admit",
    "available_bytes",
    "bucket_for",
    "current_policy",
    "describe",
    "gguf_metadata",
    "kill_tree",
    "process_footprint",
    "resolve_policy",
    "shape_from_config",
    "shape_from_gguf",
    "tree_footprint",
    "weights_size",
]
