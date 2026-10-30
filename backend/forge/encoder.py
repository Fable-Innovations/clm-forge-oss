"""Frozen text encoders (one per base model) with a persistent embedding cache.

Recipe (must match what the CLM heads are trained on, here and in the shipped package):

* tokenise with ``add_special_tokens=False`` and keep the **tail** ``max_len - 1`` tokens
  (the question sits at the end of the state text, so the tail is what matters);
* run the base transformer (no LM head) — the output is the hidden state *after* the final norm;
* take the **last token**'s vector, cast to float32, **L2-normalise**.

Backends (spec ``<backend>:<model>``; implementations in ``forge.encoder_backends``)
    ``mlx``       e.g. ``mlx-community/Qwen3-8B-4bit`` via mlx_lm on Apple Silicon.
    ``hf``        transformers bf16 on MPS/CPU (``hf:Qwen/Qwen3-8B``); not in the registry because the
                  full-precision weights are a 16 GB download.
    ``llamacpp``  a GGUF served by a private ``llama-server --embeddings --pooling last`` subprocess,
                  e.g. ``llamacpp:EldanRing/Winnow-12B:gguf/Winnow-12B-Q8_0.gguf`` (``<hf repo>:<file>``,
                  downloaded to the HF cache) or ``llamacpp:/abs/path/model.gguf``. llama.cpp's
                  pooled embedding is the last token's hidden state after the final norm, i.e. the
                  same recipe. Token ids (no special tokens, tail-truncated) are sent, not text, so
                  the server never re-tokenises or adds BOS.

Process model: the model never loads into the server process. ``Encoder`` starts an embedding
worker (``forge.embed_worker``) that takes the machine-wide model lock, passes admission control,
loads the model under framework memory caps and embeds in fixed-width, memory-budgeted batches;
the parent watches the worker's process-tree footprint and kills it past the budget, raising
``MemoryBudgetExceeded`` (see ``forge.memory`` for every layer). One model is loaded per process
at a time: starting one encoder's worker stops any other encoder's worker in this process.
Memory settings (``Settings.memory``) are read on every call; a changed policy restarts the worker.
A worker that left between requests (idle unload, or yielding the model lock to another process)
before reading the request just sent is replaced and the request resent; that is the yield protocol,
not a retry of a failure (a worker that fails sends an explicit error). ``should_stop`` (the job's
cancel flag) is polled while waiting for the lock, memory or batches; when it turns true the worker
is killed (freeing the lock and its memory) and ``InterruptedError`` is raised.

Which spec: always the caller's. An experiment embeds with its ``base_model``, an autopilot target with its own
registry key, a checkpoint with its recorded ``embed_model`` and the reference head with ``forge.encoders.REFERENCE``.
There is no process-wide or app-wide default spec.

The embedding width depends on the base model (4096 for Qwen3-8B); ``Encoder.dim`` starts at the registry's
``hidden`` for a registered spec and is confirmed by the first vector produced (a cache hit counts) or the worker.

Cache: one SQLite file ``data/cache/emb/emb.sqlite`` mapping ``sha1(model|max_len|text)`` to the
raw float32 bytes (llama.cpp with a quantized KV cache adds ``|kv=<type>`` so vectors computed with
different KV precision never mix). Fresh vectors are written per batch, as they arrive.
``embed_to_file`` streams the result into a ``.npy`` memmap instead of building it in RAM.

Status: the worker's messages (waiting for the model lock, waiting for memory, batch back-off,
peak memory) go to ``on_status(level, message)`` when given, and always to the ``forge.encoder``
logger.

Thread-safety: ``get_encoder()`` returns a process-wide singleton per spec; all model use in a
process is serialised by one lock (FastAPI runs sync handlers on a thread pool), the cache by its own.
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .embed_worker import read_frame, write_frame
from .memory import (
    MemoryAdmissionError,
    MemoryBudgetExceeded,
    MemoryPolicy,
    Watchdog,
    current_policy,
    describe,
    kill_tree,
)
from .model_lock import ModelLockTimeout
from .paths import EMB_CACHE_DIR

DEFAULT_MAX_LEN = 2048

Progress = Callable[[int, int], None]
Status = Callable[[str, str], None]          # (level, message)
StopCheck = Callable[[], bool]               # the caller's job was cancelled
log = logging.getLogger("forge.encoder")
_ERRORS = {"MemoryAdmissionError": MemoryAdmissionError, "MemoryBudgetExceeded": MemoryBudgetExceeded,
           "ModelLockTimeout": ModelLockTimeout}


class _WorkerRetired(Exception):
    """The worker exited cleanly (idle unload, or yielding the model lock to another process) before it
    read the request just sent. Nothing of the request ran; the caller starts a fresh worker and resends."""


_MAX_RESTARTS = 3


def cache_key(spec: str, max_len: int, text: str, kv: str | None = None) -> str:
    """The cache key of ``text`` embedded by ``spec`` at ``max_len`` tokens (llama.cpp: a quantized KV cache
    ``kv`` other than f16 makes its own vectors). Pure: callers that only need keys (a deletion purging a row's
    vectors) load no model."""
    base = f"{spec.partition(':')[2]}|{max_len}|{text}"
    if spec.startswith("llamacpp:") and kv and kv != "f16":
        base += f"|kv={kv}"
    return hashlib.sha1(base.encode()).hexdigest()


class EmbeddingCache:
    """sha1-keyed float32 vectors in one SQLite file (WAL mode, safe across threads/processes)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})"
                for k, v in self._db.execute(q, chunk):
                    out[k] = np.frombuffer(v, dtype=np.float32).copy()
        return out

    def delete_many(self, keys: list[str]) -> int:
        """Remove these keys' vectors (a deleted real row's embeddings). -> rows removed. The removed vectors leave no
        bytes behind: ``secure_delete`` zeroes their pages, and the WAL (which holds the page images until a
        checkpoint) is checkpointed and truncated, so a removal leaves nothing readable on disk."""
        n = 0
        with self._lock:
            self._db.execute("PRAGMA secure_delete=ON")
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                n += self._db.execute(f"DELETE FROM emb WHERE k IN ({','.join('?' * len(chunk))})", chunk).rowcount
            self._db.commit()
            if n:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return n

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                                 [(k, np.ascontiguousarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
            self._db.commit()


class _WorkerClient:
    """Parent side of one embedding worker: frames in/out, a reader thread and the budget watchdog."""

    def __init__(self, spec: str, max_len: int, policy: MemoryPolicy, status: Status, should_stop: StopCheck | None):
        self.spec, self.policy = spec, policy
        self.frames: queue.Queue = queue.Queue()
        self.info: dict = {}
        self.exit_reason = ""
        logp = EMB_CACHE_DIR.parent / "embed-worker.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "ab") as errlog:
            self.proc = subprocess.Popen([sys.executable, "-m", "forge.embed_worker"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=errlog,
                                         cwd=str(Path(__file__).resolve().parents[1]))
        self.log_path = logp
        self.watchdog = Watchdog(self.proc.pid, policy.budget_bytes, self._exceeded).start()
        threading.Thread(target=self._reader, name=f"embed-worker-{self.proc.pid}", daemon=True).start()
        write_frame(self.proc.stdin.fileno(), {"spec": spec, "max_len": max_len, "policy": policy.to_dict()})
        try:
            self.info = self._wait("ready", status, should_stop=should_stop)
        except _WorkerRetired as e:   # only a stop frame or a closed stdin makes a starting worker say bye
            raise RuntimeError(f"embedding worker for {spec} exited during start-up ({e}); see {self.log_path}") from e

    def _reader(self) -> None:
        fd = self.proc.stdout.fileno()
        while True:
            try:
                f = read_frame(fd)
            except (OSError, EOFError, ValueError):
                f = None
            if f is None:
                self.frames.put(("eof",))
                return
            if f[0] == "bye":
                self.exit_reason = f[1]
                log.info("embedding worker %s for %s exited: %s", self.proc.pid, self.spec, f[1])
            self.frames.put(f)

    @property
    def trip_message(self) -> str:
        """Why the watchdog killed the worker; readable as soon as ``watchdog.tripped`` is (the watchdog records the
        footprint before it sets the flag, and the kill that follows can take seconds)."""
        return self._trip_message(self.watchdog.exceeded_bytes) if self.watchdog.tripped else ""

    def _trip_message(self, used: int) -> str:
        return (f"embedding worker for {self.spec} used {describe(used)}, over the "
                f"{describe(self.policy.budget_bytes)} memory budget; it was terminated")

    def _exceeded(self, used: int) -> None:
        msg = self._trip_message(used)
        log.error(msg)
        self.frames.put(("error", "MemoryBudgetExceeded", msg))

    def alive(self) -> bool:
        return self.proc.poll() is None and not self.watchdog.tripped

    def abort(self, why: str) -> None:
        """Kill the worker now (cancellation): its lock and memory go with it."""
        log.info("embedding worker %s for %s aborted: %s", self.proc.pid, self.spec, why)
        self.watchdog.stop()
        kill_tree(self.proc.pid)
        self.close()

    def _wait(self, until: str, status: Status, on_batch: Callable | None = None, should_stop: StopCheck | None = None):
        got_batch = False
        while True:
            try:
                f = self.frames.get(timeout=0.5)
            except queue.Empty:
                if should_stop is not None and should_stop():
                    self.abort("cancelled by the caller")
                    raise InterruptedError(f"embedding with {self.spec} cancelled") from None
                continue
            kind = f[0]
            if kind == "status":
                status(f[1], f[2])
            elif kind == "batch" and on_batch:
                got_batch = True
                on_batch(f[1], f[2], f[3])
            elif kind == until:
                return f[1]
            elif kind == "error":
                self.close()
                raise _ERRORS.get(f[1], RuntimeError)(f[2] if f[1] in _ERRORS else f"{f[1]}: {f[2]}")
            elif kind in ("eof", "bye"):
                self.close()
                if self.watchdog.tripped:
                    continue          # the watchdog's error frame follows
                if kind == "bye" and not got_batch:
                    raise _WorkerRetired(self.exit_reason)
                raise RuntimeError(f"embedding worker for {self.spec} exited "
                                   f"({self.exit_reason or f'code {self.proc.returncode}'}); see {self.log_path}")

    def embed(self, texts: list[str], on_batch: Callable, status: Status, should_stop: StopCheck | None) -> dict:
        try:
            write_frame(self.proc.stdin.fileno(), ("embed", texts))
        except BrokenPipeError:       # it had already left (idle unload / yielded the lock)
            self.close()
            raise _WorkerRetired(self.exit_reason or "exited") from None
        return self._wait("done", status, on_batch, should_stop)

    def close(self) -> None:
        self.watchdog.stop()
        if self.proc.poll() is None:
            try:
                write_frame(self.proc.stdin.fileno(), ("stop",))
                self.proc.wait(15)
            except (OSError, subprocess.TimeoutExpired):
                kill_tree(self.proc.pid)
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except OSError:
                pass


_PROCESS_MODEL_LOCK = threading.RLock()     # one model in use per process (see module docstring)


class Encoder:
    """Last-token embedder for one base model. ``embed`` returns ``[n, dim]`` float32, L2-normalised."""

    def __init__(self, spec: str, max_len: int = DEFAULT_MAX_LEN, cache_dir: Path = EMB_CACHE_DIR,
                 policy: Callable[[int], MemoryPolicy] = current_policy):
        backend, _, model_name = spec.partition(":")
        if backend not in ("mlx", "hf", "llamacpp") or not model_name:
            raise ValueError(f"encoder spec must be 'mlx:<model>', 'hf:<model>' or 'llamacpp:<repo>:<file>', got {spec!r}")
        self.spec = spec
        self.backend, self.model_name, self.max_len = backend, model_name, max_len
        from .encoders import by_spec
        reg = by_spec(spec)
        self.dim: int | None = reg.hidden if reg else None
        self._policy_fn = policy
        self._client: _WorkerClient | None = None
        self.cache = EmbeddingCache(cache_dir / "emb.sqlite")
        self.load_seconds: float | None = None
        self.last_peak_bytes: int | None = None

    # -- lifecycle -------------------------------------------------------------------------
    def loaded(self) -> bool:
        return self._client is not None and self._client.alive()

    def policy(self) -> MemoryPolicy:
        return self._policy_fn(self.max_len)

    def load(self, on_status: Status | None = None, should_stop: StopCheck | None = None) -> None:
        """Start the worker and load the weights (idempotent); waits for the model lock and memory.
        ``should_stop()`` turning true while waiting or loading kills the worker and raises InterruptedError."""
        with _PROCESS_MODEL_LOCK:
            self._ensure(self.policy(), self._status(on_status), should_stop)

    def unload(self) -> None:
        """Stop the worker (frees the weights and the model lock); the next ``embed`` restarts it."""
        with _PROCESS_MODEL_LOCK:
            if self._client is not None:
                self._client.close()
                self._client = None

    def info(self) -> dict:
        c = self._client
        out = {"backend": self.backend, "model": self.model_name, "loaded": self.loaded(), "max_len": self.max_len,
               "dim": self.dim}
        if c is not None and self.loaded():
            out["worker"] = {"pid": c.proc.pid, "footprint_peak_bytes": c.watchdog.peak_bytes, **c.info}
        return out

    def _status(self, on_status: Status | None) -> Status:
        def cb(level: str, msg: str) -> None:
            log.log(logging.WARNING if level == "warn" else logging.INFO, "[%s] %s", self.spec, msg)
            if on_status:
                on_status(level, msg)
        return cb

    def _ensure(self, policy: MemoryPolicy, status: Status, should_stop: StopCheck | None) -> _WorkerClient:
        if self._client is not None and self._client.alive() and self._client.policy == policy:
            return self._client
        if self._client is not None:
            old, self._client = self._client, None
            old.close()
            if old.watchdog.tripped:      # killed between calls: report it to the next caller, once
                raise MemoryBudgetExceeded(old.trip_message)
        for other in list(_ENCODERS.values()):          # one loaded model per process
            if other is not self and other._client is not None:
                status("info", f"unloading {other.spec} to load {self.spec}")
                other.unload()
        t0 = time.perf_counter()
        self._client = _WorkerClient(self.spec, self.max_len, policy, status, should_stop)
        self.load_seconds = time.perf_counter() - t0
        self.dim = self._client.info.get("dim") or self.dim
        i = self._client.info
        status("info", f"{self.spec} loaded in {self.load_seconds:.1f}s: worker {describe(i['baseline_bytes'])}, "
                       f"budget {describe(policy.budget_bytes)}, up to {i['rows_max_bucket']} rows of "
                       f"{self.max_len} tokens per batch")
        return self._client

    # -- embedding -------------------------------------------------------------------------
    def key(self, text: str, policy: MemoryPolicy | None = None) -> str:
        kv = (policy or self.policy()).llama_kv_type if self.backend == "llamacpp" else None
        return cache_key(self.spec, self.max_len, text, kv)

    def _run(self, texts: list[str], write: Callable[[list[int], np.ndarray], None], on_progress: Progress | None,
             use_cache: bool, on_status: Status | None, should_stop: StopCheck | None) -> None:
        """Embed unique ``texts``; ``write(positions, vectors)`` receives cache hits, then each fresh batch."""
        policy = self.policy()
        keys = [self.key(t, policy) for t in texts]
        total, done = len(texts), 0
        todo = list(range(total))
        if use_cache:
            hit = self.cache.get_many(keys)
            have = [i for i in todo if keys[i] in hit]
            if have:
                write(have, np.stack([hit[keys[i]] for i in have]))
                self.dim = len(hit[keys[have[0]]])
            todo = [i for i in todo if keys[i] not in hit]
            done = total - len(todo)
        if on_progress:
            on_progress(done, total)
        if not todo:
            return
        status = self._status(on_status)

        def on_batch(idx: list[int], vecs: np.ndarray, stats: dict) -> None:
            nonlocal done
            pos = [todo[j] for j in idx]
            write(pos, vecs)
            if use_cache:
                self.cache.put_many({keys[p]: v for p, v in zip(pos, vecs)})
            self.dim = vecs.shape[1]
            done += len(pos)
            if on_progress:
                on_progress(done, total)

        with _PROCESS_MODEL_LOCK:
            for attempt in range(_MAX_RESTARTS + 1):
                client = self._ensure(policy, status, should_stop)
                try:
                    stats = client.embed([texts[i] for i in todo], on_batch, status, should_stop)
                    break
                except _WorkerRetired as e:
                    # The worker left between requests (the idle/yield protocol in forge.embed_worker) and
                    # never read this one; a fresh worker re-takes the model lock and runs it.
                    self._client = None
                    if attempt == _MAX_RESTARTS:
                        raise RuntimeError(f"embedding worker for {self.spec} kept exiting before serving the "
                                           f"request ({e}); see {client.log_path}") from e
                    status("info", f"embedding worker had exited ({e}); starting a new one")
                except BaseException:
                    if not client.alive():    # already reported to this caller; the next call starts fresh
                        self._client = None
                    raise
        self.last_peak_bytes = max(stats["peak_bytes"], client.watchdog.peak_bytes)
        status("info", f"embedded {len(todo)} texts in {stats['batches']} batches; worker peak "
                       f"{describe(self.last_peak_bytes)} of the {describe(policy.budget_bytes)} budget"
                       + (f", {stats['backoffs']} batch back-offs" if stats["backoffs"] else ""))

    def embed(self, texts: list[str], on_progress: Progress | None = None, use_cache: bool = True,
              on_status: Status | None = None, should_stop: StopCheck | None = None) -> np.ndarray:
        """Embed ``texts`` -> ``[n, dim]`` float32 L2-normalised.

        Duplicates are embedded once. ``use_cache=False`` bypasses the disk cache entirely
        (used for latency measurement). ``on_progress(done, total)`` counts unique texts.
        ``should_stop()`` turning true kills the worker and raises InterruptedError.
        """
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        uniq = list(dict.fromkeys(texts))
        vecs: list[np.ndarray | None] = [None] * len(uniq)

        def write(pos: list[int], v: np.ndarray) -> None:
            for p, row in zip(pos, v):
                vecs[p] = row

        self._run(uniq, write, on_progress, use_cache, on_status, should_stop)
        at = {t: i for i, t in enumerate(uniq)}
        out = np.stack([vecs[at[t]] for t in texts]).astype(np.float32, copy=False)
        self.dim = out.shape[1]
        return out

    def embed_to_file(self, texts: list[str], path: Path, on_progress: Progress | None = None,
                      on_status: Status | None = None, should_stop: StopCheck | None = None) -> np.memmap:
        """Embed ``texts`` (one row per entry, duplicates allowed) into a ``.npy`` file written batch by
        batch; returns it as a read-only memmap so callers never hold the whole matrix in RAM."""
        if not texts:
            raise ValueError("no texts to embed")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        uniq = list(dict.fromkeys(texts))
        rows: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            rows.setdefault(t, []).append(i)
        out: list[np.memmap] = []

        def write(pos: list[int], v: np.ndarray) -> None:
            if not out:
                out.append(np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(texts), v.shape[1])))
            m = out[0]
            for p, row in zip(pos, v):
                m[rows[uniq[p]]] = row

        self._run(uniq, write, on_progress, True, on_status, should_stop)
        out[0].flush()
        del out[0]
        return np.load(path, mmap_mode="r")


_ENCODERS: dict[str, Encoder] = {}
_ENCODERS_LOCK = threading.Lock()


def get_encoder(spec: str) -> Encoder:
    """Process-wide encoder per spec (lazy: the worker starts on first ``embed`` or explicit ``load()``).

    ``spec`` is the caller's (see "Which spec" above); ``FORGE_MAX_LEN`` = token budget (default 2048).
    """
    if not spec:
        raise ValueError("get_encoder needs an encoder spec (resolve it through forge.encoders)")
    with _ENCODERS_LOCK:
        if spec not in _ENCODERS:
            _ENCODERS[spec] = Encoder(spec, int(os.environ.get("FORGE_MAX_LEN", DEFAULT_MAX_LEN)))
        return _ENCODERS[spec]
