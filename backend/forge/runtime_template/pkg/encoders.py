"""Text encoders for the CLM head. The embedding recipe MUST match training:

    the training base model (``task.json`` -> ``encoder``), tokenised without special tokens,
    tail-truncated to ``max_len - 1`` tokens, final hidden state (after the final norm) of the
    LAST token, float32, L2-normalised.

``task.json`` -> ``encoder`` records which model and backend the head in this package was trained
on (``model``, ``revision``, ``backend``, ``dim``); those are the defaults here, so the package
works with no environment variables. Another build of the same base model (bf16 via vLLM, for
example) produces very close but not identical vectors; validate on ``examples/golden.jsonl``
before switching, or retrain the head on the encoder you will run in production.

Backends (pick with ``CLM_ENCODER``; default: the training backend):

``mlx``
    In-process on Apple Silicon via mlx-lm. ``CLM_MLX_MODEL`` overrides the model (a Hugging Face
    repo id or a local directory); by default the packaged ``encoder.model`` at the packaged
    ``encoder.revision`` is loaded. MLX keeps freed buffers in a cache that grows to several GB
    under steady inference; it is capped at ``CLM_MLX_CACHE_LIMIT_MB`` (default 256 MB).
``openai``
    Any OpenAI-compatible ``/v1/embeddings`` endpoint serving the same base model with last-token
    pooling, e.g. on a GPU box::

        vllm serve <base model> --runner pooling --served-model-name <name> --max-model-len 2048

    ``CLM_EMBED_URL`` (e.g. ``http://gpu-box:8090/v1/embeddings``), ``CLM_EMBED_MODEL``
    (the served model name), ``CLM_EMBED_API_KEY`` (optional). vLLM's ``truncate_prompt_tokens``
    keeps the tail, matching the recipe; set ``CLM_EMBED_TRUNCATE=0`` for servers that reject it.
``llamacpp``
    A ``llama-server`` running the training GGUF with last-token pooling::

        llama-server -m <training>.gguf --embeddings --pooling last -ub 2048 -c 8192 -np 4 --port 8090

    ``CLM_LLAMA_URL`` (default ``http://127.0.0.1:8090``). Text is tokenised by the server's
    ``/tokenize`` (no special tokens), tail-truncated here, and sent as token ids, so the recipe
    matches training exactly.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import urllib.request
from typing import Protocol

import numpy as np

DEFAULT_MAX_LEN = 2048
DEFAULT_MLX_CACHE_LIMIT_MB = 256


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


class Encoder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class MLXEncoder:
    """The training base model on Apple Silicon via mlx-lm; weights load lazily on first use.

    ``cache_limit_mb`` caps MLX's free-buffer cache (``mx.set_cache_limit``) when the weights load;
    without it steady inference grows the process footprint to roughly twice the weights.
    """

    def __init__(self, model: str, max_len: int = DEFAULT_MAX_LEN, *, revision: str | None = None,
                 cache_limit_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB):
        if not model:
            raise ValueError("MLXEncoder needs the training model id (task.json -> encoder.model)")
        if cache_limit_mb < 0:
            raise ValueError(f"cache_limit_mb must be >= 0, got {cache_limit_mb}")
        self.model_name, self.revision, self.max_len = model, revision, max_len
        self.cache_limit_mb = cache_limit_mb
        self._model = self._tok = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is None:
            try:
                import mlx.core as mx
                from mlx_lm import load
            except ImportError as e:
                raise RuntimeError("the mlx encoder needs `pip install mlx-lm` (Apple Silicon); "
                                   "or set CLM_ENCODER=openai and CLM_EMBED_URL") from e
            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
            kwargs = {"revision": self.revision} if self.revision else {}
            self._model, self._tok = load(self.model_name, **kwargs)

    def load(self) -> None:
        """Load the weights now (otherwise the first ``embed`` does)."""
        with self._lock:
            self._load()

    def embed(self, texts: list[str]) -> np.ndarray:
        import mlx.core as mx
        with self._lock:
            self._load()
            out = []
            for t in texts:
                ids = list(self._tok.encode(t, add_special_tokens=False)) or list(self._tok.encode(" ", add_special_tokens=False))
                ids = ids[-(self.max_len - 1):]
                h = self._model.model(mx.array([ids]))[0, -1].astype(mx.float32)
                mx.eval(h)
                out.append(np.array(h))
        return _l2(np.stack(out))


class OpenAIEncoder:
    """POST ``{model, input}`` to an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(self, url: str, model: str, api_key: str | None = None,
                 max_len: int = DEFAULT_MAX_LEN, truncate: bool = True, timeout: float = 60.0, batch: int = 32):
        self.url, self.model, self.api_key = url, model, api_key
        self.max_len, self.truncate, self.timeout, self.batch = max_len, truncate, timeout, batch

    def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        body: dict = {"model": self.model, "input": texts, "encoding_format": "base64"}
        if self.truncate:
            body["truncate_prompt_tokens"] = self.max_len - 1
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            j = json.loads(r.read())
        out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        for d in j["data"]:
            e = d["embedding"]
            out[d["index"]] = (np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str)
                               else np.asarray(e, dtype=np.float32))
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            vecs += self._fetch(texts[i:i + self.batch])
        return _l2(np.stack(vecs))


class LlamaCppEncoder:
    """A ``llama-server --embeddings --pooling last`` endpoint, fed token ids (see module docstring)."""

    def __init__(self, url: str = "http://127.0.0.1:8090", max_len: int = DEFAULT_MAX_LEN, timeout: float = 120.0,
                 batch: int = 16):
        self.url, self.max_len, self.timeout, self.batch = url.rstrip("/"), max_len, timeout, batch

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _ids(self, text: str) -> list[int]:
        toks = self._post("/tokenize", {"content": text, "add_special": False})["tokens"]
        ids = [t if isinstance(t, int) else t["id"] for t in toks] or self._ids(" ")
        return ids[-(self.max_len - 1):]

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            chunk = [self._ids(t) for t in texts[i:i + self.batch]]
            data = sorted(self._post("/v1/embeddings", {"input": chunk, "encoding_format": "float"})["data"],
                          key=lambda d: d["index"])
            vecs += [np.asarray(d["embedding"], dtype=np.float32) for d in data]
        return _l2(np.stack(vecs))


def _cache_limit_mb() -> int:
    raw = os.environ.get("CLM_MLX_CACHE_LIMIT_MB")
    if raw is None:
        return DEFAULT_MLX_CACHE_LIMIT_MB
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"CLM_MLX_CACHE_LIMIT_MB must be an integer (MB), got {raw!r}") from e


def encoder_from_env(max_len: int = DEFAULT_MAX_LEN, default: str = "mlx", *, model: str | None = None,
                     revision: str | None = None) -> Encoder:
    """Build the encoder selected by ``CLM_ENCODER`` (``mlx`` | ``openai`` | ``llamacpp``; default: ``default``).

    ``model`` / ``revision`` are the training encoder from ``task.json``; ``CLM_MLX_MODEL`` overrides
    the mlx model (and drops the packaged revision, which belongs to the packaged model)."""
    kind = os.environ.get("CLM_ENCODER", default).lower()
    if kind == "mlx":
        override = os.environ.get("CLM_MLX_MODEL")
        name = override or model
        if not name:
            raise RuntimeError("no mlx encoder model: task.json has no encoder.model and CLM_MLX_MODEL is unset")
        return MLXEncoder(name, max_len, revision=None if override else revision, cache_limit_mb=_cache_limit_mb())
    if kind == "llamacpp":
        return LlamaCppEncoder(os.environ.get("CLM_LLAMA_URL", "http://127.0.0.1:8090"), max_len)
    if kind == "openai":
        url = os.environ.get("CLM_EMBED_URL")
        served = os.environ.get("CLM_EMBED_MODEL")
        if not url or not served:
            raise RuntimeError("CLM_ENCODER=openai needs CLM_EMBED_URL (e.g. http://host:8090/v1/embeddings) and "
                               "CLM_EMBED_MODEL (the served name of the training base model)")
        return OpenAIEncoder(url, served, os.environ.get("CLM_EMBED_API_KEY"),
                             max_len, truncate=os.environ.get("CLM_EMBED_TRUNCATE", "1") != "0")
    raise RuntimeError(f"unknown CLM_ENCODER {kind!r}; use 'mlx', 'openai' or 'llamacpp'")
