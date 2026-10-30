"""Model backends for the frozen encoder. They run only inside the embedding worker process
(``forge.embed_worker``); the server talks to them through ``forge.encoder.Encoder``.

Every backend implements the same small surface:

    prepare() -> ModelShape     locate/download the files and describe the model (no weights in RAM)
    load()                      apply the framework memory caps, then load the weights
    tokenize(text) -> ids       no special tokens (the caller tail-truncates)
    forward(batch, width)       right-pad every row to ``width`` (a fixed bucket), return each row's
                                last-real-token hidden state after the final norm, float32 [n, dim]
    reset_peak() / peak_delta() bytes the last forward pass added at its peak, over the loaded state
    after_batch()               hand freed buffers back (MLX cache / MPS cache)
    close()                     free the weights / stop llama-server

Padding to a fixed bucket instead of the batch's own longest row is exact: with a causal mask,
right padding never reaches the real tokens and RoPE positions start at 0 either way (verified
cos >= 0.9999 against one-at-a-time). Fixed widths are what let the framework reuse its buffers.

Memory caps (verified against the installed MLX 0.32 / torch 2.14 APIs, not guessed):
    MLX    ``mx.set_memory_limit(budget)`` — a guideline: when exceeded MLX first frees its cache and
           waits for in-flight work, and only raises when no RAM is left (MLX 0.32 has no strict/
           "relaxed" switch any more; the parent's watchdog is the hard stop).
           ``mx.set_cache_limit(min(2 GB, budget/10))`` — the free-buffer cache that grew to tens of GB.
           ``mx.clear_cache()`` after every batch.
    torch  ``torch.mps.set_per_process_memory_fraction(budget / recommended_max_memory)`` — a hard
           cap: allocations past it raise. ``torch.mps.empty_cache()`` after every batch.
    llama  ``-c max_len*np -np <1-2> -b/-ub max_len`` (the largest bucket; an embedding must fit in one
           ubatch), quantized KV cache ``-ctk/-ctv`` (q8_0 default; quantized V needs ``-fa on``), and
           ``--cache-ram 0`` (the prompt cache defaults to 8 GiB and is useless for one-shot embeddings).
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from .memory import MemoryPolicy, ModelShape, shape_from_config, shape_from_gguf, tree_footprint, weights_size
from .paths import EMB_CACHE_DIR


def _pad(batch: list[list[int]], width: int) -> np.ndarray:
    arr = np.zeros((len(batch), width), dtype=np.int32)
    for i, ids in enumerate(batch):
        arr[i, :len(ids)] = ids
    return arr


class MLXBackend:
    name = "mlx"

    def __init__(self, model_name: str, max_len: int, policy: MemoryPolicy):
        self.model_name, self.max_len, self.policy = model_name, max_len, policy
        self.model = self.tok = None
        self.path: Path | None = None
        self._active0 = 0

    def prepare(self) -> ModelShape:
        from mlx_lm.utils import _download  # mlx_lm's own resolver: local dir or HF snapshot (same file set)
        self.path = _download(self.model_name)
        cfg = json.loads((self.path / "config.json").read_text())
        return shape_from_config(cfg, weights_size(sorted(self.path.glob("*.safetensors"))))

    def load(self) -> None:
        import mlx.core as mx
        from mlx_lm import load
        mx.set_memory_limit(self.policy.budget_bytes)
        mx.set_cache_limit(self.policy.cache_limit_bytes)
        self.model, self.tok = load(str(self.path))
        mx.synchronize()
        self._active0 = mx.get_active_memory()

    def tokenize(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def forward(self, batch: list[list[int]], width: int) -> np.ndarray:
        import mlx.core as mx
        h = self.model.model(mx.array(_pad(batch, width)))
        last = h[mx.arange(len(batch)), mx.array([len(ids) - 1 for ids in batch])].astype(mx.float32)
        mx.eval(last)
        out = np.array(last)
        del h, last
        return out

    def reset_peak(self) -> None:
        import mlx.core as mx
        mx.synchronize()
        mx.reset_peak_memory()

    def peak_delta(self) -> int:
        import mlx.core as mx
        return max(0, mx.get_peak_memory() - self._active0)

    def after_batch(self) -> None:
        import mlx.core as mx
        mx.clear_cache()

    def close(self) -> None:
        self.model = None
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()


class HFBackend:
    name = "hf"

    def __init__(self, model_name: str, max_len: int, policy: MemoryPolicy):
        self.model_name, self.max_len, self.policy = model_name, max_len, policy
        self.model = self.tok = None
        self.device = "cpu"
        self.path: Path | None = None
        self._driver0 = self._last_driver = 0

    def prepare(self) -> ModelShape:
        p = Path(self.model_name)
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(self.model_name, allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                                                                         "*.model", "*.txt", "*.jinja"]))
        self.path = p
        cfg = json.loads((p / "config.json").read_text())
        files = sorted(p.glob("*.safetensors"))
        # bf16 weights in RAM: a fp32 checkpoint halves on load
        size = weights_size(files) // (2 if cfg.get("torch_dtype") == "float32" else 1)
        return shape_from_config(cfg, size, attention_materialised=True)

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "mps":
            frac = min(2.0, self.policy.budget_bytes / torch.mps.recommended_max_memory())
            torch.mps.set_per_process_memory_fraction(frac)
        self.tok = AutoTokenizer.from_pretrained(str(self.path))
        self.model = AutoModel.from_pretrained(str(self.path), torch_dtype=torch.bfloat16).to(self.device).eval()
        self._driver0 = self._driver()

    def _driver(self) -> int:
        import torch
        return torch.mps.driver_allocated_memory() if self.device == "mps" else tree_footprint(os.getpid())

    def tokenize(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def forward(self, batch: list[list[int]], width: int) -> np.ndarray:
        import torch
        ids = torch.from_numpy(_pad(batch, width)).long()
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            mask[i, :len(row)] = 1
        with torch.no_grad():
            h = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        last = h[torch.arange(len(batch)), torch.tensor([len(r) - 1 for r in batch])]
        out = last.float().cpu().numpy()
        self._last_driver = self._driver()   # the caching allocator still holds the pass's blocks here
        del h, last
        return out

    def reset_peak(self) -> None:
        self._last_driver = self._driver0

    def peak_delta(self) -> int:
        return max(0, self._last_driver - self._driver0)

    def after_batch(self) -> None:
        import torch
        if self.device == "mps":
            torch.mps.empty_cache()

    def close(self) -> None:
        self.model = None
        gc.collect()
        self.after_batch()


class LlamaCppBackend:
    """A GGUF behind a private ``llama-server`` (embeddings only, last-token pooling, Metal offload)."""
    name = "llamacpp"

    def __init__(self, model_name: str, max_len: int, policy: MemoryPolicy):
        self.model_name, self.max_len, self.policy = model_name, max_len, policy
        self.proc: subprocess.Popen | None = None
        self.url = ""
        self.path = ""
        self._base = 0
        self._peak = 0

    def prepare(self) -> ModelShape:
        if self.model_name.endswith(".gguf") and os.path.isabs(self.model_name):
            self.path = self.model_name
        else:
            repo, _, fname = self.model_name.partition(":")
            if not fname:
                raise ValueError(f"llamacpp spec needs '<hf repo>:<file.gguf>' or an absolute path, got {self.model_name!r}")
            from huggingface_hub import hf_hub_download
            self.path = hf_hub_download(repo, fname)
        return shape_from_gguf(Path(self.path), self.policy.llama_kv_type)

    def command(self, exe: str, port: int) -> list[str]:
        np_ = self.policy.llama_parallel
        kv = self.policy.llama_kv_type
        cmd = [exe, "-m", self.path, "--embeddings", "--pooling", "last", "--host", "127.0.0.1",
               "--port", str(port), "-ngl", "999", "-c", str(self.max_len * np_), "-np", str(np_),
               "-b", str(self.max_len), "-ub", str(self.max_len), "-ctk", kv, "-ctv", kv,
               "--cache-ram", "0", "--no-webui"]
        if kv not in ("f16", "f32", "bf16"):
            cmd += ["-fa", "on"]       # llama.cpp requires flash attention for a quantized V cache
        return cmd

    def load(self) -> None:
        exe = os.environ.get("FORGE_LLAMA_SERVER") or shutil.which("llama-server")
        if not exe:
            raise RuntimeError("llamacpp encoder needs llama-server on PATH (brew install llama.cpp)")
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        logp = EMB_CACHE_DIR.parent / "llama-server.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "ab") as log:
            self.proc = subprocess.Popen(self.command(exe, port), stdout=log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{port}"
        import requests
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {logp}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    self._base = tree_footprint(os.getpid())
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.close()
        raise RuntimeError("llama-server did not become healthy within 10 minutes")

    def tokenize(self, text: str) -> list[int]:
        import requests
        r = requests.post(self.url + "/tokenize", json={"content": text, "add_special": False}, timeout=60)
        r.raise_for_status()
        return [t if isinstance(t, int) else t["id"] for t in r.json()["tokens"]]

    def forward(self, batch: list[list[int]], width: int) -> np.ndarray:
        import requests
        r = requests.post(self.url + "/v1/embeddings", json={"input": batch, "encoding_format": "float"}, timeout=600)
        r.raise_for_status()
        self._peak = max(self._peak, tree_footprint(os.getpid()))
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)

    def reset_peak(self) -> None:
        self._peak = self._base

    def peak_delta(self) -> int:
        return max(0, self._peak - self._base)

    def after_batch(self) -> None:
        pass

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


BACKENDS = {"mlx": MLXBackend, "hf": HFBackend, "llamacpp": LlamaCppBackend}


def make_backend(backend: str, model_name: str, max_len: int, policy: MemoryPolicy):
    return BACKENDS[backend](model_name, max_len, policy)
