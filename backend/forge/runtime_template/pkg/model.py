"""The packaged model: task definition + fine-tuned CLM head + a pluggable encoder.

``predict(state)`` -> ``{"label", "probabilities", "confidence", "source"}`` where ``confidence``
is the System One/TypeSafe confidence (top probability minus the mean of the others) and
``source`` is ``"clm"`` or ``"fallback"``. With ``fallback=`` and a threshold, low-confidence
states are handed to the caller's original code path instead.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np

from ._head import HeadPair
from ._schema import answer_from_logits, build_pairs, confidence, probabilities_of, label_of
from .encoders import Encoder, encoder_from_env

_PKG = __package__ or __name__.rpartition(".")[0]


def _asset(name: str) -> str:
    return str(resources.files(_PKG).joinpath(name))


class Model:
    """Loads ``task.json`` + ``head.pt``; option projections are computed once and cached."""

    def __init__(self, encoder: Encoder | None = None):
        with open(_asset("task.json")) as f:
            self.meta: dict = json.load(f)
        self.task: dict = self.meta["task"]
        self.question: dict = self.meta["question"]
        self.qid: str = self.task["question_id"]
        self.head = HeadPair(_asset("head.pt"))
        enc_meta = self.meta["encoder"]
        trained = enc_meta.get("backend", "mlx")
        # default to the encoder the head was trained on (model + pinned revision), never a guess
        self.encoder = encoder or encoder_from_env(int(enc_meta.get("max_len", 2048)), trained,
                                                   model=enc_meta.get("model"), revision=enc_meta.get("revision"))
        self.default_threshold: float = float(self.meta.get("fallback_threshold", 0.5))
        self._opts: dict[str, Any] = {}
        self._lock = threading.Lock()
        # Option embeddings exactly as used in training/evaluation; only valid for the training encoder.
        opt_file = _asset("options.npy")
        kind = os.environ.get("CLM_ENCODER", trained).lower()
        same_encoder = encoder is None and kind == trained and (
            kind != "mlx" or os.environ.get("CLM_MLX_MODEL", self.meta["encoder"]["model"]) == self.meta["encoder"]["model"])
        if same_encoder and os.path.exists(opt_file):
            texts = [o["text"] for o in self.meta["options"]]
            self._opts["\x00".join(texts)] = self.head.project_actions(np.load(opt_file))

    def _options(self, texts: list[str]) -> np.ndarray:
        key = "\x00".join(texts)
        with self._lock:
            if key not in self._opts:
                self._opts[key] = self.head.project_actions(self.encoder.embed(texts))
            return self._opts[key]

    def answer(self, state: Any, questions: dict[str, dict] | None = None, temperature: float = 1.0) -> dict:
        """System One answers ``{qid: Answer}``; defaults to this package's own question."""
        questions = questions or {self.qid: self.question}
        pairs = build_pairs(state, questions)
        zs = self.head.project_states(self.encoder.embed([p[0] for p in pairs.values()]))
        out = {}
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            logits = self.head.scale * (self._options(texts) @ zs[i]) / temperature
            out[qid] = answer_from_logits(questions[qid], keys, logits.tolist())
        return out

    def predict(self, state: Any, *, threshold: float | None = None,
                fallback: Callable[[Any], Any] | None = None) -> dict:
        """Answer this package's question. Below ``threshold`` (default: the packaged
        ``fallback_threshold``, only applied when ``fallback`` is given) ``fallback(state)``
        supplies the label and ``source`` becomes ``"fallback"``."""
        t0 = time.perf_counter()
        ans = self.answer(state)[self.qid]
        probs = probabilities_of(ans)
        conf = confidence(list(probs.values()))
        res = {"label": label_of(ans), "probabilities": probs, "confidence": conf, "source": "clm",
               "latency_ms": (time.perf_counter() - t0) * 1e3}
        tau = threshold if threshold is not None else (self.default_threshold if fallback else None)
        if tau is not None and conf < tau:
            res["low_confidence"] = True
            if fallback is not None:
                res.update(clm_label=res["label"], label=fallback(state), source="fallback")
        return res


_MODEL: Model | None = None
_MODEL_LOCK = threading.Lock()


def get_model() -> Model:
    """Process-wide model (the encoder loads lazily on the first call)."""
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = Model()
        return _MODEL


def predict(state: Any, *, threshold: float | None = None, fallback: Callable[[Any], Any] | None = None) -> dict:
    return get_model().predict(state, threshold=threshold, fallback=fallback)


def task() -> dict:
    """The task this package answers (question id, type, instructions, options, metrics)."""
    with open(_asset("task.json")) as f:
        meta = json.load(f)
    return {**meta["task"], "labels": meta["labels"], "fallback_threshold": meta.get("fallback_threshold"),
            "metrics": meta.get("metrics", {}), "encoder": meta["encoder"]}


if os.environ.get("CLM_PRELOAD") == "1":  # pragma: no cover - opt-in warm start at import
    get_model()
