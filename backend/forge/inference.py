"""Answer a task's question with a CLM head (fine-tuned checkpoint or the zero-shot reference).

Shared with the trainer and evaluator: ``pairs_for`` is the *only* place a state becomes the
state-head text and the options become candidate texts (via ``clm_core.schema.build_pairs``),
so training, evaluation, the playground and the shipped package all see identical inputs.

``Predictor`` loads a ``HeadPair`` (CPU by default: a single 4096->1536->1536->512 forward is
well under a millisecond there, and MPS launch overhead would dominate), embeds and projects
the task's option texts once (cached per head generation, so a hot-reloaded checkpoint
invalidates them), and scores ``exp(logit_scale) * cos(state_head(s), action_head(c))``.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .clm_core.heads import HeadPair, download
from .clm_core.schema import answer_from_logits, build_pairs, softmax
from .encoder import Encoder, get_encoder
from .encoders import head_spec
from .models import Example, TaskSchema


def pairs_for(task: TaskSchema, state: str, question: dict | None = None) -> tuple[str, list[str], list[str]]:
    """-> (state text = state + blank line + instructions, option keys, candidate texts).

    ``question`` (a row's own System One question, ``Example.meta["question"]``) replaces the task's
    question for that row: dynamic option sets (candidate tasks, projects) and per-row instructions
    (one noul per account) are asked exactly as the call site asks them."""
    return build_pairs(state, {task.question_id: question or task.question()})[task.question_id]


def row_question(ex: Example) -> dict | None:
    """The row's own question (``meta["question"]``), or None for the task's fixed question."""
    q = (ex.meta or {}).get("question")
    return q if isinstance(q, dict) and q else None


def target_for(ex: Example, keys: list[str], soft: bool = True) -> list[float]:
    """Training target over ``keys``: normalised ``ex.probabilities`` when soft and given, else one-hot."""
    if soft and ex.probabilities:
        t = [float(ex.probabilities.get(k, 0.0)) for k in keys]
        z = sum(t)
        if z > 0:
            return [v / z for v in t]
    return [float(k == ex.label) for k in keys]


class Predictor:
    """Loads a checkpoint (``ckpt=None`` -> reference head) and answers ``task`` for any state."""

    def __init__(self, ckpt: Path | None, task: TaskSchema, device: str = "cpu", encoder: Encoder | None = None):
        self.task = task
        self.question = task.question()
        path = str(ckpt) if ckpt is not None else download()
        self.name = "finetuned" if ckpt is not None else "reference"
        self.head = HeadPair(self.name, path, device).ensure()
        # the base model the head was trained on (forge.encoders.head_spec)
        self.encoder = encoder or get_encoder(head_spec(self.head.cfg, reference=ckpt is None, path=path))
        try:
            _, self.keys, self.candidates = pairs_for(task, "")
        except ValueError:      # dynamic-option task: no fixed criteria, every row brings its own question
            self.keys, self.candidates = [], []
        self._lock = threading.Lock()
        self._opt: tuple[int, object] | None = None   # (head generation, projected options [K, proj])

    # -- core ---------------------------------------------------------------------------------
    def _options(self):
        with self._lock:
            self.head.ensure()
            if self._opt is None or self._opt[0] != self.head.generation:
                e = self.encoder.embed(self.candidates)
                self._opt = (self.head.generation, self.head.project_actions(e))
            return self._opt[1]

    def logits_from_embeddings(self, states_emb: np.ndarray) -> np.ndarray:
        """``[n, dim]`` state-text embeddings -> ``[n, K]`` scaled cosine logits."""
        zo = self._options()
        zs = self.head.project_states(np.ascontiguousarray(states_emb, dtype=np.float32))
        return (self.head.scale * (zs @ zo.T)).cpu().numpy()

    def probs(self, states: list[str], on_progress=None, on_status=None) -> np.ndarray:
        """Batch: raw states -> ``[n, K]`` option probabilities (embeddings disk-cached).

        ``on_status(level, message)`` receives the encoder's memory/lock status (see ``forge.encoder``)."""
        texts = [pairs_for(self.task, s)[0] for s in states]
        lg = self.logits_from_embeddings(self.encoder.embed(texts, on_progress=on_progress, on_status=on_status))
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        return p / p.sum(1, keepdims=True)

    def row_probs(self, states: list[str], questions: list[dict | None], on_progress=None,
                  on_status=None) -> list[tuple[list[str], np.ndarray]]:
        """Per-row ``(option keys, probabilities)``; a row's own question (or None: the task's) decides
        its state text and its options, exactly as ``pairs_for`` builds them for training."""
        if not any(questions):
            p = self.probs(states, on_progress=on_progress, on_status=on_status)
            return [(list(self.keys), p[i]) for i in range(len(states))]
        pairs = [pairs_for(self.task, s, q) for s, q in zip(states, questions)]
        opts = list(dict.fromkeys(t for _, _, c in pairs for t in c))
        zs = self.head.project_states(np.ascontiguousarray(
            self.encoder.embed([p[0] for p in pairs], on_progress=on_progress, on_status=on_status), dtype=np.float32))
        zo = self.head.project_actions(np.ascontiguousarray(self.encoder.embed(opts, on_status=on_status),
                                                            dtype=np.float32))
        at = {t: i for i, t in enumerate(opts)}
        out = []
        for i, (_, keys, cands) in enumerate(pairs):
            lg = (self.head.scale * (zo[[at[t] for t in cands]] @ zs[i])).cpu().numpy().astype(np.float64)
            out.append((list(keys), np.asarray(softmax(lg.tolist()))))
        return out

    # -- public ---------------------------------------------------------------------------------
    def answer(self, state: str) -> tuple[dict, dict]:
        """-> (System One Answer dict, {embed_ms, head_ms, latency_ms})."""
        t0 = time.perf_counter()
        stext = pairs_for(self.task, state)[0]
        e = self.encoder.embed([stext])
        t1 = time.perf_counter()
        lg = self.logits_from_embeddings(e)[0]
        ans = answer_from_logits(self.question, self.keys, lg.tolist())
        t2 = time.perf_counter()
        return ans, {"embed_ms": (t1 - t0) * 1e3, "head_ms": (t2 - t1) * 1e3, "latency_ms": (t2 - t0) * 1e3}

    def rank(self, state: str, candidates: list[str], instructions: str | None = None) -> list[dict]:
        """Rank free-form candidates against ``state`` + question (default: the task's instructions)."""
        if not candidates:
            return []
        q = {"type": "choice", "instructions": instructions or self.task.instructions,
             "criteria": {str(i): c for i, c in enumerate(candidates)}}
        stext, _, cands = build_pairs(state, {"rank": q})["rank"]
        e = self.encoder.embed([stext, *cands])
        self.head.ensure()
        zs = self.head.project_states(e[:1])
        zc = self.head.project_actions(e[1:])
        probs = softmax((self.head.scale * (zs @ zc.T)).cpu().numpy()[0].tolist())
        order = sorted(range(len(candidates)), key=lambda i: -probs[i])
        return [{"rank": r + 1, "candidate": candidates[i], "prob": probs[i]} for r, i in enumerate(order)]
