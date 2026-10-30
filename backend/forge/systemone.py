"""A generic System One server: answers arbitrary closed questions with a CLM head.

    GET  /health        -> {"ok", "ready", "models": [model id], "loaded": model id | null, "encoder", "head"}
    POST /v1/systemone  {"model"?, "state", "questions": {id: Question}, "temperature"?: 1.0}
                        -> {"model", "answers": {id: Answer}, "usage"}

The packaged runtime (``runtime_template/pkg/serve.py``) serves one fine-tuned task; this server
serves *any* ``noul`` / ``choice`` / ``score`` question, which is what the zero-shot reference head
(``CLM_v0.1-8B.pt``, trained across many closed questions) is for. ``--head`` points it at another
checkpoint instead (it must have been trained on the same encoder).

Everything is reused, nothing re-implemented: ``clm_core.schema`` turns (state, question) into the
state text and candidate texts and logits into typed Answers (the same code the trainer and the
packages use), ``clm_core.heads`` loads the head, and ``forge.encoder`` embeds through the governed
embedding worker (machine-wide model lock, admission control, MLX memory/cache caps, watchdog;
see ``forge.memory``). The base model never loads into this process.

Nothing a caller sends is written to Forge's embedding cache: state texts carry private
content (mail, messages) and option texts can too (candidate task titles, project names,
account labels), so both are embedded with the disk cache bypassed.

Run: ``uv run forge-systemone --port 8711`` (from ``backend/``). ``--warm`` loads the encoder
before serving; otherwise the first request loads it and ``/health`` reports ``loaded: false``.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .clm_core.heads import HeadPair, download
from .clm_core.schema import answer_from_logits, build_pairs
from .encoder import Encoder, get_encoder
from .encoders import head_spec

log = logging.getLogger("forge.systemone")
REFERENCE_MODEL_ID = "clm-v0.1-8b-reference"


class Answerer:
    """One head + one encoder; answers System One question maps."""

    def __init__(self, head_path: str | None = None, model_id: str | None = None, encoder: Encoder | None = None):
        path = head_path or download()
        self.head = HeadPair("reference" if head_path is None else Path(path).stem, path).ensure()
        self.model_id = model_id or (REFERENCE_MODEL_ID if head_path is None else Path(path).stem)
        # the base model the head was trained on (forge.encoders.head_spec)
        self.encoder = encoder or get_encoder(head_spec(self.head.cfg, reference=head_path is None, path=path))
        self._lock = threading.Lock()   # one embedding call at a time (the worker is single-file anyway)

    def answer(self, state: Any, questions: dict[str, dict], temperature: float = 1.0) -> dict[str, dict]:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object {id: Question}")
        if not temperature > 0:
            raise ValueError("temperature must be > 0")
        pairs = build_pairs(state, questions)
        states = [p[0] for p in pairs.values()]
        options = list(dict.fromkeys(t for p in pairs.values() for t in p[2]))
        with self._lock:
            zs = self.head.project_states(self.encoder.embed(states, use_cache=False))
            zo = self.head.project_actions(self.encoder.embed(options, use_cache=False))
        at = {t: i for i, t in enumerate(options)}
        out: dict[str, dict] = {}
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            cand = zo[[at[t] for t in texts]]
            logits = (self.head.scale * (cand @ zs[i]) / temperature).cpu().numpy()
            out[qid] = answer_from_logits(questions[qid], keys, np.asarray(logits, dtype=np.float64).tolist())
        return out

    def health(self) -> dict:
        info = self.encoder.info()
        # ``loaded`` is the model now resident (System One convention: a model id, or null)
        return {"ok": True, "ready": True, "models": [self.model_id],
                "loaded": self.model_id if info["loaded"] else None,
                "encoder": {k: info[k] for k in ("backend", "model", "max_len", "dim")},
                "head": {"path": self.head.path, "params": self.head.n_params}}


def create_app(answerer: Answerer):
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    app = FastAPI(title="CLM System One", version="1")

    @app.get("/health")
    def health():
        return answerer.health()

    @app.post("/v1/systemone")
    async def systemone(body: dict):
        if "state" not in body or not isinstance(body.get("questions"), dict):
            raise HTTPException(422, "body must be {model?, state, questions: {id: Question}, temperature?}")
        model = body.get("model")
        if model not in (None, "", answerer.model_id):
            raise HTTPException(404, f"model {model!r} is not served here; this server serves {answerer.model_id!r}")
        t0 = time.perf_counter()
        try:
            answers = await run_in_threadpool(answerer.answer, body["state"], body["questions"],
                                              float(body.get("temperature", 1.0)))
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        ms = (time.perf_counter() - t0) * 1e3
        log.info("answered %d question(s) in %.0f ms", len(answers), ms)
        return JSONResponse({"model": answerer.model_id, "answers": answers,
                             "usage": {"billing_units": len(answers), "input_tokens": 0, "output_tokens": 0}},
                            headers={"X-CLM-Latency-Ms": f"{ms:.1f}"})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a CLM head as a generic System One decision server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--head", default=None, help="checkpoint path (default: the zero-shot reference head)")
    ap.add_argument("--model-id", default=None, help=f"model id to report (default: {REFERENCE_MODEL_ID})")
    ap.add_argument("--warm", action="store_true", help="load the encoder before accepting requests")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn
    answerer = Answerer(a.head, a.model_id)
    if a.warm:
        answerer.encoder.load()
    uvicorn.run(create_app(answerer), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
