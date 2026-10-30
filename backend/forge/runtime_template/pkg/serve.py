"""HTTP server for non-Python callers (``pip install '<pkg>[serve]'``).

    POST /v1/systemone  {"state": ..., "questions": {id: Question}?, "temperature"?: 1.0}
                        -> {"model", "answers": {id: Answer}, "usage"}   (System One wire format;
                           ``questions`` defaults to this package's own question)
    POST /v1/predict    {"state": ..., "threshold"?: float} -> {label, probabilities, confidence, source}
    GET  /health        -> {"ok": true, "task": ...}

The head is fine-tuned for this package's question; other questions sent to /v1/systemone are
answered by the same head and are only as good as its general (zero-shot) ability.
"""
from __future__ import annotations

import argparse
import time


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import JSONResponse
    except ImportError as e:  # pragma: no cover
        raise SystemExit("the server needs the [serve] extra: pip install '<package>[serve]'") from e
    from .model import get_model, task

    app = FastAPI(title=f"CLM {task()['name']}", version="1")
    model = get_model()

    @app.get("/health")
    def health():
        return {"ok": True, "task": task()}

    @app.post("/v1/systemone")
    async def systemone(body: dict):
        if "state" not in body:
            raise HTTPException(422, "body must be {state, questions?}")
        questions = body.get("questions") or None
        if questions is not None and not isinstance(questions, dict):
            raise HTTPException(422, "questions must be an object {id: Question}")
        t0 = time.perf_counter()
        try:
            answers = await run_in_threadpool(model.answer, body["state"], questions, float(body.get("temperature", 1.0)))
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        return JSONResponse({"model": f"clm_{task()['name']}", "answers": answers,
                             "usage": {"billing_units": len(answers), "input_tokens": 0, "output_tokens": 0}},
                            headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1e3:.1f}"})

    @app.post("/v1/predict")
    async def predict(body: dict):
        if "state" not in body:
            raise HTTPException(422, "body must be {state, threshold?}")
        return await run_in_threadpool(lambda: model.predict(body["state"], threshold=body.get("threshold")))

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve this CLM package over HTTP (System One compatible).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--warm", action="store_true", help="load the encoder before accepting requests")
    a = ap.parse_args()
    import uvicorn
    app = create_app()
    if a.warm:
        from .model import get_model
        get_model().predict("warm-up")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
