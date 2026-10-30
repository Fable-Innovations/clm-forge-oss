"""Base-model selection: which frozen encoder should this project's head be trained on?

An experiment's candidates are its own encoder (``base_model``, picked when it was created, ``forge.encoder_choice``
recommends one): with a single candidate the stage records it and returns. Given several, each candidate (an encoder
spec, see ``encoder.py``) gets a short training run on the curated train split, scored on the val split; the test
split is never looked at, so the evaluation stays an unbiased estimate for whichever model wins. The winner is the highest val accuracy, ties broken
by lower val soft cross-entropy, then by faster embedding. A candidate that fails to load (e.g.
llama-server missing) is reported and skipped; selection fails only if every candidate fails.

Embeddings are disk-cached per model, so the full training run on the winner (and any recurate)
re-embeds nothing. Only one model is loaded at a time (``forge.encoder``): loading a candidate stops
the previous candidate's worker, and the losers are unloaded at the end.

Events: ``base_model`` with the per-candidate results (``status`` running|done|failed) and, at the
end, ``chosen``. Training events from the probe runs are not forwarded (they would draw on the
Training tab); their logs are, under stage ``selecting``.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import encoders
from .encoder import get_encoder
from .models import Example, TaskSchema

STAGE = "selecting"
# Specs of the registry entries this module names (metadata lives in backend/encoders.json, forge.encoders).
QWEN = encoders.get("qwen3-8b-mlx-4bit").spec
WINNOW = encoders.get("winnow-12b-gguf-q8").spec
# Qwen3-4B: 2560-d, no reference head fits it, so heads train from scratch. The GGUF build is the one
# to ship heads on (llama.cpp runs on macOS Metal and Windows/Linux CPU alike, so train and serve
# numerics match); the MLX build is for fast Apple-Silicon iteration and ablations.
QWEN4_MLX = encoders.get("qwen3-4b").spec
QWEN4_GGUF = encoders.get("qwen3-4b-gguf-q4km").spec


def display_name(spec: str) -> str:
    return encoders.name_of_spec(spec) or spec


def describe(spec: str) -> dict:
    e = encoders.by_spec(spec)
    return {"spec": spec, "name": display_name(spec), **({"key": e.key, "detail": e.note, "url": e.url} if e else {})}


@dataclass
class SelectConfig:
    # the full training schedule: candidates without a reference head start from scratch and would
    # lose a short probe to the warm-started one for the wrong reason; head training is cheap next
    # to embedding, which is cached and reused by the real training run
    epochs: int = 30
    patience: int = 6


def select_base_model(task: TaskSchema, rows: list[Example], out_dir: Path, candidates: list[str],
                      emit: Callable[[dict], None], should_stop: Callable[[], bool] = lambda: False,
                      cfg: SelectConfig | None = None, train_fn=None) -> dict:
    """Probe-train every candidate; -> ``{"chosen", "candidates": [...]}`` (also written to base_model.json)."""
    from .trainer import TrainConfig, train
    train_fn = train_fn or train
    cfg = cfg or SelectConfig()
    out_dir = Path(out_dir)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise ValueError("base-model selection needs the experiment's encoder (none given)")
    results = [{**describe(c), "status": "pending"} for c in candidates]

    def log(msg: str, level: str = "info") -> None:
        emit({"type": "log", "stage": STAGE, "level": level, "message": msg})

    def publish(chosen: str | None = None) -> None:
        emit({"type": "base_model", "candidates": results, "chosen": chosen})

    if len(candidates) == 1:
        results[0].update(status="done", note="only candidate")
        out = {"chosen": candidates[0], "candidates": results}
        (out_dir / "base_model.json").write_text(json.dumps(out, indent=1))
        publish(candidates[0])
        return out

    log(f"comparing {len(candidates)} base models on the val split: {', '.join(r['name'] for r in results)}")
    publish()
    for res in results:
        if should_stop():
            raise InterruptedError("selection cancelled")
        spec = res["spec"]
        res["status"] = "running"
        publish()
        enc = get_encoder(spec)

        def fwd(ev: dict, _name=res["name"]) -> None:
            if ev.get("type") == "log":
                emit({**ev, "stage": STAGE, "message": f"[{_name}] {ev['message']}"})
            elif ev.get("type") == "embed":
                emit(ev)

        t0 = time.perf_counter()
        try:
            enc.load(on_status=lambda level, msg, _f=fwd: _f({"type": "log", "level": level, "message": msg}),
                     should_stop=should_stop)
            res["load_s"] = round(time.perf_counter() - t0, 1)
            texts = list(dict.fromkeys(r.state for r in rows))[:8]
            t1 = time.perf_counter()
            enc.embed(texts, use_cache=False, should_stop=should_stop)
            res["embed_ms"] = round((time.perf_counter() - t1) * 1e3 / max(1, len(texts)), 1)
            tc = TrainConfig(epochs=cfg.epochs, patience=cfg.patience, encoder=spec)
            r = train_fn(task, rows, out_dir / "selection" / res["name"], tc, fwd, should_stop)
            if getattr(r, "cancelled", False):
                raise InterruptedError("selection cancelled")
            h = r.history[r.best_epoch] if r.history else {}
            res.update(status="done", val_acc=round(r.best_val_acc, 4), val_soft_ce=round(h.get("val_soft_ce", 0.0), 4),
                       init_val_acc=round(r.init_val_acc, 4), best_epoch=r.best_epoch, dim=enc.dim,
                       seconds=round(time.perf_counter() - t0, 1))
            log(f"{res['name']}: val acc {r.best_val_acc:.4f} (epoch {r.best_epoch}), "
                f"{res['embed_ms']:.0f} ms/text, {enc.dim}-d")
        except InterruptedError:
            raise
        except Exception as e:  # noqa: BLE001 - a broken candidate is reported, not fatal
            res.update(status="failed", error=f"{type(e).__name__}: {e}")
            log(f"{res['name']} failed: {res['error']}", "warn")
            emit({"type": "log", "stage": STAGE, "level": "warn", "message": traceback.format_exc()[-1500:]})
        publish()

    done = [r for r in results if r["status"] == "done"]
    if not done:
        raise RuntimeError("every base model failed: " + "; ".join(f"{r['name']}: {r.get('error')}" for r in results))
    best = max(done, key=lambda r: (r["val_acc"], -r["val_soft_ce"], -r["embed_ms"]))
    for r in results:
        r["chosen"] = r is best
    out = {"chosen": best["spec"], "candidates": results}
    (out_dir / "base_model.json").write_text(json.dumps(out, indent=1))
    log(f"selected {best['name']}: val acc {best['val_acc']:.4f}"
        + "".join(f" vs {r['name']} {r['val_acc']:.4f}" for r in done if r is not best))
    for r in results:            # free the losers (the winner's weights stay warm for training)
        if r is not best:
            try:
                get_encoder(r["spec"]).unload()
            except Exception:  # noqa: BLE001
                pass
    publish(best["spec"])
    return out


__all__ = ["QWEN", "QWEN4_GGUF", "QWEN4_MLX", "WINNOW", "SelectConfig", "describe", "display_name", "select_base_model"]
