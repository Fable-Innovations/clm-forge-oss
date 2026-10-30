"""Evaluate a fine-tuned head on the test split against baselines, and decide the ship gate.

Models compared (all on the same test rows unless noted):
    finetuned   the trained checkpoint
    reference   the zero-shot reference head CLM_v0.1-8B (same encoder, same texts) -- or, when the
                fine-tuned head's base model is not 4096-d (e.g. Qwen3-4B, 2560-d), a freshly
                initialised head of the same architecture on that base model (seed 0), so the 8B
                encoder is never loaded and "beats reference" still means "learned something"
    majority    always predict the most frequent *train* label
    teacher     optional: the labelling LLM on a deterministic sample of ≤50 test rows
                (``teacher(list_of_raw_states) -> list_of_labels``); scored on that sample only

Gate (deterministic, mirrors rules.py): finetuned ``metric`` ≥ ``threshold`` AND finetuned beats
reference AND beats majority on that metric. ``gate_reasons`` spells out every check.

Also: confusion matrix, per-label P/R/F1, 10-bin reliability diagram + ECE (confidence = top
probability), ≤25 worst errors (lowest probability on the gold label), and latency:
uncached embed per state (disk cache bypassed) p50/p95 over 60 single-state calls after 5 untimed
warm-up calls, head-only p50/p95, the warm path (embedding cache hit + head) p50, and the embedding
worker's resident footprint. Writes ``eval_report.json``.

The gate is scored on the test split only: without a test split ``evaluate`` raises (there is no val
fallback, a gate scored on the split that picked the checkpoint would flatter it). The head is also
scored on val (``EvalReport.val``: confusion and worst errors). The reviewer that writes curation guidance sees those val rows, never test rows: guidance
turns into generated training data, so test rows in it would leak the frozen test set into train.
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .clm_core.heads import HIDDEN, make_head
from .inference import Predictor, pairs_for, row_question
from .models import (Confusion, EvalReport, ErrorCase, Example, LabelScore, Latency, ModelScore,
                     ReliabilityBin, TaskSchema, ValDiagnostics)
from .trainer import throttled_embed_progress

Emit = Callable[[dict], None]
STAGE = "evaluating"
TEACHER_MAX = 50
LATENCY_WARMUP = 5   # untimed calls first: the worker's first pass at each padding bucket compiles kernels
LATENCY_N = 60       # timed single-state calls; p95 of fewer samples is one outlier
MAX_ERRORS = 25


def _scores(gold: list[str], pred: list[str], labels: list[str]) -> tuple[float, float, list[LabelScore]]:
    """-> (accuracy, macro F1 over labels with support or predictions, per-label scores)."""
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    per, f1s = [], []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        npred, sup = sum(p == lab for p in pred), sum(g == lab for g in gold)
        prec = tp / npred if npred else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per.append(LabelScore(label=lab, precision=prec, recall=rec, f1=f1, support=sup))
        if sup or npred:
            f1s.append(f1)
    return acc, (float(np.mean(f1s)) if f1s else 0.0), per


def _reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[list[ReliabilityBin], float]:
    out, ece, n = [], 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        c = int(m.sum())
        bc = float(conf[m].mean()) if c else (lo + hi) / 2
        ba = float(correct[m].mean()) if c else 0.0
        out.append(ReliabilityBin(bin_lo=lo, bin_hi=hi, confidence=bc, accuracy=ba, count=c))
        ece += (c / n) * abs(bc - ba) if n else 0.0
    return out, ece


def _confusion(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    li = {k: i for i, k in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        mat[li[g]][li[p]] += 1
    return mat


def _worst_errors(rows: list[Example], rp: list[tuple[list[str], np.ndarray]]) -> list[ErrorCase]:
    """Misclassified rows, lowest probability on the gold label first (at most ``MAX_ERRORS``).
    ``rp`` is each row's (option keys, probabilities)."""
    pred = [keys[int(np.argmax(p))] for keys, p in rp]
    gold_p = [float(p[keys.index(r.label)]) for r, (keys, p) in zip(rows, rp)]
    wrong = sorted((i for i, r in enumerate(rows) if pred[i] != r.label), key=lambda i: gold_p[i])
    return [ErrorCase(id=rows[i].id, state=rows[i].state, gold=rows[i].label, pred=pred[i],
                      probabilities={k: float(v) for k, v in zip(*rp[i])}) for i in wrong[:MAX_ERRORS]]


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else 0.0


REF_NOTE_8B = "zero-shot CLM_v0.1-8B reference head"


def _head_width(pred: Predictor) -> int:
    cfg = pred.head.cfg
    return int(cfg.get("hidden_size") or pred.head.state_head.inp.in_features)


def reference_predictor(ft: Predictor, task: TaskSchema, out_dir: Path) -> tuple[Predictor, str]:
    """The ``reference`` baseline for ``ft``: the CLM_v0.1-8B head when ``ft`` was trained on a 4096-d
    encoder, else a seed-0 freshly initialised head of ``ft``'s architecture on ``ft``'s encoder
    (written to ``out_dir/reference_init_head.pt``). -> (predictor, note)."""
    width = _head_width(ft)
    if width == HIDDEN:
        return Predictor(None, task), REF_NOTE_8B
    import torch
    cfg = dict(ft.head.cfg)
    kw = dict(width=cfg["width"], depth=cfg["depth"], proj=cfg.get("projection_dim", 512),
              activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
              residual=cfg.get("residual", False), hidden=width)
    torch.manual_seed(0)
    sh, ah = make_head(**kw), make_head(**kw)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "reference_init_head.pt"
    torch.save({"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                "logit_scale": torch.tensor(float(np.log(1 / 0.07))),
                "cfg": {**cfg, "init_ckpt": None, "reference": "init"}}, path)
    ref = Predictor(path, task, encoder=ft.encoder)
    ref.name = "reference"
    spec = cfg.get("embed_model") or getattr(ft.encoder, "spec", "?")
    return ref, f"zero-shot init head on {spec} (8B reference head does not fit this {width}-d encoder)"


def _row_probs(pred: Predictor, rows: list[Example], **kw) -> list[tuple[list[str], np.ndarray]]:
    return pred.row_probs([e.state for e in rows], [row_question(e) for e in rows], **kw)


def _none_match(gold: str, pred: str | None = None) -> str:
    """Dynamic option sets: a key like ``candidate_3`` means nothing across rows, so a row is scored as
    ``none``, ``match`` (the gold candidate) or ``wrong`` (another candidate)."""
    if pred is None:
        return "none" if gold == "none" else "match"
    if pred == "none":
        return "none"
    return "match" if pred == gold else "wrong"


def score_rows(task: TaskSchema, rows: list[Example], ckpt: Path, **row_probs_kw) -> dict:
    """Score ``ckpt`` on ``rows`` (every row, whatever its split) the way ``evaluate`` scores the test split
    (own-option rows as none / match / wrong), without baselines, latency or a report file.
    -> ``{n, macro_f1, accuracy, per_label: {label: f1}}``. Used for the rolling real check sets
    (a head-set adapter's real check); raises ``ValueError`` on no rows."""
    if not rows:
        raise ValueError("no rows to score")
    if task.type == "choice" and any(row_question(e) for e in rows):
        labels = ["none", "match"]
        gold = [_none_match(e.label) for e in rows]
        rp = _row_probs(Predictor(Path(ckpt), task), rows, **row_probs_kw)
        pred = [_none_match(e.label, k[int(np.argmax(p))]) for e, (k, p) in zip(rows, rp)]
    else:
        labels = task.labels() or sorted({e.label for e in rows})
        gold = [e.label for e in rows]
        rp = _row_probs(Predictor(Path(ckpt), task), rows, **row_probs_kw)
        pred = [k[int(np.argmax(p))] for k, p in rp]
    acc, f1, per = _scores(gold, pred, labels)
    return {"n": len(rows), "macro_f1": f1, "accuracy": acc, "per_label": {x.label: x.f1 for x in per}}


def evaluate(task: TaskSchema, examples: list[Example], ckpt: Path, out_dir: Path, emit: Emit,
             teacher: Callable[[list[str]], list[str]] | None = None, *,
             metric: str = "accuracy", threshold: float = 0.85,
             min_confident_share: float | None = None, confident_at: float = 0.8,
             threshold_note: str | None = None) -> EvalReport:
    """Score ``ckpt`` on the test split. Raises ``ValueError`` when there is no test split.

    Rows with their own question (``meta["question"]``) are scored over their own options. For a
    choice task whose rows bring their own option sets the labels are scored as none / match / wrong
    (``_none_match``), macro-F1 over none and match. ``min_confident_share`` adds a gate check: at
    least that share of test rows must have a top probability ≥ ``confident_at``, so a head that
    answers ~0.5 on everything fails even when its argmax happens to be right. ``threshold_note``
    records why ``threshold`` is not the caller's default (a per-head exception and its approval)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str, level: str = "info") -> None:
        emit({"type": "log", "stage": STAGE, "level": level, "message": msg})

    test = [e for e in examples if e.split == "test"]
    train = [e for e in examples if e.split == "train"]
    if not test:
        raise ValueError("no test split to gate on: the gate is scored on test rows only (no val fallback)")
    dynamic_choice = task.type == "choice" and any(row_question(e) for e in examples)
    if dynamic_choice:
        labels, conf_labels = ["none", "match"], ["none", "match", "wrong"]
        mapgold = lambda rows: [_none_match(e.label) for e in rows]  # noqa: E731
        mappred = lambda rows, pred: [_none_match(e.label, p) for e, p in zip(rows, pred)]  # noqa: E731
    else:
        labels = conf_labels = (task.labels() or sorted({e.label for e in examples}))
        mapgold = lambda rows: [e.label for e in rows]  # noqa: E731
        mappred = lambda rows, pred: list(pred)  # noqa: E731
    gold = mapgold(test)
    states = [e.state for e in test]

    ft = Predictor(Path(ckpt), task)
    # the 8B reference head runs on the default encoder it was trained on; other widths get an init head
    ref, ref_note = reference_predictor(ft, task, out_dir)
    log(f"evaluating on {len(test)} test rows ({metric} ≥ {threshold:.2f}); reference baseline: {ref_note}")
    status = lambda level, msg: log(msg, level)  # noqa: E731 - encoder lock/memory status as log events
    rp_ft = _row_probs(ft, test, on_progress=throttled_embed_progress(emit), on_status=status)
    rp_ref = _row_probs(ref, test, on_status=status)
    raw_ft = [k[int(np.argmax(p))] for k, p in rp_ft]
    pred_ft = mappred(test, raw_ft)
    pred_ref = mappred(test, [k[int(np.argmax(p))] for k, p in rp_ref])

    acc_ft, f1_ft, per_label = _scores(gold, pred_ft, labels)
    acc_ref, f1_ref, _ = _scores(gold, pred_ref, labels)
    # The most frequent train *key*, predicted on every row. For per-row option sets that is a
    # positional key ("none", "candidate_0"): scored like any prediction, so "always match" (which
    # no real predictor can do without reading the row) is never a baseline.
    majority = Counter(e.label for e in (train or test)).most_common(1)[0][0]
    acc_maj, f1_maj, _ = _scores(gold, mappred(test, [majority] * len(test)), labels)
    models = [ModelScore(name="finetuned", accuracy=acc_ft, macro_f1=f1_ft, n=len(test)),
              ModelScore(name="reference", accuracy=acc_ref, macro_f1=f1_ref, n=len(test), note=ref_note),
              ModelScore(name="majority", accuracy=acc_maj, macro_f1=f1_maj, n=len(test),
                         note=f"always '{majority}' (most frequent train label)")]

    if teacher is not None:
        rng = np.random.default_rng(0)
        idx = sorted(rng.permutation(len(test))[:TEACHER_MAX].tolist())
        try:
            t0 = time.perf_counter()
            tp = [str(x) for x in teacher([states[i] for i in idx])]
            secs = time.perf_counter() - t0
            if len(tp) != len(idx):
                raise ValueError(f"teacher returned {len(tp)} labels for {len(idx)} states")
            sub = [test[i] for i in idx]
            a, f, _ = _scores(mapgold(sub), mappred(sub, tp), labels)
            # The batch above amortises one LLM call over many rows; what the call site would pay
            # is one call per decision, so time that separately (median of 3 single-row calls).
            single = []
            for i in idx[:3]:
                t1 = time.perf_counter()
                teacher([states[i]])
                single.append((time.perf_counter() - t1) * 1e3)
            models.append(ModelScore(name="teacher", accuracy=a, macro_f1=f, n=len(idx),
                                     note=f"single call p50 ~{float(np.median(single)):.0f} ms; LLM teacher on a "
                                          f"{len(idx)}-row test sample (batched ~{secs / max(1, len(idx)) * 1e3:.0f}/row)"))
        except Exception as e:  # the teacher is optional; never fail the evaluation over it
            log(f"teacher evaluation failed: {e}", "warn")

    # ---- confusion / calibration / decisiveness ------------------------------------------------------
    mat = _confusion(gold, pred_ft, conf_labels)
    conf = np.array([float(np.max(p)) for _, p in rp_ft])
    correct = np.array([r.label == p for r, p in zip(test, raw_ft)], dtype=float)
    reliability, ece = _reliability(conf, correct)
    errors = _worst_errors(test, rp_ft)
    sure = conf >= confident_at
    confident_share = float(sure.mean())
    confident_accuracy = float(correct[sure].mean()) if sure.any() else None

    # ---- gate --------------------------------------------------------------------------------------
    key = "macro_f1" if metric == "macro_f1" else "accuracy"
    v_ft, v_ref, v_maj = (getattr(m, key) for m in models[:3])
    checks = [(v_ft >= threshold, f"fine-tuned {key} {v_ft:.3f} {'≥' if v_ft >= threshold else '<'} threshold {threshold:.3f}"),
              (v_ft > v_ref, f"fine-tuned {key} {v_ft:.3f} {'>' if v_ft > v_ref else '≤'} zero-shot reference {v_ref:.3f}"),
              (v_ft > v_maj, f"fine-tuned {key} {v_ft:.3f} {'>' if v_ft > v_maj else '≤'} majority baseline {v_maj:.3f}")]
    if min_confident_share is not None:
        ok = confident_share >= min_confident_share
        checks.append((ok, f"confident share (top p ≥ {confident_at:.2f}) {confident_share:.3f} "
                           f"{'≥' if ok else '<'} {min_confident_share:.3f}"))
    gate = all(ok for ok, _ in checks)
    reasons = [("PASS: " if ok else "FAIL: ") + msg for ok, msg in checks]

    # ---- val diagnostics: the only per-row evidence curation guidance may use ---------------------------
    val = None
    val_rows = [e for e in examples if e.split == "val"]
    if val_rows:
        rp_val = _row_probs(ft, val_rows, on_status=status)    # cached since training
        vg = mapgold(val_rows)
        vp = mappred(val_rows, [k[int(np.argmax(p))] for k, p in rp_val])
        v_acc, v_f1, _ = _scores(vg, vp, labels)
        val = ValDiagnostics(n=len(val_rows), accuracy=v_acc, macro_f1=v_f1,
                             confusion=Confusion(labels=conf_labels, matrix=_confusion(vg, vp, conf_labels)),
                             errors=_worst_errors(val_rows, rp_val))

    # ---- latency ----------------------------------------------------------------------------------------
    # p_ref above may have swapped the reference model in (one model per process); load the fine-tuned
    # head's encoder now so the first timed call measures embedding, not a model load
    ft.encoder.load(on_status=status)
    for e in test[:LATENCY_WARMUP]:
        ft.encoder.embed([pairs_for(task, e.state, row_question(e))[0]], use_cache=False)
    sample = [test[i % len(test)] for i in range(LATENCY_WARMUP, LATENCY_WARMUP + LATENCY_N)]
    emb_ms, head_ms, warm_ms = [], [], []
    for e in sample:
        q = row_question(e)
        stext = pairs_for(task, e.state, q)[0]
        t0 = time.perf_counter()
        emb = ft.encoder.embed([stext], use_cache=False)
        t1 = time.perf_counter()
        if q is None:
            ft.logits_from_embeddings(emb)
        else:
            ft.head.project_states(np.ascontiguousarray(emb, dtype=np.float32))
        t2 = time.perf_counter()
        emb_ms.append((t1 - t0) * 1e3)
        head_ms.append((t2 - t1) * 1e3)
        t3 = time.perf_counter()
        ft.row_probs([e.state], [q])                     # embedding now a cache hit
        warm_ms.append((time.perf_counter() - t3) * 1e3)
    latency = Latency(embed_ms_p50=_pct(emb_ms, 50), embed_ms_p95=_pct(emb_ms, 95),
                      head_ms_p50=_pct(head_ms, 50), head_ms_p95=_pct(head_ms, 95),
                      warm_cached_ms_p50=_pct(warm_ms, 50), n=len(sample),
                      encoder_peak_bytes=ft.encoder.info().get("worker", {}).get("footprint_peak_bytes"))

    report = EvalReport(metric=key, threshold=threshold, threshold_note=threshold_note, gate_passed=gate,
                        gate_reasons=reasons, models=models,
                        confusion=Confusion(labels=conf_labels, matrix=mat), reliability=reliability, ece=ece,
                        per_label=per_label, latency=latency, errors=errors, n_test=len(test), val=val,
                        confident_at=confident_at, confident_share=confident_share,
                        confident_accuracy=confident_accuracy, min_confident_share=min_confident_share,
                        scoring="none_match" if dynamic_choice else "labels")
    (out_dir / "eval_report.json").write_text(report.model_dump_json(indent=1))
    log(f"finetuned {key} {v_ft:.3f} | reference {v_ref:.3f} | majority {v_maj:.3f} | confident "
        f"{confident_share:.2f} | ECE {ece:.3f} | gate {'PASSED' if gate else 'FAILED'} | embed p50 "
        f"{latency.embed_ms_p50:.0f} ms")
    emit({"type": "eval", "report": report.model_dump()})
    return report
