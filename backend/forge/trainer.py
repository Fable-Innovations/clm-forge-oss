"""Fine-tune a CLM head pair for one typed question (port of CLM ``train/finetune.py run_choice``).

Data path — identical to inference, so train and serve can never drift:
    Example.state --clm_core.schema.build_pairs(state, {qid: task.question()})--> state text
    (``state + "\\n\\n" + instructions``) and one candidate text per option, embedded by the frozen
    encoder (``forge.encoder``, disk-cached). The flat text table is streamed to
    ``<out_dir>/embeddings.npy`` and read memory-mapped (rows gathered per step), then deleted.

Objective — ``run_choice --loss infonce``: a flat training set where every distinct text is
embedded once and options are indices into it; each step computes a bidirectional in-batch
InfoNCE over the batch's distinct option texts (states -> options against the gold
distribution, options -> states against the states that hold them), with soft targets from
``Example.probabilities`` when present. Every row of a Forge project shares the same K options,
so the option pool per batch is those K texts.

Base model: ``TrainConfig.encoder`` (an encoder spec, required: the project's ``base_model``). The head's input
width follows the encoder's embedding width.

Optimisation: warm start from the reference head ``CLM_v0.1-8B.pt`` (its arch cfg is reused) when
its input width matches the encoder (it was trained on Qwen3-8B); other base models start from scratch.
AdamW + OneCycle (10% warm-up, cosine), grad-clip 1.0, learnable logit scale (exp clamped at 100).
Early stopping on val accuracy (ties broken by lower val soft cross-entropy, which matters on
small data where accuracy saturates); the best weights are kept in memory and written once as
``best_head.pt`` in the CLM checkpoint format (+ ``history.json``).

Events (dicts without ts/seq): ``embed``, ``log``, ``train_step`` (≤5/s), ``train_epoch`` and
``train_snapshot`` (epoch 0 = init, then every epoch). Snapshot points are a 2-D PCA of projected
val/test states + option vectors, refit every epoch and sign-aligned to the previous epoch so the
scatter animates instead of flipping.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .clm_core.heads import HIDDEN, download, make_head
from .encoder import get_encoder
from .inference import pairs_for, row_question, target_for
from .models import Example, TaskSchema

Emit = Callable[[dict], None]
STAGE = "training"
MAX_POINTS = 400


def _default_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


EPOCHS_WARM, PATIENCE_WARM = 30, 6        # warm start from the reference head
EPOCHS_SCRATCH, PATIENCE_SCRATCH = 40, 8  # scratch heads (encoder width != reference) converge later


@dataclass
class TrainConfig:
    epochs: int | None = None     # None: EPOCHS_WARM, or EPOCHS_SCRATCH when the head starts from scratch
    batch: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    patience: int | None = None   # None: PATIENCE_WARM / PATIENCE_SCRATCH, as for ``epochs``
    targets: Literal["soft", "hard"] = "soft"
    init: Literal["reference", "scratch"] = "reference"
    seed: int = 1234
    device: str = field(default_factory=_default_device)
    encoder: str | None = None   # base model spec (required by train(); a project's base_model)
    # head architecture for scratch heads (a warm start reuses the reference head's own arch)
    width: int = 1536
    depth: int = 3
    projection_dim: int = 512


@dataclass
class TrainResult:
    ckpt: Path
    best_epoch: int
    best_val_acc: float
    init_val_acc: float
    epochs_run: int
    stopped_early: bool
    cancelled: bool
    seconds: float
    n_train: int
    n_val: int
    n_test: int
    history: list[dict]


class _Throttle:
    def __init__(self, hz: float = 5.0):
        self.gap, self.last = 1.0 / hz, 0.0

    def ready(self, force: bool = False) -> bool:
        now = time.monotonic()
        if force or now - self.last >= self.gap:
            self.last = now
            return True
        return False


def throttled_embed_progress(emit: Emit) -> Callable[[int, int], None]:
    """``on_progress`` callback that emits ``embed`` events at ≤5/s (always the first and last)."""
    th = _Throttle()

    def cb(done: int, total: int) -> None:
        if th.ready(force=done in (0, total)):
            emit({"type": "embed", "done": done, "total": total})
    return cb


def _pca2(states: np.ndarray, options: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """2-D map of states + options with the modality gap removed.

    Projected states and options sit in two offset cones (the CLIP-style "modality gap"), so a plain
    PCA over both spends its first axis separating states from options and squashes the label
    structure. Instead each kind is centred on its own mean, the axes are fitted on the states (where
    the label clusters live), options are projected onto them and rescaled to the states' spread.
    Each axis is sign-flipped to correlate positively with ``prev`` so epochs animate without flips.
    """
    sc = states - states.mean(0, keepdims=True)
    oc = options - options.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(sc if len(sc) > 2 else np.vstack([sc, oc]), full_matrices=False)
    s_xy, o_xy = sc @ vt[:2].T, oc @ vt[:2].T
    s_r = float(np.linalg.norm(s_xy, axis=1).mean()) if len(s_xy) else 1.0
    o_r = float(np.linalg.norm(o_xy, axis=1).mean()) or 1.0
    xy = np.vstack([s_xy, o_xy * (1.25 * s_r / o_r)])
    if prev is not None and prev.shape == xy.shape:
        for a in range(2):
            if float(np.dot(xy[:, a], prev[:, a])) < 0:
                xy[:, a] = -xy[:, a]
    return xy


def train(task: TaskSchema, examples: list[Example], out_dir: Path, cfg: TrainConfig,
          emit: Emit, should_stop: Callable[[], bool] = lambda: False) -> TrainResult:
    """Train a head for ``task`` on ``examples`` (uses their ``split``); see module docstring."""
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str, level: str = "info") -> None:
        emit({"type": "log", "stage": STAGE, "level": level, "message": msg})

    # ---- rows -> texts -----------------------------------------------------------------------
    # A row is (example, state text, target over its options, gold option index, its option texts).
    # Every row of a fixed-option task shares the task's K options; a row with its own question
    # (``meta["question"]``: candidate tasks, projects, one account) brings its own option set.
    keys: list[str] = []
    cands: list[str] = []
    dynamic = any(row_question(ex) for ex in examples)
    rows: dict[str, list[tuple[Example, str, list[float], int, list[str]]]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        stext, rkeys, rcands = pairs_for(task, ex.state, row_question(ex))
        if ex.label not in rkeys:
            raise ValueError(f"example {ex.id}: label {ex.label!r} is not one of {rkeys}")
        tgt = target_for(ex, rkeys, soft=cfg.targets == "soft")
        rows[ex.split].append((ex, stext, tgt, rkeys.index(ex.label), rcands))
        if not dynamic:
            keys, cands = rkeys, rcands
    if not examples:
        raise ValueError("no examples to train on")
    if dynamic:
        try:
            _, keys, cands = pairs_for(task, "")     # the task's own options, when it has fixed criteria
        except ValueError:
            keys, cands = [], []
    if not rows["val"]:  # carve a deterministic 10% val split from train
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(rows["train"]))
        n_val = max(1, round(0.1 * len(perm)))
        rows["val"] = [rows["train"][i] for i in perm[:n_val]]
        rows["train"] = [rows["train"][i] for i in perm[n_val:]]
        log(f"no val split given; carved {n_val} rows from train")
    if not rows["train"]:
        raise ValueError("no training rows")
    if not dynamic:
        missing = set(range(len(keys))) - {r[3] for r in rows["train"]}
        if missing:
            log(f"labels never seen in train: {[keys[i] for i in missing]}", "warn")
    shape = "per-row options" if dynamic else f"{len(keys)} options"
    log(f"task {task.name} ({task.type}, {shape}) | train {len(rows['train'])} "
        f"val {len(rows['val'])} test {len(rows['test'])} | device {cfg.device}")

    # ---- embed (flat text table) ---------------------------------------------------------------
    # The table is streamed to a .npy memmap batch by batch and rows are gathered per step, so neither
    # the encoder nor head training ever holds a second full copy of it.
    texts = list(dict.fromkeys([*cands, *(c for s in rows.values() for r in s for c in r[4]),
                                *(r[1] for s in rows.values() for r in s)]))
    if not cfg.encoder:
        raise ValueError("TrainConfig.encoder: no base model given (a project's base_model, or a registry spec)")
    enc = get_encoder(cfg.encoder)
    t0 = time.perf_counter()
    log(f"embedding {len(texts)} unique texts with {enc.backend}:{enc.model_name}")
    table_path = out_dir / "embeddings.npy"
    E = enc.embed_to_file(texts, table_path, on_progress=throttled_embed_progress(emit),
                          on_status=lambda level, msg: log(msg, level), should_stop=should_stop)
    log(f"embedded in {time.perf_counter() - t0:.1f}s")
    try:
        return _fit(task, rows, keys, cands, texts, E, enc, out_dir, cfg, emit, should_stop, log, t_start, dynamic)
    finally:
        del E
        table_path.unlink(missing_ok=True)


def resolve_schedule(cfg: TrainConfig, scratch: bool) -> TrainConfig:
    """Fill unset ``epochs``/``patience``: scratch heads get the longer schedule (a Qwen3-4B head's
    best epoch was 27 of 30, so 30 cut it short); explicit values always win."""
    from dataclasses import replace
    return replace(cfg, epochs=cfg.epochs if cfg.epochs is not None else (EPOCHS_SCRATCH if scratch else EPOCHS_WARM),
                   patience=cfg.patience if cfg.patience is not None else (PATIENCE_SCRATCH if scratch else PATIENCE_WARM))


def _fit(task, rows, keys, cands, texts, E, enc, out_dir, cfg, emit, should_stop, log, t_start,
         dynamic: bool = False) -> TrainResult:
    """Head training on the memory-mapped embedding table ``E`` (``[len(texts), dim]``)."""
    import torch
    import torch.nn.functional as F

    if should_stop():
        raise InterruptedError("training cancelled before start")
    text_id = {t: i for i, t in enumerate(texts)}

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    hidden = int(E.shape[1])

    def emb(idx: torch.Tensor) -> torch.Tensor:
        """Rows ``idx`` (CPU long tensor) of the table, on ``device``."""
        return torch.from_numpy(np.ascontiguousarray(E[idx.numpy()])).to(device)

    opt_rows = torch.tensor([text_id[c] for c in cands], dtype=torch.long)

    def tensors(split: str):
        """-> (state ids, targets [n, Kmax], gold local index, option ids [n, Kmax] (-1 = pad), mask)."""
        rs = rows[split]
        kmax = max((len(r[4]) for r in rs), default=1)
        opt = torch.full((len(rs), kmax), -1, dtype=torch.long)
        tgt = torch.zeros((len(rs), kmax), dtype=torch.float32)
        for i, r in enumerate(rs):
            opt[i, :len(r[4])] = torch.tensor([text_id[c] for c in r[4]], dtype=torch.long)
            tgt[i, :len(r[2])] = torch.tensor(r[2], dtype=torch.float32)
        return (torch.tensor([text_id[r[1]] for r in rs], dtype=torch.long), tgt.to(device),
                torch.tensor([r[3] for r in rs], device=device, dtype=torch.long), opt, opt >= 0)

    # options as indices into the flat table: every row's own option set (for a fixed-option task,
    # the same K options on every row), padded with -1 to the widest row
    tr_st, tr_tgt, _, tr_opt, tr_mask = tensors("train")
    va_st, va_tgt, va_lab, va_opt, va_mask = tensors("val")

    # ---- heads ----------------------------------------------------------------------------------
    arch = dict(width=cfg.width, depth=cfg.depth, activation="gelu", layernorm=True, residual=False)
    proj = cfg.projection_dim
    if cfg.init == "reference" and hidden != HIDDEN:
        # the reference head was trained on 4096-d Qwen3-8B embeddings; don't even fetch it
        log(f"reference head expects {HIDDEN}-d embeddings, {enc.model_name} gives {hidden}-d; training from scratch")
    init_path = download() if cfg.init == "reference" and hidden == HIDDEN else None
    ck = torch.load(init_path, map_location="cpu", weights_only=True) if init_path else None
    if ck and ck["state_head"]["inp.weight"].shape[1] != hidden:
        log(f"reference head expects {ck['state_head']['inp.weight'].shape[1]}-d embeddings, "
            f"{enc.model_name} gives {hidden}-d; training from scratch")
        ck, init_path = None, None
    cfg = resolve_schedule(cfg, scratch=ck is None)
    log(f"schedule: {cfg.epochs} epochs, patience {cfg.patience} ({'warm start' if ck else 'from scratch'})")
    if ck:
        c0 = ck.get("cfg", {})
        arch.update({k: c0[k] for k in arch if k in c0})
        proj = c0.get("projection_dim", proj)
    sh = make_head(arch["width"], arch["depth"], proj, arch["activation"], arch["layernorm"], arch["residual"],
                   hidden=hidden).to(device)
    ah = make_head(arch["width"], arch["depth"], proj, arch["activation"], arch["layernorm"], arch["residual"],
                   hidden=hidden).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    if ck:
        sh.load_state_dict(ck["state_head"])
        ah.load_state_dict(ck["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float().to(device))
        log(f"warm start from {Path(init_path).name} {arch}")
    del ck

    def scale() -> torch.Tensor:
        return logit_scale.exp().clamp(max=100.0)

    def infonce_loss(idx: torch.Tensor) -> torch.Tensor:
        """Bidirectional in-batch InfoNCE over the batch's distinct option texts (run_choice).

        Forward (state -> options): a softmax over the row's OWN options only, exactly what inference
        computes (for a fixed-option task that is the whole pool, so nothing changes). Backward
        (option -> states): every pooled option against every state of the batch."""
        oi, tg, m = tr_opt[idx], tr_tgt[idx.to(device)], tr_mask[idx]
        pool, inv = torch.unique(oi[m], return_inverse=True)
        col = torch.zeros_like(oi)
        col[m] = inv
        col = col.to(device)
        md = m.to(device).float()
        tgt = torch.zeros(len(idx), len(pool), device=device).scatter_add_(1, col, tg * md)
        own = torch.zeros(len(idx), len(pool), device=device).scatter_add_(1, col, md) > 0
        zq = F.normalize(sh(emb(tr_st[idx])), dim=-1)
        zc = F.normalize(ah(emb(pool)), dim=-1)
        lg = scale() * zq @ zc.t()
        fwd = -(tgt * F.log_softmax(lg.masked_fill(~own, float("-inf")), 1).masked_fill(~own, 0.0)).sum(1).mean()
        w = tgt.t()
        keep = w.sum(1) > 0
        w = w[keep] / w[keep].sum(1, keepdim=True)
        bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
        return (fwd + bwd) / 2

    @torch.no_grad()
    def evaluate_val() -> dict:
        """Accuracy (argmax within each row's own options) and soft cross-entropy on val."""
        sh.eval(); ah.eval()
        pool, inv = torch.unique(va_opt[va_mask], return_inverse=True)
        col = torch.zeros_like(va_opt)
        col[va_mask] = inv
        zc = F.normalize(ah(emb(pool)), dim=-1)
        zq = F.normalize(sh(emb(va_st)), dim=-1)
        lg = scale() * torch.einsum("nd,nkd->nk", zq, zc[col.to(device)])
        lg = lg.masked_fill(~va_mask.to(device), float("-inf"))
        acc = (lg.argmax(-1) == va_lab).float().mean().item()
        ce = -(va_tgt * F.log_softmax(lg, -1).masked_fill(~va_mask.to(device), 0.0)).sum(-1).mean().item()
        sh.train(); ah.train()
        return {"acc": acc, "soft_ce": ce}

    @torch.no_grad()
    def full_train_loss() -> float:
        sh.eval(); ah.eval()
        idx = torch.arange(len(tr_st))
        tot = sum(infonce_loss(b).item() * len(b) for b in idx.split(cfg.batch))
        sh.train(); ah.train()
        return tot / len(tr_st)

    # ---- snapshot set (val first, then test; stratified-ish deterministic cap) -------------------
    snap = [(r, "val") for r in rows["val"]] + [(r, "test") for r in rows["test"]]
    cap = MAX_POINTS - (0 if dynamic else len(cands))
    if len(snap) > cap:
        pick = np.random.default_rng(cfg.seed).permutation(len(snap))[:cap]
        snap = [snap[i] for i in sorted(pick)]
    snap_idx = torch.tensor([text_id[r[1]] for r, _ in snap], dtype=torch.long)
    prev_xy: list[np.ndarray | None] = [None]

    @torch.no_grad()
    def snapshot(epoch: int) -> None:
        if dynamic:
            return _dynamic_snapshot(epoch)
        sh.eval(); ah.eval()
        zo = F.normalize(ah(emb(opt_rows)), dim=-1)
        zv = F.normalize(sh(emb(va_st)), dim=-1)
        cos = (zv @ zo.t()).cpu().numpy()
        labs = va_lab.cpu().numpy()
        row_labels, values = [], []
        for i, k in enumerate(keys):
            m = labs == i
            if m.any():
                row_labels.append(k)
                values.append([round(float(v), 4) for v in cos[m].mean(0)])
        zs = F.normalize(sh(emb(snap_idx)), dim=-1) if len(snap_idx) else zo[:0]
        xy = _pca2(zs.cpu().numpy(), zo.cpu().numpy(), prev_xy[0])
        prev_xy[0] = xy
        points = [{"x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "label": keys[r[3]],
                   "split": split, "kind": "state", "text": r[0].state[:160]} for i, (r, split) in enumerate(snap)]
        off = len(snap)
        points += [{"x": round(float(xy[off + j, 0]), 4), "y": round(float(xy[off + j, 1]), 4), "label": k,
                    "split": "option", "kind": "option", "text": cands[j][:160]} for j, k in enumerate(keys)]
        sh.train(); ah.train()
        emit({"type": "train_snapshot", "epoch": epoch,
              "similarity": {"row_labels": row_labels, "col_labels": keys, "values": values}, "points": points})

    @torch.no_grad()
    def _dynamic_snapshot(epoch: int) -> None:
        """Per-row option sets have no shared option vectors: states only, labelled by their gold key."""
        sh.eval()
        zs = F.normalize(sh(emb(snap_idx)), dim=-1).cpu().numpy() if len(snap_idx) else np.zeros((0, proj))
        xy = np.zeros((len(zs), 2))
        if len(zs) > 2:
            sc = zs - zs.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(sc, full_matrices=False)
            xy = sc @ vt[:2].T
            if prev_xy[0] is not None and prev_xy[0].shape == xy.shape:
                for a in range(xy.shape[1]):
                    if float(np.dot(xy[:, a], prev_xy[0][:, a])) < 0:
                        xy[:, a] = -xy[:, a]
        prev_xy[0] = xy
        points = [{"x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "label": r[0].label,
                   "split": split, "kind": "state", "text": r[0].state[:160]} for i, (r, split) in enumerate(snap)]
        sh.train()
        emit({"type": "train_snapshot", "epoch": epoch, "similarity": {"row_labels": [], "col_labels": [], "values": []},
              "points": points})

    # ---- optimise -------------------------------------------------------------------------------
    params = [*sh.parameters(), *ah.parameters(), logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    per_epoch = math.ceil(len(tr_st) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, per_epoch * cfg.epochs),
                                                pct_start=0.1, anneal_strategy="cos")

    def blob(epoch: int, metrics: dict) -> dict:
        return {"state_head": {k: v.detach().cpu().clone() for k, v in sh.state_dict().items()},
                "action_head": {k: v.detach().cpu().clone() for k, v in ah.state_dict().items()},
                "logit_scale": logit_scale.detach().cpu().clone(),
                "cfg": {**arch, "projection_dim": proj, "hidden_size": hidden, "task": "forge",
                        "question_type": task.type, "task_schema": task.model_dump(), "labels": keys,
                        "options": "dynamic" if dynamic else "fixed",
                        "targets": cfg.targets, "loss": "infonce", "train_config": asdict(cfg),
                        "embed_model": enc.spec, "max_len": enc.max_len,
                        "init_ckpt": Path(init_path).name if init_path else None},
                "epoch": epoch, "metrics": metrics}

    m0 = evaluate_val()
    init_loss = full_train_loss()
    history = [{"epoch": 0, "train_loss": init_loss, "val_acc": m0["acc"], "val_soft_ce": m0["soft_ce"],
                "lr": sched.get_last_lr()[0], "logit_scale": float(scale().detach())}]
    emit({"type": "train_epoch", "epochs": cfg.epochs, **history[0]})
    snapshot(0)
    log(f"init: val acc {m0['acc']:.4f} soft_ce {m0['soft_ce']:.4f}")

    best = (m0["acc"], -m0["soft_ce"])
    best_ep, bad, best_blob = 0, 0, blob(0, m0)
    g = torch.Generator().manual_seed(cfg.seed)
    th = _Throttle()
    cancelled = stopped_early = False
    ep = 0
    for ep in range(1, cfg.epochs + 1):
        tot = 0.0
        batches = torch.randperm(len(tr_st), generator=g).split(cfg.batch)
        for step, idx in enumerate(batches, 1):
            if should_stop():
                cancelled = True
                break
            loss = infonce_loss(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            tot += loss.item()
            if th.ready(force=step == len(batches)):
                emit({"type": "train_step", "epoch": ep, "step": step, "steps": len(batches), "loss": loss.item()})
        if cancelled:
            log(f"cancelled during epoch {ep}; keeping best epoch {best_ep}", "warn")
            break
        mv = evaluate_val()
        rec = {"epoch": ep, "train_loss": tot / len(batches), "val_acc": mv["acc"], "val_soft_ce": mv["soft_ce"],
               "lr": sched.get_last_lr()[0], "logit_scale": float(scale().detach())}
        history.append(rec)
        emit({"type": "train_epoch", "epochs": cfg.epochs, **rec})
        snapshot(ep)
        score = (mv["acc"], -mv["soft_ce"])
        if score[0] > best[0] + 1e-9 or (abs(score[0] - best[0]) <= 1e-9 and score[1] > best[1] + 1e-6):
            best, best_ep, bad, best_blob = score, ep, 0, blob(ep, mv)
        else:
            bad += 1
            if bad >= cfg.patience:
                stopped_early = True
                log(f"early stop at epoch {ep} (no val improvement for {cfg.patience} epochs)")
                break

    ckpt = out_dir / "best_head.pt"
    torch.save(best_blob, ckpt)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    secs = time.perf_counter() - t_start
    log(f"best epoch {best_ep}: val acc {best[0]:.4f} (init {m0['acc']:.4f}) | {secs:.1f}s total")
    return TrainResult(ckpt=ckpt, best_epoch=best_ep, best_val_acc=best[0], init_val_acc=m0["acc"],
                       epochs_run=len(history) - 1, stopped_early=stopped_early, cancelled=cancelled,
                       seconds=secs, n_train=len(rows["train"]), n_val=len(rows["val"]),
                       n_test=len(rows["test"]), history=history)

