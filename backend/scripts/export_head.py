"""Export a trained CLM head to a flat, torch-free format for non-Python runtimes (e.g. a TS/Node app).

    cd backend && uv run python scripts/export_head.py <project_dir | package src dir | head.pt> <out_dir> [--verify]

Input resolution (first match wins):
    * a ``.pt`` file: the checkpoint; ``task.json`` / ``options.npy`` are taken from the same directory
      when present;
    * a directory with ``head.pt`` (a built package's ``src/<pkg>/``): that checkpoint + its task.json/options.npy;
    * a Forge project directory: the newest built package ``package/*/src/*/head.pt`` if there is one,
      else the project's ``best_head.pt`` + ``task.json`` (no ``options.npy`` -> no ``options.bin``).

Outputs in ``out_dir``:
    head.bin     every tensor of both heads, float32 little-endian, concatenated in ``head.json`` order
    head.json    format, layer list (name, shape, byte offset), activation, LayerNorm eps, residual,
                 logit_scale (raw, log space) and scale (= min(exp(logit_scale), 100)), hidden_size,
                 projection_dim, encoder spec + revision, task + option keys/texts, file manifest
    options.bin  option-text encoder embeddings ``[K, hidden_size]`` float32 LE (row order = options)
    options_proj.bin  the same options already through the action head and L2-normalised ``[K, proj]``,
                 so a runtime that only scores states can skip the action head

Forward (per head; weights are stored PyTorch-style ``[out, in]``, so ``y = x @ W.T + b``)::

    x = act(inp(x)); for each hidden block i: h = act(norm_i(hidden_i(x))); x = x + h if residual else h
    z = l2(out(x));  logits = scale * z_state @ z_option.T;  probs = softmax(logits)

``norm_i`` is LayerNorm (eps ``layernorm_eps``) when ``layernorm`` is true, else identity; ``act`` is
exact (erf) GELU for ``gelu``. Reading ``head.pt`` uses the runtime template's restricted unpickler
(``forge/runtime_template/pkg/_head.py``): no torch import, no arbitrary globals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FORMAT = "clm-head-v1"
LAYERNORM_EPS = 1e-5
SCALE_MAX = 100.0
HEAD_PY = Path(__file__).resolve().parents[1] / "forge" / "runtime_template" / "pkg" / "_head.py"


def _head_module():
    spec = importlib.util.spec_from_file_location("_clm_export_head", HEAD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def resolve(src: Path) -> tuple[Path, Path | None, Path | None]:
    """-> (checkpoint, task.json or None, options.npy or None)."""
    src = Path(src)

    def sib(d: Path, name: str) -> Path | None:
        return d / name if (d / name).exists() else None

    if src.is_file():
        return src, sib(src.parent, "task.json"), sib(src.parent, "options.npy")
    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist")
    if (src / "head.pt").exists():
        return src / "head.pt", sib(src, "task.json"), sib(src, "options.npy")
    pkgs = sorted((src / "package").glob("*/src/*/head.pt"), key=lambda p: p.stat().st_mtime) if (src / "package").is_dir() else []
    if pkgs:
        d = pkgs[-1].parent
        return pkgs[-1], sib(d, "task.json"), sib(d, "options.npy")
    if (src / "best_head.pt").exists():
        return src / "best_head.pt", sib(src, "task.json"), None
    raise FileNotFoundError(f"no head.pt / best_head.pt under {src}")


def _task_meta(task_json: Path | None, cfg: dict) -> tuple[dict, list[dict], dict]:
    """-> (task, options [{key, text}], encoder) from a package task.json, a project task.json, or the ckpt cfg."""
    meta = json.loads(task_json.read_text()) if task_json else {}
    task = meta.get("task") if isinstance(meta.get("task"), dict) else (meta if "question_id" in meta else cfg.get("task_schema") or {})
    options = meta.get("options")
    if not options:
        crit = (task or {}).get("criteria") or {}
        labels = cfg.get("labels") or list(crit)
        options = [{"key": k, "text": crit.get(k, k)} for k in labels]
    enc = dict(meta.get("encoder") or {})
    enc.setdefault("spec", cfg.get("embed_model"))
    enc.setdefault("max_len", cfg.get("max_len"))
    return task, [{"key": o["key"], "text": o.get("text", "")} for o in options], enc


def _layers(prefix: str, sd: dict, depth: int, layernorm: bool) -> list[tuple[str, np.ndarray]]:
    names = ["inp.weight", "inp.bias"]
    for i in range(depth - 2):
        names += [f"hidden.{i}.weight", f"hidden.{i}.bias"]
        if layernorm:
            names += [f"norms.{i}.weight", f"norms.{i}.bias"]
    names += ["out.weight", "out.bias"]
    return [(f"{prefix}.{n}", np.ascontiguousarray(sd[n], dtype="<f4")) for n in names]


def export(src: Path, out_dir: Path) -> dict:
    """Write head.bin/head.json (+ options.bin/options_proj.bin when option embeddings exist) -> head.json dict."""
    hm = _head_module()
    ckpt, task_json, options_npy = resolve(src)
    ck = hm.load_checkpoint(ckpt)
    cfg = dict(ck["cfg"])
    depth, layernorm = int(cfg["depth"]), bool(cfg.get("layernorm", False))
    hidden = int(np.asarray(ck["state_head"]["inp.weight"]).shape[1])
    proj = int(np.asarray(ck["state_head"]["out.weight"]).shape[0])
    raw_scale = float(np.asarray(ck["logit_scale"]))
    task, options, enc = _task_meta(task_json, cfg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers, off = [], 0
    with open(out_dir / "head.bin", "wb") as fh:
        for head in ("state_head", "action_head"):
            for name, arr in _layers(head.split("_")[0], ck[head], depth, layernorm):
                fh.write(arr.tobytes())
                layers.append({"name": name, "shape": list(arr.shape), "offset": off, "count": int(arr.size)})
                off += arr.nbytes
    files = {"head.bin": _sha256(out_dir / "head.bin")}

    pair = hm.HeadPair(ckpt)
    if options_npy is not None:
        oe = np.load(options_npy).astype("<f4")
        if oe.shape != (len(options), hidden):
            raise ValueError(f"options.npy is {oe.shape}, expected ({len(options)}, {hidden})")
        (out_dir / "options.bin").write_bytes(np.ascontiguousarray(oe).tobytes())
        (out_dir / "options_proj.bin").write_bytes(
            np.ascontiguousarray(pair.project_actions(oe), dtype="<f4").tobytes())
        files["options.bin"] = _sha256(out_dir / "options.bin")
        files["options_proj.bin"] = _sha256(out_dir / "options_proj.bin")

    meta = {
        "format": FORMAT, "source": ckpt.name, "dtype": "float32", "byte_order": "little",
        "weight_layout": "[out, in]; y = x @ W.T + b",
        "hidden_size": hidden, "projection_dim": proj, "width": int(cfg["width"]), "depth": depth,
        "activation": cfg.get("activation", "gelu"), "gelu": "exact (erf)", "layernorm": layernorm,
        "layernorm_eps": LAYERNORM_EPS, "residual": bool(cfg.get("residual", False)),
        "logit_scale": raw_scale, "scale_max": SCALE_MAX, "scale": float(min(np.exp(raw_scale), SCALE_MAX)),
        "layers": layers,
        "encoder": {**enc, "pooling": "last token, final norm, L2-normalised"},
        "task": {k: (task or {}).get(k) for k in ("name", "question_id", "type", "instructions")},
        "options": options,
        "options_bin": {"shape": [len(options), hidden], "kind": "encoder embedding"} if options_npy else None,
        "options_proj_bin": {"shape": [len(options), proj], "kind": "action head, L2-normalised"} if options_npy else None,
        "files": files,
    }
    (out_dir / "head.json").write_text(json.dumps(meta, indent=1))
    return meta


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- independent NumPy forward over the exported files (used by --verify and the tests) ----------------

def _erf(x: np.ndarray) -> np.ndarray:
    from math import erf
    return np.vectorize(erf, otypes=[np.float64])(x)


def load_exported(out_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "head.json").read_text())
    buf = (out_dir / "head.bin").read_bytes()
    t = {l["name"]: np.frombuffer(buf, dtype="<f4", count=l["count"], offset=l["offset"]).reshape(l["shape"])
         for l in meta["layers"]}
    return meta, t


def forward(meta: dict, t: dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    """L2-normalised projection of ``x`` ``[n, hidden]`` through head ``prefix`` (``state``/``action``)."""
    acts = {"gelu": lambda v: 0.5 * v * (1.0 + _erf(v / np.sqrt(2.0))), "relu": lambda v: np.maximum(v, 0),
            "silu": lambda v: v / (1.0 + np.exp(-v))}
    act = acts[meta["activation"]]
    x = np.asarray(x, dtype=np.float64)
    x = act(x @ t[f"{prefix}.inp.weight"].T + t[f"{prefix}.inp.bias"])
    for i in range(meta["depth"] - 2):
        h = x @ t[f"{prefix}.hidden.{i}.weight"].T + t[f"{prefix}.hidden.{i}.bias"]
        if meta["layernorm"]:
            mu = h.mean(-1, keepdims=True)
            var = ((h - mu) ** 2).mean(-1, keepdims=True)
            h = (h - mu) / np.sqrt(var + meta["layernorm_eps"]) * t[f"{prefix}.norms.{i}.weight"] + t[f"{prefix}.norms.{i}.bias"]
        h = act(h)
        x = x + h if meta["residual"] else h
    z = x @ t[f"{prefix}.out.weight"].T + t[f"{prefix}.out.bias"]
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def verify(src: Path, out_dir: Path, n: int = 20, seed: int = 0) -> float:
    """Max abs difference between the exported forward and ``_head.py`` on ``n`` random vectors."""
    hm = _head_module()
    pair = hm.HeadPair(resolve(src)[0])
    meta, t = load_exported(out_dir)
    x = np.random.default_rng(seed).standard_normal((n, meta["hidden_size"])).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)           # encoder embeddings are L2-normalised
    err = max(np.abs(forward(meta, t, "state", x) - pair.project_states(x)).max(),
              np.abs(forward(meta, t, "action", x) - pair.project_actions(x)).max())
    if abs(meta["scale"] - pair.scale) > 1e-6:
        err = max(err, abs(meta["scale"] - pair.scale))
    if (Path(out_dir) / "options_proj.bin").exists():
        k, p = meta["options_proj_bin"]["shape"]
        zo = np.frombuffer((Path(out_dir) / "options_proj.bin").read_bytes(), dtype="<f4").reshape(k, p)
        oe = np.frombuffer((Path(out_dir) / "options.bin").read_bytes(), dtype="<f4").reshape(k, meta["hidden_size"])
        err = max(err, float(np.abs(forward(meta, t, "action", oe) - zo).max()))
    return float(err)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="project dir, package src dir, or head.pt")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--verify", action="store_true", help="compare the exported forward with _head.py (20 vectors)")
    a = ap.parse_args(argv)
    meta = export(a.src, a.out_dir)
    print(f"exported {meta['source']} -> {a.out_dir}: hidden {meta['hidden_size']}, proj {meta['projection_dim']}, "
          f"{len(meta['layers'])} tensors, {len(meta['options'])} options, files {sorted(meta['files'])}")
    if a.verify:
        err = verify(a.src, a.out_dir)
        print(f"verify: max abs diff vs _head.py = {err:.2e}")
        if err > 1e-5:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
