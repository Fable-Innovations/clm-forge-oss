"""Build the deliverable: a pip-installable ``clm_<name>`` package plus the hand-off documents.

Layout (``<project_dir>/package/clm_<name>/``)::

    pyproject.toml  README.md  LICENSE  NOTICE
    src/clm_<name>/  __init__.py (exports classify|judge|score)  model.py  encoders.py  serve.py
                     _head.py (NumPy head, reads head.pt without torch)  _schema.py (vendored)
                     head.pt  task.json
    dist/clm_<name>-<version>-py3-none-any.whl
    EVAL_REPORT.md  eval_report.json  INTEGRATION.md  examples/golden.jsonl

plus ``<project_dir>/package/clm_<name>-<version>.zip`` of that directory.

Sources: ``forge/runtime_template/`` (``*.tmpl`` files are Jinja2-rendered, the rest copied) and
``forge/clm_core/schema.py`` (vendored verbatim so the package and Forge share one schema).
``head.pt`` is re-saved with only plain tensors/dicts, then checked against the torch heads
through the package's own NumPy loader. The default fallback threshold is chosen on the test
split: the smallest confidence at which accuracy on the rows the CLM keeps reaches
``max(0.95, success_threshold)`` with at least half the rows kept.

INTEGRATION.md is a deterministic template (``render_integration_md``); ``build_package`` can pass
it through an optional ``polish`` callable (the agent's LLM) and falls back to the template text
if polishing fails or drops the coding-agent prompt.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jinja2 import Environment, StrictUndefined

from .clm_core.schema import confidence as s1_confidence
from .inference import Predictor, pairs_for
from .models import EvalReport, Example, ModelScore, Opportunity, PackageFile, PackageManifest, TaskSchema

Emit = Callable[[dict], None]
STAGE = "packaging"
TEMPLATE_DIR = Path(__file__).parent / "runtime_template"
SCHEMA_SRC = Path(__file__).parent / "clm_core" / "schema.py"
API_NAME = {"choice": "classify", "noul": "judge", "score": "score"}
_LANG = {".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".go": "go", ".rb": "ruby",
         ".java": "java", ".kt": "kotlin", ".rs": "rust", ".cs": "csharp", ".php": "php"}
_env = Environment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False)


def package_name(task: TaskSchema) -> str:
    """``clm_<slug>`` — a valid Python identifier."""
    slug = re.sub(r"[^0-9a-z]+", "_", task.name.lower()).strip("_") or "task"
    return f"clm_{slug}"


def _doc(text: str) -> str:
    """Make arbitrary text safe inside a triple-quoted Python docstring."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _render(name: str, **ctx) -> str:
    return _env.from_string((TEMPLATE_DIR / name).read_text()).render(**ctx)


def _options(task: TaskSchema) -> list[tuple[str, str]]:
    _, keys, cands = pairs_for(task, "")
    return list(zip(keys, cands))


def threshold_table(probs: np.ndarray, gold_idx: list[int]) -> list[dict]:
    """Coverage / accuracy-on-covered for confidence thresholds 0.0..0.9 (System One confidence)."""
    conf = np.array([s1_confidence(p.tolist()) for p in probs])
    ok = probs.argmax(1) == np.array(gold_idx)
    rows = []
    for t in np.round(np.arange(0.0, 0.91, 0.1), 2):
        m = conf >= t
        rows.append({"threshold": float(t), "coverage": float(m.mean()) if len(m) else 0.0,
                     "accuracy": float(ok[m].mean()) if m.any() else 0.0})
    return rows


MIN_FALLBACK_THRESHOLD = 0.1  # a few dozen test rows can't certify "never fall back", so never recommend 0


def recommend_threshold(table: list[dict], target: float) -> float:
    table = [r for r in table if r["threshold"] >= MIN_FALLBACK_THRESHOLD] or table
    ok = [r for r in table if r["coverage"] >= 0.5 and r["accuracy"] >= target]
    if ok:
        return ok[0]["threshold"]
    viable = [r for r in table if r["coverage"] >= 0.5] or table
    return max(viable, key=lambda r: (r["accuracy"], -r["threshold"]))["threshold"]


def render_integration_md(task: TaskSchema, opportunity: Opportunity, report: EvalReport, repo_summary: str,
                          package_name: str, *, fallback_threshold: float = 0.5,
                          table: list[dict] | None = None, wheel_name: str | None = None,
                          encoder: dict) -> str:
    """Deterministic INTEGRATION.md for a coding agent (see the template for the sections).
    ``encoder`` is the packaged ``task.json`` -> ``encoder`` block (``encoder_facts``); every encoder
    fact in the document comes from it."""
    files = [{**f.model_dump(), "lang": _LANG.get(Path(f.path).suffix, "")} for f in opportunity.files]
    python = not files or any(f["path"].endswith(".py") for f in files)
    ft = next((m for m in report.models if m.name == "finetuned"),
              ModelScore(name="finetuned", accuracy=0.0, macro_f1=0.0, n=0))
    dist = package_name.replace("_", "-")
    return _render("INTEGRATION.md.tmpl", pkg=package_name, dist=dist, api=API_NAME[task.type], task=task,
                   opp=opportunity, report=report, files=files, python=python, ft=ft,
                   labels=task.labels(), options=_options(task), threshold=fallback_threshold,
                   table=table or [], flag=package_name.upper(),
                   wheel_name=wheel_name or f"{package_name}-<version>-py3-none-any.whl",
                   repo_summary=repo_summary.strip() or "(no repository summary)",
                   encoder_backend=encoder["backend"], encoder_model=encoder["model"],
                   base_model_name=encoder["base_model"], gguf_file=encoder.get("gguf", ""),
                   max_len=int(encoder["max_len"]), encoder_dim=int(encoder["dim"]),
                   weights_gb=encoder.get("weights_bytes", 0) / 1e9,
                   cache_limit_mb=int(encoder["mlx_cache_limit_mb"]))


def render_eval_md(task: TaskSchema, report: EvalReport, package_name: str) -> str:
    return _render("EVAL_REPORT.md.tmpl", pkg=package_name, task=task, report=report)


def _find_ckpt(project_dir: Path) -> Path:
    found = [p for p in project_dir.rglob("best_head.pt") if "package" not in p.relative_to(project_dir).parts]
    if not found:
        raise FileNotFoundError(f"no best_head.pt under {project_dir}")
    return max(found, key=lambda p: p.stat().st_mtime)


def _load_template_module(name: str):
    spec = importlib.util.spec_from_file_location(f"_forge_pkg_{name}", TEMPLATE_DIR / "pkg" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_template_head_module():
    return _load_template_module("_head")


def _write_head(ckpt: Path, dest: Path) -> None:
    """Re-save the checkpoint with plain tensors only and verify the NumPy loader reproduces it."""
    import torch
    from .clm_core.heads import HeadPair
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    clean = {"state_head": {k: v.contiguous() for k, v in ck["state_head"].items()},
             "action_head": {k: v.contiguous() for k, v in ck["action_head"].items()},
             "logit_scale": torch.as_tensor(ck["logit_scale"]).float().reshape(()),
             "cfg": json.loads(json.dumps(ck["cfg"], default=str)),
             "epoch": int(ck.get("epoch", 0)), "metrics": json.loads(json.dumps(ck.get("metrics", {}), default=float))}
    torch.save(clean, dest)
    hidden = int(ck["state_head"]["inp.weight"].shape[1])
    x = np.random.default_rng(0).standard_normal((3, hidden)).astype(np.float32)
    ref = HeadPair("check", str(dest), "cpu").ensure()
    want_s, want_a = ref.project(x, x)
    got = _load_template_head_module().HeadPair(dest)
    err = max(np.abs(got.project_states(x) - want_s).max(), np.abs(got.project_actions(x) - want_a).max())
    if err > 1e-4 or abs(got.scale - ref.scale) > 1e-5:
        raise RuntimeError(f"packaged NumPy head disagrees with torch (max abs err {err:.2e})")


def head_dim(head_pt: Path) -> int:
    """Input width of the packaged head = the encoder embedding width it was trained on."""
    return int(_load_template_head_module().HeadPair(head_pt).in_dim)


def _mlx_snapshot(repo: str) -> tuple[str, int]:
    """(commit sha, total bytes) of the locally cached Hugging Face snapshot the head was trained on."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo, local_files_only=True))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())   # stat() follows the blob symlinks
    return path.name, size


def encoder_facts(enc, dim: int) -> dict:
    """``task.json`` -> ``encoder``: everything the package (and its docs) must know about the training
    encoder, taken from the encoder itself rather than assumed: model id, pinned revision, embedding
    width, weights size, recipe."""
    from .base_models import display_name
    name = display_name(enc.spec)
    recipe = ("no special tokens, tail max_len-1 tokens, final-norm hidden state of the last token, "
              "float32, L2-normalised")
    if enc.backend == "llamacpp":
        gguf = Path(enc.model_name)
        meta = {"backend": "llamacpp", "model": enc.model_name, "gguf": gguf.name,
                "recipe": f"{name} via llama.cpp --pooling last, {recipe}"}
        if gguf.is_file():
            meta["weights_bytes"] = gguf.stat().st_size
    elif enc.backend == "mlx":
        revision, size = _mlx_snapshot(enc.model_name)
        meta = {"backend": "mlx", "model": enc.model_name, "revision": revision, "weights_bytes": size,
                "recipe": f"{name}, {recipe}"}
    else:
        raise RuntimeError(f"cannot package a head trained on encoder backend {enc.backend!r} ({enc.spec})")
    meta.update({"trained_with": enc.spec, "base_model": name, "max_len": enc.max_len, "dim": dim,
                 "mlx_cache_limit_mb": _load_template_module("encoders").DEFAULT_MLX_CACHE_LIMIT_MB})
    return meta


_DESCRIPTIONS = {
    "pyproject.toml": "package metadata (hatchling)", "README.md": "package usage, encoder setup, metrics",
    "LICENSE": "Apache-2.0 license", "NOTICE": "attribution for vendored CLM code",
    "INTEGRATION.md": "integration plan for a coding agent", "EVAL_REPORT.md": "evaluation report",
    "eval_report.json": "evaluation report (machine-readable)", "examples/golden.jsonl": "test-split golden examples",
    "head.pt": "fine-tuned CLM head checkpoint", "options.npy": "option-text embeddings used in training",
    "task.json": "task definition, labels, threshold, encoder recipe",
    "__init__.py": "public API", "model.py": "model wrapper: predict / answer / fallback",
    "encoders.py": "encoder backends (mlx, openai-compatible, llama.cpp)", "serve.py": "HTTP server (System One compatible)",
    "_head.py": "NumPy projection heads (vendored from CLM)", "_schema.py": "System One schema (vendored from CLM)",
}


def _describe(rel: str) -> str:
    if rel.endswith(".whl"):
        return "built wheel"
    return _DESCRIPTIONS.get(rel, _DESCRIPTIONS.get(Path(rel).name, ""))


def build_package(project_dir: Path, task: TaskSchema, proposal_opportunity: Opportunity, report: EvalReport,
                  repo_summary: str, emit: Emit, *, ckpt: Path | None = None, examples: list[Example] | None = None,
                  polish: Callable[[str], str] | None = None, version: str = "0.1.0") -> PackageManifest:
    """Assemble, build and zip the package. ``ckpt`` defaults to the newest ``best_head.pt`` under
    ``project_dir``; ``examples`` default to ``project_dir/dataset.jsonl``."""
    project_dir = Path(project_dir)

    def log(msg: str, level: str = "info") -> None:
        emit({"type": "log", "stage": STAGE, "level": level, "message": msg})

    pkg = package_name(task)
    dist_name = pkg.replace("_", "-")
    api = API_NAME[task.type]
    ckpt = Path(ckpt) if ckpt else _find_ckpt(project_dir)
    if examples is None:
        examples = [Example.model_validate_json(line) for line in
                    (project_dir / "dataset.jsonl").read_text().splitlines() if line.strip()]
    test = [e for e in examples if e.split == "test"] or [e for e in examples if e.split == "val"]

    root = project_dir / "package"
    out = root / pkg
    shutil.rmtree(out, ignore_errors=True)
    src = out / "src" / pkg
    src.mkdir(parents=True)
    log(f"building {pkg} {version} from {ckpt}")

    # -- head + golden predictions + threshold -------------------------------------------------------
    _write_head(ckpt, src / "head.pt")
    pred = Predictor(ckpt, task)
    probs = pred.probs([e.state for e in test]) if test else np.zeros((0, len(pred.keys)))
    gold_idx = [pred.keys.index(e.label) for e in test]
    table = threshold_table(probs, gold_idx) if test else []
    target = max(0.95, proposal_opportunity.success_threshold)
    tau = recommend_threshold(table, target) if table else 0.5
    (out / "examples").mkdir()
    with open(out / "examples" / "golden.jsonl", "w") as f:
        for e, p in zip(test, probs):
            f.write(json.dumps({"id": e.id, "state": e.state, "label": e.label,
                                "clm_label": pred.keys[int(p.argmax())], "clm_confidence": s1_confidence(p.tolist()),
                                "probabilities": {k: float(v) for k, v in zip(pred.keys, p)}}) + "\n")

    # -- package sources ------------------------------------------------------------------------------
    enc = pred.encoder                        # the base model the head was trained on
    enc_meta = encoder_facts(enc, head_dim(src / "head.pt"))
    options = _options(task)
    meta = {"task": task.model_dump(), "question": task.question(), "labels": task.labels(),
            "options": [{"key": k, "text": t} for k, t in options], "fallback_threshold": tau,
            "threshold_table": table, "version": version,
            "encoder": enc_meta,
            "metrics": {m.name: {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "n": m.n} for m in report.models},
            "built_at": datetime.now(timezone.utc).isoformat()}
    (src / "task.json").write_text(json.dumps(meta, indent=1))
    np.save(src / "options.npy", enc.embed([t for _, t in options]))   # cache hit: the training vectors
    for py in (TEMPLATE_DIR / "pkg").glob("*.py"):
        shutil.copy(py, src / py.name)
    shutil.copy(SCHEMA_SRC, src / "_schema.py")
    (src / "py.typed").write_text("")
    for name in ("LICENSE", "NOTICE"):
        shutil.copy(TEMPLATE_DIR / name, out / name)

    wheel_name = f"{pkg}-{version}-py3-none-any.whl"
    ft = next(m for m in report.models if m.name == "finetuned")
    ctx = dict(pkg=pkg, dist=dist_name, api=api, version=version, qtype=task.type, question_id=task.question_id,
               instructions=task.instructions, labels=task.labels(), instructions_doc=_doc(task.instructions),
               labels_doc=_doc(", ".join(task.labels())), options=options, threshold=tau,
               state_description=task.state_description, state_example=task.state_example,
               encoder_model=enc_meta["model"], encoder_backend=enc_meta["backend"],
               base_model_name=enc_meta["base_model"], gguf_file=enc_meta.get("gguf", ""), max_len=enc.max_len,
               encoder_dim=enc_meta["dim"], weights_gb=enc_meta.get("weights_bytes", 0) / 1e9,
               cache_limit_mb=enc_meta["mlx_cache_limit_mb"], wheel_name=wheel_name, models=report.models,
               latency=report.latency, n_test=report.n_test, ft=ft)
    (src / "__init__.py").write_text(_render("pkg/__init__.py.tmpl", **ctx))
    (out / "pyproject.toml").write_text(_render("pyproject.toml.tmpl", **ctx))
    (out / "README.md").write_text(_render("README.md.tmpl", **ctx))

    # -- reports + integration plan --------------------------------------------------------------------
    (out / "eval_report.json").write_text(report.model_dump_json(indent=1))
    (out / "EVAL_REPORT.md").write_text(render_eval_md(task, report, pkg))
    md = render_integration_md(task, proposal_opportunity, report, repo_summary, pkg,
                               fallback_threshold=tau, table=table, wheel_name=wheel_name, encoder=enc_meta)
    if polish is not None:
        try:
            polished = polish(md)
            if polished and "Prompt for your coding agent" in polished:
                md = polished
            else:
                log("polished INTEGRATION.md dropped required sections; keeping the template", "warn")
        except Exception as e:  # polishing is best-effort
            log(f"INTEGRATION.md polish failed ({e}); keeping the template", "warn")
    (out / "INTEGRATION.md").write_text(md)

    # -- wheel + zip ---------------------------------------------------------------------------------
    log("building wheel")
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out / "dist"),
                        str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"wheel build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    wheels = list((out / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {[w.name for w in wheels]}")
    for stale in root.glob(f"{pkg}-*.zip"):
        stale.unlink()
    zip_path = Path(shutil.make_archive(str(root / f"{pkg}-{version}"), "zip", root_dir=root, base_dir=pkg))

    files = [PackageFile(path=str(p.relative_to(out)), bytes=p.stat().st_size, description=_describe(str(p.relative_to(out))))
             for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    manifest = PackageManifest(name=pkg, version=version, dir=str(out), zip=str(zip_path), files=files,
                               integration_md="INTEGRATION.md", created_at=datetime.now(timezone.utc).isoformat())
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=1))
    log(f"package ready: {wheels[0].name} ({wheels[0].stat().st_size / 1e6:.1f} MB), fallback threshold {tau:.2f}")
    emit({"type": "package", "manifest": manifest.model_dump()})
    return manifest
