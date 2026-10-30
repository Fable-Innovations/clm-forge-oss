"""The encoder registry: every frozen base model Forge trains heads on, one table (``backend/encoders.json``).

One owner for "which encoders exist and what they are". Each experiment (``base_model``, an encoder spec) and
each autopilot target (``HeadsetTarget.encoder``, a registry key) points into it; there is no app-wide default
encoder. A head-set adapter may read the same file for its own pin table, so the two can't drift. Which encoder a new experiment or target should start on is
``forge.encoder_choice`` (a recommendation the user can change).

* ``Encoder``: ``key``, display ``name``, ``backend`` (mlx | llamacpp), Hugging Face ``repo`` (+ ``file`` for a
  GGUF), pinned ``revision``, ``hidden`` width, ``quant``, ``approx_gb`` (measured peak, or None when never
  measured). ``spec`` is the ``forge.encoder`` spec grammar: ``mlx:<repo>`` or ``llamacpp:<repo>:<file>``.
* ``all()``, ``get(key)`` (raises ``UnknownEncoder``), ``by_spec(spec)`` (None for a spec no entry has).
* ``cached(enc)``: is the pinned snapshot (and, for a GGUF, its file) in the HF cache? A path check, no model load.
* ``output_root(base, key)``: where the autopilot writes an encoder's heads: ``<base>-<key>`` for every key, so
  a derived folder is never the base itself (for a preset the base is the reference heads folder, a protected root).
  The manual 4B pipeline's own rule (``train_heads.out_root``: the base itself for its default encoder) is not this.
* ``used_by(store, targets)``: per key, the projects whose ``base_model`` is it and the autopilot targets on it;
  ``in_use(...)`` the keys with at least one user.
* ``REFERENCE``: the encoder the CLM reference head was trained on (its 4096-d input fixes it); ``head_spec`` picks
  the encoder a head pairs with: the reference head's own, else the checkpoint's ``embed_model`` (every checkpoint
  records it; ``forge.encoder_migration`` backfilled the ones written before it did). Never a global default.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "encoders.json"
REFERENCE = "qwen3-8b-mlx-4bit"       # the CLM reference head's encoder


class UnknownEncoder(ValueError):
    """A key (or a spec) the registry has no entry for."""


class CheckpointEncoderMissing(ValueError):
    """A fine-tuned checkpoint that records no ``embed_model``: Forge cannot know which encoder it pairs with."""


class Encoder(BaseModel):
    key: str
    name: str
    backend: Literal["mlx", "llamacpp"]
    repo: str
    file: str | None = None
    revision: str
    hidden: int
    quant: str
    approx_gb: float | None = None
    approx_gb_source: str | None = None
    note: str = ""

    @property
    def spec(self) -> str:
        return f"mlx:{self.repo}" if self.backend == "mlx" else f"llamacpp:{self.repo}:{self.file}"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


@lru_cache(maxsize=1)
def _load(path: str) -> tuple[Encoder, ...]:
    raw = json.loads(Path(path).read_text())
    out = tuple(Encoder.model_validate(e) for e in raw["encoders"])
    keys = [e.key for e in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate encoder keys")
    for e in out:
        if e.backend == "llamacpp" and not e.file:
            raise ValueError(f"{path}: {e.key} is a GGUF encoder without a file")
    return out


def all() -> list[Encoder]:  # noqa: A001 - the registry's own vocabulary
    return list(_load(str(REGISTRY_PATH)))


def get(key: str) -> Encoder:
    for e in all():
        if e.key == key:
            return e
    raise UnknownEncoder(f"{key!r} is not a registered encoder (known: {', '.join(e.key for e in all())})")


def by_spec(spec: str | None) -> Encoder | None:
    return next((e for e in all() if e.spec == spec), None) if spec else None


def name_of_spec(spec: str | None) -> str | None:
    """A spec's display name: the registry name, else the spec's own tail (e.g. a ``hf:`` spec)."""
    if not spec:
        return None
    e = by_spec(spec)
    if e:
        return e.name
    tail = spec.split(":")[-1]
    return Path(tail).stem if tail.endswith(".gguf") else tail.split("/")[-1]


# ---- the HF cache --------------------------------------------------------------------------------------
def hub_dir() -> Path:
    """The Hugging Face hub cache (``HF_HUB_CACHE``, else ``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return home / "hub"


def snapshot_dir(enc: Encoder) -> Path:
    return hub_dir() / f"models--{enc.repo.replace('/', '--')}" / "snapshots" / enc.revision


def cached(enc: Encoder) -> bool:
    """The pinned snapshot is in the HF cache (for a GGUF: its file too). A path check; nothing is loaded."""
    snap = snapshot_dir(enc)
    if enc.file:
        return (snap / enc.file).exists()
    return snap.is_dir() and any(snap.iterdir())


# ---- placement and use ---------------------------------------------------------------------------------
def output_root(base: str, key: str) -> str:
    get(key)
    return f"{base}-{key}"


def used_by(store, targets) -> dict[str, dict[str, list[dict]]]:
    """key -> ``{"projects": [{id, name}], "targets": [{id, name}]}``. Projects count by their ``base_model``
    spec; head-set targets by ``headset.encoder``, project targets by their project's ``base_model``."""
    out: dict[str, dict[str, list[dict]]] = {e.key: {"projects": [], "targets": []} for e in all()}
    base: dict[str, str | None] = {}
    for rec in store.list_projects():
        base[rec["id"]] = rec.get("base_model")
        e = by_spec(rec.get("base_model"))
        if e:
            out[e.key]["projects"].append({"id": rec["id"], "name": rec.get("name") or rec["id"]})
    for t in targets.list():
        if t.headset is not None:
            key = t.headset.encoder
        else:
            e = by_spec(base.get(t.project.project_id)) if t.project else None
            key = e.key if e else None
        if key in out:
            out[key]["targets"].append({"id": t.id, "name": t.name})
    return out


def view(enc: Encoder, users: dict[str, list[dict]] | None = None) -> dict[str, Any]:
    return {**enc.model_dump(), "spec": enc.spec, "url": enc.url, "cached": cached(enc),
            "used_by": users or {"projects": [], "targets": []}}


def in_use(users: dict[str, dict[str, list[dict]]]) -> list[str]:
    """The keys of ``used_by``'s result that at least one experiment or target trains on (registry order)."""
    return [k for k, u in users.items() if u["projects"] or u["targets"]]


# ---- heads ---------------------------------------------------------------------------------------------
def head_spec(cfg: dict, *, reference: bool, path: str | None = None) -> str:
    """The encoder spec a CLM head pairs with: the reference head's own encoder (``REFERENCE``), else the
    checkpoint's ``embed_model``. A checkpoint without one raises ``CheckpointEncoderMissing`` (never a default)."""
    if reference:
        return get(REFERENCE).spec
    if cfg.get("embed_model"):
        return cfg["embed_model"]
    where = f"checkpoint {path}" if path else "the checkpoint"
    raise CheckpointEncoderMissing(f"{where} records no embed_model, so the encoder it was trained on is unknown; "
                                   "retrain it (or restart Forge so the encoder migration backfills it from its "
                                   "experiment)")


def known(key: str | None) -> bool:
    return bool(key) and any(e.key == key for e in all())
