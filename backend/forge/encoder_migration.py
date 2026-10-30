"""One-time migration to per-experiment encoders: Forge no longer has an app-wide encoder.

Before, ``settings.json`` held one encoder (``encoder`` spec, later ``default_encoder`` key, plus the ``base_models``
candidates), and an experiment created while base-model selection still had to choose stored no ``base_model``.
``migrate(store, emit)`` runs once per database (``Store.migrate_once("experiment_encoders")``, from the server's
start-up, before the experiments are restored):

1. **Experiments without an encoder** get the one their artifacts were built with, recorded in
   ``encoder_source = {from, detail, at, migration}`` and as a ``log`` event (stage ``migration``) in their stream:

   * ``checkpoint``: the ``embed_model`` of the experiment's own ``best_head.pt`` (what its head was trained on);
   * ``settings``: else the app-wide encoder the settings named at migration time (``config.legacy_settings``: the
     ``default_encoder`` key, else the older ``encoder`` spec), which is what curation embedded its dataset with.

   With neither, the experiment keeps no encoder and its stream says so (``warn``): it must be picked in its header
   before it trains (``Orchestrator.encoder_spec`` raises ``NoEncoder``); nothing is guessed.
2. **Checkpoints without ``embed_model``** (written before the trainer recorded it): every CLM checkpoint under an
   experiment's folder that lacks it gets the experiment's encoder written into its ``cfg`` once (a probe of
   base-model selection, ``selection/<name>/``, gets the encoder that name is, or is left and listed when no
   registry entry has that name). Serving, inference and System One then always read the checkpoint's own.
3. The stale ``encoder_pinned`` flag is dropped from every experiment record (every experiment owns its encoder).

-> ``{"projects": {pid: {from, spec} | {"error"}}, "checkpoints": {path: spec}, "skipped": {path: why}}``, logged.
The app-wide fields themselves leave ``settings.json`` in ``config.load_settings`` (the original file is kept as
``settings.json.bak-pre-experiment-encoders``).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from forge import config, encoders
from forge.store import Store, now

log = logging.getLogger("forge.encoder_migration")
NAME = "experiment_encoders"
DECISION = "per-experiment encoders"


def legacy_spec(raw: dict) -> tuple[str | None, str]:
    """The app-wide encoder spec stored settings named (``default_encoder`` key, else the ``encoder`` spec), and
    where it came from; (None, why) when they named none or an unregistered key."""
    key = raw.get("default_encoder")
    if key:
        if encoders.known(key):
            return encoders.get(key).spec, f"settings default_encoder {key}"
        return None, f"settings default_encoder {key!r} is not a registered encoder"
    spec = raw.get("encoder")
    if spec:
        return spec, f"settings encoder {spec}"
    return None, "the settings named no encoder"


def _cfg(path: Path) -> dict | None:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    return cfg if isinstance(cfg, dict) else None


def _backfill(path: Path, spec: str) -> None:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck["cfg"] = {**ck["cfg"], "embed_model": spec}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)


def _probe_spec(pdir: Path, ckpt: Path) -> str | None:
    """A base-model selection probe's encoder: the registry entry named like its ``selection/<name>`` folder."""
    rel = ckpt.relative_to(pdir).parts
    if len(rel) < 2 or rel[0] != "selection":
        return None
    return next((e.spec for e in encoders.all() if e.name == rel[1]), None)


def migrate_projects(store: Store, emit: Callable[[str, dict], None], legacy: dict) -> dict:
    out: dict = {"projects": {}, "checkpoints": {}, "skipped": {}}
    spec_from_settings, settings_detail = legacy_spec(legacy)
    for rec in store.list_projects():
        pid = rec["id"]
        pdir = config.project_dir(pid)
        changed = rec.pop("encoder_pinned", None) is not None
        if not rec.get("base_model"):
            best = pdir / "best_head.pt"
            cfg = _cfg(best) if best.is_file() else None
            if cfg and cfg.get("embed_model"):
                spec, source, detail = cfg["embed_model"], "checkpoint", f"embed_model of {best.name}"
            elif spec_from_settings:
                spec, source, detail = spec_from_settings, "settings", settings_detail
            else:
                spec, source, detail = None, None, settings_detail
            if spec:
                rec["base_model"] = spec
                rec["encoder_source"] = {"from": source, "detail": detail, "at": now(), "migration": DECISION}
                changed = True
                out["projects"][pid] = {"from": source, "spec": spec}
                emit(pid, {"type": "log", "stage": "migration", "level": "info",
                           "message": f"encoder set to {encoders.name_of_spec(spec)} from the {source} ({detail}): "
                                      f"{DECISION}"})
            else:
                out["projects"][pid] = {"error": detail}
                emit(pid, {"type": "log", "stage": "migration", "level": "warn",
                           "message": f"no encoder could be determined ({detail}; no checkpoint records one): pick "
                                      f"this experiment's encoder before it trains. {DECISION}"})
        if changed:
            store.save_project(rec)
        own = rec.get("base_model")
        for ckpt in sorted(pdir.rglob("*.pt")) if pdir.is_dir() else []:
            cfg = _cfg(ckpt)
            if cfg is None or cfg.get("embed_model"):
                continue
            in_selection = ckpt.relative_to(pdir).parts[0] == "selection"
            spec = _probe_spec(pdir, ckpt) if in_selection else own
            if not spec:
                out["skipped"][str(ckpt)] = ("a selection probe whose folder names no registered encoder"
                                             if in_selection else "its experiment has no encoder")
                continue
            _backfill(ckpt, spec)
            out["checkpoints"][str(ckpt)] = spec
            emit(pid, {"type": "log", "stage": "migration", "level": "info",
                       "message": f"checkpoint {ckpt.relative_to(pdir)} recorded no encoder: embed_model set to "
                                  f"{encoders.name_of_spec(spec)} ({DECISION})"})
    return out


def migrate(store: Store, emit: Callable[[str, dict], None]) -> dict | None:
    """Run the migration once per database (module docstring). -> its result, or None when it already ran."""
    result: dict = {}

    def run() -> None:
        result.update(migrate_projects(store, emit, config.legacy_settings()))
        log.warning("per-experiment encoder migration: %s", result)
    return result if store.migrate_once(NAME, run) else None


__all__ = ["DECISION", "NAME", "legacy_spec", "migrate", "migrate_projects"]
