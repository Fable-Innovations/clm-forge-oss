"""Head-set presets: the named head sets a target can be made from, so the target form never asks for argv.

A preset supplies everything a ``HeadsetTarget`` needs besides the user's choices (heads, encoder, sources,
schedule, policy): the adapter argv, its working directory and environment, the dataset root, the protected roots /
globs / files, the data sources the adapter can pull (with their settings), the encoder backends it can train on,
the data-only adapter keys a target may set, and the base output root (``forge.encoders.output_root(base, key)``
gives the encoder's folder).

Presets are operator configuration, never request data: they are read from a JSON file on the server,
``$FORGE_HEADSET_PRESETS`` when set, else ``<data dir>/headset_presets.json``::

    {"presets": [{"key": "my-heads", "name": "My heads", "description": "...",
                  "adapter": ["uv", "run", "python", "-m", "my_pkg.autopilot"], "cwd": "/abs/dir",
                  "env": {"NAME": "value"}, "env_passthrough": ["MY_TOKEN_CMD"],
                  "dataset_root": "/abs/dir", "output_base": "/abs/dir/my-heads",
                  "protected_roots": [], "protected_globs": [], "protected_files": [],
                  "sources": [{"kind": "decision_log", "label": "Decision log", "enabled_by_default": true,
                               "settings": {"stage": {"values": ["dev", "qa", "prod"], "default": "qa",
                                                      "multiple": false}}}],
                  "fixed_heads": null, "encoder_backends": ["mlx"], "adapter_config": {},
                  "adapter_config_keys": ["epochs"], "recommended_encoder": null}]}

A missing file means no presets (head-set targets cannot be created; project targets still work); a malformed one
raises ``PresetFileError`` naming the file. The file is re-read when it changes (``load``), so editing it needs no
restart. ``env_passthrough`` names shell variables ``forge service install`` copies into the launchd agent so the
adapter still finds its secrets (``forge.service``); Forge itself never names them.

The adapter contract (verbs, ``--target-config`` schema) is SPEC.md "Autopilot adapter contract".
``heads(preset, encoder, adapter_factory)`` runs the adapter's read-only ``inspect`` and returns the head names
for the form's multiselect.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
#: a source kind or setting name: an identifier the adapter reads from ``--target-config``
NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ENV_VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FILE_NAME = "headset_presets.json"


class PresetFileError(ValueError):
    """The presets file exists but cannot be used (not JSON, or a preset fails validation)."""


def _absolute(v: str, what: str) -> str:
    if not Path(v).expanduser().is_absolute():
        raise ValueError(f"{what} must be an absolute path, got {v!r}")
    return v


class SourceSetting(BaseModel):
    """One setting of a data source: an enum the form shows as a select (``multiple``: a multi-select whose
    value is a non-empty list of distinct members)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    values: tuple[str, ...] = Field(min_length=1)
    default: str | tuple[str, ...]
    multiple: bool = False

    @model_validator(mode="after")
    def _default(self) -> "SourceSetting":
        if len(set(self.values)) != len(self.values):
            raise ValueError("values must be distinct")
        if self.multiple:
            if isinstance(self.default, str):
                raise ValueError("a multiple setting's default is a list")
            if not self.default or len(set(self.default)) != len(self.default):
                raise ValueError("a multiple setting's default is a non-empty list of distinct values")
            bad = [d for d in self.default if d not in self.values]
        else:
            if not isinstance(self.default, str):
                raise ValueError("a single setting's default is one value")
            bad = [] if self.default in self.values else [self.default]
        if bad:
            raise ValueError(f"default {bad} not among values {list(self.values)}")
        return self

    def default_value(self) -> str | list[str]:
        return list(self.default) if self.multiple else self.default  # type: ignore[return-value]

    def view(self) -> dict:
        return {"values": list(self.values), "default": self.default_value(), "multiple": self.multiple}


class SourceSpec(BaseModel):
    """A data source the adapter can pull: its ``kind`` (passed through to the adapter), the form's label, whether
    a new target enables it, and its settings."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str
    label: str = ""
    enabled_by_default: bool = True
    settings: dict[str, SourceSetting] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        if not NAME.match(v):
            raise ValueError(f"source kind {v!r}: lowercase letters, digits and '_', starting with a letter")
        return v

    @field_validator("settings")
    @classmethod
    def _names(cls, v: dict) -> dict:
        for name in v:
            if not NAME.match(name):
                raise ValueError(f"setting name {name!r}: lowercase letters, digits and '_', starting with a letter")
        return v

    @property
    def title(self) -> str:
        return self.label or self.kind.replace("_", " ").capitalize()

    def defaults(self) -> dict[str, str | list[str]]:
        return {k: s.default_value() for k, s in self.settings.items()}

    def view(self) -> dict:
        return {"kind": self.kind, "label": self.title, "enabled_by_default": self.enabled_by_default,
                "settings": {k: s.view() for k, s in self.settings.items()}}


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str
    name: str
    description: str = ""
    adapter: tuple[str, ...] = Field(min_length=1)
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    #: shell variables the launchd agent copies so the adapter finds its secrets (``forge.service``)
    env_passthrough: tuple[str, ...] = ()
    dataset_root: str
    output_base: str
    protected_roots: tuple[str, ...] = ()
    protected_globs: tuple[str, ...] = ()
    protected_files: tuple[str, ...] = ()
    sources: tuple[SourceSpec, ...] = ()
    fixed_heads: tuple[str, ...] | None = None       # None: the user picks from `inspect`
    encoder_backends: tuple[str, ...] = ("mlx",)
    adapter_config: dict = Field(default_factory=dict)
    #: the data-only ``--target-config`` keys a request may set (numbers and choices, never a command, path or URL)
    adapter_config_keys: tuple[str, ...] = ()
    #: the encoder the head set is built for (the new-target form preselects it when it fits this machine's model
    #: budget; ``forge.encoder_choice.recommend`` with it preferred); the user can pick another
    recommended_encoder: str | None = None

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"key {v!r}: lowercase letters, digits, '_', '.', '-' (max 64), starting with a "
                             "letter or digit")
        return v

    @field_validator("env_passthrough")
    @classmethod
    def _env_names(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for name in v:
            if not ENV_VAR.match(name):
                raise ValueError(f"env_passthrough {name!r} is not an environment variable name")
        return v

    @model_validator(mode="after")
    def _paths(self) -> "Preset":
        for name in ("cwd", "dataset_root", "output_base"):
            _absolute(getattr(self, name), name)
        for name in ("protected_roots", "protected_files"):
            for p in getattr(self, name):
                _absolute(p, name)
        kinds = [s.kind for s in self.sources]
        if len(set(kinds)) != len(kinds):
            raise ValueError("each source kind can be declared once")
        return self

    def source(self, kind: str) -> SourceSpec | None:
        return next((s for s in self.sources if s.kind == kind), None)

    def recommendation(self) -> dict:
        """``forge.encoder_choice.recommend`` for this head set: its own encoder first, on the backends it trains on."""
        from forge.encoder_choice import recommend
        r = recommend(backends=self.encoder_backends,
                      preferred=(self.recommended_encoder,) if self.recommended_encoder else ())
        return {"key": r.key, "name": r.name, "reason": r.reason}

    def output_root_refusal(self, out: Path) -> str | None:
        """Why ``out`` (resolved) cannot be this preset's output folder, or None: it must be a
        ``<output_base>-<name>`` folder beside the base (where ``encoders.output_root`` derives one per encoder),
        so a target never writes models and summaries anywhere else on disk."""
        base = Path(self.output_base).expanduser().resolve()
        if out.parent == base.parent and out.name.startswith(base.name + "-") and len(out.name) > len(base.name) + 1:
            return None
        return (f"Output folder must be a {base.name}-<name> folder in {base.parent} (the {self.name} preset's "
                f"output folders); {out} is not")

    def encoder_keys(self) -> list[str]:
        """The registry encoders this head set's adapter can train on."""
        from forge import encoders
        return [e.key for e in encoders.all() if e.backend in self.encoder_backends]

    def view(self) -> dict:
        from forge import encoders
        keys = self.encoder_keys()
        return {"key": self.key, "name": self.name, "description": self.description,
                "dataset_root": self.dataset_root, "output_base": self.output_base,
                "protected_roots": list(self.protected_roots), "sources": [s.view() for s in self.sources],
                "fixed_heads": list(self.fixed_heads) if self.fixed_heads is not None else None,
                "encoders": keys, "output_roots": {k: encoders.output_root(self.output_base, k) for k in keys},
                "recommendation": self.recommendation()}

    def headset(self, encoder: str) -> dict:
        """A ``HeadsetTarget`` body for this preset on ``encoder`` (the form adds heads, sources and policy): every
        declared source, enabled as the preset says, with its default settings."""
        from forge import encoders
        return {"preset": self.key, "adapter": list(self.adapter), "cwd": self.cwd, "env": dict(self.env),
                "dataset_root": self.dataset_root,
                "encoder": encoder, "output_root": encoders.output_root(self.output_base, encoder),
                "heads": list(self.fixed_heads or ()), "protected_roots": list(self.protected_roots),
                "protected_globs": list(self.protected_globs), "protected_files": list(self.protected_files),
                "sources": [{"kind": s.kind, "enabled": s.enabled_by_default, "settings": s.defaults()}
                            for s in self.sources],
                "adapter_config": dict(self.adapter_config)}


class PresetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    presets: list[Preset] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> "PresetFile":
        keys = [p.key for p in self.presets]
        if dup := sorted({k for k in keys if keys.count(k) > 1}):
            raise ValueError(f"preset keys must be unique: {', '.join(dup)}")
        return self


def path() -> Path:
    """``$FORGE_HEADSET_PRESETS``, else ``<data dir>/headset_presets.json``."""
    if env := os.environ.get("FORGE_HEADSET_PRESETS"):
        return Path(env).expanduser()
    from forge import paths
    return paths.DATA_DIR / FILE_NAME


_cache: dict[tuple, dict[str, Preset]] = {}


def load(p: Path | None = None) -> dict[str, Preset]:
    """The presets in ``p`` (default ``path()``) by key, re-read when the file (path, mtime, size) changes. A missing
    file -> ``{}``; an unusable one raises ``PresetFileError`` naming it."""
    p = p or path()
    try:
        st = p.stat()
    except FileNotFoundError:
        return {}
    sig = (str(p), st.st_mtime_ns, st.st_size)
    if sig in _cache:
        return _cache[sig]
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise PresetFileError(f"head-set presets file {p}: not valid JSON ({e})") from None
    try:
        parsed = PresetFile.model_validate(raw)
    except ValidationError as e:
        first = e.errors(include_url=False)[0]
        where = ".".join(str(x) for x in first["loc"])
        raise PresetFileError(f"head-set presets file {p}: {where + ': ' if where else ''}{first['msg']}") from None
    _cache.clear()
    _cache[sig] = {x.key: x for x in parsed.presets}
    return _cache[sig]


def all() -> list[Preset]:  # noqa: A001 - mirrors forge.encoders.all
    return list(load().values())


def get(key: str) -> Preset:
    """The preset ``key``; ``KeyError`` when the file has none by that key (or there is no file)."""
    presets = load()
    if key not in presets:
        known = ", ".join(presets) or f"none; add one to {path()}"
        raise KeyError(f"no head-set preset {key!r} (known: {known})")
    return presets[key]


def env_passthrough(p: Path | None = None) -> set[str]:
    """Every preset's ``env_passthrough`` names in ``p`` (default ``path()``); empty when the file is missing or
    unusable."""
    try:
        return {n for x in load(p).values() for n in x.env_passthrough}
    except PresetFileError:
        return set()


def fill(headset: dict) -> dict:
    """A head-set body naming a ``preset``: the preset's fields under the body's own (the form sends only what it
    edits: heads, encoder, output folder, sources, policy; ``forge.api`` refuses a body naming a server field).
    Raises ``KeyError`` for an unknown preset."""
    from forge import encoders
    p = get(headset["preset"])
    enc = headset.get("encoder") or ""
    base = p.headset(enc) if encoders.known(enc) else {**p.headset(p.encoder_keys()[0]), "encoder": enc,
                                                        "output_root": ""}
    out = {**base, **{k: v for k, v in headset.items() if v is not None or k == "preset"}}
    out["adapter_config"] = {**p.adapter_config, **(headset.get("adapter_config") or {})}
    return out


async def heads(preset: Preset, encoder: str, adapter_factory) -> list[dict]:
    """``inspect`` through the preset's adapter (read-only, loads no model) -> ``[{head, group, retired}]``.
    ``inspect`` writes nothing, so the output folder is not validated here (a target's save does that); an unknown
    encoder still raises the ``ValidationError``."""
    from forge.autopilot.targets import Target
    t = Target.model_validate({"id": "preset-inspect", "kind": "headset", "headset": preset.headset(encoder)},
                              context={"inspect_only": True})
    out = await adapter_factory(t).call("inspect")
    return [{"head": s["head"], "group": s.get("group"), "retired": s.get("retired") or None}
            for s in out.get("heads") or []]
