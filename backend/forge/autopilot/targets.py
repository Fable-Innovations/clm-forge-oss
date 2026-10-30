"""Autopilot targets: what the autopilot keeps trained, and the policy it follows.

Two kinds, one loop (``forge.autopilot.runner``):

- ``project``: a Forge project, driven through the orchestrator (``ProjectTarget``);
- ``headset``: an external head set driven through its adapter, a subprocess speaking the JSON-lines contract in
  SPEC "Autopilot adapter contract" (``HeadsetTarget``), made from a preset (``forge.autopilot.presets``).

``Targets`` is the registry (``autopilot_targets`` table). A target is validated when it is saved: paths are
absolute, the output root lies outside every protected root, the encoder is a registry key (``forge.encoders``)
its preset's adapter can train on, every data source is a kind the preset declares with settings it allows, and
the kind matches its config block. Validators raise ``FieldError`` naming the field, so the form shows the
message beside it (``forge.autopilot.validation``).

A target has a display ``name`` (the id is derived from it once, at creation).

**Commands come only from presets.** What a head set runs and where it reads (``SERVER_FIELDS``: adapter argv,
working folder, environment, dataset root, protected roots / globs / files) is set by its preset, server-side; a
request body that names one is refused (``forge.api``), a stored row's copy is replaced on load
(``with_server_fields``), and a head set without a preset cannot run. A request may set only the preset's data-only
adapter keys (``Preset.adapter_config_keys``), source settings from the preset's enums, and an output folder the
preset allows (``Preset.output_root_refusal``: a ``<base>-<name>`` sibling of the preset's base). A head-set target
records the ``preset`` it was made from and its real-data ``sources``: ``Source(kind, enabled, settings)``; they
reach the adapter as ``target_config()["sources"]`` and ``pull-sources`` / ``ingest`` honour only the enabled ones.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from forge import encoders
from forge.autopilot.presets import NAME, SourceSpec
from forge.autopilot.validation import FieldError
from forge.store import Store

_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
#: what a head set runs and where it reads and must never write: set only by its preset (server-side
#: configuration), never by a request body and never trusted from a stored row (``Targets`` re-derives them)
SERVER_FIELDS = ("adapter", "cwd", "env", "dataset_root", "protected_roots", "protected_globs", "protected_files")
MIXES = ("synthetic", "synthetic+shaped", "all")


class Source(BaseModel):
    """A real-data source the adapter pulls. ``kind`` and ``settings`` are checked against the preset's declared
    sources by ``HeadsetTarget`` (where the preset is known), which also fills settings left out with defaults."""
    model_config = ConfigDict(extra="forbid")
    kind: str
    enabled: bool = True
    settings: dict[str, str | list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _kind(self) -> "Source":
        if not NAME.match(self.kind):
            raise FieldError("kind", "lowercase letters, digits and '_', starting with a letter")
        return self

    def checked(self, spec: SourceSpec) -> dict[str, str | list[str]]:
        """``settings`` validated against ``spec`` with its defaults filled in; ``FieldError`` names
        ``settings.<name>``."""
        out: dict[str, str | list[str]] = {}
        for name, value in self.settings.items():
            s = spec.settings.get(name)
            if s is None:
                raise FieldError(f"settings.{name}", f"{spec.title} has no setting {name!r} (settings: "
                                 f"{', '.join(spec.settings) or 'none'})")
            allowed = ", ".join(s.values)
            if s.multiple:
                if not isinstance(value, list) or not value:
                    raise FieldError(f"settings.{name}", f"choose at least one of {allowed}")
                if len(set(value)) != len(value):
                    raise FieldError(f"settings.{name}", "each value can be listed once")
                if bad := [v for v in value if v not in s.values]:
                    raise FieldError(f"settings.{name}", f"{', '.join(bad)} is not one of {allowed}")
            elif not isinstance(value, str) or value not in s.values:
                raise FieldError(f"settings.{name}", f"must be one of {allowed}")
            out[name] = list(value) if isinstance(value, list) else value
        return {**spec.defaults(), **out}


class Policy(BaseModel):
    """Numbers the detectors, balance maths, rounds and promotion use (SPEC "Autopilot")."""
    min_count: int = Field(default=150, ge=1, description="minimum train rows per label group")
    gen_cap: int = Field(default=600, ge=1, description="generated rows per head per round; the rest carries over")
    max_rounds: int = Field(default=3, ge=1, le=10)
    attempts: int = Field(default=3, ge=1, le=10, description="seeds / variants per training round (val-ranked)")
    exemplars: int = Field(default=4, ge=1, le=8, description="scrubbed real exemplars per generation prompt")
    adjudicate: bool = True
    tol: float = Field(default=0.01, ge=0, description="promotion: candidate test macro-F1 >= incumbent - tol")
    tol_per_head: dict[str, float] = Field(default_factory=dict)
    shape_template_share: float = 0.2      # shape_gap: share of real key templates absent from synthetic
    shape_ks: float = 0.25                 # shape_gap: length KS D
    noise_disagreement: float = 0.15       # label_noise: teacher disagreement rate
    noise_excluded: float = 0.10           # label_noise: excluded share of real rows
    confusable_share: float = 0.25         # confusable: one pair's share of a label's val errors
    confusable_min_errors: int = 5         # ... counted only once the label has this many val errors
    real_hurts_margin: float = 0.02        # real_hurts: candidate val macro-F1 < control - margin
    real_prior_min: int = 30               # below this many real labelled rows, balance uses the synthetic prior
    promote_over_failing_incumbent: bool = Field(
        default=False, description="promote a candidate that misses its test gate when the incumbent misses its own "
        "and the candidate beats it on test (journalled gate_policy when it applies)")
    generate_kinds: list[str] = Field(default_factory=list, description="head kinds (inspect `kind`) the adapter "
                                      "can generate rows for; empty = every kind")

    def tol_for(self, head: str) -> float:
        return self.tol_per_head.get(head, self.tol)


class HeadsetTarget(BaseModel):
    preset: str | None = Field(default=None, description="the head-set preset this target was made from "
                               "(forge.autopilot.presets); None for a hand-registered adapter")
    adapter: list[str] = Field(min_length=1, description="argv prefix of the adapter, e.g. "
                               "[\"uv\", \"run\", \"python\", \"-m\", \"my_pkg.autopilot\"]")
    cwd: str = Field(description="working directory the adapter runs in (absolute)")
    env: dict[str, str] = Field(default_factory=dict)
    dataset_root: str
    encoder: str = Field(description="encoder registry key (backend/encoders.json); the adapter's pin table is "
                         "the registry's MLX entries")
    output_root: str = Field(description="per-encoder output root; promoted heads land in <output_root>/<head>")
    heads: list[str] = Field(default_factory=list, description="heads to keep trained; empty = every head "
                             "`inspect` reports")
    protected_roots: list[str] = Field(default_factory=list, description="roots the autopilot never writes "
                                       "(e.g. shipped head artifacts); output_root must lie outside them")
    protected_globs: list[str] = Field(default_factory=lambda: ["*/rows.jsonl"],
                                       description="files under dataset_root no step may change (writer guard)")
    protected_files: list[str] = Field(default_factory=list, description="absolute paths no step may change "
                                       "(frozen test sets referenced by path)")
    sources: list[Source] = Field(default_factory=list, description="real-data sources pull-sources and ingest "
                                  "read (only the enabled ones)")
    adapter_config: dict = Field(default_factory=dict, description="adapter-only --target-config keys "
                                 "(the preset's, plus its adapter_config_keys); see SPEC \"Autopilot adapter "
                                 "contract\"")
    policy: Policy = Field(default_factory=Policy)

    @model_validator(mode="after")
    def _paths(self, info: ValidationInfo) -> "HeadsetTarget":
        """Field checks. Validated with context ``{"inspect_only": True}`` (``presets.heads``: a read-only
        ``inspect`` that writes nothing) the output folder is not checked against the protected roots; with
        ``{"stored": True}`` (``Targets.list``) not against the preset's output folders (the runner checks that
        before it runs the target)."""
        inspect_only = bool((info.context or {}).get("inspect_only"))
        try:
            enc = encoders.get(self.encoder)
        except encoders.UnknownEncoder as e:
            raise FieldError("encoder", str(e)) from None
        labels = {"cwd": "Working folder", "dataset_root": "Dataset folder", "output_root": "Output folder"}
        for name in ("cwd", "dataset_root", "output_root"):
            if not Path(getattr(self, name)).expanduser().is_absolute():
                raise FieldError(name, f"{labels[name]} must be an absolute path")
        out = Path(self.output_root).expanduser().resolve()
        for root in () if inspect_only else self.protected_roots:
            r = Path(root).expanduser().resolve()
            if out == r or r in out.parents:
                raise FieldError("output_root", f"Output folder is inside a protected folder ({r}); the autopilot "
                                 "never writes there")
        for i, f in enumerate(self.protected_files):
            if not Path(f).expanduser().is_absolute():
                raise FieldError(f"protected_files.{i}", f"protected file {f!r} must be an absolute path")
        if self.preset is not None:
            from forge.autopilot import presets
            try:
                p = presets.get(self.preset)
            except KeyError as e:
                raise FieldError("preset", str(e.args[0])) from None
            stored = bool((info.context or {}).get("stored"))
            if not inspect_only and not stored and (why := p.output_root_refusal(out)):
                raise FieldError("output_root", why)
            if extra := sorted(set(self.adapter_config) - set(p.adapter_config_keys) - set(p.adapter_config)):
                raise FieldError("adapter_config", f"{', '.join(extra)} cannot be set on a target (allowed: "
                                 f"{', '.join(sorted(p.adapter_config_keys)) or 'none'}); commands and paths "
                                 f"come from the {p.name} preset")
            if enc.backend not in p.encoder_backends:
                raise FieldError("encoder", f"{p.name} train on {'/'.join(p.encoder_backends)} encoders; "
                                 f"{enc.name} is a {enc.backend} build")
            for i, src in enumerate(self.sources):
                spec = p.source(src.kind)
                if spec is None:
                    raise FieldError(f"sources.{i}.kind", f"{p.name} can't pull from {src.kind!r} (offered: "
                                     f"{', '.join(s.kind for s in p.sources) or 'none'})")
                try:
                    src.settings = src.checked(spec)
                except FieldError as e:
                    raise FieldError(f"sources.{i}.{e.field}", e.msg) from None
        kinds = [s.kind for s in self.sources]
        if len(set(kinds)) != len(kinds):
            raise FieldError("sources", "each source kind can be listed once")
        self.target_config()
        return self

    def target_config(self) -> dict:
        """The adapter's ``--target-config`` (its schema: SPEC "Autopilot adapter contract")."""
        shared = {"encoder": self.encoder, "data_root": self.dataset_root, "output_root": self.output_root,
                  "heads": self.heads, "attempts": self.policy.attempts, "tol": self.policy.tol_per_head,
                  "sources": [s.model_dump() for s in self.sources]}
        clash = sorted(set(shared) & set(self.adapter_config) | ({"stage"} & set(self.adapter_config)))
        if clash:
            raise FieldError("adapter_config", f"adapter_config may not set {clash}: they come from the target "
                             "itself (the stage is a source setting)")
        return {**shared, **self.adapter_config}


class ProjectTarget(BaseModel):
    project_id: str
    policy: Policy = Field(default_factory=Policy)


class Target(BaseModel):
    id: str
    name: str = Field(default="", description="display name; empty = the id")
    kind: Literal["project", "headset"]
    enabled: bool = True
    headset: HeadsetTarget | None = None
    project: ProjectTarget | None = None

    @model_validator(mode="after")
    def _kind(self) -> "Target":
        if not _ID.match(self.id):
            raise FieldError("id", "lowercase letters, digits, '_', '.', '-' (max 64), starting with a letter or digit")
        self.name = self.name.strip() or self.id
        if self.kind == "headset" and (self.headset is None or self.project is not None):
            raise FieldError("headset", "a head-set target needs a head set (and no project)")
        if self.kind == "project" and (self.project is None or self.headset is not None):
            raise FieldError("project", "a project target needs a project (and no head set)")
        return self

    @property
    def policy(self) -> Policy:
        return (self.headset or self.project).policy  # type: ignore[union-attr]


def id_from_name(name: str) -> str:
    """The id a new target gets from its name: lowercase, runs of other characters become '-'."""
    return re.sub(r"[^a-z0-9_.-]+", "-", name.strip().lower()).strip("-._")[:64]


def with_server_fields(cfg: dict) -> dict:
    """A stored target config with its head set's ``SERVER_FIELDS`` taken from its preset (whatever the row holds).
    A head set without a preset keeps its stored fields; it cannot run (``runner.default_adapter``)."""
    hs = cfg.get("headset")
    if not hs or not hs.get("preset"):
        return cfg
    from forge.autopilot import presets
    try:
        p = presets.get(hs["preset"])
    except KeyError:
        return cfg                           # validation names the unknown preset
    base = p.headset(p.encoder_keys()[0])          # the server fields do not depend on the encoder
    return {**cfg, "headset": {**hs, **{k: base[k] for k in SERVER_FIELDS},
                               "adapter_config": {**p.adapter_config, **(hs.get("adapter_config") or {})}}}


class Targets:
    def __init__(self, store: Store):
        self.store = store

    def list(self) -> list[Target]:
        # a stored row loads even when its output folder predates the preset rule; the runner refuses to run it
        return [Target.model_validate({"id": t["id"], "kind": t["kind"], "enabled": t["enabled"],
                                       **with_server_fields(t["config"])}, context={"stored": True})
                for t in self.store.targets()]

    def get(self, tid: str) -> Target | None:
        return next((t for t in self.list() if t.id == tid), None)

    def put(self, t: Target) -> Target:
        cfg = t.model_dump(include={"name", "headset", "project"}, exclude_none=True)
        if cfg.get("headset") and t.headset.preset is not None:       # derived from the preset on every load
            cfg["headset"] = {k: v for k, v in cfg["headset"].items() if k not in SERVER_FIELDS}
        self.store.save_target(t.id, t.kind, cfg, t.enabled)
        return t

    def delete(self, tid: str) -> None:
        self.store.delete_target(tid)
