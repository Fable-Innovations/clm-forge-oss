"""Which encoder a new experiment or autopilot target should train on: Forge's recommendation, which the user can
change (there is no app-wide default encoder; every experiment and target keeps its own, ``forge.encoders``).

``recommend(task=None, *, backends=None, preferred=(), budget_bytes=None) -> Recommendation`` ranks the registry
(``backend/encoders.json``) with the facts it has, never a guess:

* **memory**: an encoder is eligible only when its measured embedding-worker peak (``approx_gb``) fits this machine's
  model-job budget (``Settings.memory`` resolved, ``forge.memory.resolve_policy``). An encoder whose peak was never
  measured is listed, not recommended (it may still be picked by hand).
* **backends**: what the caller can train on (a head-set preset's adapter trains on MLX builds only).
* **task**: a ``TaskSchema`` (from the proposal) sets the profile. ``rich`` = a choice among more than four options,
  options that come with each call (dynamic), or an ordered score: these gain from the widest encoder that fits,
  and the reference CLM head's own encoder (``encoders.REFERENCE``) is preferred among equals because training
  warm-starts from that head. ``simple`` = a yes/no question or a choice among at most four fixed options, and
  ``general`` = no task yet (the create dialog, before analysis): the smallest encoder at least ``SIMPLE_MIN_HIDDEN``
  wide, the GGUF (llama.cpp) build first among equals because that is the build packages ship on (macOS Metal and
  Windows/Linux CPU alike), then one that is already downloaded, then the smaller peak.
* **preferred**: keys the caller names first (a preset's own encoder, ``Preset.recommended_encoder``); the first
  eligible one wins, else the profile's ranking decides.

The result says why, in words, and lists every encoder considered with its facts, so the UI can show the reasoning
beside the pick.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from forge import encoders
from forge.memory import GiB

Profile = Literal["rich", "simple", "general"]
#: the narrowest encoder width a simple or general task is recommended (Qwen3-4B's); narrower ones stay pickable by
#: hand
SIMPLE_MIN_HIDDEN = 2560
RICH_OPTIONS = 4


class Considered(BaseModel):
    key: str
    name: str
    approx_gb: float | None
    fits: bool
    cached: bool
    why_not: str | None = None


class Recommendation(BaseModel):
    key: str | None
    name: str | None
    profile: Profile
    reason: str
    budget_gb: float
    considered: list[Considered]


def profile_of(task) -> Profile:
    """The task profile (module docstring) of a ``TaskSchema``, or ``general`` without one."""
    if task is None:
        return "general"
    if task.type == "score":
        return "rich"
    if task.type == "choice":
        if not task.criteria or len(task.labels()) > RICH_OPTIONS:
            return "rich"
    return "simple"


def _budget_bytes() -> int:
    from forge import config
    from forge.memory import resolve_policy
    return resolve_policy(config.load_settings().memory, int(os.environ.get("FORGE_MAX_LEN", 2048))).budget_bytes


def _rank(e: encoders.Encoder, profile: Profile) -> tuple:
    cached = encoders.cached(e)
    if profile == "rich":
        return (-e.hidden, e.key != encoders.REFERENCE, not cached, e.approx_gb or 0.0)
    return (e.hidden < SIMPLE_MIN_HIDDEN, e.hidden, e.backend != "llamacpp", not cached, e.approx_gb or 0.0)


def _why(e: encoders.Encoder, profile: Profile, budget_gb: float, preferred: bool) -> str:
    size = f"its measured peak ~{e.approx_gb:.1f} GB fits the {budget_gb:.1f} GB model budget"
    if preferred:
        return f"{e.name}: the encoder this head set is built for; {size}"
    if profile == "rich":
        extra = (" and the reference CLM head was trained on it, so training warm-starts"
                 if e.key == encoders.REFERENCE else "")
        return (f"{e.name}: the widest encoder that fits ({e.hidden}-d){extra}; a question with many or per-call "
                f"options, or an ordered score, gains from the width; {size}")
    what = ("a yes/no question or a choice among a few fixed options" if profile == "simple"
            else "a new experiment before its task is known")
    ship = " and the GGUF build is the one packages ship on" if e.backend == "llamacpp" else ""
    return f"{e.name}: {e.hidden}-d is wide enough for {what}{ship}; {size}"


def recommend(task=None, *, backends: tuple[str, ...] | list[str] | None = None, preferred: tuple[str, ...] = (),
              budget_bytes: int | None = None) -> Recommendation:
    """The encoder to start on (module docstring). ``key`` is None only when nothing in the registry fits the
    budget and the allowed backends; ``reason`` then says so."""
    profile = profile_of(task)
    budget = budget_bytes if budget_bytes is not None else _budget_bytes()
    budget_gb = round(budget / GiB, 1)
    considered: list[Considered] = []
    eligible: list[encoders.Encoder] = []
    for e in encoders.all():
        why_not = None
        if backends is not None and e.backend not in backends:
            why_not = f"a {e.backend} build; this trains on {'/'.join(backends)} encoders"
        elif e.approx_gb is None:
            why_not = "its memory peak was never measured on this machine"
        elif e.approx_gb * GiB > budget:
            why_not = f"its measured peak ~{e.approx_gb:.1f} GB exceeds the {budget_gb:.1f} GB model budget"
        considered.append(Considered(key=e.key, name=e.name, approx_gb=e.approx_gb, fits=why_not is None,
                                     cached=encoders.cached(e), why_not=why_not))
        if why_not is None:
            eligible.append(e)
    if not eligible:
        return Recommendation(key=None, name=None, profile=profile, budget_gb=budget_gb, considered=considered,
                              reason=f"no registered encoder fits the {budget_gb:.1f} GB model budget"
                                     + (f" on {'/'.join(backends)}" if backends is not None else ""))
    by_key = {e.key: e for e in eligible}
    pick = next((by_key[k] for k in preferred if k in by_key), None)
    is_preferred = pick is not None
    pick = pick or min(eligible, key=lambda e: _rank(e, profile))
    return Recommendation(key=pick.key, name=pick.name, profile=profile, budget_gb=budget_gb, considered=considered,
                          reason=_why(pick, profile, budget_gb, is_preferred))


__all__ = ["Considered", "Profile", "Recommendation", "profile_of", "recommend"]
