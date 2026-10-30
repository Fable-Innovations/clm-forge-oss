"""Runtime settings, persisted as JSON in output/settings.json.

Paths come from `forge.paths` (owned by the ML side); until it exists we fall back to the same
layout so this module always imports cleanly.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from forge.memory import MemorySettings

ROOT = Path(__file__).resolve().parents[2]  # the repo root

try:  # pragma: no cover - depends on the ML side having landed
    from forge.paths import CACHE_DIR, DATA_DIR, project_dir  # type: ignore
except Exception:  # noqa: BLE001
    DATA_DIR = Path(os.environ.get("FORGE_DATA_DIR", ROOT / "output")).resolve()
    CACHE_DIR = DATA_DIR / "cache"

    def project_dir(pid: str) -> Path:
        p = DATA_DIR / "projects" / pid
        p.mkdir(parents=True, exist_ok=True)
        return p


REPOS_DIR = DATA_DIR / "repos"
DB_PATH = DATA_DIR / "forge.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"

Provider = Literal["claude-subscription", "codex-subscription", "openai", "ollama"]


class Settings(BaseModel):
    provider: Provider = "claude-subscription"
    model: str = Field(default="", description="model id as listed by the provider; empty means the provider's default")
    ollama_url: str = "http://127.0.0.1:11434"
    openai_api_key: str = Field(default="", description="never returned by the API; OPENAI_API_KEY is used when empty")
    openai_base_url: str = Field(default="", description="OpenAI-compatible server; empty means api.openai.com")
    openai_api_key_url: str = Field(default="", description="base URL the saved key was entered for; the key is "
                                    "only ever sent there, so changing openai_base_url can't redirect it")
    success_threshold: float = 0.85
    target_examples: int = 600
    max_target_examples: int = 3000
    max_attempts: int = 3
    min_examples: int = 120
    llm_concurrency: int = 4
    gen_concurrency: int = 8     # parallel generator agents during curation's generate step
    qa_min_pass: float = 0.75    # orchestrator QA: share of a label's sample that must pass before labelling
    explore_max_turns: int = 40
    teacher_eval_n: int = 50
    labelset_roots: list[str] = Field(
        default_factory=list,
        description="directories of head dirs (<root>/<head_id>/task.json + real_rows.jsonl) whose labelling queues "
                    "the Labelling page shows; FORGE_LABELSETS (os.pathsep-separated) adds more")
    job_defer_s: float = Field(default=60.0, description="model-job queue: seconds between preflight re-checks "
                               "while a job is deferred (forge.jobqueue)")
    preflight_max_pressure_level: int = Field(default=1, description="highest macOS memory pressure level a model "
                                              "job may start at (1 normal, 2 warn, 4 critical)")
    preflight_ignore: list[str] = Field(
        default_factory=list,
        description="command-line markers of model processes the preflight does not count as foreign "
                    "(e.g. an app whose own encoders should not hold a job back)")
    dedup_cosine: float = 0.975  # raw cosine on Qwen3 last-token embeddings (see curator.dedup)
    memory: MemorySettings = Field(default_factory=MemorySettings,
                                   description="model-job memory budget, reserve, padding buckets (see forge.memory)")


_lock = threading.Lock()
_cached: Settings | None = None


#: settings fields from when Forge had one app-wide encoder (``encoder`` spec, then ``default_encoder`` key and the
#: ``base_models`` candidates); each experiment and target now carries its own (``forge.encoders``)
LEGACY_ENCODER_FIELDS = ("encoder", "default_encoder", "unregistered_encoder", "base_models",
                         "unregistered_base_models")


def legacy_backup() -> Path:
    """Where the settings file is kept as it was before the app-wide encoder fields were stripped."""
    return SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".bak-pre-experiment-encoders")


def strip_legacy_encoder_fields(raw: dict) -> bool:
    """Drop the app-wide encoder fields from stored settings (``LEGACY_ENCODER_FIELDS``). -> whether ``raw``
    changed. Their values are read once, by ``forge.encoder_migration``, from ``legacy_backup()`` (written before the
    stripped file replaces the original), to give experiments created before per-experiment encoders theirs."""
    found = [k for k in LEGACY_ENCODER_FIELDS if k in raw]
    for k in found:
        raw.pop(k)
    return bool(found)


def legacy_settings() -> dict:
    """The stored settings as they were before the app-wide encoder fields were stripped: ``legacy_backup()`` when
    the strip already happened, else ``settings.json`` itself ({} without either)."""
    for f in (legacy_backup(), SETTINGS_PATH):
        if f.exists():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                return {}
    return {}


def load_settings() -> Settings:
    global _cached
    with _lock:
        if _cached is None:
            if SETTINGS_PATH.exists():
                try:
                    raw = json.loads(SETTINGS_PATH.read_text())
                    stripped = strip_legacy_encoder_fields(raw)
                    _cached = Settings.model_validate(raw)
                except Exception:  # noqa: BLE001 - corrupt settings fall back to defaults
                    _cached, stripped = Settings(), False
                if stripped:
                    backup = legacy_backup()
                    if not backup.exists():
                        backup.write_text(SETTINGS_PATH.read_text())
                        backup.chmod(0o600)
                    _write(_cached)
            else:
                _cached = Settings()
        return _cached.model_copy()


def _write(s: Settings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s.model_dump(), indent=2))
    SETTINGS_PATH.chmod(0o600)  # may hold an API key


def save_settings(s: Settings) -> Settings:
    global _cached
    with _lock:
        _write(s)
        _cached = s.model_copy()
        return _cached.model_copy()


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, REPOS_DIR, DATA_DIR / "projects"):
        d.mkdir(parents=True, exist_ok=True)
