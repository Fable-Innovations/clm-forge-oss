"""Filesystem layout for runtime data (kept tiny so the ML modules don't depend on config.py).

``FORGE_DATA`` overrides the data root (default ``<repo>/data``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FORGE_DATA", str(REPO_ROOT / "data"))).expanduser()
CACHE_DIR = DATA_DIR / "cache"
EMB_CACHE_DIR = CACHE_DIR / "emb"
PROJECTS_DIR = DATA_DIR / "projects"


def project_dir(pid: str) -> Path:
    """``data/projects/<pid>`` (created on demand)."""
    p = PROJECTS_DIR / pid
    p.mkdir(parents=True, exist_ok=True)
    return p
