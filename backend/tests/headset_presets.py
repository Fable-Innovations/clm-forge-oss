"""Head-set presets files for tests (``forge.autopilot.presets``): ``preset(root)`` is one preset whose folders are
under ``root``; ``use(monkeypatch, path, *presets)`` writes them to ``path`` and points ``FORGE_HEADSET_PRESETS`` at
it (the autouse fixture in ``conftest`` points it at a missing file otherwise)."""
from __future__ import annotations

import json
from pathlib import Path

#: two sources: one single-valued setting, enabled by default; one multi-valued setting, off by default
SOURCES = [
    {"kind": "event_log", "label": "Event log", "enabled_by_default": True,
     "settings": {"stage": {"values": ["dev", "qa", "prod"], "default": "qa", "multiple": False}}},
    {"kind": "shared_export", "label": "Shared export", "enabled_by_default": False,
     "settings": {"stages": {"values": ["qa", "prod"], "default": ["qa", "prod"], "multiple": True}}},
]


def preset(root: Path, key: str = "test-heads", name: str = "Test heads", **kw) -> dict:
    root = Path(root)
    return {"key": key, "name": name, "description": "", "adapter": ["x"], "cwd": str(root),
            "dataset_root": str(root / "data"), "output_base": str(root / "heads"),
            "protected_roots": [str(root / "heads")], "protected_globs": ["*/rows.jsonl"], "protected_files": [],
            "sources": SOURCES, "adapter_config_keys": ["epochs"], **kw}


def use(monkeypatch, path: Path, *presets: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"presets": list(presets)}))
    monkeypatch.setenv("FORGE_HEADSET_PRESETS", str(path))
    return path
