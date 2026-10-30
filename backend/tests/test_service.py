"""forge/service.py: the launchd agent that keeps Forge (and its 03:00 schedule) running without a terminal.
launchctl is replaced by a recorder; nothing is installed or loaded on this machine."""
from __future__ import annotations

import json
import os
import plistlib
import subprocess

from forge import instance_lock, service


def _presets(tmp_path):
    """A presets file whose one preset passes ``MY_TOKEN_CMD`` and ``MY_SRC`` through to its adapter."""
    f = tmp_path / "presets.json"
    if not f.exists():
        f.write_text(json.dumps({"presets": [{
            "key": "my-heads", "name": "My heads", "adapter": ["my-adapter"], "cwd": str(tmp_path),
            "dataset_root": str(tmp_path / "ds"), "output_base": str(tmp_path / "out" / "heads"),
            "env_passthrough": ["MY_TOKEN_CMD", "MY_SRC"]}]}))
    return f


def _env(tmp_path):
    return {"PATH": "/usr/bin:/bin", "HOME": "/home/x", "FORGE_DATA": str(tmp_path / "data"), "FORGE_PORT": "8765",
            "FORGE_HEADSET_PRESETS": str(_presets(tmp_path)),
            "MY_TOKEN_CMD": "op read op://x/y", "MY_SRC": "/src", "SECRET_THING": "no", "OPENAI_API_KEY": "sk-no"}


class Launchctl:
    def __init__(self, rc=0):
        self.calls, self.rc = [], rc

    def __call__(self, cmd):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.rc, "", "")


def test_the_plist_runs_this_checkouts_server_with_only_forge_and_preset_settings(tmp_path):
    p = plistlib.loads(service.render(_env(tmp_path), python="/venv/bin/python").encode())
    assert p["Label"] == service.LABEL and p["ProgramArguments"] == ["/venv/bin/python", "-m", "forge.api"]
    assert p["WorkingDirectory"] == str(service.BACKEND) and p["RunAtLoad"] and p["KeepAlive"]
    env = p["EnvironmentVariables"]
    assert env["FORGE_DATA"] == str((tmp_path / "data").resolve()) and env["MY_SRC"] == "/src"
    assert "SECRET_THING" not in env and "OPENAI_API_KEY" not in env           # only what Forge and adapters read
    assert p["StandardOutPath"] == str((tmp_path / "data").resolve() / "logs" / "forge-api.log")


def test_install_uninstall_and_status_through_launchctl(tmp_path):
    lc = Launchctl()
    base = tmp_path / "LaunchAgents"
    assert service.install(run=lc, base=base, env=_env(tmp_path), dry_run=True).startswith("<?xml")
    assert lc.calls == [] and not service.plist_path(base).exists()               # a dry run changes nothing
    msg = service.install(run=lc, base=base, env=_env(tmp_path))
    assert "installed and loaded" in msg and service.plist_path(base).is_file()
    assert lc.calls == [["launchctl", "bootstrap", f"gui/{os.getuid()}", str(service.plist_path(base))]]
    data = tmp_path / "data"
    held = instance_lock.acquire(data, 8765)
    try:
        st = service.status(run=lc, base=base, env=_env(tmp_path))
        assert st["installed"] and st["loaded"] and st["server"] == {"pid": os.getpid(), "port": 8765}
    finally:
        held.close()
    assert service.status(run=lc, base=base, env=_env(tmp_path))["server"] is None
    assert "unloaded and removed" in service.uninstall(run=lc, base=base) and not service.plist_path(base).exists()
    assert service.uninstall(run=Launchctl(rc=113), base=base) == "Forge is not installed as a launchd agent"


def test_a_failed_load_says_so(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="launchctl could not load"):
        service.install(run=Launchctl(rc=5), base=tmp_path / "LA", env=_env(tmp_path))


def test_no_secret_reaches_the_plist_and_plist_and_log_are_owner_only(tmp_path):
    """Only the settings Forge reads and the names a preset passes through, by exact name: a token or key exported
    under a Forge or adapter prefix is never written into the plist (which launchd keeps on disk); plist and log
    are owner-only."""
    import stat
    env = {**_env(tmp_path), "MY_TOKEN": "tok-secret", "FORGE_API_KEY": "sk-secret", "MY_SRC_PASSWORD": "pw"}
    env_out = plistlib.loads(service.render(env, python="/venv/bin/python").encode())["EnvironmentVariables"]
    assert env_out["MY_TOKEN_CMD"] == "op read op://x/y"                              # a command, not a token
    assert env_out["FORGE_HEADSET_PRESETS"] == env["FORGE_HEADSET_PRESETS"]
    for name in ("MY_TOKEN", "FORGE_API_KEY", "MY_SRC_PASSWORD"):
        assert name not in env_out
    base = tmp_path / "LaunchAgents"
    base.mkdir()
    service.plist_path(base).write_text("old")
    os.chmod(service.plist_path(base), 0o644)                                       # an older, world-readable one
    log = tmp_path / "data" / "logs" / "forge-api.log"
    log.parent.mkdir(parents=True)
    log.write_text("earlier lines\n")
    os.chmod(log, 0o644)
    service.install(run=Launchctl(), base=base, env=env)
    text = service.plist_path(base).read_text()
    assert "tok-secret" not in text and "sk-secret" not in text
    assert stat.S_IMODE(service.plist_path(base).stat().st_mode) == 0o600
    assert stat.S_IMODE(log.stat().st_mode) == 0o600 and log.read_text() == "earlier lines\n"
    assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700


def test_without_a_presets_file_nothing_is_passed_through(tmp_path):
    env = {**_env(tmp_path), "FORGE_HEADSET_PRESETS": str(tmp_path / "missing.json")}
    out = service.environment(env)
    assert "MY_TOKEN_CMD" not in out and "MY_SRC" not in out and out["PATH"] == "/usr/bin:/bin"
