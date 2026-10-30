"""Forge as a launchd agent, so the autopilot's schedule (e.g. weekly at 03:00) runs without a terminal.

    forge service install [--dry-run]    write ~/Library/LaunchAgents/<label>.plist and load it
    forge service uninstall              unload it and remove the plist
    forge service status                 the plist, launchd's view, the data folder's server lock, /api/health

One owner for the agent definition: ``launchd/forge-api.plist.tmpl`` rendered by ``render`` with this checkout's
interpreter (``python -m forge.api``), its backend folder, the data folder, port and the environment the server
and its adapters read: exactly the names in ``ENV_KEYS`` and ``ENV_NAMES``, plus the names head-set presets list in
``env_passthrough`` (``forge.autopilot.presets``: an adapter's own settings and secret-fetching commands, named by
the operator's presets file, never by Forge), taken from the shell that runs ``install``. Nothing else of that
shell's environment is copied, whatever its prefix, so a token or key exported there (``FORGE_API_KEY``,
``OPENAI_API_KEY``) is never written into the plist unless a preset names it: an adapter that needs a secret should
list the command that fetches it, not the secret itself. The plist is written owner-only (0600). Output goes to
``<data>/logs/forge-api.log`` (the uvicorn access log included, so who called which route is on disk; never a
session token, which travels in a header), created owner-only (0600) in an owner-only folder (0700) before launchd
opens it. The server holds ``<data>/forge-server.lock`` (``forge.instance_lock``): with a terminal server running on the same data folder the
agent's server refuses to start and launchd retries every ``THROTTLE_S`` seconds; stop the terminal one first.

``install`` and ``uninstall`` call ``launchctl`` (``bootstrap`` / ``bootout`` in the ``gui/<uid>`` domain); the
command runner is injectable for tests. Nothing here runs at import.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from string import Template
from xml.sax.saxutils import escape

from forge import instance_lock

LABEL = "com.clm-forge.forge-api"
THROTTLE_S = 60
TEMPLATE = Path(__file__).resolve().parent / "launchd" / "forge-api.plist.tmpl"
BACKEND = Path(__file__).resolve().parents[1]
#: environment the server reads, by exact name (an allowlist: nothing is copied by prefix); head-set presets add
#: their adapters' names (``passthrough``)
ENV_KEYS = ("PATH", "HOME", "LANG")
ENV_NAMES = frozenset({
    "FORGE_DATA", "FORGE_DATA_DIR", "FORGE_PORT", "FORGE_UI_PORT", "FORGE_MAX_LEN", "FORGE_MODEL_LOCK_DIR",
    "FORGE_MLFLOW", "FORGE_LLAMA_SERVER", "FORGE_LABELSETS", "FORGE_HEADSET_PRESETS"})

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(base: Path | None = None) -> Path:
    return (base or agents_dir()) / f"{LABEL}.plist"


def _data_dir(env: dict[str, str]) -> Path:
    from forge import paths
    return Path(env.get("FORGE_DATA") or paths.DATA_DIR).expanduser().resolve()


def passthrough(env: dict[str, str]) -> set[str]:
    """The ``env_passthrough`` names of the presets file the agent will read (``FORGE_HEADSET_PRESETS`` in ``env``,
    else ``<data>/headset_presets.json``); empty when there is none or it is unusable."""
    from forge.autopilot import presets
    f = env.get("FORGE_HEADSET_PRESETS")
    return presets.env_passthrough(Path(f).expanduser() if f else _data_dir(env) / presets.FILE_NAME)


def environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """The variables the agent gets: ``ENV_KEYS``, ``ENV_NAMES`` and the presets' ``env_passthrough`` names only,
    plus the data folder."""
    env = dict(os.environ if env is None else env)
    extra = passthrough(env)
    out = {k: v for k, v in env.items() if k in ENV_KEYS or k in ENV_NAMES or k in extra}
    out["FORGE_DATA"] = str(_data_dir(env))
    out.setdefault("FORGE_PORT", "8765")
    return dict(sorted(out.items()))


def render(env: dict[str, str] | None = None, python: str | None = None) -> str:
    """The agent's plist for this checkout and environment."""
    e = environment(env)
    log = Path(e["FORGE_DATA"]) / "logs" / "forge-api.log"
    lines = "\n".join(f"    <key>{escape(k)}</key>\n    <string>{escape(v)}</string>" for k, v in e.items())
    return Template(TEMPLATE.read_text()).substitute(
        label=LABEL, python=escape(python or sys.executable), backend=escape(str(BACKEND)), environment=lines,
        throttle=THROTTLE_S, log=escape(str(log)))


def _write_private(path: Path, text: str) -> None:
    """Write ``path`` owner-only (0600), also when it exists with wider permissions."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


def prepare_log(data: Path) -> Path:
    """``<data>/logs`` owner-only (0700) and ``forge-api.log`` in it owner-only (0600), created before launchd
    opens it (launchd keeps an existing file's mode). -> the log path."""
    logs = Path(data) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    os.chmod(logs, 0o700)
    log = logs / "forge-api.log"
    os.close(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
    os.chmod(log, 0o600)
    return log


def install(*, run: Runner = _run, base: Path | None = None, env: dict[str, str] | None = None,
            dry_run: bool = False) -> str:
    """Write the plist and load it (``launchctl bootstrap gui/<uid>``). -> what was done, in words."""
    text = render(env)
    if dry_run:
        return text
    p = plist_path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    prepare_log(Path(environment(env)["FORGE_DATA"]))
    if p.exists():
        run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"])        # replace a loaded older definition
    _write_private(p, text)
    r = run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(p)])
    if r.returncode != 0:
        raise RuntimeError(f"launchctl could not load {p}: {(r.stderr or r.stdout).strip()}")
    return f"installed and loaded {p}; Forge now starts at login and restarts if it stops"


def uninstall(*, run: Runner = _run, base: Path | None = None) -> str:
    p = plist_path(base)
    r = run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"])
    gone = p.exists()
    if gone:
        p.unlink()
    if not gone and r.returncode != 0:
        return "Forge is not installed as a launchd agent"
    return f"unloaded and removed {p}"


def status(*, run: Runner = _run, base: Path | None = None, env: dict[str, str] | None = None) -> dict:
    """``{installed, plist, loaded, launchd_pid, server: {pid, port} | None, healthy}``."""
    p = plist_path(base)
    r = run(["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"])
    pid = next((ln.split("=", 1)[1].strip() for ln in (r.stdout or "").splitlines() if ln.strip().startswith("pid =")),
               None)
    e = environment(env)
    holder = instance_lock.holder(Path(e["FORGE_DATA"]))
    healthy = False
    if holder:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{holder.get('port')}/api/health", timeout=3) as h:
                healthy = h.status == 200
        except OSError:
            healthy = False
    return {"installed": p.exists(), "plist": str(p), "loaded": r.returncode == 0, "launchd_pid": pid,
            "data": e["FORGE_DATA"], "server": holder, "healthy": healthy}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge", description="Forge command line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    svc = sub.add_parser("service", help="Forge as a launchd agent (macOS)")
    act = svc.add_subparsers(dest="action", required=True)
    act.add_parser("install").add_argument("--dry-run", action="store_true", help="print the plist, change nothing")
    act.add_parser("uninstall")
    act.add_parser("status")
    a = ap.parse_args(argv)
    try:
        if a.action == "install":
            print(install(dry_run=a.dry_run))
        elif a.action == "uninstall":
            print(uninstall())
        else:
            print(json.dumps(status(), indent=1))
    except RuntimeError as e:
        print(f"forge service {a.action}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
