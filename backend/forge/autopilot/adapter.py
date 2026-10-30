"""Subprocess client for the head-set adapter contract (SPEC "Autopilot adapter contract").

``<preset adapter argv> <verb> --target-config <json> [--run <id>] [--flag value]...``

- stdout is JSON lines: ``{"event": "progress", "verb", "message", ...}`` any number of times, then exactly
  one ``{"event": "result", "verb", "data": {...}}``; ``call`` returns ``data``;
- a non-zero exit is a failure, stderr (ending ``ERROR <type>: <message>``) says why.

Fail-fast: a non-zero exit raises ``AdapterError`` (the verb failed: the runner fails that head and goes on);
a line that is not a JSON object with ``event`` progress|result, a result without a ``data`` object, a second
result or no result at all raise ``AdapterContractError`` (the adapter is broken: the run fails). Each verb is
its own process, so a model it loads is released when it exits. Cancelling a call terminates the child.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Progress = Callable[[dict], None]
STDOUT_LIMIT = 64 * 1024 * 1024


class AdapterError(RuntimeError):
    """The adapter failed: the verb exited non-zero (stderr says why). The runner isolates this per head."""

    def __init__(self, message: str, *, verb: str | None = None, stderr: str = ""):
        super().__init__(message)
        self.verb, self.stderr = verb, stderr


class AdapterContractError(AdapterError):
    """The adapter broke the output contract (malformed lines, no or two results). Never isolated per head: the
    adapter itself is wrong, so the whole run fails."""


def flags(kw: dict[str, Any]) -> list[str]:
    """``{"head": "x", "adjudicate": True, "heads": ["a", "b"], "focus_template": ("t1", "t2"), "n": 3}`` ->
    ``--head x --adjudicate --heads a,b --focus-template t1 --focus-template t2 --n 3``. A list is one
    comma-joined value, a tuple repeats the flag (argparse ``append``); False / None / empty are omitted."""
    out: list[str] = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        name = "--" + k.replace("_", "-")
        if v is True:
            out.append(name)
        elif isinstance(v, tuple):
            for x in v:
                out += [name, str(x)]
        elif isinstance(v, list):
            if v:
                out += [name, ",".join(map(str, v))]
        else:
            out += [name, str(v)]
    return out


def failure_line(stderr: str) -> str:
    """What the journal may keep of a failed verb's stderr: the adapter's own last ``ERROR <Type>: <message>``
    line (its messages name ids, counts and reasons, never row text), cut to 600 characters. Everything else on
    stderr (library output, a teacher CLI's reply) can quote real rows and stays in the local log file."""
    lines = [x for x in stderr.splitlines() if x.startswith("ERROR ")]
    if not lines:
        return "(no ERROR line on stderr)"
    line = lines[-1]
    return line if len(line) <= 600 else line[:600] + f"… ({len(line)} chars)"


def log_dir() -> Path:
    from forge import paths
    return paths.DATA_DIR / "logs" / "autopilot"


def keep_stderr(verb: str, run_id: str | None, stderr: str) -> str | None:
    """The full stderr of a failed verb, kept for debugging in ``<FORGE_DATA>/logs/autopilot/`` (mode 0600; it may
    quote real rows, so it stays local, like the rows). -> its path."""
    if not stderr:
        return None
    d = log_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{time.strftime('%Y%m%dT%H%M%S')}-{run_id or 'adhoc'}-{verb}.stderr.log"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(stderr)
    return str(p)


class Adapter:
    def __init__(self, argv: list[str], cwd: str, env: dict[str, str] | None, target_config: dict):
        self.argv, self.cwd, self.env, self.target_config = list(argv), cwd, dict(env or {}), target_config

    def command(self, verb: str, run_id: str | None = None, config_overrides: dict | None = None,
                **kw: Any) -> list[str]:
        cfg = {**self.target_config, **(config_overrides or {})}
        cmd = [*self.argv, verb, "--target-config", json.dumps(cfg)]
        if run_id:
            cmd += ["--run", run_id]
        return cmd + flags(kw)

    async def call(self, verb: str, *, run_id: str | None = None, on_progress: Progress | None = None,
                   config_overrides: dict | None = None, **kw: Any) -> dict:
        """Run one verb; ``config_overrides`` replace target-config keys for this call only."""
        cmd = self.command(verb, run_id, config_overrides, **kw)
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.cwd, env={**os.environ, **self.env}, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=STDOUT_LIMIT)
        err_chunks: list[bytes] = []

        async def drain_err():
            assert proc.stderr
            while chunk := await proc.stderr.read(65536):
                err_chunks.append(chunk)
                if sum(map(len, err_chunks)) > 1_000_000:
                    err_chunks.pop(0)

        err_task = asyncio.create_task(drain_err())
        result: dict | None = None
        problem: str | None = None
        n = 0
        try:
            assert proc.stdout
            async for raw in proc.stdout:
                n += 1
                line = raw.decode(errors="replace").strip()
                if not line or problem:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:       # the line is not shown: it may hold data (it is in the output log)
                    problem = f"line {n} is not JSON ({len(line)} characters)"
                    err_chunks.append(f"\n[stdout line {n}, not JSON] {line}\n".encode())
                    continue
                ev = msg.get("event") if isinstance(msg, dict) else None
                if ev == "progress":
                    if on_progress:
                        on_progress({k: v for k, v in msg.items() if k != "event"})
                elif ev == "result":
                    if result is not None:
                        problem = f"line {n}: a second result"
                    elif not isinstance(msg.get("data"), dict):
                        problem = f"line {n}: the result has no data object"
                    else:
                        result = msg["data"]
                else:
                    problem = f"line {n}: event must be progress or result, got {ev!r}"
            rc = await proc.wait()
            await err_task
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 30)
                except asyncio.TimeoutError:
                    proc.kill()
            err_task.cancel()
            raise
        stderr = b"".join(err_chunks).decode(errors="replace").strip()
        if rc != 0:
            why = failure_line(stderr)
            log = keep_stderr(verb, run_id, stderr)
            raise AdapterError(f"adapter {verb} exited {rc}: {why}" + (f" (full output: {log})" if log else ""),
                               verb=verb, stderr=why + (f" (full output: {log})" if log else ""))
        if problem:
            log = keep_stderr(verb, run_id, stderr)
            raise AdapterContractError(f"adapter {verb} broke the output contract: {problem}"
                                       + (f" (full output: {log})" if log else ""))
        if result is None:
            raise AdapterContractError(f"adapter {verb} exited 0 without a result line")
        return result
