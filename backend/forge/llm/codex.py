"""Codex via the local ``codex`` CLI, authenticated with its ChatGPT subscription login.

The counterpart of ``claude.py``: no API key is used (OPENAI_API_KEY / CODEX_API_KEY are removed from
the subprocess env so the CLI can't silently bill one). Each call is one ``codex exec --json`` run.

Confinement. Codex's ``read-only`` sandbox blocks writes and network but not reads, and its shell
can ``cat`` anything on disk; with repositories cloned from untrusted URLs a prompt-injected README
could exfiltrate ``~/.ssh``. So every run:
- switches off the shell (``shell_tool``, ``unified_exec``) and every other built-in that reaches
  outside (browser, computer use, apps, plugins, web search, image viewing, memories);
- keeps code mode, a JS isolate with no fs/process/network that can only call tools (verified:
  ``require``/``import``/``fetch``/``process`` are all unavailable);
- serves the repository through ``forge.llm.repo_mcp`` (list_files, grep, read_file confined by
  ``repo.safe_path``) as the only way to read code;
- runs in an empty temp dir, with ``--ignore-user-config``, ``--ignore-rules`` and ``--ephemeral``,
  so neither the user's Codex config/MCP servers nor the repo's AGENTS.md are loaded.

Structured output uses ``--output-schema`` (strict JSON schema), validated with Pydantic and re-asked
once on failure. ``codex exec`` doesn't stream tokens, so chat delivers its reply in one delta.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forge.llm.base import LLMError, OnDelta, OnEvent, T, short, traced

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Built-ins that could read, write or reach beyond the repository tools we provide.
# Every feature enabled by default in codex 0.161 that forge doesn't need is switched off: the outward
# ones (shell, browser, computer use, apps, plugins, web, images, memories) and everything else too
# (hooks, multi-agent, worktrees, workspace dependencies, voice/UI features ...), so the remaining
# surface is exactly ALLOWED_ENABLED below.
DISABLED_FEATURES = [
    "shell_tool", "unified_exec", "shell_snapshot", "unified_exec_tty", "unified_exec_zsh_fork", "write_stdin_approval",
    "apps", "plugins", "remote_plugin", "plugin_sharing", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "browser_annotation_api", "in_app_browser", "computer_use",
    "image_generation", "view_image", "memories", "tool_suggest", "goals", "skill_mcp_dependency_install",
    "skill_search", "hooks", "multi_agent", "worktrees", "workspace_dependencies", "in_app_local_automation",
    "in_app_chat", "in_app_dictation", "in_app_updates", "in_app_voice", "realtime_conversation",
    "mentions_v2", "sleep_tool", "guardian_approval", "guardian_reuse_parent_compaction", "auth_elicitation",
    "tool_call_mcp_elicitation", "api_key_model_discovery", "daemon_auto_start", "system_proxy_fallback",
    "unbounded_connection_retries", "fast_mode", "compaction_image_budget", "content_item_kinds",
    "enable_request_compression",
]
# The only non-"removed" features allowed to be on, each vetted against codex 0.161:
ALLOWED_ENABLED = {
    # needed: Codex reaches MCP tools through it. A JS isolate with no fs, process, network or imports
    # (require/import/fetch/process/Deno all unavailable; globals are tools, text, timers, store, exit).
    "code_mode_host",
}
# Disabled features that codex 0.161 ignores --disable for. We still always pass the flag (a later
# Codex may honour it); reading true is tolerated only while the feature it depends on reads false.
# unified_exec only backs shell_tool: with shell_tool off the tool inventory has no exec/shell tool
# (apply_patch, clock, MCP resource helpers, mcp__repo__*).
DISABLE_IGNORED = {"unified_exec": "shell_tool"}
assert not ALLOWED_ENABLED & set(DISABLED_FEATURES), "a feature can't be both disabled and allowed"
assert set(DISABLE_IGNORED) <= set(DISABLED_FEATURES) and set(DISABLE_IGNORED.values()) <= set(DISABLED_FEATURES)
SCRUB_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")

# Fail closed. Codex updates itself, so the flags above are checked rather than trusted:
# 1. Preventive, before any run (cached per codex binary + version): `codex features list`, taken with
#    exactly our --disable flags and an empty CODEX_HOME (the same defaults `exec --ignore-user-config`
#    runs with), must succeed (an unknown --disable name makes Codex exit non-zero), every disabled
#    feature must read false, and every other enabled feature must be in ALLOWED_ENABLED or at stage
#    "removed". Any new or unvetted feature therefore refuses all Codex runs until it is reviewed.
# 2. Backstop at run time (detective, not preventive): only these event item types are tolerated;
#    anything else (command_execution, file_change, web_search, a new type, or an MCP call to another
#    server) kills the process as soon as the item starts and fails the call.
ALLOWED_ITEMS = {"agent_message", "reasoning", "mcp_tool_call", "todo_list", "error"}
MCP_SERVER = "repo"
_preflight: dict[tuple[str, str], str | None] = {}   # (binary, version) -> None (ok) or why runs are refused


async def _maybe_await(x):
    if inspect.isawaitable(x):
        await x


def binary() -> str | None:
    return shutil.which("codex") or next((p for p in (os.path.expanduser("~/.local/bin/codex"),) if os.path.exists(p)), None)


def env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}


def _toml(s: str) -> str:
    return json.dumps(s)  # a JSON string literal is a valid TOML basic string


def strict_schema(schema_model) -> dict:
    """The JSON schema in the strict form OpenAI structured outputs require."""
    from openai.lib._parsing._completions import type_to_response_format_param
    return type_to_response_format_param(schema_model)["json_schema"]["schema"]  # type: ignore[index]


class CodexLLM:
    provider = "codex-subscription"

    def __init__(self, model: str = "default", timeout: float = 1800):
        self.model = model
        self.timeout = timeout
        self.tool_calls = 0

    def _cmd(self, workdir: str, schema_path: str | None, repo_path: str | None) -> list[str]:
        exe = binary()
        if not exe:
            raise LLMError("codex CLI not found. Install Codex and run `codex login`.")
        cmd = [exe, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
               "--ignore-rules", "--sandbox", "read-only", "--color", "never",
               "-c", 'approval_policy="never"', "-c", 'web_search="disabled"', "-C", workdir]
        for f in DISABLED_FEATURES:
            cmd += ["--disable", f]
        if self.model and self.model != "default":
            cmd += ["-m", self.model]
        if schema_path:
            cmd += ["--output-schema", schema_path]
        if repo_path:
            cmd += ["-c", f"mcp_servers.repo.command={_toml(sys.executable)}",
                    "-c", f"mcp_servers.repo.args=[{_toml('-m')}, {_toml('forge.llm.repo_mcp')}, {_toml(str(repo_path))}]",
                    "-c", f"mcp_servers.repo.cwd={_toml(str(BACKEND_DIR))}"]
        return cmd + ["-"]

    async def _run(self, prompt: str, schema: dict | None = None, repo_path: str | None = None,
                   on_event: OnEvent | None = None) -> str:
        """One ``codex exec`` run; returns the final agent message (repo tool calls go to self.tool_calls)."""
        self.tool_calls = 0
        reason = await preflight()
        if reason:
            raise LLMError(reason)
        with tempfile.TemporaryDirectory(prefix="forge-codex-") as work:
            schema_path = None
            if schema is not None:
                schema_path = os.path.join(work, ".schema.json")
                Path(schema_path).write_text(json.dumps(schema))
            proc = await asyncio.create_subprocess_exec(
                *self._cmd(work, schema_path, repo_path), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env(), cwd=work,
                limit=16 * 1024 * 1024)
            assert proc.stdin and proc.stdout and proc.stderr
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            last_msg, failure = "", None
            try:
                async with asyncio.timeout(self.timeout):
                    async for raw in proc.stdout:
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        kind, item = ev.get("type"), ev.get("item") or {}
                        if kind in ("item.started", "item.updated", "item.completed"):
                            bad = _disallowed(item)
                            if bad:
                                proc.kill()
                                await proc.wait()
                                raise LLMError(f"codex tried a tool outside the allowed set ({bad}); run aborted. "
                                               "A Codex update may have re-enabled it: see forge/llm/codex.py.")
                        if kind == "item.completed" and item.get("type") == "agent_message":
                            last_msg = item.get("text") or ""
                            if on_event and last_msg.strip():
                                await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "info",
                                                             "message": short(last_msg, 1200)}))
                        elif kind == "item.completed" and item.get("type") == "mcp_tool_call":
                            self.tool_calls += 1
                            if not on_event:
                                continue
                            res = item.get("result") or {}
                            text = " ".join(c.get("text", "") for c in res.get("content", []) if isinstance(c, dict))
                            await _maybe_await(on_event({
                                "type": "tool", "name": item.get("tool", "tool"), "input": _label(item),
                                "output": ("ERROR " if item.get("error") else "") + short(text or item.get("error") or "", 300)}))
                        elif kind in ("turn.failed", "error"):
                            err = ev.get("error") or {}
                            failure = (err.get("message") if isinstance(err, dict) else None) or ev.get("message") or str(ev)
                    rc = await proc.wait()
            except TimeoutError:
                proc.kill()
                raise LLMError(f"codex timed out after {self.timeout:.0f}s") from None
            stderr = (await proc.stderr.read()).decode(errors="ignore")
            if rc != 0 or (failure and not last_msg):
                tail = failure or stderr.strip().splitlines()[-1:] or [f"exit {rc}"]
                raise LLMError(f"codex failed: {short(tail if isinstance(tail, str) else tail[0], 600)}")
            return last_msg

    async def _structured(self, prompt: str, schema_model: type[T], repo_path: str | None = None,
                          on_event: OnEvent | None = None) -> T:
        schema = strict_schema(schema_model)
        last = None
        seen = 0
        for _ in range(2):
            text = await self._run(prompt if last is None else
                                   f"{prompt}\n\nYour previous answer failed validation: {last}. Return corrected JSON.",
                                   schema, repo_path, on_event)
            seen += self.tool_calls
            self.tool_calls = seen
            try:
                return schema_model.model_validate_json(_strip_fence(text))
            except ValidationError as e:
                last = short(str(e), 1200)
        raise LLMError(f"codex structured output invalid: {last}")

    @traced("structured")
    async def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        return await self._structured(_with_system(prompt, system), schema)

    @traced("chat")
    async def chat(self, messages: list[dict], system: str | None = None, on_delta: OnDelta | None = None) -> str:
        convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[:-1])
        prompt = (f"Conversation so far:\n{convo}\n\nUSER: {messages[-1]['content']}" if convo
                  else messages[-1]["content"])
        prompt += "\n\nReply to the user directly. Do not call any tools."
        text = (await self._run(_with_system(prompt, system))).strip()
        if on_delta and text:
            await _maybe_await(on_delta(text))
        return text

    @traced("explore")
    async def explore(self, task: str, repo_path: str | None, schema: type[T], on_event: OnEvent,
                      system: str | None = None) -> T:
        if not repo_path:
            return await self.structured(task, schema, system=system)
        guide = ("Explore the repository only with the repo tools (list_files, grep, read_file); paths are "
                 "relative to the repository root. Investigate until you can answer, citing real file paths "
                 "and line numbers you have seen, then give the final answer.")
        prompt = _with_system(f"{task}\n\n{guide}", system)
        # Codex occasionally starts a session without discovering the MCP tools and then answers blind.
        # An analysis that never looked at the code is worthless, so retry once, then fail loudly.
        for attempt in range(2):
            out = await self._structured(prompt, schema, repo_path, on_event)
            if self.tool_calls:
                return out
            await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "warn",
                                         "message": "codex answered without reading the repository; retrying"}))
            prompt = _with_system(f"{task}\n\n{guide}\n\nYou must call the repo tools (they are exposed as "
                                  "mcp__repo__list_files, mcp__repo__grep and mcp__repo__read_file) before "
                                  "answering.", system)
        raise LLMError("codex could not reach the repository tools in two attempts; try again or use another provider")


def _disallowed(item: dict) -> str | None:
    t = item.get("type")
    if t not in ALLOWED_ITEMS:
        return str(t)
    if t == "mcp_tool_call" and item.get("server") != MCP_SERVER:
        return f"mcp:{item.get('server')}"
    return None


async def version() -> str:
    exe = binary()
    if not exe:
        return ""
    p = await asyncio.create_subprocess_exec(exe, "--version", stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.DEVNULL, env=env())
    return (await p.communicate())[0].decode(errors="ignore").strip()


def check_features(listing: str) -> str | None:
    """Refusal reason for a ``codex features list`` output taken with our flags, or None if safe."""
    state: dict[str, tuple[str, bool]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] in ("true", "false"):
            state[parts[0]] = (" ".join(parts[1:-1]), parts[-1] == "true")
    missing = [f for f in DISABLED_FEATURES if f not in state]
    if missing:
        return f"Codex no longer has features {missing}; refusing to run until forge's sandbox flags are updated."
    still_on = [f for f in DISABLED_FEATURES if state[f][1]
                and not (f in DISABLE_IGNORED and not state[DISABLE_IGNORED[f]][1])]
    if still_on:
        return f"Codex ignored --disable for {still_on}; refusing to run."
    unvetted = sorted(f for f, (stage, on) in state.items()
                      if on and f not in ALLOWED_ENABLED and f not in DISABLED_FEATURES and stage != "removed")
    if unvetted:
        return (f"Codex has enabled features forge hasn't vetted: {unvetted}. Refusing to run until each is "
                "reviewed and added to DISABLED_FEATURES or ALLOWED_ENABLED in forge/llm/codex.py.")
    return None


async def preflight() -> str | None:
    """Cached per Codex binary and version: None when the confinement holds, else why runs are refused."""
    exe = binary()
    if not exe:
        return "codex CLI not found. Install Codex and run `codex login`."
    key = (exe, await version())
    if key in _preflight:
        return _preflight[key]
    args = [exe]
    for f in DISABLED_FEATURES:
        args += ["--disable", f]
    with tempfile.TemporaryDirectory(prefix="forge-codex-home-") as home:
        # empty CODEX_HOME: evaluate the defaults that `exec --ignore-user-config` runs with, not the
        # user's config.toml (features list has no --ignore-user-config of its own)
        p = await asyncio.create_subprocess_exec(*args, "features", "list", stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT, env={**env(), "CODEX_HOME": home})
        out = (await p.communicate())[0].decode(errors="ignore")
    if p.returncode != 0:
        reason = ("Codex rejected forge's sandbox flags (a feature was probably renamed or removed); refusing "
                  f"to run. codex said: {short(out.strip().splitlines()[0] if out.strip() else 'nothing', 300)}")
    else:
        reason = check_features(out)
    _preflight[key] = reason
    return reason


def _with_system(prompt: str, system: str | None) -> str:
    return f"{system}\n\n---\n\n{prompt}" if system else prompt


def _label(item: dict) -> str:
    a = item.get("arguments") or {}
    if isinstance(a, dict):
        return short(" ".join(str(v) for v in (a.get("path"), a.get("pattern"), a.get("glob"), a.get("offset")) if v) or "*", 200)
    return short(a, 200)


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return s.strip()


async def login_status() -> tuple[bool, str]:
    exe = binary()
    if not exe:
        return False, "No codex CLI found. Install Codex and run `codex login`."
    proc = await asyncio.create_subprocess_exec(exe, "login", "status", stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, env=env())
    out = (await proc.communicate())[0].decode(errors="ignore").strip()
    if proc.returncode != 0 or "logged in" not in out.lower():
        return False, "Codex is not logged in. Run `codex login` and sign in with ChatGPT."
    return True, "Using your local codex CLI login" + (" (ChatGPT)." if "chatgpt" in out.lower() else ".")


async def models() -> list[dict]:
    """The models the logged-in account can use, from ``codex app-server``'s ``model/list``."""
    exe = binary()
    if not exe:
        return []
    proc = await asyncio.create_subprocess_exec(exe, "app-server", stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                                                env=env(), limit=16 * 1024 * 1024)
    assert proc.stdin and proc.stdout

    async def send(o: dict) -> None:
        proc.stdin.write((json.dumps(o) + "\n").encode())  # type: ignore[union-attr]
        await proc.stdin.drain()  # type: ignore[union-attr]

    async def recv(want: int) -> dict[str, Any]:
        async for line in proc.stdout:  # type: ignore[union-attr]
            m = json.loads(line)
            if m.get("id") == want:
                return m
        raise LLMError("codex app-server closed")

    try:
        await send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"clientInfo": {"name": "clm-forge", "version": "1"}}})
        await recv(1)
        await send({"jsonrpc": "2.0", "method": "initialized"})
        await send({"jsonrpc": "2.0", "id": 2, "method": "model/list", "params": {}})
        r = await recv(2)
    finally:
        proc.kill()
        await proc.wait()
    if r.get("error"):
        raise LLMError(str(r["error"].get("message", r["error"])))
    data = [m for m in (r.get("result") or {}).get("data", []) if not m.get("hidden")]
    data.sort(key=lambda m: not m.get("isDefault"))
    return [{"id": m.get("model") or m["id"], "label": m.get("displayName") or m["id"],
             "description": m.get("description")} for m in data]
