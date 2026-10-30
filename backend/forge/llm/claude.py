"""Claude via the Agent SDK, authenticated with the local `claude` CLI's subscription login.

ANTHROPIC_API_KEY is blanked in the subprocess env so the CLI never silently bills an API key.
Structured output uses `output_format={"type": "json_schema"}` -> `ResultMessage.structured_output`.

`explore` runs an agentic session that reads the repository only through forge's own repo-confined
tools (list_files, grep, read_file; see ``openai.RepoTools``), served in-process over MCP. Claude
Code's built-in Read/Grep/Glob are switched off: they accept absolute paths, so a prompt-injected
README in a cloned repo could have them read any file the user can (verified). A PreToolUse hook
denies every tool that isn't one of the three before it runs, so a tool Claude Code adds later is
refused too. The session's cwd is an empty temp dir, never the repository.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from forge.llm.base import LLMError, OnDelta, OnEvent, T, short, traced

REPO_SERVER = "repo"
REPO_TOOLS = [f"mcp__{REPO_SERVER}__{n}" for n in ("list_files", "grep", "read_file")]
FORBIDDEN_TOOLS = ["Read", "Grep", "Glob", "Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "BashOutput",
                   "KillShell", "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite", "Skill", "LS",
                   "NotebookRead", "ListMcpResourcesTool", "ReadMcpResourceTool"]


def _repo_server(repo_path: str):
    """In-process MCP server exposing the confined repo tools for one repository."""
    import asyncio
    from pathlib import Path

    from claude_agent_sdk import create_sdk_mcp_server, tool

    from forge.llm.openai import TOOLS, RepoTools

    tools = RepoTools(Path(repo_path))

    def make(spec: dict):
        name = spec["function"]["name"]

        @tool(name, spec["function"]["description"], spec["function"]["parameters"])
        async def run(args: dict[str, Any]) -> dict[str, Any]:
            out, _label = await asyncio.to_thread(tools.run, name, args or {})
            return {"content": [{"type": "text", "text": out}],
                    "is_error": out.startswith(("error:", "invalid regex"))}
        return run

    return create_sdk_mcp_server(name=REPO_SERVER, tools=[make(t) for t in TOOLS])


async def _only_repo_tools(input_data: dict, tool_use_id, context) -> dict:
    """PreToolUse hook: deny anything but the confined repo tools, before it executes."""
    name = (input_data or {}).get("tool_name", "")
    if name in REPO_TOOLS or name == "StructuredOutput":
        return {}
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": f"{name} is not available; use the repo tools "
                                   "(list_files, grep, read_file) with paths relative to the repository root."}}


async def _maybe_await(x):
    if inspect.isawaitable(x):
        await x


def _env() -> dict[str, str]:
    # Subscription login only: never let an inherited API key take over billing.
    return {"ANTHROPIC_API_KEY": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


class ClaudeLLM:
    provider = "claude-subscription"

    def __init__(self, model: str = "default", max_turns: int = 40):
        self.model = model
        self.max_turns = max_turns
        self.total_cost_usd = 0.0

    def _options(self, **kw):
        from claude_agent_sdk import ClaudeAgentOptions

        # "default" lets the CLI use its own recommended model
        base: dict[str, Any] = dict(model=None if self.model == "default" else self.model, env=_env(), setting_sources=[],
                                    disallowed_tools=FORBIDDEN_TOOLS)
        base.update(kw)
        return ClaudeAgentOptions(**base)

    async def _run(self, prompt: str, options, on_event: OnEvent | None = None,
                   on_delta: OnDelta | None = None) -> tuple[Any, str, list[str]]:
        """Run one query; return (structured_output, final_text, transcript_snippets)."""
        from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent, TextBlock,
                                      ToolResultBlock, ToolUseBlock, UserMessage, query)

        pending: dict[str, tuple[str, dict]] = {}
        texts: list[str] = []
        transcript: list[str] = []
        result = None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, StreamEvent) and on_delta:
                ev = msg.event or {}
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    await _maybe_await(on_delta(ev["delta"].get("text", "")))
            elif isinstance(msg, AssistantMessage):
                if msg.error:
                    raise LLMError(f"Claude error: {msg.error}")
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        texts.append(b.text)
                        transcript.append(f"[assistant] {b.text[:1500]}")
                        if on_event:
                            await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "info",
                                                         "message": short(b.text, 1200)}))
                    elif isinstance(b, ToolUseBlock) and b.name != "StructuredOutput":
                        pending[b.id] = (b.name, b.input)
            elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for b in msg.content:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in pending:
                        name, inp = pending.pop(b.tool_use_id)
                        out = b.content
                        if isinstance(out, list):
                            out = " ".join(str(x.get("text", "")) for x in out if isinstance(x, dict))
                        out = out or ""
                        transcript.append(f"[{name} {short(inp, 200)}] {str(out)[:1500]}")
                        if on_event:
                            await _maybe_await(on_event({
                                "type": "tool", "name": name.removeprefix(f"mcp__{REPO_SERVER}__"),
                                "input": _tool_input(name, inp),
                                "output": ("ERROR " if b.is_error else "") + short(out, 300)}))
            elif isinstance(msg, ResultMessage):
                result = msg
        if result is None:
            raise LLMError("Claude session ended without a result")
        self.total_cost_usd += result.total_cost_usd or 0.0
        if result.is_error and result.structured_output is None:
            raise LLMError(f"Claude session failed: {result.subtype} {result.errors or result.result or ''}")
        return result.structured_output, (result.result or "\n".join(texts)), transcript

    @traced("structured")
    async def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        opts = self._options(tools=[], max_turns=4, system_prompt=system,
                             output_format={"type": "json_schema", "schema": schema.model_json_schema()})
        last_err = None
        for _ in range(2):
            out, _text, _ = await self._run(prompt if last_err is None else
                                            f"{prompt}\n\nYour previous output failed validation: {last_err}. Fix it.",
                                            opts)
            if out is None:
                last_err = "no structured output was returned"
                continue
            try:
                return schema.model_validate(out)
            except ValidationError as e:
                last_err = short(str(e), 1500)
        raise LLMError(f"structured output invalid: {last_err}")

    @traced("chat")
    async def chat(self, messages: list[dict], system: str | None = None, on_delta: OnDelta | None = None) -> str:
        convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[:-1])
        prompt = (f"Conversation so far:\n{convo}\n\nUSER: {messages[-1]['content']}" if convo
                  else messages[-1]["content"])
        opts = self._options(tools=[], max_turns=1, system_prompt=system,
                             include_partial_messages=on_delta is not None)
        _, text, _ = await self._run(prompt, opts, on_delta=on_delta)
        return text.strip()

    @traced("explore")
    async def explore(self, task: str, repo_path: str | None, schema: type[T], on_event: OnEvent,
                      system: str | None = None) -> T:
        if not repo_path:
            return await self.structured(task, schema, system=system)
        import tempfile

        from claude_agent_sdk import HookMatcher
        guide = ("Read the repository only with the repo tools (list_files, grep, read_file); paths are "
                 "relative to the repository root.")
        with tempfile.TemporaryDirectory(prefix="forge-claude-") as work:
            opts = self._options(
                tools=[], allowed_tools=REPO_TOOLS, permission_mode="dontAsk", cwd=work,
                mcp_servers={REPO_SERVER: _repo_server(repo_path)}, strict_mcp_config=True,
                hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_only_repo_tools])]},
                max_turns=self.max_turns, system_prompt=f"{system}\n\n{guide}" if system else guide,
                output_format={"type": "json_schema", "schema": schema.model_json_schema()})
            try:
                out, text, transcript = await self._run(task, opts, on_event=on_event)
            except LLMError as e:
                # e.g. max turns hit before the final answer: salvage from what it found.
                await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "warn",
                                             "message": f"exploration ended early ({e}); summarising findings"}))
                out, transcript = None, []
        if out is not None:
            try:
                return schema.model_validate(out)
            except ValidationError as e:
                return await self.structured(
                    f"{task}\n\nA draft answer failed validation ({short(str(e), 1200)}). Draft:\n{out}\n"
                    "Return a corrected, complete answer.", schema, system=system)
        notes = "\n".join(transcript)[-30000:]
        return await self.structured(f"{task}\n\nFindings from exploring the repository:\n{notes}", schema,
                                     system=system)


def _tool_input(name: str, inp: dict) -> str:
    if name.startswith(f"mcp__{REPO_SERVER}__"):
        return short(" ".join(str(v) for v in (inp.get("path"), inp.get("pattern"), inp.get("glob"),
                                               inp.get("offset")) if v) or "*", 200)
    if name == "Read":
        s = inp.get("file_path", "")
        if inp.get("offset"):
            s += f" @{inp['offset']}"
        return short(s, 200)
    if name == "Grep":
        return short(f"{inp.get('pattern', '')}  {inp.get('path', '') or ''} {inp.get('glob', '') or ''}".strip(), 200)
    if name == "Glob":
        return short(inp.get("pattern", ""), 200)
    return short(inp, 200)


def available() -> bool:
    import shutil
    return shutil.which("claude") is not None or os.path.exists(os.path.expanduser("~/.local/bin/claude"))
