"""OpenAI provider (and any OpenAI-compatible server) via the Chat Completions API.

Auth: the API key from Settings, else ``OPENAI_API_KEY``. ``openai_base_url`` points it at a
compatible server instead (vLLM, LM Studio, a gateway); empty means api.openai.com.

Structured output asks for a strict ``json_schema`` response format first; schemas that strict mode
rejects, and servers without it, fall back to JSON mode with the schema in the prompt. Either way the
result is validated with Pydantic and re-asked on failure.

``explore`` is a real tool loop over three read-only tools (list_files, grep, read_file) executed here
in Python against the repository, every path checked by ``repo.safe_path``. The model never gets a
shell or write access; the loop is bounded by ``max_turns``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forge.llm.base import LLMError, OnDelta, OnEvent, T, short, traced

MAX_READ_LINES = 400
MAX_TOOL_CHARS = 16_000
MAX_GREP_HITS = 60
MAX_PATTERN = 500
MAX_LINE_CHARS = 2000
GREP_BUDGET_S = 5.0
#: one line's match budget: the ``regex`` module checks its timeout only now and then and overshoots it by ~40%
#: (5 s asked took 7 s here), so a short per-line slice keeps the whole search within ~GREP_BUDGET_S + a fraction
#: of a second; a line that needs more than this is a catastrophic pattern anyway
GREP_LINE_BUDGET_S = 0.5

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List repository files (relative paths). Optional glob filter such as '**/*.py' or 'src/*'.",
        "parameters": {"type": "object", "properties": {"glob": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents with a Python regular expression. Returns path:line: text matches.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "glob": {"type": "string", "description": "optional file filter, e.g. '**/*.ts'"}},
            "required": ["pattern"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"Read a text file by relative path. Returns numbered lines, at most {MAX_READ_LINES} per call.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line (default 1)"},
            "limit": {"type": "integer", "description": f"number of lines (default and max {MAX_READ_LINES})"}},
            "required": ["path"], "additionalProperties": False}}},
]


async def _maybe_await(x):
    if inspect.isawaitable(x):
        await x


def norm_url(u: str | None) -> str:
    return (u or "").strip().rstrip("/")


def api_key(settings=None) -> str:
    """The key to send to the configured base URL, or "".

    A saved key goes only to the URL it was entered for, and OPENAI_API_KEY only to api.openai.com.
    Otherwise anyone able to change ``openai_base_url`` (or a user pointing it at a local server)
    would have the OpenAI key sent to that server."""
    s = settings
    if s is None:
        from forge import config
        s = config.load_settings()
    base = norm_url(getattr(s, "openai_base_url", ""))
    saved = (getattr(s, "openai_api_key", "") or "").strip()
    if saved and norm_url(getattr(s, "openai_api_key_url", "")) == base:
        return saved
    if not base:
        return os.environ.get("OPENAI_API_KEY", "").strip()
    return ""


def make_client(settings=None, timeout: float = 900):
    from openai import AsyncOpenAI

    s = settings
    if s is None:
        from forge import config
        s = config.load_settings()
    base = (getattr(s, "openai_base_url", "") or "").strip() or None
    # Compatible servers often need no key; the SDK still insists on a non-empty one.
    return AsyncOpenAI(api_key=api_key(s) or ("unused" if base else None), base_url=base, timeout=timeout,
                       max_retries=3)


class OpenAILLM:
    provider = "openai"

    def __init__(self, model: str, settings=None, max_turns: int = 40):
        self.model = model
        self.settings = settings
        self.max_turns = max_turns

    def _client(self):
        if not api_key(self.settings) and not (getattr(self.settings, "openai_base_url", "") or "").strip():
            raise LLMError("No OpenAI API key: add one in Settings or set OPENAI_API_KEY.")
        return make_client(self.settings)

    @staticmethod
    def _msgs(prompt_or_msgs, system: str | None) -> list[dict]:
        msgs = [{"role": "user", "content": prompt_or_msgs}] if isinstance(prompt_or_msgs, str) else list(prompt_or_msgs)
        return ([{"role": "system", "content": system}] if system else []) + msgs

    async def _complete(self, client, **kw):
        try:
            return await client.chat.completions.create(model=self.model, **kw)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"OpenAI call failed: {_err(e)}") from e

    async def _structured_msgs(self, msgs: list[dict], schema: type[T]) -> T:
        client = self._client()
        strict = True
        last = None
        for _ in range(3):
            content = None
            if strict:
                try:
                    r = await client.chat.completions.parse(model=self.model, messages=msgs, response_format=schema)
                    m = r.choices[0].message
                    if m.parsed is not None:
                        return m.parsed  # type: ignore[return-value]
                    if getattr(m, "refusal", None):
                        raise LLMError(f"OpenAI refused: {m.refusal}")
                    content = m.content or ""
                except LLMError:
                    raise
                except ValidationError as e:
                    last = short(str(e), 1200)
                    content = None
                except Exception:  # noqa: BLE001 - strict schema or server unsupported: JSON mode instead
                    strict = False
            if content is None and not strict:
                hint = ("Respond only with a JSON object matching this JSON schema:\n"
                        + json.dumps(schema.model_json_schema()))
                r = await self._complete(client, messages=msgs + [{"role": "system", "content": hint}],
                                         response_format={"type": "json_object"})
                content = r.choices[0].message.content or ""
            if content is not None:
                try:
                    return schema.model_validate_json(_strip_fence(content))
                except ValidationError as e:
                    last = short(str(e), 1200)
                msgs = msgs + [{"role": "assistant", "content": content},
                               {"role": "user", "content": f"That failed validation: {last}. Return corrected JSON."}]
        raise LLMError(f"OpenAI structured output invalid: {last}")

    @traced("structured")
    async def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        return await self._structured_msgs(self._msgs(prompt, system), schema)

    @traced("chat")
    async def chat(self, messages: list[dict], system: str | None = None, on_delta: OnDelta | None = None) -> str:
        client = self._client()
        msgs = self._msgs(messages, system)
        if on_delta is None:
            r = await self._complete(client, messages=msgs)
            return (r.choices[0].message.content or "").strip()
        out: list[str] = []
        stream = await self._complete(client, messages=msgs, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta.content or ""
            if d:
                out.append(d)
                await _maybe_await(on_delta(d))
        return "".join(out).strip()

    @traced("explore")
    async def explore(self, task: str, repo_path: str | None, schema: type[T], on_event: OnEvent,
                      system: str | None = None) -> T:
        if not repo_path:
            return await self.structured(task, schema, system=system)
        root = Path(repo_path)
        tools = RepoTools(root)
        client = self._client()
        guide = ("You are exploring a code repository with read-only tools (list_files, grep, read_file). "
                 "Investigate until you can answer, then reply with a short summary of your findings and no "
                 "further tool calls. Cite real file paths and line numbers you have seen.")
        msgs = self._msgs(task, f"{system}\n\n{guide}" if system else guide)
        for turn in range(self.max_turns):
            r = await self._complete(client, messages=msgs, tools=TOOLS, parallel_tool_calls=True)
            m = r.choices[0].message
            calls = m.tool_calls or []
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                       for c in calls]
            msgs.append(entry)
            if m.content and m.content.strip():
                await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "info",
                                             "message": short(m.content, 1200)}))
            if not calls:
                break
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                # off the event loop: a slow tool must not stall every other request
                out, label = await asyncio.to_thread(tools.run, c.function.name, args)
                await _maybe_await(on_event({"type": "tool", "name": c.function.name, "input": label,
                                             "output": short(out, 300)}))
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
        else:
            await _maybe_await(on_event({"type": "log", "stage": "agent", "level": "warn",
                                         "message": f"exploration hit {self.max_turns} turns; summarising findings"}))
        msgs.append({"role": "user", "content": "Now give your final answer as JSON matching the required schema."})
        # Tool-call history can be large; JSON mode needs only the text, so keep it but drop it if rejected.
        try:
            return await self._structured_msgs(msgs, schema)
        except LLMError:
            notes = "\n".join(str(m.get("content", ""))[:1500] for m in msgs if m.get("role") in ("assistant", "tool"))
            return await self.structured(f"{task}\n\nFindings from exploring the repository:\n{notes[-30000:]}",
                                         schema, system=system)


class RepoTools:
    """The three read-only tools, confined to ``root``."""

    def __init__(self, root: Path):
        self.root = root
        self._files: list[str] | None = None

    def files(self) -> list[str]:
        if self._files is None:
            from forge import repo as repo_mod
            self._files = repo_mod.list_files(self.root)
        return self._files

    def _match(self, glob: str | None) -> list[str]:
        if not glob:
            return self.files()
        import fnmatch
        pats = [glob] + ([glob[3:]] if glob.startswith("**/") else [])
        return [f for f in self.files() if any(fnmatch.fnmatch(f, p) for p in pats)]

    def run(self, name: str, args: dict) -> tuple[str, str]:
        """(tool output for the model, short label for the UI)."""
        try:
            if name == "list_files":
                fs = self._match(args.get("glob"))
                body = "\n".join(fs[:500]) + (f"\n… {len(fs) - 500} more" if len(fs) > 500 else "")
                return (body or "no files match"), (args.get("glob") or "*")
            if name == "grep":
                pat = str(args.get("pattern", ""))
                if len(pat) > MAX_PATTERN:
                    return f"pattern too long (max {MAX_PATTERN} chars)", pat[:80]
                # The pattern comes from the model, which reads untrusted repo text: use `regex` for its
                # matching timeout so a catastrophic pattern can't pin a thread.
                import regex
                try:
                    rx = regex.compile(pat)
                except regex.error as e:
                    return f"invalid regex: {e}", pat
                from forge import repo as repo_mod
                hits: list[str] = []
                deadline = time.monotonic() + GREP_BUDGET_S
                for f in self._match(args.get("glob")):
                    for i, line in enumerate(repo_mod.read_text(self.root, f).splitlines(), 1):
                        left = deadline - time.monotonic()
                        if left <= 0:
                            hits.append(f"… search stopped after {GREP_BUDGET_S:g}s; narrow the pattern or glob")
                            return "\n".join(hits), f"{pat}  {args.get('glob') or ''}".strip()
                        try:
                            found = rx.search(line[:MAX_LINE_CHARS], timeout=min(left, GREP_LINE_BUDGET_S))
                        except TimeoutError:
                            return ("pattern too slow (catastrophic backtracking?); simplify it",
                                    f"{pat}  {args.get('glob') or ''}".strip())
                        if found:
                            hits.append(f"{f}:{i}: {line.strip()[:200]}")
                            if len(hits) >= MAX_GREP_HITS:
                                break
                    if len(hits) >= MAX_GREP_HITS:
                        hits.append("… more matches; narrow the pattern or glob")
                        break
                return ("\n".join(hits) or "no matches"), f"{pat}  {args.get('glob') or ''}".strip()
            if name == "read_file":
                from forge import repo as repo_mod
                path = str(args.get("path", ""))
                path = path[2:] if path.startswith("./") else path
                if repo_mod.safe_path(self.root, path) is None:
                    return "error: path is outside the repository or not readable", path
                text = repo_mod.read_text(self.root, path)
                if not text:
                    return "error: file not found or empty", path
                lines = text.splitlines()
                off = max(1, int(args.get("offset") or 1))
                lim = max(1, min(MAX_READ_LINES, int(args.get("limit") or MAX_READ_LINES)))
                chunk = lines[off - 1: off - 1 + lim]
                body = "\n".join(f"{off + i:>5}  {ln}" for i, ln in enumerate(chunk))[:MAX_TOOL_CHARS]
                if off - 1 + lim < len(lines):
                    body += f"\n… {len(lines) - (off - 1 + lim)} more lines (use offset)"
                return body, path + (f" @{off}" if off > 1 else "")
            return f"unknown tool {name}", name
        except Exception as e:  # noqa: BLE001 - a tool error goes back to the model, not up the stack
            return f"error: {e}", name


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _err(e: Exception) -> str:
    msg = getattr(e, "message", None) or str(e)
    code = getattr(e, "status_code", None)
    return f"{code} {msg}" if code else msg


_CHAT_RX = re.compile(r"^(gpt-|o\d|chatgpt-)")
_SKIP_RX = re.compile(r"(audio|realtime|transcribe|tts|image|embedding|search|moderation|instruct|dall-e|whisper|codex-mini|computer-use)")


async def models(settings=None) -> list[dict]:
    """Chat-capable models the configured key can use. For a custom base URL every listed model is kept."""
    s = settings
    if s is None:
        from forge import config
        s = config.load_settings()
    custom = bool((getattr(s, "openai_base_url", "") or "").strip())
    if not api_key(s) and not custom:
        return []
    page = await make_client(s, timeout=8).models.list()
    out = []
    async for m in page:
        mid = m.id
        if not custom and (not _CHAT_RX.search(mid) or _SKIP_RX.search(mid)):
            continue
        out.append({"id": mid, "created": getattr(m, "created", 0) or 0})
    out.sort(key=lambda x: (-x["created"], x["id"]))
    return [{"id": x["id"], "label": x["id"]} for x in out]

