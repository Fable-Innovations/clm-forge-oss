"""Which providers are usable right now and which models each one offers, asked of the provider itself.

Nothing here hard-codes model names: Claude's list comes from the local ``claude`` CLI (the same
list its /model picker shows), Ollama's from ``/api/tags``, OpenAI's from ``/v1/models``. Results are
cached briefly because the Claude probe starts a CLI process and the others are network calls.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

PROVIDERS = {
    "claude-subscription": "Claude (subscription)",
    "codex-subscription": "Codex (ChatGPT subscription)",
    "openai": "OpenAI",
    "ollama": "Ollama (local)",
}

_TTL = 600.0
_cache: dict[str, tuple[float, dict]] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _claude(_s) -> dict:
    from forge.llm import claude as claude_mod
    if not claude_mod.available():
        return {"available": False, "detail": "No claude CLI found. Install Claude Code and run `claude login`.",
                "models": []}
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    opts = ClaudeAgentOptions(env=claude_mod._env(), setting_sources=[], tools=[])
    async with ClaudeSDKClient(options=opts) as c:
        info = await c.get_server_info() or {}
    models = [{"id": m["value"], "label": m.get("displayName") or m["value"], "description": m.get("description")}
              for m in info.get("models", []) if m.get("value")]
    return {"available": True, "detail": "Using your local claude CLI login.", "models": models}


async def _codex(_s) -> dict:
    from forge.llm import codex as codex_mod
    ok, detail = await codex_mod.login_status()
    if not ok:
        return {"available": False, "detail": detail, "models": []}
    refused = await codex_mod.preflight()
    if refused:
        return {"available": False, "detail": refused, "models": []}
    return {"available": True, "detail": detail, "models": await codex_mod.models()}


async def _ollama(s) -> dict:
    from forge.llm import ollama as ollama_mod
    names = await asyncio.to_thread(ollama_mod.models, s.ollama_url)
    if not names:
        return {"available": False, "detail": f"Ollama isn't reachable at {s.ollama_url}, or has no models pulled.",
                "models": []}
    return {"available": True, "detail": f"Ollama at {s.ollama_url}.", "models": [{"id": n, "label": n} for n in names]}


async def _openai(s) -> dict:
    from forge.llm import openai as openai_mod
    custom = (s.openai_base_url or "").strip()
    if not openai_mod.api_key(s) and not custom:
        return {"available": False, "detail": "No API key. Add one below or set OPENAI_API_KEY.", "models": []}
    models = await openai_mod.models(s)
    where = custom or "api.openai.com"
    return {"available": bool(models), "detail": f"{len(models)} models at {where}." if models else f"No chat models listed at {where}.",
            "models": models}


_PROBES = {"claude-subscription": _claude, "codex-subscription": _codex, "ollama": _ollama, "openai": _openai}


def _key(provider: str, s) -> str:
    if provider == "ollama":
        return f"ollama|{s.ollama_url}"
    if provider == "openai":
        from forge.llm import openai as openai_mod
        return f"openai|{s.openai_base_url}|{hash(openai_mod.api_key(s))}"
    return provider


async def provider_info(provider: str, settings=None, refresh: bool = False) -> dict[str, Any]:
    from forge import config
    s = settings or config.load_settings()
    k = _key(provider, s)
    hit = _cache.get(k)
    if hit and not refresh and time.monotonic() - hit[0] < _TTL:
        return hit[1]
    lock = _locks.setdefault(k, asyncio.Lock())
    async with lock:
        hit = _cache.get(k)
        if hit and not refresh and time.monotonic() - hit[0] < _TTL:
            return hit[1]
        try:
            info = await asyncio.wait_for(_PROBES[provider](s), timeout=20)
        except Exception as e:  # noqa: BLE001 - report, don't fail the page
            msg = getattr(e, "message", None) or str(e) or type(e).__name__
            info = {"available": False, "detail": f"Couldn't list models: {msg}", "models": []}
        info = {"id": provider, "label": PROVIDERS[provider], **info}
        # Failures are retried sooner than successes.
        _cache[k] = (time.monotonic() - (0 if info["available"] else _TTL - 30), info)
        return info


async def all_providers(settings=None, refresh: bool = False) -> list[dict]:
    return list(await asyncio.gather(*(provider_info(p, settings, refresh) for p in PROVIDERS)))


async def default_model(provider: str, settings=None) -> str | None:
    info = await provider_info(provider, settings)
    return info["models"][0]["id"] if info["models"] else None
