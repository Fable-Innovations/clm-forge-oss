"""Ollama provider. No tool use: `explore` analyses a deterministic repo digest (forge.repo) and
requests schema-constrained JSON (`format=<json schema>`), thinking disabled for speed."""
from __future__ import annotations

import inspect

import httpx
from pydantic import ValidationError

from forge.llm.base import LLMError, OnDelta, OnEvent, T, short, traced


async def _maybe_await(x):
    if inspect.isawaitable(x):
        await x


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str = "qwen3:8b", url: str = "http://127.0.0.1:11434", num_ctx: int = 32768,
                 temperature: float = 0.2):
        self.model, self.url, self.num_ctx, self.temperature = model, url, num_ctx, temperature

    def _client(self):
        from ollama import AsyncClient
        return AsyncClient(host=self.url, timeout=900)

    @staticmethod
    def _msgs(prompt_or_msgs, system: str | None) -> list[dict]:
        msgs = [{"role": "user", "content": prompt_or_msgs}] if isinstance(prompt_or_msgs, str) else list(prompt_or_msgs)
        return ([{"role": "system", "content": system}] if system else []) + msgs

    @traced("structured")
    async def structured(self, prompt: str, schema: type[T], system: str | None = None,
                         temperature: float | None = None) -> T:
        sys_ = (system or "") + "\n\nRespond only with JSON matching the required schema."
        msgs = self._msgs(prompt, sys_.strip())
        last = None
        for _ in range(3):
            try:
                r = await self._client().chat(
                    model=self.model, messages=msgs, format=schema.model_json_schema(), think=False,
                    options={"temperature": self.temperature if temperature is None else temperature,
                             "num_ctx": self.num_ctx})
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"ollama call failed: {e}") from e
            content = r.message.content or ""
            try:
                return schema.model_validate_json(content)
            except ValidationError as e:
                last = short(str(e), 1200)
                msgs = msgs + [{"role": "assistant", "content": content},
                               {"role": "user", "content": f"That failed validation: {last}. Return corrected JSON."}]
        raise LLMError(f"ollama structured output invalid: {last}")

    @traced("chat")
    async def chat(self, messages: list[dict], system: str | None = None, on_delta: OnDelta | None = None) -> str:
        msgs = self._msgs(messages, system)
        opts = {"temperature": 0.4, "num_ctx": self.num_ctx}
        if on_delta is None:
            r = await self._client().chat(model=self.model, messages=msgs, think=False, options=opts)
            return (r.message.content or "").strip()
        out = []
        async for part in await self._client().chat(model=self.model, messages=msgs, think=False, options=opts,
                                                     stream=True):
            d = part.message.content or ""
            if d:
                out.append(d)
                await _maybe_await(on_delta(d))
        return "".join(out).strip()

    @traced("explore")
    async def explore(self, task: str, repo_path: str | None, schema: type[T], on_event: OnEvent,
                      system: str | None = None) -> T:
        if not repo_path:
            return await self.structured(task, schema, system=system)
        from pathlib import Path

        from forge import repo as repo_mod
        sc = repo_mod.scan(Path(repo_path))
        await _maybe_await(on_event({"type": "tool", "name": "scan", "input": repo_path,
                                     "output": f"{len(sc.files)} files; languages {sc.languages}"}))
        for c in sc.candidates[:25]:
            await _maybe_await(on_event({"type": "tool", "name": "grep", "input": c.category,
                                         "output": f"{c.path}:{c.line}"}))
        prompt = (f"{task}\n\nYou cannot open files. Everything you know about the repository is in this digest; "
                  "only cite file paths and line numbers that appear in it.\n\n" + sc.digest())
        return await self.structured(prompt, schema, system=system)


def models(url: str) -> list[str]:
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []
