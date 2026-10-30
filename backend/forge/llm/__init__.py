"""LLM providers. `get_llm(settings, provider=None, model=None)` returns one implementing `LLM`."""
from __future__ import annotations

from forge.llm.base import LLM, LLMError  # noqa: F401


def get_llm(settings=None, provider: str | None = None, model: str | None = None) -> LLM:
    """Model ids come from the provider's own listing (see ``catalog``); an experiment stores the id it
    was created with. The subscription CLIs (Claude, Codex) have a usable implicit default."""
    from forge import config

    s = settings or config.load_settings()
    provider = provider or s.provider
    m = model or (s.model if s.provider == provider else None) or None
    if provider == "claude-subscription":
        from forge.llm.claude import ClaudeLLM
        return ClaudeLLM(model=m or "default", max_turns=s.explore_max_turns)
    if provider == "codex-subscription":
        from forge.llm.codex import CodexLLM
        return CodexLLM(model=m or "default")
    if not m:
        raise LLMError(f"no model chosen for {provider}; pick one in Settings or when creating the experiment")
    if provider == "openai":
        from forge.llm.openai import OpenAILLM
        return OpenAILLM(model=m, settings=s, max_turns=s.explore_max_turns)
    if provider == "ollama":
        from forge.llm.ollama import OllamaLLM
        return OllamaLLM(model=m, url=s.ollama_url)
    raise ValueError(f"unknown provider {provider!r}")
