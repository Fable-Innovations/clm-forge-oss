"""OpenAI provider: structured output (strict and JSON-mode fallback), the read-only explore loop,
repo confinement of its tools, model listing, and the write-only API key in settings."""
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient  # noqa: F401

import forge_client  # noqa: E402
from pydantic import BaseModel

from test_agent_fakes import isolated  # noqa: F401  (fixture)


class Answer(BaseModel):
    label: str
    files: list[str]


def msg(content=None, tool_calls=None, parsed=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls, parsed=parsed, refusal=None))])


def call(id_, name, args):
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))


class FakeCompletions:
    def __init__(self, creates, parse_error=None):
        self.creates = list(creates)
        self.parse_error = parse_error
        self.seen = []

    async def create(self, **kw):
        self.seen.append(kw)
        return self.creates.pop(0)

    async def parse(self, **kw):
        if self.parse_error:
            raise self.parse_error
        return msg(parsed=kw["response_format"](label="billing", files=["a.py"]))


def llm_with(completions):
    from forge.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-test", settings=NS(openai_api_key="sk-test", openai_base_url=""), max_turns=5)
    llm._client = lambda: NS(chat=NS(completions=completions))
    return llm


async def test_structured_strict_path():
    out = await llm_with(FakeCompletions([])).structured("q", Answer)
    assert out == Answer(label="billing", files=["a.py"])


async def test_structured_falls_back_to_json_mode_and_reasks_on_invalid():
    fc = FakeCompletions([msg('{"label": 3}'), msg('```json\n{"label": "refunds", "files": []}\n```')],
                         parse_error=ValueError("schema not strict-compatible"))
    out = await llm_with(fc).structured("q", Answer)
    assert out.label == "refunds"
    assert fc.seen[0]["response_format"] == {"type": "json_object"}
    assert "failed validation" in fc.seen[1]["messages"][-2]["content"] or "failed validation" in str(fc.seen[1]["messages"])


async def test_explore_runs_tools_inside_repo_only(tmp_path):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "router.py").write_text("def route(t):\n    return llm('classify ticket')\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOKEN=hunter2")
    (root / "app" / "link.txt").symlink_to(secret)

    fc = FakeCompletions([
        msg(tool_calls=[call("1", "grep", {"pattern": r"llm\(", "glob": "**/*.py"}),
                        call("2", "read_file", {"path": "./app/router.py"})]),
        msg(tool_calls=[call("3", "read_file", {"path": "../secret.txt"}),
                        call("4", "read_file", {"path": "app/link.txt"}),
                        call("5", "list_files", {})]),
        msg(content="The router calls an LLM to classify tickets."),
    ])
    events = []
    out = await llm_with(fc).explore("find decisions", str(root), Answer, events.append)
    assert out.label == "billing"
    tools = [e for e in events if e["type"] == "tool"]
    assert [t["name"] for t in tools] == ["grep", "read_file", "read_file", "read_file", "list_files"]
    tool_msgs = [m for m in fc.seen[-1]["messages"] if m["role"] == "tool"]
    assert "app/router.py:2:" in tool_msgs[0]["content"]
    assert "classify ticket" in tool_msgs[1]["content"]
    assert "hunter2" not in json.dumps(fc.seen[-1]["messages"])
    assert "outside the repository" in tool_msgs[2]["content"]
    assert "link.txt" not in tool_msgs[4]["content"]          # symlink never listed
    assert all("tools" in kw for kw in fc.seen)                # every explore turn offered tools


async def test_explore_stops_at_max_turns():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "x.py").write_text("x = 1\n")
        fc = FakeCompletions([msg(tool_calls=[call(str(i), "list_files", {})]) for i in range(5)])
        events = []
        out = await llm_with(fc).explore("t", d, Answer, events.append)
        assert out.label == "billing" and len(fc.seen) == 5
        assert any(e.get("level") == "warn" for e in events)


async def test_model_listing_filters_non_chat(monkeypatch):
    from forge.llm import openai as mod

    class Page:
        def __init__(self, ids):
            self.items = [NS(id=i, created=n) for n, i in enumerate(ids)]

        def __aiter__(self):
            async def gen():
                for x in self.items:
                    yield x
            return gen()

    ids = ["gpt-x", "gpt-x-mini", "text-embedding-3-large", "gpt-x-realtime", "dall-e-3", "whisper-1", "o9"]
    monkeypatch.setattr(mod, "make_client", lambda s, timeout=8: NS(models=NS(list=lambda: _aw(Page(ids)))))
    got = [m["id"] for m in await mod.models(NS(openai_api_key="sk", openai_base_url=""))]
    assert got == ["o9", "gpt-x-mini", "gpt-x"]
    # custom server: keep everything it lists
    got = [m["id"] for m in await mod.models(NS(openai_api_key="", openai_base_url="http://localhost:1234/v1"))]
    assert len(got) == len(ids)
    assert await mod.models(NS(openai_api_key="", openai_base_url="")) == []


async def _aw(x):
    return x


def test_get_llm_requires_model_for_openai(isolated):  # noqa: F811
    from forge.llm import LLMError, get_llm
    with pytest.raises(LLMError):
        get_llm(provider="openai", model=None)
    assert get_llm(provider="openai", model="gpt-x").provider == "openai"
    assert get_llm(provider="claude-subscription").model == "default"


def test_api_key_is_write_only(isolated, monkeypatch):  # noqa: F811
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from forge.api import create_app
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    from forge import config
    c = forge_client.client(create_app(Orchestrator(store=Store(config.DB_PATH))))
    r = c.put("/api/settings", json={"provider": "openai", "model": "gpt-x", "openai_api_key": "sk-secret"}).json()
    assert "openai_api_key" not in r and r["openai_api_key_set"] is True
    assert "sk-secret" not in c.get("/api/settings").text
    # sending the masked settings back ("" key) keeps the stored key
    c.put("/api/settings", json={**r, "openai_api_key": ""})
    assert config.load_settings().openai_api_key == "sk-secret"
    assert oct(config.SETTINGS_PATH.stat().st_mode & 0o777) == "0o600"
    # null clears it
    assert c.put("/api/settings", json={"openai_api_key": None}).json()["openai_api_key_set"] is False
    assert c.post("/api/projects", json={"task": "classify tickets please", "provider": "nope",
                                          "encoder": "qwen3-4b"}).status_code == 422


def test_saved_key_only_goes_to_the_url_it_was_entered_for(monkeypatch):
    from forge.llm.openai import api_key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    official = NS(openai_api_key="sk-saved", openai_api_key_url="", openai_base_url="")
    assert api_key(official) == "sk-saved"
    # base URL changed after the key was saved: the key must not follow it
    moved = NS(openai_api_key="sk-saved", openai_api_key_url="", openai_base_url="https://evil.example/v1")
    assert api_key(moved) == ""
    # the environment key is for api.openai.com only
    assert api_key(NS(openai_api_key="", openai_api_key_url="", openai_base_url="http://localhost:1234/v1")) == ""
    assert api_key(NS(openai_api_key="", openai_api_key_url="", openai_base_url="")) == "sk-env"
    # a key entered for a custom server is used there (trailing slash ignored)
    local = NS(openai_api_key="sk-local", openai_api_key_url="http://localhost:1234/v1/", openai_base_url="http://localhost:1234/v1")
    assert api_key(local) == "sk-local"


def test_api_binds_key_to_base_url(isolated, monkeypatch):  # noqa: F811
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from forge.api import create_app
    from forge.llm.openai import api_key
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    from forge import config
    c = forge_client.client(create_app(Orchestrator(store=Store(config.DB_PATH))))
    r = c.put("/api/settings", json={"openai_api_key": "sk-secret"}).json()
    assert r["openai_api_key_applies"] is True
    # redirecting the base URL (even with a forged openai_api_key_url) does not carry the key along
    r = c.put("/api/settings", json={"openai_base_url": "https://evil.example/v1",
                                     "openai_api_key_url": "https://evil.example/v1"}).json()
    assert r["openai_api_key_applies"] is False
    assert api_key(config.load_settings()) == ""


def test_grep_survives_catastrophic_regex(tmp_path):
    import time
    from forge.llm import openai as mod
    (tmp_path / "a.txt").write_text(("a" * 40 + "b\n") * 3)
    tools = mod.RepoTools(tmp_path)
    t0 = time.monotonic()
    out, _ = tools.run("grep", {"pattern": r"(a|aa)+$"})   # exponential even for the regex module
    # one line's slice (plus the regex module's own overshoot, under a loaded machine), never the whole budget
    assert time.monotonic() - t0 < mod.GREP_LINE_BUDGET_S * 3 + 1
    assert "too slow" in out
    assert "too long" in tools.run("grep", {"pattern": "a" * 600})[0]
    assert "a.txt:1:" in tools.run("grep", {"pattern": "b$"})[0]


async def test_slow_tool_does_not_block_event_loop(tmp_path, monkeypatch):
    import asyncio
    import time as _t
    from forge.llm import openai as mod
    (tmp_path / "x.py").write_text("x = 1\n")
    monkeypatch.setattr(mod.RepoTools, "run", lambda self, n, a: (_t.sleep(0.5), ("ok", n))[1])
    fc = FakeCompletions([msg(tool_calls=[call("1", "list_files", {})]), msg(content="done")])
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1
    t = asyncio.create_task(ticker())
    await llm_with(fc).explore("t", str(tmp_path), Answer, lambda e: None)
    t.cancel()
    assert ticks >= 5   # the loop kept running while the tool slept
