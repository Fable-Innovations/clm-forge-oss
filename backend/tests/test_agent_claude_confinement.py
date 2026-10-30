"""Claude provider: explore reads the repo only through the confined repo tools.

Regression for a verified leak: Claude Code's built-in Read accepted an absolute path outside the
repository (a README pointing at ../shared/labels.yaml was enough), so a cloned repo could steer the
analysis into reading any file the user can read."""
import asyncio

from pydantic import BaseModel

from forge.llm import claude as mod


class Finding(BaseModel):
    labels: list[str]


def test_hook_denies_everything_but_repo_tools():
    async def decide(name):
        return await mod._only_repo_tools({"tool_name": name}, "id", None)

    for name in mod.REPO_TOOLS + ["StructuredOutput"]:
        assert asyncio.run(decide(name)) == {}
    for name in ("Read", "Grep", "Glob", "Bash", "WebFetch", "Task", "mcp__other__read", "BrandNewTool"):
        out = asyncio.run(decide(name))["hookSpecificOutput"]
        assert out["permissionDecision"] == "deny" and out["hookEventName"] == "PreToolUse", name


def test_explore_options_disable_builtins_and_never_use_repo_as_cwd(tmp_path, monkeypatch):
    seen = {}

    async def fake_run(self, prompt, options, on_event=None, on_delta=None):
        seen["opts"] = options
        return {"labels": ["billing"]}, "", []
    monkeypatch.setattr(mod.ClaudeLLM, "_run", fake_run)
    out = asyncio.run(mod.ClaudeLLM().explore("task", str(tmp_path), Finding, lambda e: None))
    assert out.labels == ["billing"]
    o = seen["opts"]
    assert o.tools == []                                         # no built-in tools at all
    assert o.allowed_tools == mod.REPO_TOOLS
    assert {"Read", "Grep", "Glob", "Bash"} <= set(o.disallowed_tools)
    assert o.strict_mcp_config is True and list(o.mcp_servers) == [mod.REPO_SERVER]
    assert str(o.cwd) != str(tmp_path) and "forge-claude-" in str(o.cwd)
    (matcher,) = o.hooks["PreToolUse"]
    assert matcher.matcher is None and mod._only_repo_tools in matcher.hooks
    assert o.env["ANTHROPIC_API_KEY"] == ""


def test_repo_server_tools_are_confined(tmp_path, monkeypatch):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "a.py").write_text("x = 1\n")
    (tmp_path / "secret.txt").write_text("CANARY")
    captured = {}
    import claude_agent_sdk
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server",
                        lambda name, version="1.0.0", tools=None: captured.setdefault("tools", tools))
    mod._repo_server(str(tmp_path / "repo"))
    by_name = {t.name: t for t in captured["tools"]}
    assert set(by_name) == {"list_files", "grep", "read_file"}

    def call(name, args):
        return asyncio.run(by_name[name].handler(args))

    assert "x = 1" in call("read_file", {"path": "a.py"})["content"][0]["text"]
    for p in ("../secret.txt", str(tmp_path / "secret.txt")):
        r = call("read_file", {"path": p})
        assert r["is_error"] and "CANARY" not in r["content"][0]["text"]
    assert "secret" not in call("list_files", {})["content"][0]["text"]
